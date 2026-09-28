"""Adaptador para velas REST y precios en vivo de Twelve Data."""
import asyncio
import json
import math
from collections import deque
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit

import httpx
import websockets

from market_data.models import MarketDataCandle

_INTERVAL_SECONDS = {
    "1min": 60,
    "5min": 300,
    "15min": 900,
    "30min": 1800,
    "45min": 2700,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "8h": 28800,
}
_TIME_SERIES_URL = "https://api.twelvedata.com/time_series"
_WEBSOCKET_URL = "wss://ws.twelvedata.com/v1/quotes/price"


class _WebSocket(Protocol):
    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...

    def __aiter__(self) -> AsyncIterator[str | bytes]: ...


class TwelveDataProvider:
    def __init__(
        self,
        api_key: str,
        symbol: str,
        interval: str = "1min",
        history_size: int = 500,
        websocket_url: str = _WEBSOCKET_URL,
    ) -> None:
        if not api_key.strip():
            raise ValueError("MARKET_DATA_API_KEY es obligatorio para Twelve Data.")
        if not symbol.strip():
            raise ValueError("MARKET_DATA_SYMBOL es obligatorio para Twelve Data.")
        if interval not in _INTERVAL_SECONDS:
            raise ValueError(
                "Twelve Data admite aquí intervalos intradía de 1min a 8h."
            )
        if (
            isinstance(history_size, bool)
            or not isinstance(history_size, int)
            or not 1 <= history_size <= 5000
        ):
            raise ValueError("MARKET_DATA_HISTORY_SIZE debe estar entre 1 y 5000.")
        parsed_websocket_url = urlsplit(websocket_url)
        if (
            parsed_websocket_url.scheme != "wss"
            or parsed_websocket_url.hostname != "ws.twelvedata.com"
            or parsed_websocket_url.path != "/v1/quotes/price"
            or parsed_websocket_url.query
        ):
            raise ValueError("Solo se permite el WebSocket oficial de Twelve Data.")

        self._api_key = api_key.strip()
        self._symbol = symbol.strip()
        self._interval = interval
        self._interval_seconds = _INTERVAL_SECONDS[interval]
        self._history_size = history_size
        self._websocket_url = websocket_url
        self._seed: list[MarketDataCandle] = []
        self._socket: _WebSocket | None = None
        self._pending_quotes: deque[str | bytes] = deque()
        self._spread_pct: float | None = None

    @property
    def name(self) -> str:
        return "Twelve Data"

    @property
    def spread_pct(self) -> float | None:
        return self._spread_pct

    async def connect(self) -> None:
        self._pending_quotes.clear()
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                _TIME_SERIES_URL,
                params={
                    "symbol": self._symbol,
                    "interval": self._interval,
                    "outputsize": self._history_size,
                    "timezone": "UTC",
                    "order": "asc",
                },
                headers={"Authorization": f"apikey {self._api_key}"},
            )
            response.raise_for_status()
            payload = response.json()

        if not isinstance(payload, dict) or payload.get("status") == "error":
            message = payload.get("message", "respuesta inválida") if isinstance(payload, dict) else "respuesta inválida"
            raise RuntimeError(f"Twelve Data rechazó la consulta de velas: {message}")
        values = payload.get("values")
        if not isinstance(values, list) or not values:
            raise RuntimeError("Twelve Data no devolvió velas para el símbolo configurado.")
        self._seed = self._contiguous_suffix(
            self.parse_time_series(values, self._symbol, self._interval_seconds),
            self._interval_seconds,
        )
        if not self._seed:
            raise RuntimeError("Twelve Data no devolvió velas cronológicas utilizables.")

        socket_url = f"{self._websocket_url}?{urlencode({'apikey': self._api_key})}"
        self._socket = await websockets.connect(
            socket_url,
            ping_interval=None,
            close_timeout=5,
        )
        await self._socket.send(
            json.dumps(
                {
                    "action": "subscribe",
                    "params": {"symbols": self._symbol},
                }
            )
        )
        await self._confirm_subscription()

    async def stream(self) -> AsyncIterator[list[MarketDataCandle]]:
        if self._socket is None:
            raise RuntimeError("Twelve Data no está conectado.")

        yield self._seed
        current = self._seed[-1]
        receive_task: asyncio.Task[str | bytes] | None = None
        heartbeat_task = asyncio.create_task(self._send_heartbeat())
        try:
            while True:
                if self._pending_quotes:
                    raw_message = self._pending_quotes.popleft()
                else:
                    if receive_task is None:
                        receive_task = asyncio.create_task(self._socket.recv())
                    done, _ = await asyncio.wait(
                        (receive_task, heartbeat_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if heartbeat_task in done:
                        await heartbeat_task
                        heartbeat_task = asyncio.create_task(self._send_heartbeat())
                    if receive_task not in done:
                        continue
                    raw_message = receive_task.result()
                    receive_task = None

                message = json.loads(raw_message)
                if not isinstance(message, dict):
                    raise TypeError("Twelve Data envió un mensaje WebSocket inválido.")
                if message.get("event") != "price":
                    if message.get("status") == "error" or message.get("event") == "error":
                        raise RuntimeError("Twelve Data reportó un error en el feed WebSocket.")
                    continue

                current, candle_batch = self._aggregate_price_event(message, current)
                yield candle_batch
        finally:
            tasks = [heartbeat_task]
            if receive_task is not None:
                tasks.append(receive_task)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def disconnect(self) -> None:
        if self._socket is not None:
            await self._socket.close()
            self._socket = None

    async def _confirm_subscription(self) -> None:
        if self._socket is None:
            raise RuntimeError("Twelve Data no está conectado.")
        while True:
            raw_message = await asyncio.wait_for(self._socket.recv(), timeout=10)
            message = json.loads(raw_message)
            if not isinstance(message, dict):
                raise TypeError("Twelve Data envió un acuse de suscripción inválido.")
            if message.get("event") == "price":
                self._pending_quotes.append(raw_message)
                continue
            if message.get("event") == "subscribe-status":
                successful = message.get("success", [])
                failed = message.get("fails", [])
                if (
                    isinstance(successful, list)
                    and isinstance(failed, list)
                    and message.get("status") == "ok"
                    and not failed
                    and any(
                        isinstance(item, dict)
                        and isinstance(item.get("symbol"), str)
                        and item["symbol"].casefold() == self._symbol.casefold()
                        for item in successful
                    )
                ):
                    return
                raise RuntimeError("Twelve Data no aceptó la suscripción al símbolo.")
            if message.get("event") == "error" or message.get("status") == "error":
                raise RuntimeError("Twelve Data rechazó la suscripción WebSocket.")

    async def _send_heartbeat(self) -> None:
        await asyncio.sleep(10)
        if self._socket is not None:
            await self._socket.send(json.dumps({"action": "heartbeat"}))

    def _aggregate_price_event(
        self,
        message: dict[str, Any],
        current: MarketDataCandle,
    ) -> tuple[MarketDataCandle, list[MarketDataCandle]]:
        quote = self._parse_price_event(message)
        self._spread_pct = quote["spread_pct"]
        bucket_epoch = int(quote["timestamp"].timestamp()) // self._interval_seconds
        bucket_start = datetime.fromtimestamp(
            bucket_epoch * self._interval_seconds,
            tz=timezone.utc,
        )
        if bucket_start < current.timestamp:
            return current, [
                MarketDataCandle(
                    symbol=current.symbol,
                    timestamp=bucket_start,
                    interval_seconds=current.interval_seconds,
                    open=quote["price"],
                    high=quote["price"],
                    low=quote["price"],
                    close=quote["price"],
                    volume=0,
                )
            ]
        if bucket_start == current.timestamp:
            current = MarketDataCandle(
                symbol=current.symbol,
                timestamp=current.timestamp,
                interval_seconds=current.interval_seconds,
                open=current.open,
                high=max(current.high, quote["price"]),
                low=min(current.low, quote["price"]),
                close=quote["price"],
                volume=0,
            )
        else:
            current = MarketDataCandle(
                symbol=self._symbol,
                timestamp=bucket_start,
                interval_seconds=self._interval_seconds,
                open=quote["price"],
                high=quote["price"],
                low=quote["price"],
                close=quote["price"],
                volume=0,
            )
        return current, [current]

    @staticmethod
    def parse_time_series(
        values: list[dict[str, Any]],
        symbol: str,
        interval_seconds: int,
    ) -> list[MarketDataCandle]:
        candles = []
        for item in values:
            if not isinstance(item, dict):
                raise TypeError("Twelve Data devolvió una vela con formato inválido.")
            timestamp = datetime.fromisoformat(str(item["datetime"]))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            volume_value = item.get("volume")
            candles.append(
                MarketDataCandle(
                    symbol=symbol,
                    timestamp=timestamp.astimezone(timezone.utc),
                    interval_seconds=interval_seconds,
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(volume_value) if volume_value not in (None, "") else 0,
                )
            )
        return sorted(candles, key=lambda candle: candle.timestamp)

    @staticmethod
    def _contiguous_suffix(
        candles: list[MarketDataCandle],
        interval_seconds: int,
    ) -> list[MarketDataCandle]:
        if not candles:
            return []
        start = len(candles) - 1
        for index in range(len(candles) - 1, 0, -1):
            elapsed = (candles[index].timestamp - candles[index - 1].timestamp).total_seconds()
            if elapsed <= 0 or elapsed > interval_seconds + 1:
                break
            start = index - 1
        return candles[start:]

    def _parse_price_event(self, message: dict[str, Any]) -> dict[str, Any]:
        symbol = message.get("symbol")
        if not isinstance(symbol, str) or symbol.casefold() != self._symbol.casefold():
            raise RuntimeError("Twelve Data envió un precio de un símbolo inesperado.")
        if isinstance(message.get("price"), bool) or isinstance(
            message.get("timestamp"), bool
        ):
            raise TypeError("Twelve Data envió un precio o timestamp no válido.")
        try:
            price = float(message["price"])
            timestamp_value = float(message["timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Twelve Data envió un precio sin timestamp o valor válido.") from exc
        if not math.isfinite(price) or price <= 0 or not math.isfinite(timestamp_value):
            raise RuntimeError("Twelve Data envió un precio o timestamp no válido.")
        spread_pct = None
        if "bid" in message and "ask" in message:
            try:
                bid = float(message["bid"])
                ask = float(message["ask"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Twelve Data envió bid/ask con formato inválido.") from exc
            midpoint = (bid + ask) / 2
            spread_pct = (
                (ask - bid) / midpoint * 100
                if math.isfinite(bid)
                and math.isfinite(ask)
                and bid > 0
                and ask >= bid
                and midpoint > 0
                else math.nan
            )
        return {
            "price": price,
            "timestamp": datetime.fromtimestamp(timestamp_value, tz=timezone.utc),
            "spread_pct": spread_pct,
        }
