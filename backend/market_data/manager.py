"""Coordinación de conexión y validación de lotes de datos de mercado."""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from market_data.models import DataQualityResult, MarketDataCandle
from market_data.provider import MarketDataProvider
from market_data.quality_gate import DataQualityGate

if TYPE_CHECKING:
    from analysis.economic_calendar import EconomicCalendarManager
    from analysis.indicators import TechnicalIndicatorEngine
    from analysis.market_regime import MarketRegimeDetector

logger = structlog.get_logger(__name__)


class MarketDataManager:
    def __init__(
        self,
        provider: MarketDataProvider | None = None,
        quality_gate: DataQualityGate | None = None,
        indicator_engine: TechnicalIndicatorEngine | None = None,
        regime_detector: MarketRegimeDetector | None = None,
        economic_calendar: EconomicCalendarManager | None = None,
        history_size: int = 240,
        setup_error: str | None = None,
    ) -> None:
        from analysis.economic_calendar import EconomicCalendarManager
        from analysis.indicators import TechnicalIndicatorEngine
        from analysis.market_regime import MarketRegimeDetector

        if history_size <= 0:
            raise ValueError("history_size debe ser un entero positivo.")
        self._provider = provider
        self._quality_gate = quality_gate or DataQualityGate()
        self._indicator_engine = indicator_engine or TechnicalIndicatorEngine()
        self._regime_detector = regime_detector or MarketRegimeDetector()
        self._economic_calendar = economic_calendar or EconomicCalendarManager()
        self._history_size = history_size
        self._candles: deque[MarketDataCandle] = deque(maxlen=history_size)
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._connected = False
        self._last_update: datetime | None = None
        self._last_result: DataQualityResult | None = None
        self._connection_error = setup_error
        self._recovery_candle: MarketDataCandle | None = None

    def get_status(self) -> dict:
        result = self._last_result
        reasons = list(result.reasons) if result else []
        latest_candle = self._candles[-1] if self._candles else None
        is_fresh = True
        if self._connected and latest_candle is not None:
            age_seconds = (
                datetime.now(timezone.utc)
                - latest_candle.timestamp.astimezone(timezone.utc)
            ).total_seconds()
            is_fresh = age_seconds <= max(latest_candle.interval_seconds * 2, 30)
            if not is_fresh:
                reasons.append("Último dato de mercado desactualizado; NO_TRADE.")
        if not self._connected:
            if self._connection_error:
                reasons.append(self._connection_error)
            elif self._provider is None:
                reasons.append("No hay un adaptador de proveedor configurado.")
            else:
                reasons.append("Feed de datos desconectado; NO_TRADE.")
        elif result is None:
            reasons.append("Esperando la primera vela validada; NO_TRADE.")
        return {
            "connected": self._connected,
            "provider": self._provider.name if self._provider else None,
            "last_update": self._last_update,
            "quality_status": (
                "OK"
                if result and result.accepted and self._connected and is_fresh
                else "NO_TRADE"
            ),
            "quality_accepted": bool(
                result and result.accepted and self._connected and is_fresh
            ),
            "fresh": is_fresh and self._connected,
            "quality_reasons": list(dict.fromkeys(reasons)),
            "connection_error": self._connection_error,
        }

    @property
    def has_provider(self) -> bool:
        return self._provider is not None

    @property
    def has_calendar_provider(self) -> bool:
        return self._economic_calendar.has_provider

    async def refresh_economic_calendar(self) -> None:
        try:
            await self._economic_calendar.refresh()
        finally:
            self._publish_snapshot()

    def ingest(self, candles: Sequence[MarketDataCandle]) -> DataQualityResult:
        batch = list(candles)
        result = self._quality_gate.evaluate(batch, feed_connected=self._connected)
        recovery_context: list[MarketDataCandle] | None = None
        if result.accepted and len(batch) == 1 and self._recovery_candle is not None:
            recovery_context = (
                [batch[0]]
                if batch[0].timestamp == self._recovery_candle.timestamp
                else [self._recovery_candle, batch[0]]
            )
            result = self._quality_gate.evaluate(
                recovery_context,
                feed_connected=self._connected,
            )
        elif result.accepted and len(batch) == 1 and self._candles:
            history = list(self._candles)
            latest = history[-1]
            if batch[0].timestamp == latest.timestamp and len(history) > 1:
                context = [history[-2], batch[0]]
            else:
                context = [latest, batch[0]]
            result = self._quality_gate.evaluate(context, feed_connected=self._connected)

        if (
            not result.accepted
            and len(batch) == 1
            and any("gap temporal" in reason for reason in result.reasons)
            and self._quality_gate.evaluate(batch, feed_connected=self._connected).accepted
        ):
            self._recovery_candle = batch[0]

        self._last_result = result
        if result.accepted:
            history = (
                list(recovery_context)
                if recovery_context is not None
                else list(self._candles)
            )
            if recovery_context is not None:
                self._candles.clear()
                self._recovery_candle = None
            for candle in batch:
                matching_index = next(
                    (
                        index
                        for index, existing in enumerate(history)
                        if existing.timestamp == candle.timestamp
                    ),
                    None,
                )
                if matching_index is None:
                    history.append(candle)
                else:
                    history[matching_index] = candle
            history.sort(key=lambda candle: candle.timestamp)
            self._candles.clear()
            self._candles.extend(history[-self._history_size :])
            self._last_update = max(candle.timestamp for candle in batch)
        self._publish_snapshot()
        return result

    def get_snapshot(self) -> dict[str, Any]:
        status = self.get_status()
        candles = list(self._candles)
        indicators = self._indicator_engine.calculate(
            candles,
            feed_connected=bool(status["fresh"]),
            data_quality_accepted=bool(status["quality_accepted"]),
        )
        currencies = self._symbol_currencies(candles[-1].symbol) if candles else None
        economic_calendar = self._economic_calendar.get_status(currencies=currencies)
        regime = self._regime_detector.detect(
            candles,
            feed_connected=bool(status["fresh"]),
            data_quality_accepted=bool(status["quality_accepted"]),
            indicators=indicators,
            economic_calendar=economic_calendar,
            spread_pct=(
                getattr(self._provider, "spread_pct", None)
                if self._provider is not None
                else None
            ),
        )
        return {
            "type": "market_data",
            "status": status,
            "indicators": indicators,
            "economic_calendar": economic_calendar,
            "market_regime": regime,
            "candles": [
                {
                    "symbol": candle.symbol,
                    "timestamp": candle.timestamp.isoformat(),
                    "interval_seconds": candle.interval_seconds,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
                for candle in self._candles
            ],
        }

    @staticmethod
    def _symbol_currencies(symbol: str) -> set[str] | None:
        normalized = symbol.upper().replace(":", "/")
        components = normalized.split("/")
        if len(components) == 2 and all(len(component) == 3 for component in components):
            return set(components)
        return None

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        queue.put_nowait(self.get_snapshot())
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def _publish_snapshot(self) -> None:
        snapshot = self.get_snapshot()
        for queue in tuple(self._subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(snapshot)

    async def run(self) -> None:
        if self._provider is None:
            raise RuntimeError("No hay un adaptador de proveedor de datos configurado.")

        self._recovery_candle = None
        run_failed = False
        try:
            await self._provider.connect()
            self._connected = True
            self._connection_error = None
            self._publish_snapshot()
            async for candles in self._provider.stream():
                result = self.ingest(candles)
                if not result.accepted:
                    logger.warning(
                        "market_data_quality_rejected",
                        provider=self._provider.name,
                        reasons=result.reasons,
                    )
        except asyncio.CancelledError:
            run_failed = True
            raise
        except Exception as exc:
            run_failed = True
            self._connection_error = "Error de conexión con el proveedor."
            logger.error(
                "market_data_provider_failed",
                provider=self._provider.name,
                error_type=type(exc).__name__,
            )
            raise
        finally:
            self._connected = False
            self._publish_snapshot()
            try:
                await self._provider.disconnect()
            except Exception as exc:
                self._connection_error = "Error al cerrar la conexión del proveedor."
                logger.error(
                    "market_data_provider_disconnect_failed",
                    provider=self._provider.name,
                    error_type=type(exc).__name__,
                )
                if not run_failed:
                    raise

    def mark_disconnected(self) -> None:
        """Permite a un adaptador indicar una desconexión detectada externamente."""
        self._connected = False
        self._connection_error = "Desconexión reportada por el proveedor."
        logger.warning(
            "market_data_provider_disconnected",
            provider=self._provider.name if self._provider else None,
        )
        self._publish_snapshot()
