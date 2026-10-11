import asyncio
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from analysis.indicators import TechnicalIndicatorEngine
from api.main import app
from market_data.manager import MarketDataManager
from market_data.models import MarketDataCandle
from market_data.quality_gate import DataQualityGate
from market_data.twelve_data import TwelveDataProvider


def candle(timestamp: datetime, close: float = 100.0) -> MarketDataCandle:
    return MarketDataCandle(
        symbol="TEST",
        timestamp=timestamp,
        interval_seconds=60,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=10,
    )


def test_accepts_ordered_candles_when_feed_is_connected():
    start = datetime.now(timezone.utc) - timedelta(seconds=60)
    result = DataQualityGate().evaluate(
        [candle(start), candle(start + timedelta(seconds=60), 101)],
        feed_connected=True,
    )
    assert result.accepted is True
    assert result.status == "OK"
    assert result.reasons == ()


def test_rejects_disconnected_feed_and_empty_data():
    result = DataQualityGate().evaluate([], feed_connected=False)
    assert result.accepted is False
    assert result.status == "NO_TRADE"
    assert any("desconectado" in reason for reason in result.reasons)
    assert any("No hay velas" in reason for reason in result.reasons)


def test_rejects_duplicate_and_out_of_order_timestamps():
    start = datetime.now(timezone.utc)
    result = DataQualityGate().evaluate(
        [candle(start), candle(start), candle(start - timedelta(seconds=60))],
        feed_connected=True,
    )
    assert result.accepted is False
    assert any("duplicados" in reason for reason in result.reasons)
    assert any("fuera de orden" in reason for reason in result.reasons)


def test_rejects_time_gaps_and_price_outliers():
    start = datetime.now(timezone.utc)
    result = DataQualityGate().evaluate(
        [candle(start), candle(start + timedelta(seconds=120), 150)],
        feed_connected=True,
    )
    assert result.accepted is False
    assert any("gap temporal" in reason for reason in result.reasons)
    assert any("outlier" in reason for reason in result.reasons)


def test_manager_stays_no_trade_without_provider():
    manager = MarketDataManager()
    result = manager.ingest([candle(datetime.now(timezone.utc))])
    status = manager.get_status()
    assert result.accepted is False
    assert status["connected"] is False
    assert status["quality_accepted"] is False
    assert status["provider"] is None


def test_status_and_latest_signal_expose_no_trade_without_provider():
    with TestClient(app) as client:
        status = client.get("/api/status/").json()["data_feed"]
        signal = client.get("/api/signals/latest").json()
        with client.websocket_connect("/api/market-data/ws") as websocket:
            snapshot = websocket.receive_json()
    assert status["connected"] is False
    assert status["quality_accepted"] is False
    assert signal["signal"] == "NO_TRADE"
    assert signal["should_trade"] is False
    assert signal["market_regime"] == "unknown"
    assert any("feed externo está desactivado" in reason for reason in signal["reasons"])
    assert any("Calendario económico" in reason for reason in signal["reasons"])
    assert snapshot["type"] == "market_data"
    assert snapshot["status"]["connected"] is False
    assert snapshot["candles"] == []
    assert snapshot["economic_calendar"]["available"] is False
    assert snapshot["market_regime"]["trade_blocked"] is True


def test_rejects_invalid_ohlcv_and_naive_timestamp():
    start = datetime.now(timezone.utc)
    invalid_candle = MarketDataCandle(
        symbol="TEST",
        timestamp=start,
        interval_seconds=60,
        open=100,
        high=99,
        low=98,
        close=100,
        volume=10,
    )
    naive_candle = candle(start.replace(tzinfo=None))
    gate = DataQualityGate()
    invalid_result = gate.evaluate([invalid_candle], feed_connected=True)
    naive_result = gate.evaluate([naive_candle], feed_connected=True)
    assert invalid_result.accepted is False
    assert any("OHLC" in reason for reason in invalid_result.reasons)
    assert naive_result.accepted is False
    assert any("zona horaria" in reason for reason in naive_result.reasons)


def test_provider_stream_is_validated_and_disconnects_cleanly():
    start = datetime.now(timezone.utc) - timedelta(seconds=60)

    class Provider:
        name = "test-provider"
        disconnected = False

        async def connect(self):
            return None

        async def stream(self):
            yield [candle(start), candle(start + timedelta(seconds=60), 101)]

        async def disconnect(self):
            self.disconnected = True

    provider = Provider()
    manager = MarketDataManager(provider=provider)
    queue = manager.subscribe()
    asyncio.run(manager.run())
    snapshot = queue.get_nowait()
    manager.unsubscribe(queue)
    assert provider.disconnected is True
    assert manager.get_status()["connected"] is False
    assert manager.get_status()["quality_status"] == "NO_TRADE"
    assert manager.get_status()["quality_accepted"] is False
    assert len(snapshot["candles"]) == 2


def test_manager_replaces_live_updates_for_the_current_candle():
    start = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(seconds=120)

    class Provider:
        name = "test-provider"
        disconnected = False

        async def connect(self):
            return None

        async def stream(self):
            yield [candle(start, 100), candle(start + timedelta(seconds=60), 101)]
            yield [candle(start + timedelta(seconds=60), 102)]

        async def disconnect(self):
            self.disconnected = True

    manager = MarketDataManager(provider=Provider())
    asyncio.run(manager.run())
    candles = manager.get_snapshot()["candles"]
    assert len(candles) == 2
    assert candles[-1]["close"] == 102


def test_rejects_non_numeric_market_data_without_raising():
    invalid_candle = MarketDataCandle(
        symbol="TEST",
        timestamp=datetime.now(timezone.utc),
        interval_seconds="60",
        open="invalid",
        high=101,
        low=99,
        close=100,
        volume=10,
    )
    result = DataQualityGate().evaluate([invalid_candle], feed_connected=True)
    assert result.accepted is False
    assert any("intervalo" in reason for reason in result.reasons)
    assert any("no finitos" in reason for reason in result.reasons)


def test_rejects_future_market_data_timestamps():
    future = datetime.now(timezone.utc) + timedelta(minutes=1)
    result = DataQualityGate().evaluate([candle(future)], feed_connected=True)
    assert result.accepted is False
    assert any("futuro" in reason for reason in result.reasons)


def test_technical_indicators_require_valid_feed_and_sufficient_history():
    start = datetime.now(timezone.utc)
    history = [candle(start + timedelta(seconds=60 * index), 100 + index) for index in range(60)]
    engine = TechnicalIndicatorEngine()
    result = engine.calculate(
        history,
        feed_connected=True,
        data_quality_accepted=True,
    )
    assert result["ready"] is True
    assert result["status"] == "OK"
    assert result["samples"] == 60
    assert result["values"]["sma_20"] == sum(c.close for c in history[-20:]) / 20
    assert 0 <= result["values"]["rsi_14"] <= 100
    assert result["values"]["macd"]["line"] > 0
    assert result["values"]["atr_14"] > 0
    assert result["values"]["price_action"]["structure"] == "higher_highs_and_lows"
    assert result["values"]["candle_pattern"] == "doji"
    assert result["values"]["ichimoku"]["senkou_span_b"] is not None

    blocked = engine.calculate(
        history[:20],
        feed_connected=True,
        data_quality_accepted=True,
    )
    assert blocked["ready"] is False
    assert blocked["status"] == "NO_TRADE"
    assert blocked["values"]["sma_50"] is None


def test_vwap_is_unavailable_without_reported_volume():
    start = datetime.now(timezone.utc)
    candles = [
        replace(candle(start + timedelta(seconds=60 * index)), volume=0)
        for index in range(52)
    ]
    engine = TechnicalIndicatorEngine()
    result = engine.calculate(
        candles,
        feed_connected=True,
        data_quality_accepted=True,
    )
    assert result["ready"] is True
    assert result["values"]["ichimoku"]["senkou_span_b"] is not None
    assert result["values"]["vwap_session"] is None


def test_twelve_data_rest_bars_are_sorted_and_normalized_to_utc():
    bars = TwelveDataProvider.parse_time_series(
        [
            {
                "datetime": "2026-09-27 18:02:00",
                "open": "102",
                "high": "103",
                "low": "101",
                "close": "102.5",
                "volume": "12",
            },
            {
                "datetime": "2026-09-27 18:01:00",
                "open": "101",
                "high": "102",
                "low": "100",
                "close": "101.5",
            },
        ],
        "AAPL",
        60,
    )
    assert [bar.timestamp.minute for bar in bars] == [1, 2]
    assert all(bar.timestamp.utcoffset() == timedelta(0) for bar in bars)
    assert bars[0].volume == 0
    assert bars[1].volume == 12


def test_twelve_data_provider_rejects_nonofficial_websocket_and_bad_interval():
    with pytest.raises(ValueError, match="oficial"):
        TwelveDataProvider(
            "secret",
            "EUR/USD",
            websocket_url="wss://attacker.example/ws",
        )
    with pytest.raises(ValueError, match="intervalos"):
        TwelveDataProvider("secret", "EUR/USD", interval="3min")


def test_twelve_data_provider_seeds_and_aggregates_official_price_events(monkeypatch):
    start = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
    rest_response = {
        "status": "ok",
        "values": [
            {
                "datetime": (start + timedelta(minutes=index)).strftime("%Y-%m-%d %H:%M:%S"),
                "open": str(100 + index),
                "high": str(101 + index),
                "low": str(99 + index),
                "close": str(100 + index),
                "volume": "20",
            }
            for index in range(2)
        ],
    }

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return rest_response

    class HttpClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, *, params, headers):
            assert url.endswith("/time_series")
            assert params["timezone"] == "UTC"
            assert headers["Authorization"] == "apikey test-key"
            return Response()

    class Socket:
        def __init__(self):
            self.sent = []
            self.messages = [
                json.dumps({
                    "event": "subscribe-status",
                    "status": "ok",
                    "success": [{"symbol": "TEST"}],
                    "fails": [],
                }),
                json.dumps({
                    "event": "price",
                    "symbol": "TEST",
                    "timestamp": int((start + timedelta(minutes=1, seconds=20)).timestamp()),
                    "price": "103",
                    "bid": "102",
                    "ask": "104",
                }),
                json.dumps({
                    "event": "price",
                    "symbol": "TEST",
                    "timestamp": int((start + timedelta(minutes=2, seconds=5)).timestamp()),
                    "price": "104",
                    "bid": "103.9",
                    "ask": "104.1",
                }),
            ]
            self.message_index = 0

        async def send(self, message):
            self.sent.append(json.loads(message))

        async def recv(self):
            message = self.messages[self.message_index]
            self.message_index += 1
            return message

        async def close(self):
            return None

    socket = Socket()

    async def connect_socket(url, **kwargs):
        assert url.startswith("wss://ws.twelvedata.com/v1/quotes/price?apikey=")
        return socket

    monkeypatch.setattr("market_data.twelve_data.httpx.AsyncClient", HttpClient)
    monkeypatch.setattr("market_data.twelve_data.websockets.connect", connect_socket)

    async def consume():
        provider = TwelveDataProvider("test-key", "TEST", history_size=10)
        await provider.connect()
        stream = provider.stream()
        batches = [await anext(stream), await anext(stream), await anext(stream)]
        await stream.aclose()
        spread_pct = provider.spread_pct
        await provider.disconnect()
        return batches, socket.sent, spread_pct

    batches, sent, spread_pct = asyncio.run(consume())
    assert len(batches[0]) == 2
    assert batches[1][0].timestamp == start + timedelta(minutes=1)
    assert batches[1][0].high == 103
    assert batches[1][0].volume == 0
    assert batches[2][0].timestamp == start + timedelta(minutes=2)
    assert sent == [{"action": "subscribe", "params": {"symbols": "TEST"}}]
    assert spread_pct == pytest.approx(0.2 / 104 * 100)


def test_manager_resets_history_after_gap_only_after_a_fresh_candle():
    start = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(seconds=660)

    class Provider:
        name = "test-provider"

        async def connect(self):
            return None

        async def stream(self):
            yield [candle(start), candle(start + timedelta(seconds=60))]
            yield [candle(start + timedelta(seconds=600), 101)]
            yield [candle(start + timedelta(seconds=600), 101.1)]

        async def disconnect(self):
            return None

    manager = MarketDataManager(provider=Provider())
    asyncio.run(manager.run())
    snapshot = manager.get_snapshot()
    assert len(snapshot["candles"]) == 1
    assert snapshot["candles"][0]["timestamp"] == (start + timedelta(seconds=600)).isoformat()


def test_manager_forces_no_trade_when_connected_feed_is_stale():
    stale_candle = candle(datetime(2020, 1, 1, tzinfo=timezone.utc))

    class Provider:
        name = "test-provider"

        async def connect(self):
            return None

        async def stream(self):
            yield [stale_candle]
            await asyncio.Event().wait()

        async def disconnect(self):
            return None

    async def check_status():
        manager = MarketDataManager(provider=Provider())
        task = asyncio.create_task(manager.run())
        await asyncio.sleep(0)
        status = manager.get_status()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return status

    status = asyncio.run(check_status())
    assert status["connected"] is True
    assert status["fresh"] is False
    assert status["quality_accepted"] is False
    assert status["quality_status"] == "NO_TRADE"
    assert any("desactualizado" in reason for reason in status["quality_reasons"])
