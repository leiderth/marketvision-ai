import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from analysis.economic_calendar import EconomicCalendarManager, EconomicEvent
from analysis.market_regime import MarketRegimeDetector
from api.main import app
from fastapi.testclient import TestClient
from market_data.models import MarketDataCandle


class EmptyCalendarProvider:
    async def fetch_events(self, start: datetime, end: datetime):
        return []


def candles_for_trend(count: int = 60) -> list[MarketDataCandle]:
    start = datetime.now(timezone.utc) - timedelta(minutes=count + 5)
    result = []
    for index in range(count):
        close = 100 + index * 0.05
        result.append(
            MarketDataCandle(
                symbol="EUR/USD",
                timestamp=start + timedelta(minutes=index),
                interval_seconds=60,
                open=close - 0.01,
                high=close + 0.02,
                low=close - 0.02,
                close=close,
                volume=100 + index,
            )
        )
    return result


def available_calendar(
    events: list[EconomicEvent] | None = None,
    *,
    fetched_at: datetime | None = None,
) -> EconomicCalendarManager:
    calendar = EconomicCalendarManager(provider=EmptyCalendarProvider())
    calendar.load_events(events or [], fetched_at=fetched_at)
    return calendar


def test_economic_calendar_filters_currencies_and_finds_high_impact_window():
    now = datetime.now(timezone.utc)
    calendar = available_calendar(
        [
            EconomicEvent("Employment report", "USD", "high", now + timedelta(minutes=5)),
            EconomicEvent("CPI", "EUR", "high", now + timedelta(minutes=12)),
            EconomicEvent("Retail sales", "USD", "low", now + timedelta(minutes=3)),
        ],
        fetched_at=now,
    )
    status = calendar.get_status(currencies={"USD"}, now=now)
    assert status["available"] is True
    assert len(status["active_events"]) == 1
    assert status["active_events"][0]["name"] == "Employment report"
    assert len(status["upcoming_events"]) == 1
    assert status["upcoming_events"][0]["name"] == "Retail sales"


def test_economic_calendar_unavailable_or_stale_fails_closed():
    now = datetime.now(timezone.utc)
    assert EconomicCalendarManager().get_status(now=now)["available"] is False

    stale = available_calendar(fetched_at=now - timedelta(hours=7))
    status = stale.get_status(now=now)
    assert status["available"] is False
    assert status["fresh"] is False
    assert any("no actualizado" in reason for reason in status["reasons"])


def test_economic_calendar_rejects_timezone_naive_events():
    calendar = available_calendar()
    with pytest.raises(ValueError, match="zona horaria"):
        calendar.load_events(
            [
                EconomicEvent(
                    "CPI",
                    "USD",
                    "high",
                    datetime(2026, 9, 27, 12, tzinfo=timezone.utc).replace(tzinfo=None),
                )
            ]
        )


def test_market_regime_classifies_trend_but_calendar_availability_is_required():
    candles = candles_for_trend()
    detector = MarketRegimeDetector(extreme_atr_pct=10)
    indicators = {
        "values": {"macd": {"histogram": 0.1}},
    }
    result = detector.detect(
        candles,
        feed_connected=True,
        data_quality_accepted=True,
        indicators=indicators,
        economic_calendar={"available": True, "active_events": []},
        spread_pct=0.01,
    )
    assert result["regime"] == "trending_up"
    assert result["ready"] is True
    assert result["trade_blocked"] is False

    missing_calendar = detector.detect(
        candles,
        feed_connected=True,
        data_quality_accepted=True,
        indicators=indicators,
        economic_calendar={"available": False, "active_events": []},
    )
    assert missing_calendar["regime"] == "trending_up"
    assert missing_calendar["ready"] is False
    assert missing_calendar["trade_blocked"] is True
    assert "economic_calendar_unavailable" in missing_calendar["flags"]


def test_market_regime_blocks_extreme_volatility_false_breakout_and_spread():
    candles = candles_for_trend()
    last = candles[-1]
    candles[-1] = MarketDataCandle(
        symbol=last.symbol,
        timestamp=last.timestamp,
        interval_seconds=last.interval_seconds,
        open=last.open,
        high=max(item.high for item in candles[-21:-1]) + 1,
        low=last.low,
        close=max(item.high for item in candles[-21:-1]) - 0.1,
        volume=last.volume,
    )
    detector = MarketRegimeDetector(extreme_atr_pct=0.001, max_spread_pct=0.05)
    result = detector.detect(
        candles,
        feed_connected=True,
        data_quality_accepted=True,
        indicators={"values": {"macd": {"histogram": -0.1}}},
        economic_calendar={"available": True, "active_events": []},
        spread_pct=0.2,
    )
    assert result["regime"] == "extreme_volatility"
    assert result["trade_blocked"] is True
    assert "extreme_volatility" in result["flags"]
    assert "possible_false_breakout" in result["flags"]
    assert "elevated_spread" in result["flags"]


def test_market_regime_blocks_news_window_and_contradictory_momentum():
    candles = candles_for_trend()
    now = datetime.now(timezone.utc)
    calendar = available_calendar(
        [EconomicEvent("Rate decision", "USD", "high", now + timedelta(minutes=2))],
        fetched_at=now,
    )
    calendar_status = calendar.get_status(currencies={"EUR", "USD"}, now=now)
    result = MarketRegimeDetector(extreme_atr_pct=10).detect(
        candles,
        feed_connected=True,
        data_quality_accepted=True,
        indicators={"values": {"macd": {"histogram": -0.1}}},
        economic_calendar=calendar_status,
    )
    assert result["trade_blocked"] is True
    assert "high_impact_event_window" in result["flags"]
    assert "contradictory_trend_momentum" in result["flags"]


def test_market_regime_classifies_low_efficiency_price_as_noise():
    start = datetime.now(timezone.utc) - timedelta(minutes=60)
    flat_market = [
        MarketDataCandle(
            symbol="EUR/USD",
            timestamp=start + timedelta(minutes=index),
            interval_seconds=60,
            open=100,
            high=100.01,
            low=99.99,
            close=100,
            volume=0,
        )
        for index in range(60)
    ]
    result = MarketRegimeDetector().detect(
        flat_market,
        feed_connected=True,
        data_quality_accepted=True,
        indicators={"values": {"macd": None}},
        economic_calendar={"available": True, "active_events": []},
    )
    assert result["regime"] == "noisy"
    assert "high_noise" in result["flags"]
    assert "liquidity_unavailable" in result["flags"]
    assert result["trade_blocked"] is True


def test_market_regime_and_calendar_api_report_unconfigured_source():
    with TestClient(app) as client:
        regime = client.get("/api/market-data/regime")
        calendar = client.get("/api/market-data/calendar")
    assert regime.status_code == 200
    assert regime.json()["regime"] == "unknown"
    assert regime.json()["trade_blocked"] is True
    assert calendar.status_code == 200
    assert calendar.json()["available"] is False


def test_calendar_provider_refresh_updates_freshness():
    now = datetime.now(timezone.utc)

    class Provider:
        async def fetch_events(self, start: datetime, end: datetime):
            assert start < now < end
            return [EconomicEvent("Rate decision", "USD", "high", now)]

    calendar = EconomicCalendarManager(provider=Provider())
    asyncio.run(calendar.refresh(now=now))
    status = calendar.get_status(currencies={"USD"}, now=now)
    assert status["available"] is True
    assert len(status["active_events"]) == 1
