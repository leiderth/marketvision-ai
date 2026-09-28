import asyncio
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from analysis.economic_calendar import EconomicCalendarManager
from analysis.trading_economics_calendar import TradingEconomicsCalendarProvider
from httpx import HTTPStatusError, Request, Response


def event_row(
    *,
    country: str = "United States",
    currency: str = "",
    date: str = "2026-09-27T12:30:00",
    importance: int = 3,
) -> dict:
    return {
        "Country": country,
        "Currency": currency,
        "Date": date,
        "Event": "Employment report",
        "Importance": importance,
    }


def test_trading_economics_maps_importance_and_naive_api_timestamps_to_utc():
    events = TradingEconomicsCalendarProvider.parse_events(
        [
            event_row(importance=1),
            event_row(importance=2, currency="EUR"),
            event_row(importance=3),
        ]
    )

    assert [(event.currency, event.impact) for event in events] == [
        ("USD", "low"),
        ("EUR", "medium"),
        ("USD", "high"),
    ]
    assert events[0].timestamp == datetime(2026, 9, 27, 12, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "row",
    [
        event_row(country="Unknown"),
        event_row(date="2026-09-27"),
        event_row(importance=4),
        event_row(importance=2.5),
    ],
)
def test_trading_economics_rejects_untrustworthy_event_fields(row):
    with pytest.raises(ValueError):
        TradingEconomicsCalendarProvider.parse_events([row])


def test_trading_economics_rejects_unsupported_calendar_currency():
    with pytest.raises(ValueError, match="divisa"):
        TradingEconomicsCalendarProvider("secret", currencies=["XXX"])


def test_trading_economics_calendar_fails_closed_when_symbol_currency_is_uncovered():
    provider = TradingEconomicsCalendarProvider("private-key", currencies=["USD"])
    calendar = EconomicCalendarManager(provider=provider)
    calendar.load_events([], fetched_at=datetime.now(timezone.utc))

    status = calendar.get_status(currencies={"USD", "TRY"})

    assert status["available"] is False
    assert status["coverage_complete"] is False
    assert "Cobertura de divisas incompleta en el calendario económico." in status["reasons"]


def test_trading_economics_fetches_documented_country_range_without_leaking_key(monkeypatch):
    request_details = {}

    class MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [event_row()]

    class MockClient:
        def __init__(self, *, timeout):
            request_details["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url, *, params):
            request_details["url"] = url
            request_details["params"] = params
            return MockResponse()

    monkeypatch.setattr(
        "analysis.trading_economics_calendar.httpx.AsyncClient",
        MockClient,
    )
    provider = TradingEconomicsCalendarProvider("private-key", currencies=["USD"])
    events = asyncio.run(
        provider.fetch_events(
            datetime(2026, 9, 27, tzinfo=timezone.utc),
            datetime(2026, 10, 4, tzinfo=timezone.utc),
        )
    )

    assert len(events) == 1
    assert request_details["url"].endswith(
        "/united%20states/2026-09-27/2026-10-04"
    )
    assert request_details["params"] == {"c": "private-key", "f": "json"}


def test_trading_economics_http_error_leaves_calendar_unavailable(monkeypatch):
    class MockResponse:
        def raise_for_status(self):
            request = Request("GET", "https://api.tradingeconomics.com/calendar")
            response = Response(401, request=request)
            raise HTTPStatusError("unauthorized", request=request, response=response)

        def json(self):
            return []

    class MockClient:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, *_args, **_kwargs):
            return MockResponse()

    monkeypatch.setattr(
        "analysis.trading_economics_calendar.httpx.AsyncClient",
        MockClient,
    )
    provider = TradingEconomicsCalendarProvider("private-key", currencies=["USD"])
    manager = EconomicCalendarManager(provider=provider)
    with pytest.raises(HTTPStatusError):
        asyncio.run(manager.refresh())
    assert manager.get_status()["available"] is False
    assert manager.get_status()["error"] == "Error al actualizar el calendario económico (HTTPStatusError)."
