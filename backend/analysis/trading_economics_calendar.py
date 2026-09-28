"""Official Trading Economics economic-calendar REST adapter."""
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from analysis.economic_calendar import EconomicEvent

_CALENDAR_URL = "https://api.tradingeconomics.com/calendar/country"
_CURRENCY_COUNTRIES = {
    "AUD": "australia",
    "CAD": "canada",
    "CHF": "switzerland",
    "CNY": "china",
    "EUR": "euro area",
    "GBP": "united kingdom",
    "HKD": "hong kong",
    "INR": "india",
    "JPY": "japan",
    "MXN": "mexico",
    "NOK": "norway",
    "NZD": "new zealand",
    "SEK": "sweden",
    "SGD": "singapore",
    "USD": "united states",
    "ZAR": "south africa",
}
_COUNTRY_CURRENCIES = {
    country: currency for currency, country in _CURRENCY_COUNTRIES.items()
}
_IMPORTANCE = {1: "low", 2: "medium", 3: "high"}


class TradingEconomicsCalendarProvider:
    def __init__(
        self,
        api_key: str,
        *,
        currencies: Sequence[str] = tuple(_CURRENCY_COUNTRIES),
        timeout_seconds: float = 15.0,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("ECONOMIC_CALENDAR_API_KEY es obligatorio.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds debe ser positivo.")
        normalized = tuple(dict.fromkeys(currency.upper() for currency in currencies))
        if not normalized or any(currency not in _CURRENCY_COUNTRIES for currency in normalized):
            raise ValueError("Se configuró una divisa sin país compatible en Trading Economics.")
        self._api_key = api_key.strip()
        self._supported_currencies = frozenset(normalized)
        self._countries = tuple(_CURRENCY_COUNTRIES[currency] for currency in normalized)
        self._timeout_seconds = timeout_seconds

    @property
    def supported_currencies(self) -> frozenset[str]:
        return self._supported_currencies

    async def fetch_events(
        self,
        start: datetime,
        end: datetime,
    ) -> Sequence[EconomicEvent]:
        if (
            start.tzinfo is None
            or start.utcoffset() is None
            or end.tzinfo is None
            or end.utcoffset() is None
            or end < start
        ):
            raise ValueError("El rango del calendario debe tener fechas válidas con zona horaria.")
        countries = quote(",".join(self._countries), safe=",")
        url = (
            f"{_CALENDAR_URL}/{countries}/"
            f"{start.astimezone(timezone.utc):%Y-%m-%d}/"
            f"{end.astimezone(timezone.utc):%Y-%m-%d}"
        )
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.get(
                url,
                params={"c": self._api_key, "f": "json"},
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, list):
            raise TypeError("Trading Economics devolvió una respuesta de calendario inválida.")
        return self.parse_events(payload)

    @staticmethod
    def parse_events(payload: list[Any]) -> tuple[EconomicEvent, ...]:
        events = []
        for row in payload:
            if not isinstance(row, dict):
                raise TypeError("Trading Economics devolvió un evento con formato inválido.")
            event_name = row.get("Event")
            country = row.get("Country")
            raw_timestamp = row.get("Date")
            raw_importance = row.get("Importance")
            if not isinstance(event_name, str) or not event_name.strip():
                raise ValueError("Trading Economics devolvió un evento sin nombre.")
            if not isinstance(country, str) or country.strip().casefold() not in _COUNTRY_CURRENCIES:
                raise ValueError("Trading Economics devolvió un país sin divisa mapeada.")
            if not isinstance(raw_timestamp, str) or "T" not in raw_timestamp:
                raise ValueError("Trading Economics devolvió un evento sin hora precisa.")
            try:
                timestamp = datetime.fromisoformat(raw_timestamp)
            except ValueError as exc:
                raise ValueError("Trading Economics devolvió una fecha de evento inválida.") from exc
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            if isinstance(raw_importance, bool):
                raise TypeError("Trading Economics devolvió un nivel de impacto inválido.")
            if isinstance(raw_importance, int):
                importance = raw_importance
            elif (
                isinstance(raw_importance, str)
                and raw_importance.strip().isdigit()
            ):
                importance = int(raw_importance.strip())
            else:
                raise ValueError("Trading Economics devolvió un nivel de impacto inválido.")
            impact = _IMPORTANCE.get(importance)
            if impact is None:
                raise ValueError("Trading Economics devolvió un nivel de impacto desconocido.")

            currency = row.get("Currency")
            if not isinstance(currency, str) or currency.strip().upper() not in _CURRENCY_COUNTRIES:
                currency = _COUNTRY_CURRENCIES[country.strip().casefold()]
            events.append(
                EconomicEvent(
                    name=event_name.strip(),
                    currency=currency.strip().upper(),
                    impact=impact,
                    timestamp=timestamp.astimezone(timezone.utc),
                )
            )
        return tuple(events)
