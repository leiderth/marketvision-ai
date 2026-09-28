"""Economic-event contracts and a fail-closed, freshness-aware calendar."""
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from httpx import HTTPError


@dataclass(frozen=True)
class EconomicEvent:
    name: str
    currency: str
    impact: str
    timestamp: datetime


class EconomicCalendarProvider(Protocol):
    @property
    def supported_currencies(self) -> frozenset[str]: ...

    async def fetch_events(
        self,
        start: datetime,
        end: datetime,
    ) -> Sequence[EconomicEvent]: ...


class EconomicCalendarManager:
    def __init__(
        self,
        provider: EconomicCalendarProvider | None = None,
        *,
        configuration_error: str | None = None,
        event_window_minutes: int = 30,
        max_age_hours: int = 6,
    ) -> None:
        if event_window_minutes < 0 or max_age_hours <= 0:
            raise ValueError("La ventana y vigencia del calendario deben ser válidas.")
        self._provider = provider
        self._event_window = timedelta(minutes=event_window_minutes)
        self._max_age = timedelta(hours=max_age_hours)
        self._events: tuple[EconomicEvent, ...] = ()
        self._fetched_at: datetime | None = None
        self._last_error = configuration_error

    @property
    def has_provider(self) -> bool:
        return self._provider is not None

    async def refresh(self, *, now: datetime | None = None) -> None:
        if self._provider is None:
            raise RuntimeError("No hay un proveedor de calendario económico configurado.")
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None or current_time.utcoffset() is None:
            raise ValueError("La fecha de actualización debe incluir zona horaria.")
        try:
            events = await self._provider.fetch_events(
                current_time - timedelta(days=1),
                current_time + timedelta(days=7),
            )
            validated = self._validate_events(events)
        except (HTTPError, TimeoutError, OSError, RuntimeError, ValueError, TypeError) as exc:
            self._last_error = f"Error al actualizar el calendario económico ({type(exc).__name__})."
            raise
        self._events = validated
        self._fetched_at = current_time.astimezone(timezone.utc)
        self._last_error = None

    def load_events(
        self,
        events: Sequence[EconomicEvent],
        *,
        fetched_at: datetime | None = None,
    ) -> None:
        """Allows a selected trusted provider or controlled test to publish events."""
        current_time = fetched_at or datetime.now(timezone.utc)
        if current_time.tzinfo is None or current_time.utcoffset() is None:
            raise ValueError("fetched_at debe incluir zona horaria.")
        self._events = self._validate_events(events)
        self._fetched_at = current_time.astimezone(timezone.utc)
        self._last_error = None

    def get_status(
        self,
        *,
        currencies: set[str] | None = None,
        now: datetime | None = None,
    ) -> dict:
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None or current_time.utcoffset() is None:
            raise ValueError("now debe incluir zona horaria.")
        normalized_now = current_time.astimezone(timezone.utc)
        age = (
            normalized_now - self._fetched_at
            if self._fetched_at is not None
            else None
        )
        fresh = age is not None and timedelta(0) <= age <= self._max_age
        available = self._provider is not None and fresh
        normalized_currencies = (
            {currency.upper() for currency in currencies}
            if currencies
            else None
        )
        supported_currencies = (
            getattr(self._provider, "supported_currencies", None)
            if self._provider is not None
            else None
        )
        coverage_complete = (
            normalized_currencies is None
            or supported_currencies is None
            or normalized_currencies.issubset(supported_currencies)
        )
        available = available and coverage_complete
        active_events = []
        upcoming_events = []
        for event in self._events:
            if (
                normalized_currencies
                and event.currency.upper() not in normalized_currencies
            ):
                continue
            event_time = event.timestamp.astimezone(timezone.utc)
            seconds_until = (event_time - normalized_now).total_seconds()
            entry = {
                "name": event.name,
                "currency": event.currency.upper(),
                "impact": event.impact,
                "timestamp": event_time.isoformat(),
                "minutes_until": round(seconds_until / 60, 1),
            }
            if (
                event.impact == "high"
                and abs(event_time - normalized_now) <= self._event_window
            ):
                active_events.append(entry)
            elif timedelta(0) <= event_time - normalized_now <= timedelta(days=1):
                upcoming_events.append(entry)

        reasons = []
        if not self._provider:
            reasons.append("Proveedor de calendario económico no configurado.")
        elif not fresh:
            reasons.append("Calendario económico no actualizado.")
        if not coverage_complete:
            reasons.append("Cobertura de divisas incompleta en el calendario económico.")
        if self._last_error:
            reasons.append(self._last_error)
        return {
            "available": available,
            "fresh": fresh,
            "fetched_at": self._fetched_at.isoformat() if self._fetched_at else None,
            "age_seconds": age.total_seconds() if age else None,
            "event_window_minutes": self._event_window.total_seconds() / 60,
            "currencies": sorted(normalized_currencies) if normalized_currencies else [],
            "coverage_complete": coverage_complete,
            "covered_currencies": sorted(supported_currencies or ()),
            "active_events": active_events,
            "upcoming_events": upcoming_events,
            "events_considered": len(active_events) + len(upcoming_events),
            "error": self._last_error,
            "reasons": reasons,
        }

    @staticmethod
    def _validate_events(events: Sequence[EconomicEvent]) -> tuple[EconomicEvent, ...]:
        validated = []
        for event in events:
            if not isinstance(event, EconomicEvent):
                raise TypeError("El proveedor devolvió un evento con formato inválido.")
            if (
                not isinstance(event.name, str)
                or not isinstance(event.currency, str)
                or not isinstance(event.impact, str)
            ):
                raise TypeError("El proveedor devolvió campos de evento con tipo inválido.")
            if not event.name.strip() or not event.currency.strip():
                raise ValueError("Cada evento debe incluir nombre y divisa.")
            if event.impact not in {"low", "medium", "high"}:
                raise ValueError("El impacto del evento debe ser low, medium o high.")
            if (
                not isinstance(event.timestamp, datetime)
                or event.timestamp.tzinfo is None
                or event.timestamp.utcoffset() is None
            ):
                raise ValueError("Todos los eventos deben incluir zona horaria.")
            validated.append(
                EconomicEvent(
                    name=event.name.strip(),
                    currency=event.currency.strip().upper(),
                    impact=event.impact,
                    timestamp=event.timestamp.astimezone(timezone.utc),
                )
            )
        return tuple(sorted(validated, key=lambda event: event.timestamp))
