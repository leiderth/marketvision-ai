"""Validaciones conservadoras de integridad para lotes de velas."""
import math
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from numbers import Real

from market_data.models import DataQualityResult, MarketDataCandle


class DataQualityGate:
    def __init__(self, max_price_change_pct: float = 20.0) -> None:
        if not math.isfinite(max_price_change_pct) or max_price_change_pct <= 0:
            raise ValueError("max_price_change_pct debe ser un número positivo.")
        self._max_price_change_pct = max_price_change_pct

    def evaluate(
        self,
        candles: Sequence[MarketDataCandle],
        *,
        feed_connected: bool,
    ) -> DataQualityResult:
        reasons: list[str] = []
        if not feed_connected:
            reasons.append("Feed de datos desconectado; NO_TRADE.")
        if not candles:
            reasons.append("No hay velas disponibles; NO_TRADE.")
            return self._result(False, "NO_TRADE", reasons, 0)

        first = candles[0]
        invalid_interval = (
            isinstance(first.interval_seconds, bool)
            or not isinstance(first.interval_seconds, int)
            or first.interval_seconds <= 0
        )
        if invalid_interval:
            reasons.append("El intervalo de las velas debe ser positivo.")

        previous: MarketDataCandle | None = None
        seen_timestamps: set[datetime] = set()
        for candle in candles:
            if not isinstance(candle.symbol, str) or not candle.symbol.strip():
                reasons.append("Cada vela debe incluir un símbolo válido.")
            if candle.symbol != first.symbol:
                reasons.append("El lote contiene símbolos mezclados.")
            if candle.interval_seconds != first.interval_seconds:
                reasons.append("El lote contiene intervalos mezclados.")

            timestamp = candle.timestamp
            if (
                not isinstance(timestamp, datetime)
                or timestamp.tzinfo is None
                or timestamp.utcoffset() is None
            ):
                reasons.append("Todas las marcas de tiempo deben incluir zona horaria.")
                previous = None
                continue
            normalized_timestamp = timestamp.astimezone(timezone.utc)
            if normalized_timestamp > datetime.now(timezone.utc) + timedelta(seconds=5):
                reasons.append("Se detectó un timestamp futuro no válido.")
            if normalized_timestamp in seen_timestamps:
                reasons.append("Se detectaron timestamps duplicados.")
            seen_timestamps.add(normalized_timestamp)

            values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
            numeric_values = all(
                isinstance(value, Real) and not isinstance(value, bool)
                for value in values
            )
            if not numeric_values or not all(math.isfinite(value) for value in values):
                reasons.append("La vela contiene valores no finitos.")
            elif (
                min(candle.open, candle.high, candle.low, candle.close) <= 0
                or candle.volume < 0
                or candle.high < max(candle.open, candle.close)
                or candle.low > min(candle.open, candle.close)
                or candle.high < candle.low
            ):
                reasons.append("La vela contiene precios OHLC o volumen inválidos.")

            if previous is not None:
                previous_timestamp = previous.timestamp
                if previous_timestamp.tzinfo is not None and previous_timestamp.utcoffset() is not None:
                    elapsed = (
                        normalized_timestamp
                        - previous_timestamp.astimezone(timezone.utc)
                    ).total_seconds()
                    if elapsed <= 0:
                        reasons.append("Las velas están duplicadas o fuera de orden.")
                    elif (
                        not invalid_interval
                        and elapsed > first.interval_seconds + 1
                    ):
                        reasons.append("Se detectó un gap temporal en las velas.")
                    if (
                        elapsed > 0
                        and isinstance(previous.close, Real)
                        and not isinstance(previous.close, bool)
                        and math.isfinite(previous.close)
                        and previous.close > 0
                        and isinstance(candle.close, Real)
                        and not isinstance(candle.close, bool)
                        and math.isfinite(candle.close)
                    ):
                        change_pct = abs(candle.close - previous.close) / previous.close * 100
                        if change_pct > self._max_price_change_pct:
                            reasons.append(
                                "El cambio entre cierres supera el umbral de outlier "
                                f"({self._max_price_change_pct:g}%)."
                            )
            previous = candle

        unique_reasons = tuple(dict.fromkeys(reasons))
        accepted = not unique_reasons
        return self._result(
            accepted,
            "OK" if accepted else "NO_TRADE",
            list(unique_reasons),
            len(candles),
        )

    @staticmethod
    def _result(
        accepted: bool,
        status: str,
        reasons: list[str],
        candles_checked: int,
    ) -> DataQualityResult:
        return DataQualityResult(
            accepted=accepted,
            status=status,
            checked_at=datetime.now(timezone.utc),
            reasons=tuple(reasons),
            candles_checked=candles_checked,
        )
