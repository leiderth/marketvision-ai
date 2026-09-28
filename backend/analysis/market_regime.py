"""Market regime classification and conservative trade-blocking diagnostics."""
import math
from collections.abc import Sequence
from itertools import pairwise
from numbers import Real
from typing import Any

from market_data.models import MarketDataCandle


class MarketRegimeDetector:
    REQUIRED_CANDLES = 52

    def __init__(
        self,
        extreme_atr_pct: float = 1.0,
        noisy_efficiency_ratio: float = 0.15,
        ranging_efficiency_ratio: float = 0.30,
        trend_atr_separation: float = 0.5,
        max_spread_pct: float = 0.1,
    ) -> None:
        if not math.isfinite(extreme_atr_pct) or extreme_atr_pct <= 0:
            raise ValueError("extreme_atr_pct debe ser positivo.")
        if not 0 <= noisy_efficiency_ratio < ranging_efficiency_ratio <= 1:
            raise ValueError("Los umbrales de eficiencia del régimen son inválidos.")
        if not math.isfinite(trend_atr_separation) or trend_atr_separation < 0:
            raise ValueError("trend_atr_separation no puede ser negativo.")
        if not math.isfinite(max_spread_pct) or max_spread_pct <= 0:
            raise ValueError("max_spread_pct debe ser positivo.")
        self._extreme_atr_pct = extreme_atr_pct
        self._noisy_efficiency_ratio = noisy_efficiency_ratio
        self._ranging_efficiency_ratio = ranging_efficiency_ratio
        self._trend_atr_separation = trend_atr_separation
        self._max_spread_pct = max_spread_pct

    def detect(
        self,
        candles: Sequence[MarketDataCandle],
        *,
        feed_connected: bool,
        data_quality_accepted: bool,
        indicators: dict[str, Any],
        economic_calendar: dict[str, Any],
        spread_pct: float | None = None,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        flags: list[str] = []
        if not feed_connected:
            reasons.append("Feed desconectado; régimen no operativo.")
        if not data_quality_accepted:
            reasons.append("Datos no aprobados por DataQualityGate.")
        if len(candles) < self.REQUIRED_CANDLES:
            reasons.append(
                f"Información insuficiente: se requieren {self.REQUIRED_CANDLES} "
                f"velas y hay {len(candles)}."
            )

        calendar_available = bool(economic_calendar.get("available"))
        if not calendar_available:
            flags.append("economic_calendar_unavailable")
            reasons.append("Calendario económico no disponible; no se pueden descartar eventos.")
        if economic_calendar.get("error"):
            reasons.append(str(economic_calendar["error"]))

        if spread_pct is None:
            flags.append("spread_unavailable")
            reasons.append("Spread no informado; información insuficiente para el régimen.")
        elif (
            isinstance(spread_pct, bool)
            or not isinstance(spread_pct, Real)
            or not math.isfinite(spread_pct)
            or spread_pct < 0
        ):
            flags.append("spread_invalid")
            reasons.append("Spread inválido; régimen no operativo.")
        elif spread_pct > self._max_spread_pct:
            flags.append("elevated_spread")
            reasons.append("Spread superior al umbral configurado.")

        computed = len(candles) >= self.REQUIRED_CANDLES
        metrics: dict[str, float | None] = {
            "atr_pct": None,
            "efficiency_ratio": None,
            "sma_separation_atr": None,
            "spread_pct": spread_pct,
        }
        regime = "unknown"
        if computed:
            closes = [candle.close for candle in candles]
            atr = self._atr(candles, 14)
            atr_pct = atr / closes[-1] * 100 if closes[-1] > 0 else math.inf
            efficiency_ratio = self._efficiency_ratio(closes[-50:])
            sma20 = sum(closes[-20:]) / 20
            sma50 = sum(closes[-50:]) / 50
            separation_atr = abs(sma20 - sma50) / atr if atr > 0 else 0.0
            metrics.update(
                {
                    "atr_pct": atr_pct,
                    "efficiency_ratio": efficiency_ratio,
                    "sma_separation_atr": separation_atr,
                }
            )

            if atr_pct >= self._extreme_atr_pct:
                regime = "extreme_volatility"
                flags.append("extreme_volatility")
                reasons.append("Volatilidad extrema frente al umbral configurado.")
            elif efficiency_ratio < self._noisy_efficiency_ratio:
                regime = "noisy"
                flags.append("high_noise")
                reasons.append("Movimiento errático con baja eficiencia direccional.")
            elif (
                efficiency_ratio <= self._ranging_efficiency_ratio
                and separation_atr < self._trend_atr_separation
            ):
                regime = "ranging"
            elif sma20 > sma50 and separation_atr >= self._trend_atr_separation:
                regime = "trending_up"
            elif sma20 < sma50 and separation_atr >= self._trend_atr_separation:
                regime = "trending_down"
            else:
                regime = "transition"

            if self._is_false_breakout(candles):
                flags.append("possible_false_breakout")
                reasons.append("Posible ruptura falsa del rango reciente.")

            macd = indicators.get("values", {}).get("macd")
            if isinstance(macd, dict):
                histogram = macd.get("histogram")
                if (
                    isinstance(histogram, (int, float))
                    and math.isfinite(histogram)
                    and (sma20 > sma50 and histogram < 0 or sma20 < sma50 and histogram > 0)
                ):
                    flags.append("contradictory_trend_momentum")
                    reasons.append("Tendencia y momentum presentan direcciones contradictorias.")

            volumes = [candle.volume for candle in candles[-20:]]
            if not any(volume > 0 for volume in volumes):
                flags.append("liquidity_unavailable")
                reasons.append("Volumen de mercado no disponible para evaluar liquidez.")
            else:
                average_volume = sum(volumes) / len(volumes)
                latest_volume = volumes[-1]
                if average_volume > 0 and latest_volume < average_volume * 0.1:
                    flags.append("low_liquidity")
                    reasons.append("Volumen reciente muy inferior a la media; posible baja liquidez.")

        active_events = economic_calendar.get("active_events", [])
        if isinstance(active_events, list) and active_events:
            flags.append("high_impact_event_window")
            reasons.extend(
                f"Ventana de evento económico de alto impacto: {event.get('name', 'evento')}."
                for event in active_events
                if isinstance(event, dict)
            )

        blocking_flags = {
            "economic_calendar_unavailable",
            "spread_unavailable",
            "liquidity_unavailable",
            "extreme_volatility",
            "high_noise",
            "possible_false_breakout",
            "contradictory_trend_momentum",
            "high_impact_event_window",
            "low_liquidity",
            "spread_invalid",
            "elevated_spread",
        }
        blocked = bool(blocking_flags.intersection(flags))
        ready = (
            computed
            and feed_connected
            and data_quality_accepted
            and calendar_available
            and not blocked
        )
        return {
            "status": "OK" if ready else "NO_TRADE",
            "ready": ready,
            "regime": regime,
            "as_of": candles[-1].timestamp.isoformat() if candles else None,
            "metrics": metrics,
            "flags": list(dict.fromkeys(flags)),
            "trade_blocked": blocked or not ready,
            "reasons": list(dict.fromkeys(reasons)),
        }

    @staticmethod
    def _atr(candles: Sequence[MarketDataCandle], period: int) -> float:
        true_ranges = []
        for index, candle in enumerate(candles):
            previous_close = candles[index - 1].close if index else candle.close
            true_ranges.append(
                max(
                    candle.high - candle.low,
                    abs(candle.high - previous_close),
                    abs(candle.low - previous_close),
                )
            )
        average = sum(true_ranges[:period]) / period
        for true_range in true_ranges[period:]:
            average = (average * (period - 1) + true_range) / period
        return average

    @staticmethod
    def _efficiency_ratio(closes: Sequence[float]) -> float:
        path = sum(abs(current - previous) for previous, current in pairwise(closes))
        if path == 0:
            return 0.0
        return abs(closes[-1] - closes[0]) / path

    @staticmethod
    def _is_false_breakout(candles: Sequence[MarketDataCandle]) -> bool:
        previous_range = candles[-21:-1]
        current = candles[-1]
        highest = max(candle.high for candle in previous_range)
        lowest = min(candle.low for candle in previous_range)
        failed_up_break = current.high > highest and current.close <= highest
        failed_down_break = current.low < lowest and current.close >= lowest
        return failed_up_break or failed_down_break
