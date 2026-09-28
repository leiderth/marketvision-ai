"""Indicadores técnicos descriptivos para velas ya aprobadas por el gate."""
from collections.abc import Sequence
from datetime import timezone
from itertools import pairwise
from typing import Any

from market_data.models import MarketDataCandle


class TechnicalIndicatorEngine:
    REQUIRED_CANDLES = 52

    def calculate(
        self,
        candles: Sequence[MarketDataCandle],
        *,
        feed_connected: bool,
        data_quality_accepted: bool,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        if not feed_connected:
            reasons.append("Feed desconectado; indicadores no disponibles.")
        if not data_quality_accepted:
            reasons.append("Datos no aprobados por DataQualityGate.")
        if len(candles) < self.REQUIRED_CANDLES:
            reasons.append(
                f"Información insuficiente: se requieren {self.REQUIRED_CANDLES} "
                f"velas y hay {len(candles)}."
            )

        values: dict[str, Any] = {
            "sma_20": None,
            "sma_50": None,
            "ema_20": None,
            "ema_50": None,
            "rsi_14": None,
            "macd": None,
            "atr_14": None,
            "vwap_session": None,
            "price_action": None,
            "candle_pattern": None,
            "ichimoku": None,
        }
        enough_data = len(candles) >= self.REQUIRED_CANDLES
        if enough_data:
            closes = [candle.close for candle in candles]
            values["sma_20"] = sum(closes[-20:]) / 20
            values["sma_50"] = sum(closes[-50:]) / 50
            values["ema_20"] = self._ema(closes, 20)[-1]
            values["ema_50"] = self._ema(closes, 50)[-1]
            values["rsi_14"] = self._rsi(closes, 14)
            macd_lines = [
                fast - slow
                for fast, slow in zip(self._ema(closes, 12), self._ema(closes, 26))
            ]
            signal = self._ema(macd_lines, 9)[-1]
            macd_line = macd_lines[-1]
            values["macd"] = {
                "line": macd_line,
                "signal": signal,
                "histogram": macd_line - signal,
            }
            values["atr_14"] = self._atr(candles, 14)
            latest_day = candles[-1].timestamp.astimezone(timezone.utc).date()
            session_candles = [
                candle
                for candle in candles
                if candle.timestamp.astimezone(timezone.utc).date() == latest_day
            ]
            volume_sum = sum(candle.volume for candle in session_candles)
            if volume_sum > 0:
                values["vwap_session"] = sum(
                    ((candle.high + candle.low + candle.close) / 3) * candle.volume
                    for candle in session_candles
                ) / volume_sum
            values["price_action"] = self._price_action(candles)
            values["candle_pattern"] = self._candle_pattern(candles[-2:])
            values["ichimoku"] = self._ichimoku(candles)

        ready = enough_data and feed_connected and data_quality_accepted
        return {
            "status": "OK" if ready else "NO_TRADE",
            "ready": ready,
            "as_of": candles[-1].timestamp.isoformat() if candles else None,
            "samples": len(candles),
            "required_samples": self.REQUIRED_CANDLES,
            "values": values,
            "reasons": list(dict.fromkeys(reasons)),
        }

    @staticmethod
    def _ema(values: Sequence[float], period: int) -> list[float]:
        alpha = 2 / (period + 1)
        ema_values = [values[0]]
        for value in values[1:]:
            ema_values.append(value * alpha + ema_values[-1] * (1 - alpha))
        return ema_values

    @staticmethod
    def _rsi(closes: Sequence[float], period: int) -> float:
        changes = [current - previous for previous, current in pairwise(closes)]
        gains = [max(change, 0.0) for change in changes]
        losses = [max(-change, 0.0) for change in changes]
        average_gain = sum(gains[:period]) / period
        average_loss = sum(losses[:period]) / period
        for gain, loss in zip(gains[period:], losses[period:]):
            average_gain = (average_gain * (period - 1) + gain) / period
            average_loss = (average_loss * (period - 1) + loss) / period
        if average_gain == 0 and average_loss == 0:
            return 50.0
        if average_loss == 0:
            return 100.0
        relative_strength = average_gain / average_loss
        return 100 - (100 / (1 + relative_strength))

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
    def _price_action(candles: Sequence[MarketDataCandle]) -> dict[str, Any]:
        current = candles[-1]
        candle_range = current.high - current.low
        body = abs(current.close - current.open)
        if len(candles) >= 6:
            previous_group = candles[-6:-3]
            current_group = candles[-3:]
            higher_highs = max(item.high for item in current_group) > max(
                item.high for item in previous_group
            )
            higher_lows = min(item.low for item in current_group) > min(
                item.low for item in previous_group
            )
            lower_highs = max(item.high for item in current_group) < max(
                item.high for item in previous_group
            )
            lower_lows = min(item.low for item in current_group) < min(
                item.low for item in previous_group
            )
            if higher_highs and higher_lows:
                structure = "higher_highs_and_lows"
            elif lower_highs and lower_lows:
                structure = "lower_highs_and_lows"
            else:
                structure = "mixed"
        else:
            structure = "insufficient_swings"
        return {
            "structure": structure,
            "candle_bias": (
                "bullish"
                if current.close > current.open
                else "bearish"
                if current.close < current.open
                else "neutral"
            ),
            "close_position": (
                (current.close - current.low) / candle_range
                if candle_range > 0
                else 0.5
            ),
            "body_pct": body / candle_range if candle_range > 0 else 0.0,
        }

    @staticmethod
    def _candle_pattern(candles: Sequence[MarketDataCandle]) -> str:
        previous, current = candles
        candle_range = current.high - current.low
        if candle_range <= 0:
            return "none"
        body = abs(current.close - current.open)
        upper_shadow = current.high - max(current.open, current.close)
        lower_shadow = min(current.open, current.close) - current.low
        if body / candle_range <= 0.1:
            return "doji"
        if (
            previous.close < previous.open
            and current.close > current.open
            and current.open <= previous.close
            and current.close >= previous.open
        ):
            return "bullish_engulfing"
        if (
            previous.close > previous.open
            and current.close < current.open
            and current.open >= previous.close
            and current.close <= previous.open
        ):
            return "bearish_engulfing"
        if lower_shadow >= 2 * body and upper_shadow <= body:
            return "hammer"
        if upper_shadow >= 2 * body and lower_shadow <= body:
            return "shooting_star"
        return "none"

    @staticmethod
    def _ichimoku(candles: Sequence[MarketDataCandle]) -> dict[str, float]:
        def midpoint(period: int) -> float:
            window = candles[-period:]
            highest = max(candle.high for candle in window)
            lowest = min(candle.low for candle in window)
            return (highest + lowest) / 2

        tenkan_sen = midpoint(9)
        kijun_sen = midpoint(26)
        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": (tenkan_sen + kijun_sen) / 2,
            "senkou_span_b": midpoint(52),
        }
