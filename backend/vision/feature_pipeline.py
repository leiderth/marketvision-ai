"""Deterministic, scale-independent features extracted from visible candles."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

FEATURE_NAMES = (
    "call_score",
    "put_score",
    "read_quality",
    "candle_count",
    "trend_up",
    "trend_down",
    "trend_sideways",
    "momentum_bullish",
    "momentum_bearish",
    "pattern_bullish",
    "pattern_bearish",
    "candle_bias_5",
    "candle_bias_10",
    "body_ratio_mean_10",
    "upper_wick_ratio_mean_10",
    "lower_wick_ratio_mean_10",
    "range_ratio_mean_10",
    "return_mean_5",
    "return_std_10",
    "support_distance",
    "resistance_distance",
)


def extract_visual_features(state: Mapping[str, Any]) -> dict[str, Any]:
    candles = state.get("candles")
    if not isinstance(candles, Sequence) or isinstance(candles, (str, bytes)):
        candles = []
    normalized_candles = [
        _normalize_candle(item)
        for item in candles
        if isinstance(item, Mapping)
    ]
    normalized_candles = [item for item in normalized_candles if item is not None][-40:]
    scores = state.get("directional_scores")
    scores = scores if isinstance(scores, Mapping) else {}
    trend = state.get("trend")
    momentum = state.get("momentum")
    pattern = state.get("pattern")
    return {
        "schema_version": 1,
        "price_basis": "normalized_chart_pixels",
        "features": {
            "call_score": _unit(scores.get("CALL")),
            "put_score": _unit(scores.get("PUT")),
            "read_quality": _unit(state.get("read_quality_score"), scale=100),
            "candle_count": _unit(state.get("candle_count"), scale=100),
            "trend_up": float(trend == "up"),
            "trend_down": float(trend == "down"),
            "trend_sideways": float(trend == "sideways"),
            "momentum_bullish": float(momentum == "bullish"),
            "momentum_bearish": float(momentum == "bearish"),
            "pattern_bullish": float(pattern == "bullish_engulfing_shape"),
            "pattern_bearish": float(pattern == "bearish_engulfing_shape"),
            **_candle_features(
                normalized_candles,
                support_levels=state.get("support_levels", []),
                resistance_levels=state.get("resistance_levels", []),
            ),
        },
        "candles": normalized_candles,
        "ocr_context": _ocr_context(state.get("ocr")),
    }


def _normalize_candle(candle: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        normalized = {
            key: float(candle[key])
            for key in ("x", "top", "bottom", "body_top", "body_bottom", "wick_top", "wick_bottom", "body_size")
        }
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in normalized.values()):
        return None
    direction = candle.get("direction")
    if direction not in {"CALL", "PUT"}:
        return None
    if not (
        normalized["wick_top"] <= normalized["body_top"]
        <= normalized["body_bottom"] <= normalized["wick_bottom"]
        and normalized["top"] <= normalized["wick_top"]
        and normalized["bottom"] >= normalized["wick_bottom"]
    ):
        return None
    return {**normalized, "direction": direction}


def _candle_features(
    candles: list[dict[str, Any]],
    *,
    support_levels: Sequence[Any],
    resistance_levels: Sequence[Any],
) -> dict[str, float]:
    selected = candles[-10:]
    if not selected:
        return dict.fromkeys(FEATURE_NAMES[11:], 0.0)
    directions = [1.0 if item["direction"] == "CALL" else -1.0 for item in selected]
    body_ratios = [
        min(1.0, candle["body_size"] / max(candle["bottom"] - candle["top"], 1e-6))
        for candle in selected
    ]
    ranges = [candle["bottom"] - candle["top"] for candle in selected]
    upper_wicks = [
        max(0.0, candle["body_top"] - candle["wick_top"])
        / max(candle["bottom"] - candle["top"], 1e-6)
        for candle in selected
    ]
    lower_wicks = [
        max(0.0, candle["wick_bottom"] - candle["body_bottom"])
        / max(candle["bottom"] - candle["top"], 1e-6)
        for candle in selected
    ]
    closes = [
        candle["body_bottom"] if candle["direction"] == "CALL" else candle["body_top"]
        for candle in selected
    ]
    returns = [previous - current for previous, current in zip(closes, closes[1:])]
    support = _nearest_level_distance(selected[-1]["body_bottom"], support_levels)
    resistance = _nearest_level_distance(selected[-1]["body_top"], resistance_levels)
    return {
        "candle_bias_5": _mean(directions[-5:]),
        "candle_bias_10": _mean(directions),
        "body_ratio_mean_10": _mean(body_ratios),
        "upper_wick_ratio_mean_10": min(1.0, _mean(upper_wicks)),
        "lower_wick_ratio_mean_10": min(1.0, _mean(lower_wicks)),
        "range_ratio_mean_10": min(1.0, _mean(ranges)),
        "return_mean_5": max(-1.0, min(1.0, _mean(returns[-5:]) * 10)),
        "return_std_10": min(1.0, _std(returns) * 10),
        "support_distance": support,
        "resistance_distance": resistance,
    }


def _nearest_level_distance(value: float, levels: Sequence[Any]) -> float:
    normalized: list[float] = []
    for level in levels:
        try:
            number = float(level)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and 0 <= number <= 1:
            normalized.append(number)
    return min((abs(value - level) for level in normalized), default=1.0)


def _ocr_context(ocr: Any) -> dict[str, Any]:
    if not isinstance(ocr, Mapping):
        return {}
    fields = ocr.get("fields")
    if not isinstance(fields, Mapping):
        return {}
    context: dict[str, Any] = {}
    for name in ("asset", "timer", "payout", "amount"):
        field = fields.get(name)
        if not isinstance(field, Mapping):
            continue
        context[name] = {
            "value": field.get("value")
            or field.get("seconds_remaining")
            or field.get("percent"),
            "confidence": _unit(field.get("confidence")),
            "stable": bool(field.get("stable", False)),
        }
    return context


def _unit(value: Any, *, scale: float = 1.0) -> float:
    try:
        number = float(value) / scale
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
    return max(0.0, min(1.0, number)) if math.isfinite(number) else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
