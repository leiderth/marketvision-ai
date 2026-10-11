import json
import sqlite3
from datetime import datetime, timezone

from vision.feature_pipeline import FEATURE_NAMES, extract_visual_features
from vision.decision_journal import VisionDecisionJournal
from vision.ohlc_store import VisionCandleStore
from vision.series_labeler import VisionSeriesLabeler


def _candles():
    values = []
    for index in range(12):
        top = 0.35 + (index % 4) * 0.005
        bottom = top + 0.04
        values.append(
            {
                "x": 0.1 + index * 0.05,
                "top": top - 0.02,
                "bottom": bottom + 0.02,
                "body_top": top,
                "body_bottom": bottom,
                "wick_top": top - 0.02,
                "wick_bottom": bottom + 0.02,
                "body_size": 0.04,
                "direction": "CALL" if index % 2 == 0 else "PUT",
            }
        )
    return values


def _state():
    return {
        "readable": True,
        "captured_at": "2026-10-11T12:00:18+00:00",
        "asset": "EUR/USD",
        "timeframe_seconds": 60,
        "market_regime": "visual_transition",
        "directional_scores": {"CALL": 0.7, "PUT": 0.3},
        "trend": "up",
        "momentum": "bullish",
        "pattern": "bullish_engulfing_shape",
        "candle_count": 12,
        "read_quality_score": 88,
        "candles": _candles(),
        "support_levels": [0.38],
        "resistance_levels": [0.58],
        "ocr": {
            "fields": {
                "asset": {"value": "EUR/USD", "confidence": 0.9, "stable": True},
                "timer": {"seconds_remaining": 42, "confidence": 0.9, "stable": True},
                "payout": {"percent": 82, "confidence": 0.9, "stable": True},
                "amount": {"value": 5, "confidence": 0.9, "stable": True},
            }
        },
    }


def _pixel_candles(target_direction: str = "CALL"):
    return [
        {
            "body_top": 0.3 if target_direction == "CALL" else 0.6,
            "body_bottom": 0.4 if target_direction == "CALL" else 0.7,
            "wick_top": 0.2 if target_direction == "CALL" else 0.5,
            "wick_bottom": 0.5 if target_direction == "CALL" else 0.8,
            "direction": target_direction,
        },
        {
            "body_top": 0.4,
            "body_bottom": 0.5,
            "wick_top": 0.3,
            "wick_bottom": 0.6,
            "direction": "CALL",
        },
    ]


def test_visual_feature_pipeline_is_finite_normalized_and_carries_no_price_claims():
    snapshot = extract_visual_features(_state())

    assert snapshot["schema_version"] == 1
    assert snapshot["price_basis"] == "normalized_chart_pixels"
    assert set(snapshot["features"]) == set(FEATURE_NAMES)
    assert all(
        isinstance(value, float) and -1 <= value <= 1
        for value in snapshot["features"].values()
    )
    assert len(snapshot["candles"]) == 12
    assert snapshot["features"]["candle_bias_10"] == 0
    assert snapshot["features"]["pattern_bullish"] == 1


def test_next_candle_label_requires_a_completed_candle_and_two_stable_observations(tmp_path):
    database = tmp_path / "vision.sqlite"
    store = VisionCandleStore(database)
    labeler = VisionSeriesLabeler(database)
    state = _state()
    labeler.watch_decision("candidate-1", state)

    too_early = store.record_frame(
        asset="EUR/USD",
        timeframe_seconds=60,
        candles=_pixel_candles(),
        observed_at="2026-10-11T12:01:55+00:00",
        timer_seconds_remaining=5,
        chart_width=800,
        chart_height=480,
        read_quality_score=90,
    )
    assert too_early
    assert labeler.observe_completed_candles(
        store,
        now=datetime(2026, 10, 11, 12, 1, 55, tzinfo=timezone.utc),
    ) == []

    store.record_frame(
        asset="EUR/USD",
        timeframe_seconds=60,
        candles=_pixel_candles(),
        observed_at="2026-10-11T12:02:05+00:00",
        timer_seconds_remaining=55,
        chart_width=800,
        chart_height=480,
        read_quality_score=90,
    )
    assert labeler.observe_completed_candles(
        store,
        now=datetime(2026, 10, 11, 12, 2, 5, tzinfo=timezone.utc),
    ) == []
    assert labeler.get_dataset_status()["pending"] == 1

    store.record_frame(
        asset="EUR/USD",
        timeframe_seconds=60,
        candles=_pixel_candles(),
        observed_at="2026-10-11T12:02:07+00:00",
        timer_seconds_remaining=53,
        chart_width=800,
        chart_height=480,
        read_quality_score=90,
    )
    labeled = labeler.observe_completed_candles(
        store,
        now=datetime(2026, 10, 11, 12, 2, 7, tzinfo=timezone.utc),
    )

    assert len(labeled) == 1
    assert labeled[0]["target_direction"] == "CALL"
    assert labeled[0]["target"] == 1
    status = labeler.get_dataset_status()
    assert status["total_labeled"] == 1
    assert status["by_direction"]["CALL"] == 1
    assert status["pending"] == 0
    assert status["price_basis"] == "normalized_chart_pixels"
    assert "no son resultados de bróker" in status["note"]
    sample = labeler.get_samples()[0]
    assert sample["source"] == "pixel_series"
    assert sample["created_at"] == state["captured_at"]
    assert sample["features"]["price_basis"] == "normalized_chart_pixels"


def test_series_labels_are_separate_from_trade_outcomes_and_restart_safe(tmp_path):
    database = tmp_path / "vision.sqlite"
    store = VisionCandleStore(database)
    labeler = VisionSeriesLabeler(database)
    journal = VisionDecisionJournal(database)
    state = _state()
    decision = journal.record_decision(state)
    labeler.watch_decision(decision["decision_id"], state)
    labeler = VisionSeriesLabeler(database)

    store.record_frame(
        asset="EUR/USD",
        timeframe_seconds=60,
        candles=_pixel_candles("PUT"),
        observed_at="2026-10-11T12:02:05+00:00",
        timer_seconds_remaining=55,
        chart_width=800,
        chart_height=480,
        read_quality_score=90,
    )
    labeler.observe_completed_candles(
        store,
        now=datetime(2026, 10, 11, 12, 2, 5, tzinfo=timezone.utc),
    )
    store.record_frame(
        asset="EUR/USD",
        timeframe_seconds=60,
        candles=_pixel_candles("PUT"),
        observed_at="2026-10-11T12:02:07+00:00",
        timer_seconds_remaining=53,
        chart_width=800,
        chart_height=480,
        read_quality_score=90,
    )
    result = labeler.observe_completed_candles(
        store,
        now=datetime(2026, 10, 11, 12, 2, 7, tzinfo=timezone.utc),
    )

    assert result[0]["target_direction"] == "PUT"
    assert result[0]["target"] == 0
    with sqlite3.connect(database) as connection:
        trade_outcome = connection.execute(
            "SELECT outcome FROM vision_decisions WHERE decision_id = ?",
            (decision["decision_id"],),
        ).fetchone()
    assert trade_outcome is not None
    assert trade_outcome[0] is None


def test_series_sample_snapshot_is_valid_json_and_no_training_is_claimed(tmp_path):
    database = tmp_path / "vision.sqlite"
    labeler = VisionSeriesLabeler(database)
    result = labeler.watch_decision("candidate-3", _state())

    assert result["watching"] is True
    status = labeler.get_dataset_status()
    assert status["model_training_enabled"] is False
    with sqlite3.connect(database) as connection:
        raw = connection.execute(
            "SELECT features_json FROM vision_series_outcome_watch WHERE decision_id = ?",
            ("candidate-3",),
        ).fetchone()[0]
    assert json.loads(raw)["features"]["body_ratio_mean_10"] > 0
