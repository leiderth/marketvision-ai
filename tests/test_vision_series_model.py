import json
import sqlite3
from datetime import datetime, timedelta, timezone

from ml.vision_series_model import VisionSeriesModel
from vision.feature_pipeline import FEATURE_NAMES


def _build_model(tmp_path):
    database = tmp_path / "vision.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE vision_series_samples (
                decision_id TEXT PRIMARY KEY,
                asset TEXT NOT NULL,
                timeframe_seconds INTEGER NOT NULL,
                market_regime TEXT NOT NULL,
                prediction_direction TEXT NOT NULL,
                target_direction TEXT NOT NULL,
                target INTEGER,
                target_candle_open_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                labeled_at TEXT NOT NULL,
                source TEXT NOT NULL,
                price_basis TEXT NOT NULL,
                features_json TEXT NOT NULL
            );
            """
        )
    return database, VisionSeriesModel(database, tmp_path / "registry")


def _seed_samples(database, count=300):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(count):
        target = index % 2
        created = start + timedelta(seconds=index * 10)
        candle_open = created + timedelta(seconds=60)
        features = dict.fromkeys(FEATURE_NAMES, 0.0)
        features.update(
            call_score=0.9 if target else 0.1,
            put_score=0.1 if target else 0.9,
            read_quality=0.9,
            candle_count=0.3,
            trend_up=float(bool(target)),
            trend_down=float(not target),
            momentum_bullish=float(bool(target)),
            momentum_bearish=float(not target),
            candle_bias_5=1.0 if target else -1.0,
            candle_bias_10=1.0 if target else -1.0,
        )
        snapshot = {
            "schema_version": 1,
            "price_basis": "normalized_chart_pixels",
            "features": features,
            "candles": [],
            "ocr_context": {},
        }
        rows.append(
            (
                f"series-{index:04d}",
                "EUR/USD",
                60,
                "visual_trending",
                "CALL",
                "CALL" if target else "PUT",
                target,
                candle_open.isoformat(),
                created.isoformat(),
                (candle_open + timedelta(seconds=60)).isoformat(),
                "pixel_series",
                "normalized_chart_pixels",
                json.dumps(snapshot),
            )
        )
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO vision_series_samples(
                decision_id, asset, timeframe_seconds, market_regime,
                prediction_direction, target_direction, target,
                target_candle_open_at, created_at, labeled_at,
                source, price_basis, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def test_series_model_requires_minimum_samples_and_does_not_fabricate_prediction(tmp_path):
    database, model = _build_model(tmp_path)
    result = model.train(force=True)
    status = model.get_status()

    assert result["trained"] is False
    assert result["minimum_samples"] == 200
    assert status["active_version"] is None
    assert status["prediction"] is None
    assert status["training_source"] == "visual_next_candle_labels_only"


def test_series_model_uses_purged_chronological_holdout_and_calibrates_probabilities(tmp_path):
    database, model = _build_model(tmp_path)
    _seed_samples(database)

    result = model.train(force=True)
    status = model.get_status(
        {
            "readable": True,
            "directional_scores": {"CALL": 0.8, "PUT": 0.2},
            "read_quality_score": 90,
            "candle_count": 20,
            "candles": [],
            "trend": "up",
            "momentum": "bullish",
            "pattern": "bullish_engulfing_shape",
        }
    )

    assert result["trained"] is True
    assert result["sample_count"] == 300
    assert result["test_samples"] >= 30
    assert result["decision"] == "promoted"
    assert result["brier_score"] < result["baseline_brier_score"]
    assert 0 <= result["log_loss"] < 1
    assert status["active_version"] == result["version"]
    assert status["prediction"]["probability_correct"] > 0.7
    assert status["prediction"]["calibrated"] is True
    assert status["prediction"]["interval_95"] is not None
    assert status["prediction"]["price_basis"] == "normalized_chart_pixels"
    assert status["recent_evaluations"][0]["decision"] == "promoted"
    artifact_path = tmp_path / "registry" / f"{result['version']}.json"
    assert artifact_path.is_file()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["validation_policy"] == "chronological_60_20_20_label_horizon_purged"


def test_series_model_schedule_requires_new_samples_or_daily_boundary(tmp_path):
    database, model = _build_model(tmp_path)
    _seed_samples(database)
    first = model.maybe_train(
        now=datetime(2026, 10, 1, tzinfo=timezone.utc),
    )
    second = model.maybe_train(
        now=datetime(2026, 10, 1, 1, tzinfo=timezone.utc),
    )

    assert first["trained"] is True
    assert second["trained"] is False
    assert "intervalo" in second["reason"]


def test_series_model_defers_failed_splits_until_dataset_changes(tmp_path):
    database, model = _build_model(tmp_path)
    _seed_samples(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE vision_series_samples SET target = 1, target_direction = 'CALL'"
        )
    now = datetime(2026, 10, 1, tzinfo=timezone.utc)

    first = model.maybe_train(now=now)
    second = model.maybe_train(now=now + timedelta(minutes=2))
    next_day = model.maybe_train(now=now + timedelta(days=1))

    assert first["trained"] is False
    assert "requiere etiquetas" in first["reason"]
    assert second["trained"] is False
    assert "intervalo" in second["reason"]
    assert next_day["trained"] is False
    assert "intervalo" in next_day["reason"]
