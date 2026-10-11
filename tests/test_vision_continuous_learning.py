import json
import sqlite3
from datetime import datetime, timedelta, timezone

from ml.vision_learning import VisionLearningManager
from vision.decision_journal import VisionDecisionJournal


def _build_learning(tmp_path):
    database = tmp_path / "vision.sqlite"
    journal = VisionDecisionJournal(database)
    learner = VisionLearningManager(
        database,
        tmp_path / "vision_training.csv",
        tmp_path / "registry",
    )
    return journal, learner, database


def _seed_labeled_decisions(journal, count=200, start_index=0):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with journal._connect() as connection:
        for index in range(start_index, start_index + count):
            wins = index % 2 == 0
            score = 0.9 if wins else 0.1
            timestamp = start + timedelta(seconds=index)
            features = {
                "trend": "up" if wins else "down",
                "momentum": "bullish" if wins else "bearish",
                "pattern": "bullish_engulfing_shape" if wins else "bearish_engulfing_shape",
                "candle_count": 24,
                "read_quality_score": 92,
            }
            connection.execute(
                """
                INSERT INTO vision_decisions (
                    decision_id, created_at, asset, timeframe_seconds,
                    signal_window, direction, call_score, put_score,
                    confidence_pct, market_regime, features_json,
                    outcome, outcome_at, outcome_source, source
                ) VALUES (?, ?, 'EUR/USD', 60, ?, ?, ?, ?, 80,
                          'visual_trending_up', ?, ?, ?, 'manual', 'demo_manual')
                """,
                (
                    f"decision-{index:04d}",
                    timestamp.isoformat(),
                    f"window-{index}",
                    "CALL" if wins else "PUT",
                    score,
                    1 - score,
                    json.dumps(features),
                    "win" if wins else "loss",
                    (timestamp + timedelta(milliseconds=100)).isoformat(),
                ),
            )


def test_walk_forward_model_promotes_only_after_beating_baseline(tmp_path):
    journal, learner, _database = _build_learning(tmp_path)
    _seed_labeled_decisions(journal)

    result = learner.train_batch()
    dashboard = learner.get_dashboard()

    assert result["trained"] is True
    assert result["decision"] == "promoted"
    assert result["brier_score"] + 0.002 < result["baseline_brier_score"]
    assert dashboard["active_model_version"] == result["version"]
    assert dashboard["model_evolution"][0]["decision"] == "promoted"
    assert dashboard["calibration"]
    assert len(dashboard["online_learning"]["feature_importance"]) > 0


def test_scheduler_waits_for_minimum_sample_and_retrains_by_batch_size(tmp_path):
    journal, learner, _database = _build_learning(tmp_path)
    _seed_labeled_decisions(journal, count=80)

    insufficient = learner.maybe_retrain()
    assert insufficient["trained"] is False
    assert "100" in insufficient["reason"]

    _seed_labeled_decisions(journal, count=120, start_index=80)
    first = learner.maybe_retrain(force=True)
    second = learner.maybe_retrain()

    assert first["trained"] is True
    assert second["trained"] is False
    assert "cada 50" in second["reason"]


def test_degraded_active_model_rolls_back_to_safe_baseline(tmp_path):
    journal, learner, database = _build_learning(tmp_path)
    _seed_labeled_decisions(journal)
    learner.train_batch()
    active = learner._load_active_model()
    assert active is not None

    assert learner._rollback_active_model(
        active,
        sample_count=40,
        brier_score=0.9,
        baseline_brier=0.1,
    )
    assert learner.get_dashboard()["active_model_version"] is None
    with sqlite3.connect(database) as connection:
        decision = connection.execute(
            "SELECT decision FROM vision_model_evaluations ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
    assert decision == "rollback"
