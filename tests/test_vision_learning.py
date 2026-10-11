import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np

from vision.decision_journal import VisionDecisionJournal
from vision.screen_capture import ScreenVisionService, analyze_chart_image
from ml.vision_learning import VisionLearningManager


def _seed_directional_examples(journal: VisionDecisionJournal, direction: str) -> None:
    with journal._connect() as connection:
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for index in range(100):
            high_score_bucket = index % 2 == 1
            score = 0.8 if high_score_bucket else 0.2
            bucket_index = index // 2 if index < 80 else (index - 80) // 2
            win = bucket_index < (
                (36 if high_score_bucket else 4)
                if index < 80
                else (9 if high_score_bucket else 1)
            )
            call_score = score if direction == "CALL" else 1 - score
            put_score = score if direction == "PUT" else 1 - score
            connection.execute(
                """
                INSERT INTO vision_decisions (
                    decision_id, created_at, asset, direction, call_score,
                    put_score, features_json, outcome, outcome_at, source
                ) VALUES (?, ?, ?, ?, ?, ?, '{}', ?, ?, 'paper')
                """,
                (
                    f"{direction}-{index:03d}",
                    (start + timedelta(minutes=index)).isoformat(),
                    "TEST",
                    direction,
                    call_score,
                    put_score,
                    "win" if win else "loss",
                    (start + timedelta(minutes=index, seconds=30)).isoformat(),
                ),
            )


def _seed_context_examples(journal: VisionDecisionJournal, direction: str) -> None:
    with journal._connect() as connection:
        start = datetime(2025, 1, 1, 10, tzinfo=timezone.utc)
        high_count = 0
        low_count = 0
        for index in range(100):
            if index >= 80:
                high_score = (index - 80) % 20 < 10
                bucket_index = (index - 80) % 10
            else:
                high_score = index < (70 if direction == "CALL" else 10)
                if high_score:
                    bucket_index = high_count
                    high_count += 1
                else:
                    bucket_index = low_count
                    low_count += 1
            score = 0.8 if high_score else 0.2
            win = bucket_index % 10 < (9 if high_score else 1)
            call_score = score if direction == "CALL" else 1 - score
            put_score = score if direction == "PUT" else 1 - score
            connection.execute(
                """
                INSERT INTO vision_decisions (
                    decision_id, created_at, asset, timeframe_seconds,
                    signal_window, direction, call_score, put_score,
                    features_json, market_regime, outcome, outcome_at, source
                ) VALUES (?, ?, 'EUR/USD', 60, ?, ?, ?, ?, '{}',
                          'visual_trending_up', ?, ?, 'demo_manual')
                """,
                (
                    f"context-{direction}-{index:03d}",
                    (start + timedelta(days=index)).isoformat(),
                    f"context-{direction}-{index}",
                    direction,
                    call_score,
                    put_score,
                    "win" if win else "loss",
                    (start + timedelta(days=index, seconds=30)).isoformat(),
                ),
            )


def test_contextual_probability_uses_wilson_interval_and_payout_breakeven(tmp_path):
    journal = VisionDecisionJournal(tmp_path / "vision.sqlite")
    _seed_context_examples(journal, "CALL")
    _seed_context_examples(journal, "PUT")

    calibration = journal.get_calibration(
        call_score=0.8,
        put_score=0.2,
        asset="EUR/USD",
        timeframe_seconds=60,
        market_regime="visual_trending_up",
        hour_utc=10,
        payout_pct=82,
        read_quality_score=90,
    )

    assert calibration["calibrated"] is True
    assert calibration["estimated_accuracy_pct"] == {"CALL": 80.0, "PUT": 20.0}
    assert calibration["contextual_estimates"]["CALL"]["samples"] == 80
    assert calibration["contextual_estimates"]["CALL"]["lower_95_pct"] < 80
    assert calibration["contextual_estimates"]["CALL"]["upper_95_pct"] > 80
    assert calibration["break_even_pct"] == 54.95
    assert calibration["required_probability_pct"] == 59.95
    assert calibration["decision"] == "CALL"
    assert calibration["conservative_margin_pct"] >= 15
    stricter_threshold = journal.get_calibration(
        call_score=0.8,
        put_score=0.2,
        asset="EUR/USD",
        timeframe_seconds=60,
        market_regime="visual_trending_up",
        hour_utc=10,
        payout_pct=82,
        read_quality_score=90,
        minimum_probability_pct=85,
    )
    assert stricter_threshold["decision"] == "NO_TRADE"
    assert stricter_threshold["required_probability_pct"] == 85


def test_directional_calibration_stays_blocked_below_sample_requirement(tmp_path):
    journal = VisionDecisionJournal(tmp_path / "vision.sqlite")
    _seed_directional_examples(journal, "CALL")

    calibration = journal.get_calibration(call_score=0.8, put_score=0.2)

    assert calibration["calibrated"] is False
    assert calibration["decision"] == "NO_TRADE"
    assert calibration["validation"]["PUT"]["samples"] == 0


def test_context_decision_fails_closed_for_low_quality_missing_payout_or_other_hour(tmp_path):
    journal = VisionDecisionJournal(tmp_path / "vision.sqlite")
    _seed_context_examples(journal, "CALL")
    _seed_context_examples(journal, "PUT")
    context = {
        "asset": "EUR/USD",
        "timeframe_seconds": 60,
        "market_regime": "visual_trending_up",
        "hour_utc": 10,
        "payout_pct": 82,
        "read_quality_score": 90,
        "call_score": 0.8,
        "put_score": 0.2,
    }

    low_quality = journal.get_calibration(**{**context, "read_quality_score": 40})
    no_payout = journal.get_calibration(**{**context, "payout_pct": None})
    other_hour = journal.get_calibration(**{**context, "hour_utc": 11})

    assert low_quality["decision"] == "NO_TRADE"
    assert "Calidad" in low_quality["reason"]
    assert no_payout["decision"] == "NO_TRADE"
    assert "payout" in no_payout["reason"]
    assert other_hour["decision"] == "NO_TRADE"
    assert other_hour["contextual_estimates"]["CALL"]["samples"] == 0


def test_contextual_decision_requires_visual_direction_to_agree(tmp_path):
    journal = VisionDecisionJournal(tmp_path / "vision.sqlite")
    _seed_context_examples(journal, "CALL")
    _seed_context_examples(journal, "PUT")

    calibration = journal.get_calibration(
        call_score=0.2,
        put_score=0.8,
        asset="EUR/USD",
        timeframe_seconds=60,
        market_regime="visual_trending_up",
        hour_utc=10,
        payout_pct=82,
        read_quality_score=90,
    )

    assert calibration["decision"] == "NO_TRADE"
    assert calibration["visual_direction"] == "PUT"
    assert "favorece PUT" in calibration["reason"]


def test_unreadable_chart_does_not_create_a_directional_score():
    result = analyze_chart_image(np.zeros((240, 320, 3), dtype=np.uint8))

    assert result["candle_count"] == 0
    assert result["directional_scores"] == {"CALL": None, "PUT": None}


def test_corrupt_screen_calibration_is_reported_to_the_ui(tmp_path):
    calibration_path = tmp_path / "screen_vision.json"
    calibration_path.write_text("{invalid json", encoding="utf-8")
    service = ScreenVisionService(calibration_path)

    state = service.capture_and_analyze()

    assert state["status"] == "invalid_screen_calibration"
    assert "JSONDecodeError" in state["calibration_error"]
    assert service.get_calibration()["calibrated"] is False


def test_capture_pause_stops_vision_until_resumed(tmp_path):
    service = ScreenVisionService(tmp_path / "screen_vision.json")

    assert service.set_paused(True)["status"] == "capture_paused"
    assert service.capture_and_analyze()["status"] == "capture_paused"
    service.set_paused(False)
    assert service.capture_and_analyze()["status"] == "screen_area_not_calibrated"


def test_manual_demo_outcome_keeps_timeframe_payout_and_duration(tmp_path):
    journal = VisionDecisionJournal(tmp_path / "vision.sqlite")
    result = journal.record_decision(
        {
            "readable": True,
            "asset": "EUR/USD",
            "timeframe_seconds": 60,
            "captured_at": "2025-01-01T00:00:10+00:00",
            "market_regime": "visual_unclassified",
            "directional_scores": {"CALL": 0.7, "PUT": 0.3},
            "trend": "up",
            "momentum": "bullish",
            "pattern": "none_detected",
        }
    )

    journal.record_outcome(
        result["decision_id"],
        "win",
        payout_pct=82,
        duration_seconds=60,
    )
    saved = journal.get_recent()[0]

    assert saved["timeframe_seconds"] == 60
    assert saved["market_regime"] == "visual_unclassified"
    assert saved["outcome"] == "win"
    assert saved["payout_pct"] == 82
    assert saved["duration_seconds"] == 60
    assert saved["outcome_source"] == "manual"


def test_screen_result_auto_labels_only_after_expiry_and_two_stable_reads(tmp_path):
    database = tmp_path / "vision.sqlite"
    journal = VisionDecisionJournal(database)
    learner = VisionLearningManager(
        database,
        tmp_path / "vision_training.csv",
        tmp_path / "models",
    )
    decision = journal.record_decision(
        {
            "readable": True,
            "asset": "EUR/USD",
            "timeframe_seconds": 60,
            "captured_at": "2025-01-01T00:00:10+00:00",
            "market_regime": "visual_trending_up",
            "directional_scores": {"CALL": 0.7, "PUT": 0.3},
        }
    )
    first_ocr_at = datetime.now(timezone.utc) - timedelta(seconds=5)
    watch = journal.arm_auto_outcome(
        decision["decision_id"],
        60,
        {
            "configured": True,
            "available": True,
            "result": None,
            "captured_at": first_ocr_at.isoformat(),
        },
    )
    with journal._connect() as connection:
        connection.execute(
            "UPDATE vision_outcome_watch SET expected_at = ? WHERE id = 1",
            ("2020-01-01T00:00:00+00:00",),
        )

    observation = {
        "result": "loss",
        "confidence": 0.95,
        "text": "Cerrado LOSS -$10.00",
        "captured_at": (first_ocr_at + timedelta(seconds=2)).isoformat(),
    }
    assert watch["decision_id"] == decision["decision_id"]
    assert journal.observe_auto_outcome(observation, payout_pct=82) is None
    assert journal.observe_auto_outcome(observation, payout_pct=82) is None
    second_observation = {
        **observation,
        "captured_at": (first_ocr_at + timedelta(seconds=3)).isoformat(),
    }
    labeled = journal.observe_auto_outcome(second_observation, payout_pct=82)
    learner.on_outcome(decision["decision_id"])
    saved = journal.get_recent()[0]

    assert labeled["outcome"] == "loss"
    assert journal.get_auto_outcome_watch() is None
    assert saved["outcome_source"] == "screen_ocr"
    assert saved["payout_pct"] == 82
    assert learner.get_dashboard()["online_learning"]["updates"] == 1
    assert learner.get_dashboard()["online_learning"]["confidence_threshold"] == 0.705
    assert learner.get_context_edge_margin(
        asset="EUR/USD",
        timeframe_seconds=60,
        market_regime="visual_trending_up",
        hour_utc=0,
        configured_margin_pct=5,
    ) == 5.25


def test_auto_outcome_watch_can_be_cancelled_without_adding_a_label(tmp_path):
    journal = VisionDecisionJournal(tmp_path / "vision.sqlite")
    decision = journal.record_decision(
        {
            "readable": True,
            "asset": "EUR/USD",
            "timeframe_seconds": 60,
            "captured_at": "2025-01-01T00:00:10+00:00",
            "market_regime": "visual_trending_up",
            "directional_scores": {"CALL": 0.7, "PUT": 0.3},
        }
    )
    journal.arm_auto_outcome(
        decision["decision_id"],
        60,
        {
            "configured": True,
            "available": True,
            "result": None,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    assert journal.cancel_auto_outcome_watch(decision["decision_id"]) is True
    assert journal.get_auto_outcome_watch() is None
    assert journal.get_recent()[0]["outcome"] is None


def test_manual_outcome_correction_is_audited_and_rebuilds_incremental_state(tmp_path):
    database = tmp_path / "vision.sqlite"
    journal = VisionDecisionJournal(database)
    learner = VisionLearningManager(
        database,
        tmp_path / "vision_training.csv",
        tmp_path / "models",
    )
    decision = journal.record_decision(
        {
            "readable": True,
            "asset": "EUR/USD",
            "timeframe_seconds": 60,
            "captured_at": "2025-01-01T00:00:10+00:00",
            "market_regime": "visual_trending_up",
            "directional_scores": {"CALL": 0.7, "PUT": 0.3},
        }
    )
    journal.record_outcome(decision["decision_id"], "win")
    learner.on_outcome(decision["decision_id"])

    journal.correct_outcome(decision["decision_id"], "loss", note="Se leyó mal el cierre.")
    learner.rebuild_online_state()

    assert journal.get_recent()[0]["outcome"] == "loss"
    assert journal.get_recent()[0]["outcome_source"] == "manual_correction"
    with journal._connect() as connection:
        audit = connection.execute(
            "SELECT previous_outcome, new_outcome, source, note FROM vision_outcome_audit ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert tuple(audit) == ("win", "loss", "manual_correction", "Se leyó mal el cierre.")


def test_same_asset_timeframe_signal_is_only_recorded_once_per_window(tmp_path):
    journal = VisionDecisionJournal(tmp_path / "vision.sqlite")
    state = {
        "readable": True,
        "asset": "EUR/USD",
        "timeframe_seconds": 60,
        "captured_at": "2025-01-01T00:00:10+00:00",
        "directional_scores": {"CALL": 0.7, "PUT": 0.3},
    }

    first = journal.record_decision(state)
    duplicate = journal.record_decision({**state, "directional_scores": {"CALL": 0.3, "PUT": 0.7}})

    assert "duplicate" not in first
    assert duplicate["duplicate"] is True
    assert duplicate["decision_id"] == first["decision_id"]
    assert len(journal.get_recent()) == 1


def test_existing_vision_database_receives_additive_columns(tmp_path):
    database = tmp_path / "vision.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE vision_decisions (
                decision_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                asset TEXT NOT NULL, direction TEXT NOT NULL,
                call_score REAL NOT NULL, put_score REAL NOT NULL,
                features_json TEXT NOT NULL, outcome TEXT, outcome_at TEXT,
                source TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO vision_decisions VALUES (
                'legacy', '2025-01-01T00:00:00+00:00', 'TEST', 'CALL',
                0.7, 0.3, '{}', 'win', '2025-01-01T00:01:00+00:00', 'paper'
            )
            """
        )

    journal = VisionDecisionJournal(database)
    saved = journal.get_recent()[0]

    assert saved["decision_id"] == "legacy"
    assert saved["timeframe_seconds"] is None
    assert saved["market_regime"] == "unknown"
    assert saved["outcome_source"] == "manual"
