import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fastapi.testclient import TestClient
from api.main import app
from ml.drift import FeatureDriftMonitor
from ml.schema import FEATURE_NAMES


def test_drift_monitor_requires_distinct_samples_and_latches_detection():
    monitor = FeatureDriftMonitor(
        tuple(0.0 for _ in FEATURE_NAMES),
        tuple(1.0 for _ in FEATURE_NAMES),
        window_size=4,
        min_samples=3,
        threshold=2.0,
    )
    start = datetime(2026, 9, 27, tzinfo=timezone.utc)
    baseline = dict.fromkeys(FEATURE_NAMES, 0.0)
    shifted = dict.fromkeys(FEATURE_NAMES, 3.0)

    assert monitor.observe(baseline, start.isoformat())["available"] is False
    assert monitor.observe(shifted, start.isoformat())["samples"] == 1
    monitor.observe(shifted, (start + timedelta(minutes=1)).isoformat())
    detected = monitor.observe(shifted, (start + timedelta(minutes=2)).isoformat())
    assert detected["drifted"] is True
    assert detected["drifted_features"] == list(FEATURE_NAMES)
    reset_observation = monitor.observe(
        baseline,
        (start + timedelta(minutes=3)).isoformat(),
    )
    assert reset_observation["drifted"] is True


def test_drift_latch_survives_restart_and_corrupt_state_fails_closed(tmp_path):
    state_path = tmp_path / "drift.json"
    monitor = FeatureDriftMonitor(
        tuple(0.0 for _ in FEATURE_NAMES),
        tuple(1.0 for _ in FEATURE_NAMES),
        window_size=3,
        min_samples=2,
        threshold=1.0,
        state_path=state_path,
    )
    start = datetime(2026, 9, 27, tzinfo=timezone.utc)
    shifted = dict.fromkeys(FEATURE_NAMES, 2.0)
    monitor.observe(shifted, start.isoformat())
    assert monitor.observe(shifted, (start + timedelta(minutes=1)).isoformat())["drifted"]
    restarted = FeatureDriftMonitor(
        tuple(0.0 for _ in FEATURE_NAMES),
        tuple(1.0 for _ in FEATURE_NAMES),
        window_size=3,
        min_samples=2,
        threshold=1.0,
        state_path=state_path,
    )
    assert restarted.get_status()["drifted"] is True

    state_path.write_text("invalid", encoding="utf-8")
    corrupted = FeatureDriftMonitor(
        tuple(0.0 for _ in FEATURE_NAMES),
        tuple(1.0 for _ in FEATURE_NAMES),
        state_path=state_path,
    )
    assert corrupted.get_status()["drifted"] is True


def test_observability_endpoints_report_fail_closed_defaults():
    with TestClient(app) as client:
        health = client.get("/api/observability/health")
        metrics = client.get("/api/observability/metrics")

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["components"]["data_quality"] == "blocked"
    assert metrics.status_code == 200
    assert "marketvision_feed_connected 0" in metrics.text
    assert "marketvision_paper_mode 0" in metrics.text
