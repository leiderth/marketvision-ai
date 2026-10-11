from datetime import datetime, timezone
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.vision import router
from vision.ohlc_store import VisionCandleStore
from vision.screen_capture import (
    _stabilize_ocr_results,
    detect_chart_candidates,
)


def _chart() -> np.ndarray:
    image = np.full((480, 800, 3), 245, dtype=np.uint8)
    for index in range(30):
        x = 20 + index * 24
        center = 250 + (index % 7 - 3) * 9
        color = (0, 190, 0) if index % 2 else (0, 0, 210)
        cv2.line(image, (x + 3, center - 24), (x + 3, center + 27), color, 1)
        cv2.rectangle(image, (x, center - 12), (x + 6, center + 12), color, -1)
    return image


def test_chart_detection_returns_a_reviewable_region_with_global_monitor_origin():
    desktop = np.full((900, 1500, 3), 245, dtype=np.uint8)
    desktop[180:660, 260:1060] = _chart()

    candidates = detect_chart_candidates(
        desktop,
        desktop_left=-1280,
        desktop_top=-200,
    )

    assert candidates
    candidate = candidates[0]
    assert candidate["candle_count"] >= 8
    assert candidate["visual_quality_score"] >= 35
    assert candidate["left"] < 0
    assert candidate["top"] >= -200
    assert candidate["top"] + candidate["height"] <= 700
    assert candidate["width"] >= 320
    assert candidate["height"] >= 160


def test_temporal_ocr_requires_matching_static_fields_and_consistent_timer():
    history = {}
    first = {
        "asset": {"text": "EUR/USD", "confidence": 0.9, "available": True},
        "timer": {"text": "00:42", "confidence": 0.9, "available": True},
        "payout": {"text": "82%", "confidence": 0.9, "available": True},
        "amount": {"text": "$5.00", "confidence": 0.9, "available": True},
    }
    initial = _stabilize_ocr_results(
        first,
        history,
        observed_at=10.0,
        timeframe_seconds=60,
    )
    assert not any(result["stable"] for result in initial.values())

    second = {
        **first,
        "timer": {"text": "00:41", "confidence": 0.9, "available": True},
    }
    stable = _stabilize_ocr_results(
        second,
        history,
        observed_at=11.0,
        timeframe_seconds=60,
    )

    assert all(stable[name]["stable"] for name in ("asset", "timer", "payout", "amount"))
    assert stable["timer"]["text"] == "00:41"
    assert stable["asset"]["stable_samples"] == 2

    inconsistent = {
        **first,
        "timer": {"text": "00:20", "confidence": 0.9, "available": True},
    }
    unstable_timer = _stabilize_ocr_results(
        inconsistent,
        history,
        observed_at=12.0,
        timeframe_seconds=60,
    )
    assert unstable_timer["timer"]["stable"] is False


def test_pixel_ohlc_store_uses_pixel_normalized_values_and_upserts_frames(tmp_path):
    store = VisionCandleStore(tmp_path / "trades.sqlite3")
    candles = [
        {
            "body_top": 0.3,
            "body_bottom": 0.4,
            "wick_top": 0.2,
            "wick_bottom": 0.5,
            "direction": "CALL",
        },
        {
            "body_top": 0.5,
            "body_bottom": 0.7,
            "wick_top": 0.4,
            "wick_bottom": 0.8,
            "direction": "PUT",
        },
    ]
    observed_at = datetime(2026, 10, 11, 12, 0, 30, tzinfo=timezone.utc).isoformat()
    arguments = {
        "asset": "EUR/USD",
        "timeframe_seconds": 60,
        "candles": candles,
        "observed_at": observed_at,
        "timer_seconds_remaining": 30,
        "chart_width": 800,
        "chart_height": 480,
        "read_quality_score": 88,
    }

    first_write = store.record_frame(**arguments)
    store.record_frame(**arguments)
    saved = store.list_candles(asset="EUR/USD", timeframe_seconds=60)

    assert len(first_write) == 2
    assert store.count(asset="EUR/USD") == 2
    assert len(saved) == 2
    assert saved[0]["price_basis"] == "normalized_chart_pixels"
    assert saved[0]["open"] == 0.6
    assert saved[0]["high"] == 0.8
    assert saved[0]["low"] == 0.5
    assert saved[0]["close"] == 0.7
    assert saved[1]["open"] == 0.5
    assert saved[1]["close"] == pytest.approx(0.3)


def test_chart_detection_and_pixel_candle_history_api_are_local_and_reviewable(tmp_path):
    store = VisionCandleStore(tmp_path / "trades.sqlite3")
    store.record_frame(
        asset="EUR/USD",
        timeframe_seconds=60,
        candles=[
            {
                "body_top": 0.3,
                "body_bottom": 0.4,
                "wick_top": 0.2,
                "wick_bottom": 0.5,
                "direction": "CALL",
            }
        ],
        observed_at="2026-10-11T12:00:00+00:00",
        timer_seconds_remaining=30,
        chart_width=800,
        chart_height=480,
        read_quality_score=85,
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/vision")
    app.state.screen_vision = SimpleNamespace(
        detect_chart_candidates=lambda: {
            "chart_candidates": [
                {
                    "left": -100,
                    "top": 20,
                    "width": 800,
                    "height": 480,
                    "candle_count": 20,
                    "visual_quality_score": 85,
                }
            ]
        }
    )
    app.state.vision_candle_store = store

    with TestClient(app) as client:
        detection = client.get("/api/vision/detect-chart")
        series = client.get("/api/vision/candles?asset=EUR%2FUSD&limit=10")

    assert detection.status_code == 200
    assert detection.json()["chart_candidates"][0]["left"] == -100
    assert series.status_code == 200
    assert series.json()["price_basis"] == "normalized_chart_pixels"
    assert series.json()["candles"][0]["asset"] == "EUR/USD"


def test_pixel_series_dataset_endpoints_report_non_trade_labels(tmp_path):
    from vision.series_labeler import VisionSeriesLabeler

    app = FastAPI()
    app.include_router(router, prefix="/api/vision")
    app.state.vision_series_labeler = VisionSeriesLabeler(tmp_path / "vision.sqlite")

    with TestClient(app) as client:
        status = client.get("/api/vision/series/dataset-status")
        samples = client.get("/api/vision/series/samples?limit=10")

    assert status.status_code == 200
    assert status.json()["source"] == "pixel_series"
    assert status.json()["model_training_enabled"] is False
    assert samples.status_code == 200
    assert samples.json()["samples"] == []
