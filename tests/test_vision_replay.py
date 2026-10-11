import base64

import cv2
import numpy as np
import pytest

from vision.replay import VisionReplayError, VisionReplayStore


def _chart_capture_base64() -> str:
    image = np.full((480, 800, 3), 245, dtype=np.uint8)
    for index in range(30):
        x = 20 + index * 24
        center = 250 + (index % 7 - 3) * 9
        color = (0, 190, 0) if index % 2 else (0, 0, 210)
        cv2.line(image, (x + 3, center - 24), (x + 3, center + 27), color, 1)
        cv2.rectangle(image, (x, center - 12), (x + 6, center + 12), color, -1)
    encoded_ok, encoded = cv2.imencode(".jpg", image)
    assert encoded_ok
    return base64.b64encode(encoded).decode("ascii")


def test_replay_saves_lists_and_reproduces_capture_without_live_screen(tmp_path, monkeypatch):
    store = VisionReplayStore(tmp_path / "vision_replay")
    capture = store.save_capture(
        _chart_capture_base64(),
        {
            "captured_at": "2026-07-15T12:30:00+00:00",
            "asset": "EUR/USD",
            "timeframe_seconds": 60,
            "visual_quality_score": 88,
        },
    )
    monkeypatch.setattr(
        "vision.screen_capture.mss.mss",
        lambda: (_ for _ in ()).throw(AssertionError("Replay must not capture the live screen.")),
    )

    listed = store.list_captures()
    replayed = store.replay_capture(capture["capture_id"])

    assert listed[0]["capture_id"] == capture["capture_id"]
    assert replayed["source"] == "saved_chart_capture_replay"
    assert replayed["asset"] == "EUR/USD"
    assert replayed["candle_count"] >= 8
    assert replayed["decision"] == "NO_TRADE"
    assert "no ejecuta operaciones" in replayed["reason"]


def test_replay_requires_a_captured_image_and_valid_capture_id(tmp_path):
    store = VisionReplayStore(tmp_path / "vision_replay")

    with pytest.raises(ValueError, match="captura reciente"):
        store.save_capture(None, {})
    with pytest.raises(ValueError, match="identificador"):
        store.replay_capture("..\\outside")


def test_replay_surfaces_corrupt_saved_image(tmp_path):
    store = VisionReplayStore(tmp_path / "vision_replay")
    capture = store.save_capture(
        _chart_capture_base64(),
        {"captured_at": "2026-07-15T12:30:00+00:00"},
    )
    (tmp_path / "vision_replay" / f"{capture['capture_id']}.jpg").write_bytes(b"not an image")

    with pytest.raises(VisionReplayError, match="No se pudo leer"):
        store.replay_capture(capture["capture_id"])
