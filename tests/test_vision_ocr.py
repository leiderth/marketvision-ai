import csv
import io
import json

import cv2
import numpy as np

from vision.ocr_reader import (
    OCRReader,
    _parse_tsv,
    parse_amount,
    parse_asset,
    parse_percent,
    parse_timer,
    parse_trade_outcome,
)
from vision.screen_capture import (
    ScreenVisionService,
    _interpret_ocr,
    _monitor_layout,
    _same_monitor_layout,
    _screen_change_status,
    analyze_chart_image,
)
from vision.ohlc_store import VisionCandleStore


class _FakeScreen:
    monitors = [
        {"left": -1920, "top": 0, "width": 4480, "height": 1440},
        {"left": -1920, "top": 0, "width": 1920, "height": 1080},
        {"left": 0, "top": 0, "width": 2560, "height": 1440},
    ]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def grab(self, _region):
        return np.zeros((20, 40, 4), dtype=np.uint8)


class _ChartCaptureScreen(_FakeScreen):
    def grab(self, region):
        if region["left"] == 20 and region["top"] == 100:
            image = _synthetic_chart()
            return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        return np.full((region["height"], region["width"], 4), 255, dtype=np.uint8)


class _FakeOCR:
    available = True

    def __init__(self, *, asset="EUR/USD"):
        self._asset = asset
        self._index = 0
        self._results = [
            self._asset,
            "00:42",
            "82%",
            "$5.00",
        ]

    def read(self, _image, *, whitelist=None):
        result = self._results[self._index % len(self._results)]
        self._index += 1
        return {
            "text": result,
            "confidence": 0.9,
            "available": True,
            "error": None,
        }


def _ocr_regions():
    return {
        "asset": {"left": 40, "top": 30, "width": 180, "height": 36},
        "timer": {"left": 240, "top": 30, "width": 120, "height": 36},
        "payout": {"left": 390, "top": 30, "width": 100, "height": 36},
        "amount": {"left": 500, "top": 30, "width": 100, "height": 36},
    }


def _synthetic_chart():
    image = np.full((480, 800, 3), 245, dtype=np.uint8)
    for index in range(30):
        x = 20 + index * 24
        center = 250 + (index % 7 - 3) * 9
        color = (0, 190, 0) if index % 2 else (0, 0, 210)
        cv2.line(image, (x + 3, center - 24), (x + 3, center + 27), color, 1)
        cv2.rectangle(image, (x, center - 12), (x + 6, center + 12), color, -1)
    return image


def test_ocr_extractors_parse_asset_timer_payout_and_amount():
    assert parse_asset("EUR/USD OTC", "EUR/USD OTC")["matches_calibration"] is True
    assert parse_asset("GBP/USD", "EUR/USD")["matches_calibration"] is False
    assert parse_timer("00:42") == 42
    assert parse_timer("01:02:03") == 3723
    assert parse_timer("99:99") is None
    assert parse_percent("Payout 82.5%") == 82.5
    assert parse_percent("130%") is None
    assert parse_amount("$12.50") == 12.5
    assert parse_amount("0,75") == 0.75
    assert parse_amount("stake") is None
    assert parse_trade_outcome("Operación cerrada: WIN +$4.10") == "win"
    assert parse_trade_outcome("LOSS -$5.00") == "loss"
    assert parse_trade_outcome("operación abierta") is None
    assert parse_trade_outcome("+$0.00") is None
    assert parse_trade_outcome("no win") is None


def test_tsv_parser_ignores_non_word_rows_and_averages_confidence():
    output = io.StringIO()
    writer = csv.writer(output, delimiter="\t")
    writer.writerow(["level", "conf", "text"])
    writer.writerow(["5", "80", "EUR/USD"])
    writer.writerow(["5", "60", "OTC"])
    writer.writerow(["5", "-1", ""])

    result = _parse_tsv(output.getvalue())

    assert result["text"] == "EUR/USD OTC"
    assert result["confidence"] == 0.7
    assert result["available"] is True


def test_ocr_unavailable_is_reported_without_fabricating_text(tmp_path):
    reader = OCRReader(tmp_path / "missing-tesseract.exe")

    result = reader.read(np.zeros((30, 100), dtype=np.uint8))

    assert result["available"] is False
    assert result["text"] == ""
    assert result["confidence"] == 0


def test_interpreted_ocr_requires_all_fields_and_expected_asset():
    raw = {
        "asset": {"text": "EUR/USD", "confidence": 0.9, "available": True},
        "timer": {"text": "00:30", "confidence": 0.8, "available": True},
        "payout": {"text": "82%", "confidence": 0.9, "available": True},
        "amount": {"text": "$5.00", "confidence": 0.85, "available": True},
    }

    result = _interpret_ocr(raw, "EUR/USD")

    assert result["available"] is True
    assert result["valid"] is True
    assert result["fields"]["timer"]["seconds_remaining"] == 30
    assert result["fields"]["amount"]["value"] == 5
    with_result = _interpret_ocr(
        {
            **raw,
            "outcome": {
                "text": "WIN",
                "confidence": 0.4,
                "available": True,
                "error": None,
            },
        },
        "EUR/USD",
    )
    assert with_result["valid"] is True
    assert with_result["fields"]["outcome"]["result"] == "win"
    assert _interpret_ocr({**raw, "asset": {**raw["asset"], "text": "GBP/USD"}}, "EUR/USD")["valid"] is False


def test_chart_quality_extracts_bodies_and_wicks_from_multiple_candles():
    image = np.full((260, 480, 3), 245, dtype=np.uint8)
    for index in range(24):
        x = 12 + index * 19
        center = 135 + (index % 7 - 3) * 8
        color = (0, 190, 0) if index % 2 else (0, 0, 210)
        cv2.line(image, (x + 2, center - 24), (x + 2, center + 27), color, 1)
        cv2.rectangle(image, (x, center - 12), (x + 4, center + 12), color, -1)

    result = analyze_chart_image(image)

    assert result["candle_count"] >= 20
    assert result["read_quality_score"] >= 55
    assert result["candles"][0]["body_detected"] is True
    assert result["candles"][0]["wick_top"] < result["candles"][0]["body_top"]
    assert result["candles"][0]["wick_bottom"] > result["candles"][0]["body_bottom"]


def test_scene_change_and_monitor_topology_are_detected():
    first_image = np.zeros((160, 320, 3), dtype=np.uint8)
    second_image = np.full((160, 320, 3), 255, dtype=np.uint8)
    first_signature = cv2.resize(
        cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY),
        (32, 32),
        interpolation=cv2.INTER_AREA,
    ) // 16

    change = _screen_change_status(second_image, first_signature, 80)

    assert change["status"] == "chart_moved"
    assert _same_monitor_layout(
        {
            "monitors": [
                {"left": -1920, "top": 0, "width": 1920, "height": 1080, "dpi": 96},
                {"left": 0, "top": 0, "width": 2560, "height": 1440, "dpi": 144},
            ]
        },
        {"left": -1920, "top": 0, "width": 4480, "height": 1440},
        [
            {"left": -1920, "top": 0, "width": 1920, "height": 1080, "dpi": 96},
            {"left": 0, "top": 0, "width": 2560, "height": 1440, "dpi": 144},
        ],
    )


def test_monitor_layout_keeps_each_displays_origin_and_dpi():
    class Screen:
        monitors = [
            {"left": -1920, "top": 0, "width": 4480, "height": 1440},
            {"left": -1920, "top": 0, "width": 1920, "height": 1080},
            {"left": 0, "top": 0, "width": 2560, "height": 1440},
        ]

    layout = _monitor_layout(Screen())

    assert [monitor["left"] for monitor in layout] == [-1920, 0]
    assert all(monitor["dpi"] == 96 for monitor in layout)


def test_calibration_persists_four_ocr_regions_and_virtual_monitor_topology(
    tmp_path, monkeypatch
):
    import vision.screen_capture as capture

    monkeypatch.setattr(capture.mss, "mss", _FakeScreen)
    monkeypatch.setattr(capture, "_system_dpi", lambda: 96)
    service = ScreenVisionService(tmp_path / "calibration.json")

    saved = service.calibrate_region(
        left=20,
        top=100,
        width=800,
        height=480,
        asset="EUR/USD",
        timeframe_seconds=60,
        ocr_regions=_ocr_regions(),
    )

    assert saved["ocr_regions"] == _ocr_regions()
    assert len(saved["monitors"]) == 2
    assert json.loads((tmp_path / "calibration.json").read_text())["asset"] == "EUR/USD"


def test_button_calibration_updates_saved_coordinates_without_changing_chart(
    tmp_path, monkeypatch
):
    import vision.screen_capture as capture

    monkeypatch.setattr(capture.mss, "mss", _FakeScreen)
    monkeypatch.setattr(capture, "_system_dpi", lambda: 96)
    path = tmp_path / "calibration.json"
    service = ScreenVisionService(path)
    service.calibrate_region(
        left=20,
        top=100,
        width=800,
        height=480,
        asset="EUR/USD",
        timeframe_seconds=60,
        ocr_regions=_ocr_regions(),
    )

    result = service.calibrate_buttons(
        click_call=(-1800, 100),
        click_put=(100, 1000),
    )

    assert result["buttons"] == {
        "CALL": {"x": -1800, "y": 100},
        "PUT": {"x": 100, "y": 1000},
    }
    assert service.get_button_position("PUT") == (100, 1000)
    saved = json.loads(path.read_text())
    assert saved["asset"] == "EUR/USD"
    assert saved["ocr_regions"] == _ocr_regions()


def test_button_probe_requires_current_monitor_layout(tmp_path, monkeypatch):
    import vision.screen_capture as capture

    monkeypatch.setattr(capture.mss, "mss", _FakeScreen)
    monkeypatch.setattr(capture, "_system_dpi", lambda: 96)
    service = ScreenVisionService(tmp_path / "calibration.json")
    service.calibrate_region(
        left=20,
        top=100,
        width=800,
        height=480,
        asset="EUR/USD",
        timeframe_seconds=60,
        ocr_regions=_ocr_regions(),
        click_call=(-1800, 100),
        click_put=(100, 1000),
    )

    class ChangedLayoutScreen(_FakeScreen):
        monitors = [
            {"left": -1920, "top": 0, "width": 4480, "height": 1440},
            {"left": -1920, "top": 0, "width": 1920, "height": 1080},
            {"left": 0, "top": 0, "width": 2560, "height": 1200},
        ]

    monkeypatch.setattr(capture.mss, "mss", ChangedLayoutScreen)
    try:
        service.get_button_position("CALL")
    except ValueError as error:
        assert "recalibra" in str(error)
    else:
        raise AssertionError("Button probes must fail when monitor geometry or DPI changes.")


def test_outcome_ocr_region_is_optional_and_persists_for_demo_tracking(tmp_path, monkeypatch):
    import vision.screen_capture as capture

    monkeypatch.setattr(capture.mss, "mss", _FakeScreen)
    monkeypatch.setattr(capture, "_system_dpi", lambda: 96)
    service = ScreenVisionService(tmp_path / "calibration.json")
    regions = {
        **_ocr_regions(),
        "outcome": {"left": 700, "top": 30, "width": 140, "height": 36},
    }

    saved = service.calibrate_region(
        left=20,
        top=100,
        width=800,
        height=480,
        asset="EUR/USD",
        timeframe_seconds=60,
        ocr_regions=regions,
    )

    assert saved["ocr_regions"]["outcome"] == regions["outcome"]


def test_calibration_rejects_a_region_spanning_two_monitors(tmp_path, monkeypatch):
    import vision.screen_capture as capture

    monkeypatch.setattr(capture.mss, "mss", _FakeScreen)
    monkeypatch.setattr(capture, "_system_dpi", lambda: 96)
    service = ScreenVisionService(tmp_path / "calibration.json")

    try:
        service.calibrate_region(
            left=-100,
            top=100,
            width=400,
            height=480,
            asset="EUR/USD",
            timeframe_seconds=60,
            ocr_regions=_ocr_regions(),
        )
    except ValueError as error:
        assert "un solo monitor" in str(error)
    else:
        raise AssertionError("A graph region spanning monitors must be rejected.")


def test_capture_requires_quality_and_matching_ocr_to_be_readable(tmp_path, monkeypatch):
    import vision.screen_capture as capture

    monkeypatch.setattr(capture.mss, "mss", _ChartCaptureScreen)
    monkeypatch.setattr(capture, "_system_dpi", lambda: 96)
    candle_store = VisionCandleStore(tmp_path / "vision.sqlite3")
    service = ScreenVisionService(
        tmp_path / "calibration.json",
        ocr_reader=_FakeOCR(),
        candle_store=candle_store,
    )
    service.calibrate_region(
        left=20,
        top=100,
        width=800,
        height=480,
        asset="EUR/USD",
        timeframe_seconds=60,
        ocr_regions=_ocr_regions(),
    )
    service.OCR_INTERVAL_SECONDS = 0

    first_state = service.capture_and_analyze()
    state = service.capture_and_analyze()

    assert first_state["status"] == "ocr_temporal_unstable"
    assert state["readable"] is True
    assert state["read_quality_score"] >= 55
    assert state["ocr"]["fields"]["timer"]["seconds_remaining"] == 42
    assert state["ocr"]["fields"]["payout"]["percent"] == 82
    assert state["ocr"]["fields"]["amount"]["value"] == 5
    assert state["ohlc_price_basis"] == "normalized_chart_pixels"
    assert state["ohlc_stored_count"] >= 8
    assert candle_store.count(asset="EUR/USD") >= 8


def test_capture_fails_closed_when_ocr_detects_a_different_asset(tmp_path, monkeypatch):
    import vision.screen_capture as capture

    monkeypatch.setattr(capture.mss, "mss", _ChartCaptureScreen)
    monkeypatch.setattr(capture, "_system_dpi", lambda: 96)
    service = ScreenVisionService(
        tmp_path / "calibration.json",
        ocr_reader=_FakeOCR(asset="GBP/USD"),
    )
    service.calibrate_region(
        left=20,
        top=100,
        width=800,
        height=480,
        asset="EUR/USD",
        timeframe_seconds=60,
        ocr_regions=_ocr_regions(),
    )
    service.OCR_INTERVAL_SECONDS = 0

    service.capture_and_analyze()
    state = service.capture_and_analyze()

    assert state["readable"] is False
    assert state["status"] == "asset_changed_recalibrate"
    assert state["needs_recalibration"] is True
