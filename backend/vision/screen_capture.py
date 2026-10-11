"""Local desktop capture and conservative candlestick-shape extraction."""
from __future__ import annotations

import base64
import json
import math
import os
import sqlite3
import tempfile
import threading
import time
from collections import deque
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import mss
import numpy as np

from vision.ocr_reader import (
    OCRReader,
    parse_amount,
    parse_asset,
    parse_percent,
    parse_timer,
    parse_trade_outcome,
)
from vision.ohlc_store import VisionCandleStore

REQUIRED_OCR_REGIONS = {"asset", "timer", "payout", "amount"}
OPTIONAL_OCR_REGIONS = {"outcome"}


class ScreenVisionError(RuntimeError):
    pass


class ScreenVisionService:
    OCR_INTERVAL_SECONDS = 1.0

    def __init__(
        self,
        calibration_path: str | Path,
        *,
        ocr_reader: OCRReader | None = None,
        candle_store: VisionCandleStore | None = None,
    ) -> None:
        self._calibration_path = Path(calibration_path)
        self._lock = threading.RLock()
        self._calibration_error: str | None = None
        self._dpi_error = _enable_per_monitor_dpi_awareness()
        self._calibration = self._load_calibration()
        self._ocr = ocr_reader or OCRReader()
        self._candle_store = candle_store
        self._latest: dict[str, Any] = self._empty_state()
        self._latest_preview: str | None = None
        self._paused = False
        self._last_ocr_at = 0.0
        self._last_candle_persist_at = 0.0
        self._last_ocr_captured_at: str | None = None
        self._ocr_cache: dict[str, dict[str, Any]] = {}
        self._ocr_history: dict[str, deque[dict[str, Any]]] = {}
        self._last_scene_signature: np.ndarray | None = None

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._latest))

    def set_paused(self, paused: bool) -> dict[str, Any]:
        with self._lock:
            self._paused = paused
            if paused:
                self._latest = self._empty_state("capture_paused")
            elif self._calibration is not None:
                self._latest = self._empty_state("waiting_for_first_capture")
            return json.loads(json.dumps(self._latest))

    def get_calibration(self) -> dict[str, Any]:
        with self._lock:
            if self._calibration is None:
                result: dict[str, Any] = {"calibrated": False}
                if self._calibration_error:
                    result["error"] = self._calibration_error
                return result
            return {"calibrated": True, **self._calibration}

    def capture_desktop_preview(self) -> dict[str, Any]:
        with mss.mss() as screen:
            desktop = screen.monitors[0]
            image = np.asarray(screen.grab(desktop))
            monitors = _monitor_layout(screen)
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 72],
        )
        if not encoded_ok:
            raise ScreenVisionError("No se pudo codificar la captura de pantalla.")
        return {
            "image": base64.b64encode(encoded).decode("ascii"),
            "left": desktop["left"],
            "top": desktop["top"],
            "width": desktop["width"],
            "height": desktop["height"],
            "dpi": _system_dpi(),
            "monitors": monitors,
            "dpi_aware": self._dpi_error is None,
            "dpi_error": self._dpi_error,
        }

    def detect_chart_candidates(self) -> dict[str, Any]:
        preview = self.capture_desktop_preview()
        encoded = base64.b64decode(preview["image"], validate=True)
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ScreenVisionError("No se pudo analizar la captura del escritorio.")
        preview["chart_candidates"] = detect_chart_candidates(
            image,
            desktop_left=preview["left"],
            desktop_top=preview["top"],
        )
        return preview

    def calibrate_region(
        self,
        *,
        left: int,
        top: int,
        width: int,
        height: int,
        asset: str,
        timeframe_seconds: int,
        ocr_regions: Mapping[str, Mapping[str, int]],
        click_call: tuple[int, int] | None = None,
        click_put: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in (left, top, width, height))
            or width < 320
            or height < 160
            or width > 8192
            or height > 8192
        ):
            raise ValueError("Selecciona un área de gráfico de al menos 320×160 px.")
        if not isinstance(asset, str) or not asset.strip() or len(asset.strip()) > 80:
            raise ValueError("Indica manualmente el activo mostrado en el gráfico.")
        if (
            isinstance(timeframe_seconds, bool)
            or not isinstance(timeframe_seconds, int)
            or not 1 <= timeframe_seconds <= 86_400
        ):
            raise ValueError("El timeframe debe estar entre 1 y 86400 segundos.")
        if not REQUIRED_OCR_REGIONS.issubset(ocr_regions) or not set(ocr_regions).issubset(
            REQUIRED_OCR_REGIONS | OPTIONAL_OCR_REGIONS
        ):
            raise ValueError("Calibra las cuatro regiones OCR: activo, temporizador, payout y monto.")
        with mss.mss() as screen:
            desktop = screen.monitors[0]
            monitors = _monitor_layout(screen)
        if (
            left < desktop["left"]
            or top < desktop["top"]
            or left + width > desktop["left"] + desktop["width"]
            or top + height > desktop["top"] + desktop["height"]
        ):
            raise ValueError("El área seleccionada queda fuera del escritorio actual.")
        if not any(
            _rectangle_inside_monitor(
                {"left": left, "top": top, "width": width, "height": height},
                monitor,
            )
            for monitor in monitors
        ):
            raise ValueError("El área del gráfico debe quedar dentro de un solo monitor.")
        checked_ocr_regions: dict[str, dict[str, int]] = {}
        for name, region in ocr_regions.items():
            if set(region) != {"left", "top", "width", "height"}:
                raise ValueError(f"La región OCR {name} tiene coordenadas incompletas.")
            roi_left, roi_top, roi_width, roi_height = (
                region["left"], region["top"], region["width"], region["height"]
            )
            if (
                any(isinstance(value, bool) or not isinstance(value, int)
                    for value in (roi_left, roi_top, roi_width, roi_height))
                or roi_width < 12
                or roi_height < 10
                or roi_width > 1600
                or roi_height > 500
                or roi_left < desktop["left"]
                or roi_top < desktop["top"]
                or roi_left + roi_width > desktop["left"] + desktop["width"]
                or roi_top + roi_height > desktop["top"] + desktop["height"]
            ):
                raise ValueError(f"La región OCR {name} está fuera de pantalla o tiene tamaño inválido.")
            if not any(
                _rectangle_inside_monitor(
                    {
                        "left": roi_left,
                        "top": roi_top,
                        "width": roi_width,
                        "height": roi_height,
                    },
                    monitor,
                )
                for monitor in monitors
            ):
                raise ValueError(f"La región OCR {name} debe quedar dentro de un solo monitor.")
            checked_ocr_regions[name] = {
                "left": roi_left,
                "top": roi_top,
                "width": roi_width,
                "height": roi_height,
            }
        buttons = _validated_button_positions(
            click_call,
            click_put,
            desktop=desktop,
            monitors=monitors,
        )
        calibration = {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "screen_width": desktop["width"],
            "screen_height": desktop["height"],
            "screen_left": desktop["left"],
            "screen_top": desktop["top"],
            "dpi": _system_dpi(),
            "monitors": monitors,
            "asset": asset.strip(),
            "timeframe_seconds": timeframe_seconds,
            "ocr_regions": checked_ocr_regions,
            "buttons": buttons,
        }
        _write_json_atomic(self._calibration_path, calibration)
        with self._lock:
            self._calibration = calibration
            self._calibration_error = None
            self._latest_preview = None
            self._last_ocr_at = 0.0
            self._last_candle_persist_at = 0.0
            self._ocr_cache.clear()
            self._ocr_history.clear()
            self._last_scene_signature = None
            self._latest = self._empty_state("waiting_for_first_capture")
        return {"calibrated": True, **calibration}

    def calibrate_buttons(
        self,
        *,
        click_call: tuple[int, int],
        click_put: tuple[int, int],
    ) -> dict[str, Any]:
        with self._lock:
            if self._calibration is None:
                raise ValueError("Primero calibra el área del gráfico y las regiones OCR.")
            calibration = dict(self._calibration)
            with mss.mss() as screen:
                desktop = screen.monitors[0]
                monitors = _monitor_layout(screen)
            if not _same_monitor_layout(calibration, desktop, monitors):
                raise ValueError(
                    "Cambió la resolución, distribución de monitores o DPI; recalibra antes de marcar botones."
                )
            buttons = _validated_button_positions(
                click_call,
                click_put,
                desktop=desktop,
                monitors=monitors,
            )
            updated_calibration = {**calibration, "buttons": buttons}
            _write_json_atomic(self._calibration_path, updated_calibration)
            self._calibration = updated_calibration
            return {
                "calibrated": True,
                "buttons": buttons,
                "screen_width": desktop["width"],
                "screen_height": desktop["height"],
                "screen_left": desktop["left"],
                "screen_top": desktop["top"],
                "dpi": _system_dpi(),
                "monitors": monitors,
            }

    def get_button_position(self, side: str) -> tuple[int, int]:
        if side not in {"CALL", "PUT"}:
            raise ValueError("El botón debe ser CALL o PUT.")
        with self._lock:
            if self._calibration is None:
                raise ValueError("No hay calibración de pantalla guardada.")
            calibration = dict(self._calibration)
            buttons = calibration.get("buttons")
            if not isinstance(buttons, dict) or not {"CALL", "PUT"}.issubset(buttons):
                raise ValueError("Calibra CALL y PUT antes de probar el cursor.")

            call_point = _button_point(buttons["CALL"], "CALL")
            put_point = _button_point(buttons["PUT"], "PUT")
            with mss.mss() as screen:
                desktop = screen.monitors[0]
                monitors = _monitor_layout(screen)
            if not _same_monitor_layout(calibration, desktop, monitors):
                raise ValueError(
                    "Cambió la resolución, distribución de monitores o DPI; recalibra los botones."
                )
            validated = _validated_button_positions(
                call_point,
                put_point,
                desktop=desktop,
                monitors=monitors,
            )
            point = validated[side]
            return point["x"], point["y"]

    def capture_and_analyze(self) -> dict[str, Any]:
        with self._lock:
            if self._paused:
                return json.loads(json.dumps(self._latest))
        calibration = self._calibration
        if calibration is None:
            reason = (
                "invalid_screen_calibration"
                if self._calibration_error
                else "screen_area_not_calibrated"
            )
            state = self._empty_state(
                reason,
                calibration_error=self._calibration_error,
            )
            with self._lock:
                self._latest = state
            return state
        try:
            with mss.mss() as screen:
                desktop = screen.monitors[0]
                monitors = _monitor_layout(screen)
                if not _same_monitor_layout(calibration, desktop, monitors):
                    state = self._empty_state(
                        "screen_layout_or_dpi_changed",
                        needs_recalibration=True,
                        reason_detail="Cambió la resolución, la distribución de monitores o el DPI.",
                    )
                    with self._lock:
                        self._latest = state
                    return state
                if "ocr_regions" not in calibration:
                    state = self._empty_state(
                        "ocr_regions_not_calibrated",
                        needs_recalibration=True,
                        reason_detail="Vuelve a calibrar para marcar activo, temporizador, payout y monto.",
                    )
                    with self._lock:
                        self._latest = state
                    return state
                region = {
                    "left": calibration["left"],
                    "top": calibration["top"],
                    "width": calibration["width"],
                    "height": calibration["height"],
                }
                image = cv2.cvtColor(
                    np.asarray(screen.grab(region)),
                    cv2.COLOR_BGRA2BGR,
                )
                now = time.monotonic()
                if now - self._last_ocr_at >= self.OCR_INTERVAL_SECONDS or not self._ocr_cache:
                    raw_ocr = {
                        name: self._ocr.read(
                            cv2.cvtColor(
                                np.asarray(screen.grab(ocr_region)),
                                cv2.COLOR_BGRA2BGR,
                            ),
                            whitelist=(
                                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/.-"
                                if name == "asset"
                                else (
                                    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                                    "0123456789+-$€£.,% "
                                    if name == "outcome"
                                    else "0123456789:$€£.,% "
                                )
                            ),
                        )
                        for name, ocr_region in calibration["ocr_regions"].items()
                    }
                    self._ocr_cache = _stabilize_ocr_results(
                        raw_ocr,
                        self._ocr_history,
                        observed_at=now,
                        timeframe_seconds=calibration["timeframe_seconds"],
                    )
                    self._last_ocr_at = now
                    self._last_ocr_captured_at = _utc_now()
            analysis = analyze_chart_image(image)
            screen_change = _screen_change_status(
                image,
                self._last_scene_signature,
                analysis["read_quality_score"],
            )
            if self._last_scene_signature is None or screen_change["status"] == "stable":
                self._last_scene_signature = _scene_signature(image)
            ocr_state = _interpret_ocr(self._ocr_cache, calibration["asset"])
            ocr_state["fields"]["outcome"]["captured_at"] = self._last_ocr_captured_at
            ocr_quality = min(
                (
                    ocr_state["fields"][name]["confidence"]
                    for name in REQUIRED_OCR_REGIONS
                ),
                default=0.0,
            )
            quality_score = min(analysis["read_quality_score"], round(ocr_quality * 100))
            block_reason = _vision_block_reason(
                analysis["read_quality_score"],
                ocr_state,
                screen_change,
                dpi_error=self._dpi_error,
            )
            stored_candles: list[dict[str, Any]] = []
            storage_error: str | None = None
            if (
                self._candle_store is not None
                and block_reason is None
                and now - self._last_candle_persist_at >= self.OCR_INTERVAL_SECONDS
                and ocr_state["fields"]["timer"]["seconds_remaining"] is not None
            ):
                try:
                    stored_candles = self._candle_store.record_frame(
                        asset=calibration["asset"],
                        timeframe_seconds=calibration["timeframe_seconds"],
                        candles=analysis.get("candles", []),
                        observed_at=_utc_now(),
                        timer_seconds_remaining=ocr_state["fields"]["timer"]["seconds_remaining"],
                        chart_width=region["width"],
                        chart_height=region["height"],
                        read_quality_score=quality_score,
                    )
                    self._last_candle_persist_at = now
                except sqlite3.Error as exc:
                    storage_error = f"{type(exc).__name__}: {exc}"
                    block_reason = "candle_storage_failed"
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 78],
            )
            if not encoded_ok:
                raise ScreenVisionError("No se pudo codificar la vista previa del gráfico.")
            state = {
                **analysis,
                "readable": block_reason is None,
                "status": block_reason or "reading",
                "reason": _vision_reason(block_reason),
                "asset": calibration["asset"],
                "timeframe_seconds": calibration["timeframe_seconds"],
                "ocr": ocr_state,
                "read_quality_score": quality_score,
                "visual_quality_score": analysis["read_quality_score"],
                "needs_recalibration": block_reason in {
                    "asset_changed_recalibrate",
                    "screen_layout_or_dpi_changed",
                    "chart_moved_recalibrate",
                    "chart_covered_or_unreadable",
                },
                "screen_change": screen_change,
                "monitor_layout": monitors,
                "dpi_aware": self._dpi_error is None,
                "ohlc_price_basis": "normalized_chart_pixels",
                "ohlc_stored_count": (
                    self._candle_store.count(asset=calibration["asset"])
                    if self._candle_store is not None and storage_error is None
                    else 0
                ),
                "ohlc_candles": stored_candles[-8:],
                "ohlc_storage_error": storage_error,
                "calibration": self.get_calibration(),
                "captured_at": _utc_now(),
                "preview": base64.b64encode(encoded).decode("ascii"),
            }
        except (OSError, cv2.error, ScreenVisionError, ValueError, sqlite3.Error) as exc:
            state = self._empty_state("screen_capture_failed", error=str(exc))
        with self._lock:
            self._latest_preview = state.pop("preview", None)
            self._latest = state
        return self.get_state()

    def get_chart_preview(self) -> str | None:
        with self._lock:
            return self._latest_preview

    @staticmethod
    def _empty_state(reason: str = "screen_area_not_calibrated", **extra: Any) -> dict[str, Any]:
        return {
            "source": "local_screen_capture",
            "readable": False,
            "status": reason,
            "reason": _state_reason(reason),
            "asset": None,
            "timeframe_seconds": None,
            "candle_count": 0,
            "trend": "unknown",
            "momentum": "unknown",
            "support_levels": [],
            "resistance_levels": [],
            "pattern": "unknown",
            "directional_scores": {"CALL": None, "PUT": None},
            "read_quality_score": 0,
            "ohlc_price_basis": "normalized_chart_pixels",
            "ohlc_stored_count": 0,
            "ohlc_candles": [],
            "ohlc_storage_error": None,
            "decision": "NO_TRADE",
            "confidence_calibrated": False,
            "calibration_reason": "No hay historial de resultados visuales validado.",
            **extra,
        }

    def _load_calibration(self) -> dict[str, Any] | None:
        if not self._calibration_path.exists():
            return None
        try:
            value = json.loads(self._calibration_path.read_text(encoding="utf-8"))
            required = {
                "left",
                "top",
                "width",
                "height",
                "screen_width",
                "screen_height",
                "screen_left",
                "screen_top",
                "dpi",
                "asset",
                "timeframe_seconds",
                "buttons",
            }
            if (
                not isinstance(value, dict)
                or not required.issubset(value)
                or any(
                    isinstance(value[key], bool) or not isinstance(value[key], int)
                    for key in ("left", "top", "width", "height", "screen_width", "screen_height", "dpi")
                )
                or not isinstance(value["buttons"], dict)
                or ("ocr_regions" in value and not isinstance(value["ocr_regions"], dict))
            ):
                raise ValueError("El archivo de calibración de pantalla es inválido.")
            ocr_regions = value.get("ocr_regions")
            if ocr_regions is not None:
                if (
                    not REQUIRED_OCR_REGIONS.issubset(ocr_regions)
                    or not set(ocr_regions).issubset(
                        REQUIRED_OCR_REGIONS | OPTIONAL_OCR_REGIONS
                    )
                ):
                    raise ValueError("El archivo de calibración no contiene las cuatro regiones OCR.")
                for name, region in ocr_regions.items():
                    if (
                        not isinstance(region, dict)
                        or set(region) != {"left", "top", "width", "height"}
                        or any(
                            isinstance(region[key], bool) or not isinstance(region[key], int)
                            for key in ("left", "top", "width", "height")
                        )
                        or region["width"] < 12
                        or region["height"] < 10
                    ):
                        raise ValueError(f"La región OCR calibrada {name} no es válida.")
            return value
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self._calibration_error = f"{type(exc).__name__}: {exc}"
            return None


def detect_chart_candidates(
    image: np.ndarray,
    *,
    desktop_left: int = 0,
    desktop_top: int = 0,
) -> list[dict[str, Any]]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("La captura de escritorio debe ser BGR de tres canales.")
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation = hsv[:, :, 0], hsv[:, :, 1]
    green = (hue >= 32) & (hue <= 100) & (saturation >= 65)
    red = ((hue <= 12) | (hue >= 168)) & (saturation >= 65)
    colored = green | red
    minimum_column_pixels = max(3, round(height * 0.006))
    active_columns = np.flatnonzero(colored.sum(axis=0) >= minimum_column_pixels)
    horizontal_gap = min(48, max(24, round(width * 0.008)))
    regions: list[dict[str, Any]] = []

    for start, end in _column_groups(active_columns, maximum_gap=horizontal_gap):
        candle_columns = active_columns[
            (active_columns >= start) & (active_columns < end)
        ]
        candle_groups = _column_groups(candle_columns, maximum_gap=3)
        if len(candle_groups) < 8:
            continue
        section = colored[:, start:end]
        active_rows = np.flatnonzero(section.sum(axis=1) >= 2)
        if len(active_rows) < 12:
            continue

        top, bottom = int(active_rows.min()), int(active_rows.max()) + 1
        x_padding = max(24, round((end - start) * 0.08))
        y_padding = max(16, round((bottom - top) * 0.12))
        left = max(0, start - x_padding)
        right = min(width, end + x_padding)
        top = max(0, top - y_padding)
        bottom = min(height, bottom + y_padding)
        if right - left < 320:
            center = (left + right) // 2
            left = max(0, center - 160)
            right = min(width, left + 320)
            left = max(0, right - 320)
        if bottom - top < 160:
            center = (top + bottom) // 2
            top = max(0, center - 80)
            bottom = min(height, top + 160)
            top = max(0, bottom - 160)
        candidate_image = image[top:bottom, left:right]
        try:
            analysis = analyze_chart_image(candidate_image)
        except ValueError:
            continue
        if analysis["candle_count"] < 8:
            continue
        regions.append(
            {
                "left": desktop_left + left,
                "top": desktop_top + top,
                "width": right - left,
                "height": bottom - top,
                "candle_count": analysis["candle_count"],
                "visual_quality_score": analysis["read_quality_score"],
            }
        )
    return sorted(
        regions,
        key=lambda region: (
            region["visual_quality_score"],
            region["candle_count"],
        ),
        reverse=True,
    )[:5]


def _stabilize_ocr_results(
    raw_results: Mapping[str, Mapping[str, Any]],
    history: dict[str, deque[dict[str, Any]]],
    *,
    observed_at: float,
    timeframe_seconds: int,
) -> dict[str, dict[str, Any]]:
    parsers = {
        "asset": lambda text: parse_asset(text, text)["normalized"],
        "timer": parse_timer,
        "payout": parse_percent,
        "amount": parse_amount,
        "outcome": parse_trade_outcome,
    }
    stabilized: dict[str, dict[str, Any]] = {}
    for name, result in raw_results.items():
        parser = parsers.get(name)
        if parser is None:
            stabilized[name] = {**result, "stable": False, "stable_samples": 0}
            continue
        value = parser(str(result.get("text", "")))
        samples = history.setdefault(name, deque(maxlen=3))
        if (
            value is not None
            and result.get("available")
            and float(result.get("confidence", 0)) >= 0.35
        ):
            samples.append(
                {
                    "value": value,
                    "text": str(result.get("text", "")),
                    "confidence": float(result.get("confidence", 0)),
                    "observed_at": observed_at,
                }
            )

        stable = False
        stable_samples = 0
        selected = dict(result)
        if name == "timer":
            if value is not None and samples and samples[-1]["observed_at"] == observed_at:
                stable, stable_samples = _timer_reading_is_stable(
                    list(samples),
                    timeframe_seconds=timeframe_seconds,
                )
            if stable and samples:
                latest = samples[-1]
                selected.update(
                    text=latest["text"],
                    confidence=latest["confidence"],
                )
        elif value is not None:
            matching = [sample for sample in samples if sample["value"] == value]
            counts: dict[Any, int] = {}
            for sample in samples:
                counts[sample["value"]] = counts.get(sample["value"], 0) + 1
            winner = max(counts, key=counts.get) if counts else None
            matching = [sample for sample in samples if sample["value"] == winner]
            stable = value == winner and len(matching) >= 2
            stable_samples = len(matching)
            if stable:
                latest = matching[-1]
                selected.update(
                    text=latest["text"],
                    confidence=min(sample["confidence"] for sample in matching),
                )
        selected["stable"] = stable
        selected["stable_samples"] = stable_samples
        if not stable and result.get("error") is None:
            selected["error"] = "Lectura OCR aún sin confirmar temporalmente."
        stabilized[name] = selected
    return stabilized


def _timer_reading_is_stable(
    samples: list[dict[str, Any]],
    *,
    timeframe_seconds: int,
) -> tuple[bool, int]:
    if len(samples) < 2:
        return False, len(samples)
    previous, current = samples[-2:]
    elapsed = current["observed_at"] - previous["observed_at"]
    if elapsed <= 0 or elapsed > 10:
        return False, 1
    previous_value, current_value = previous["value"], current["value"]
    countdown = (
        current_value <= previous_value
        and previous_value - current_value <= max(3, math.ceil(elapsed) + 2)
    )
    rollover = (
        previous_value <= max(2, math.ceil(elapsed))
        and timeframe_seconds - 3 <= current_value <= timeframe_seconds + 2
    )
    return countdown or rollover, 2


def analyze_chart_image(image: np.ndarray) -> dict[str, Any]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("La imagen de gráfico debe ser BGR de tres canales.")
    height, width = image.shape[:2]
    if width < 320 or height < 160:
        raise ValueError("El recorte del gráfico debe ser al menos 320×160 px.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    green_mask = (hue >= 32) & (hue <= 100) & (saturation >= 65)
    red_mask = ((hue <= 12) | (hue >= 168)) & (saturation >= 65)
    colored = green_mask | red_mask
    active_columns = np.flatnonzero(colored.sum(axis=0) >= max(4, round(height * 0.015)))
    groups = _column_groups(active_columns, maximum_gap=3)
    candles: list[dict[str, Any]] = []
    for start, end in groups:
        if end - start < 2 or end - start > max(28, round(width * 0.04)):
            continue
        section = colored[:, start:end]
        ys, xs = np.where(section)
        if len(ys) < max(8, round(height * 0.025)):
            continue
        mid_x = start + (end - start) // 2
        local_green = int(green_mask[:, start:end].sum())
        local_red = int(red_mask[:, start:end].sum())
        if max(local_green, local_red) < 6:
            continue
        row_density = section.sum(axis=1)
        body_rows = np.flatnonzero(row_density >= max(2, math.ceil((end - start) * 0.55)))
        body_top = int(body_rows.min()) if len(body_rows) else int(ys.min())
        body_bottom = int(body_rows.max()) if len(body_rows) else int(ys.max())
        candles.append(
            {
                "x": mid_x / width,
                "top": float(ys.min() / height),
                "bottom": float(ys.max() / height),
                "center": float((body_top + body_bottom) / (2 * height)),
                "body_size": float(max(body_bottom - body_top, 1) / height),
                "wick_top": float(ys.min() / height),
                "wick_bottom": float(ys.max() / height),
                "body_top": float(body_top / height),
                "body_bottom": float(body_bottom / height),
                "body_detected": bool(len(body_rows)),
                "color_ratio": max(local_green, local_red) / max(local_green + local_red, 1),
                "direction": "CALL" if local_green > local_red else "PUT",
            }
        )
    candles.sort(key=lambda candle: candle["x"])
    if len(candles) < 8:
        return {
            "candle_count": len(candles),
            "read_quality_score": min(35, len(candles) * 4),
            "market_regime": "unknown",
            "trend": "unknown",
            "momentum": "unknown",
            "support_levels": [],
            "resistance_levels": [],
            "pattern": "unknown",
            "directional_scores": {"CALL": None, "PUT": None},
        }

    centers = np.asarray([candle["center"] for candle in candles], dtype=float)
    tail = centers[-min(20, len(centers)) :]
    x = np.arange(len(tail), dtype=float)
    slope = float(np.polyfit(x, tail, 1)[0])
    normalized_slope = slope / max(float(np.std(tail)), 0.01)
    trend = "up" if normalized_slope < -0.12 else "down" if normalized_slope > 0.12 else "sideways"
    recent = candles[-min(5, len(candles)) :]
    recent_call = sum(candle["direction"] == "CALL" for candle in recent)
    recent_put = len(recent) - recent_call
    momentum = "bullish" if recent_call >= 4 else "bearish" if recent_put >= 4 else "mixed"
    pivots = _pivot_levels(centers)
    last = candles[-1]
    previous = candles[-2]
    pattern = (
        "bullish_engulfing_shape"
        if previous["direction"] == "PUT"
        and last["direction"] == "CALL"
        and last["body_size"] > previous["body_size"] * 1.25
        else "bearish_engulfing_shape"
        if previous["direction"] == "CALL"
        and last["direction"] == "PUT"
        and last["body_size"] > previous["body_size"] * 1.25
        else "none_detected"
    )

    candle_pressure = (recent_call - recent_put) / len(recent)
    trend_pressure = max(-1.0, min(1.0, -normalized_slope * 3))
    call_raw_score = max(
        0.0,
        min(1.0, 0.5 + 0.25 * candle_pressure + 0.25 * trend_pressure),
    )
    put_raw_score = 1.0 - call_raw_score
    color_density = float(colored.mean())
    count_quality = min(100.0, len(candles) * 5.0)
    density_quality = (
        100.0
        if 0.015 <= color_density <= 0.35
        else max(0.0, min(100.0, color_density / 0.015 * 100))
    )
    shape_quality = 100.0 * sum(candle["body_detected"] for candle in candles) / len(candles)
    read_quality = round(
        0.55 * count_quality + 0.25 * density_quality + 0.20 * shape_quality
    )
    return {
        "candle_count": len(candles),
        "read_quality_score": read_quality,
        "market_regime": _visual_market_regime(trend, momentum),
        "trend": trend,
        "momentum": momentum,
        "support_levels": pivots["support"],
        "resistance_levels": pivots["resistance"],
        "pattern": pattern,
        "directional_scores": {"CALL": call_raw_score, "PUT": put_raw_score},
        "raw_score_note": (
            "Puntuaciones heurísticas de forma visual; no son probabilidades ni "
            "habilitan operaciones sin calibración histórica walk-forward."
        ),
        "candles": candles[-40:],
    }


def _visual_market_regime(trend: str, momentum: str) -> str:
    if trend == "up" and momentum == "bullish":
        return "visual_trending_up"
    if trend == "down" and momentum == "bearish":
        return "visual_trending_down"
    if trend == "sideways" and momentum == "mixed":
        return "visual_range"
    return "visual_transition"


def _interpret_ocr(
    ocr_results: Mapping[str, Mapping[str, Any]],
    expected_asset: str,
) -> dict[str, Any]:
    asset_result = ocr_results.get("asset", {})
    timer_result = ocr_results.get("timer", {})
    payout_result = ocr_results.get("payout", {})
    amount_result = ocr_results.get("amount", {})
    outcome_result = ocr_results.get("outcome", {})
    asset = parse_asset(str(asset_result.get("text", "")), expected_asset)
    timer_seconds = parse_timer(str(timer_result.get("text", "")))
    payout_pct = parse_percent(str(payout_result.get("text", "")))
    amount = parse_amount(str(amount_result.get("text", "")))
    fields = {
        "asset": {
            **asset,
            "confidence": round(float(asset_result.get("confidence", 0)), 3),
            "error": asset_result.get("error"),
            "stable": bool(asset_result.get("stable", True)),
            "stable_samples": int(asset_result.get("stable_samples", 0)),
        },
        "timer": {
            "text": timer_result.get("text", ""),
            "seconds_remaining": timer_seconds,
            "confidence": round(float(timer_result.get("confidence", 0)), 3),
            "error": timer_result.get("error"),
            "stable": bool(timer_result.get("stable", True)),
            "stable_samples": int(timer_result.get("stable_samples", 0)),
        },
        "payout": {
            "text": payout_result.get("text", ""),
            "percent": payout_pct,
            "confidence": round(float(payout_result.get("confidence", 0)), 3),
            "error": payout_result.get("error"),
            "stable": bool(payout_result.get("stable", True)),
            "stable_samples": int(payout_result.get("stable_samples", 0)),
        },
        "amount": {
            "text": amount_result.get("text", ""),
            "value": amount,
            "confidence": round(float(amount_result.get("confidence", 0)), 3),
            "error": amount_result.get("error"),
            "stable": bool(amount_result.get("stable", True)),
            "stable_samples": int(amount_result.get("stable_samples", 0)),
        },
        "outcome": {
            "configured": "outcome" in ocr_results,
            "available": bool(outcome_result.get("available")),
            "text": outcome_result.get("text", ""),
            "result": parse_trade_outcome(str(outcome_result.get("text", ""))),
            "confidence": round(float(outcome_result.get("confidence", 0)), 3),
            "error": outcome_result.get("error"),
            "stable": bool(outcome_result.get("stable", True)),
            "stable_samples": int(outcome_result.get("stable_samples", 0)),
        },
    }
    available = all(
        bool(ocr_results.get(name, {}).get("available"))
        for name in REQUIRED_OCR_REGIONS
    )
    valid = (
        asset["matches_calibration"]
        and timer_seconds is not None
        and payout_pct is not None
        and amount is not None
        and all(
            fields[name]["confidence"] >= 0.35
            and fields[name]["stable"]
            for name in REQUIRED_OCR_REGIONS
        )
    )
    return {
        "available": available,
        "valid": valid,
        "fields": fields,
    }


def _vision_block_reason(
    visual_quality: int,
    ocr: Mapping[str, Any],
    screen_change: Mapping[str, str],
    *,
    dpi_error: str | None,
) -> str | None:
    if dpi_error:
        return "dpi_awareness_unavailable"
    if screen_change["status"] == "chart_moved":
        return "chart_moved_recalibrate"
    if visual_quality < 55:
        return "chart_covered_or_unreadable"
    if not ocr.get("available"):
        return "ocr_unavailable"
    if any(
        not ocr["fields"][name]["stable"]
        for name in REQUIRED_OCR_REGIONS
    ):
        return "ocr_temporal_unstable"
    asset = ocr["fields"]["asset"]
    if (
        asset["stable"]
        and asset["normalized"]
        and not asset["matches_calibration"]
    ):
        return "asset_changed_recalibrate"
    if not ocr.get("valid"):
        return "ocr_low_quality"
    return None


def _vision_reason(status: str | None) -> str | None:
    reasons = {
        "dpi_awareness_unavailable": "No se pudo asegurar el escalado DPI por monitor; lectura bloqueada.",
        "chart_moved_recalibrate": "La imagen cambió bruscamente; revisa el gráfico y recalibra si se movió.",
        "chart_covered_or_unreadable": "Gráfico tapado o ilegible; NO OPERAR hasta recuperar una lectura clara.",
        "ocr_unavailable": "OCR local no disponible; no se puede verificar activo, timer, payout y monto.",
        "ocr_temporal_unstable": "Esperando lecturas OCR consecutivas coherentes; operación bloqueada hasta confirmar los campos.",
        "candle_storage_failed": "No se pudieron guardar las velas reconstruidas; la lectura queda bloqueada.",
        "asset_changed_recalibrate": "El activo leído no coincide con el calibrado; confirma la pantalla y recalibra.",
        "ocr_low_quality": "OCR incompleto o de baja confianza; verifica sus cuatro regiones calibradas.",
    }
    return reasons.get(status) if status else None


def _state_reason(status: str) -> str:
    return (
        _vision_reason(status)
        or {
            "screen_area_not_calibrated": "Selecciona y calibra el área del gráfico.",
            "invalid_screen_calibration": "La calibración guardada es inválida; vuelve a calibrar.",
            "screen_layout_or_dpi_changed": "Cambió el diseño o DPI de las pantallas; vuelve a calibrar.",
            "screen_resolution_or_dpi_changed": "Cambió la resolución o DPI; vuelve a calibrar.",
            "ocr_regions_not_calibrated": "Marca las regiones de activo, temporizador, payout y monto.",
            "candle_storage_failed": "No se pudieron guardar las velas reconstruidas; lectura bloqueada.",
            "waiting_for_first_capture": "Esperando la primera captura del gráfico.",
            "capture_paused": "Captura pausada.",
            "screen_capture_failed": "Falló la captura local de pantalla.",
        }.get(status, status.replace("_", " "))
    )


def _scene_signature(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reduced = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return reduced // 16


def _screen_change_status(
    image: np.ndarray,
    previous_signature: np.ndarray | None,
    visual_quality: int,
) -> dict[str, str]:
    if visual_quality < 35:
        return {"status": "chart_covered", "reason": "La región ya no parece un gráfico legible."}
    if previous_signature is None:
        return {"status": "initializing", "reason": "Se estableció la referencia visual."}
    distance = float(np.mean(_scene_signature(image) != previous_signature))
    if distance >= 0.78:
        return {
            "status": "chart_moved",
            "reason": f"Cambio visual abrupto detectado ({distance:.0%}).",
        }
    return {"status": "stable", "reason": "La región conserva una estructura visual similar."}


def _monitor_layout(screen: Any) -> list[dict[str, int]]:
    monitors = screen.monitors[1:] or screen.monitors[:1]
    return [
        {
            "left": int(monitor["left"]),
            "top": int(monitor["top"]),
            "width": int(monitor["width"]),
            "height": int(monitor["height"]),
            "dpi": _monitor_dpi(
                int(monitor["left"]) + int(monitor["width"]) // 2,
                int(monitor["top"]) + int(monitor["height"]) // 2,
            ),
        }
        for monitor in monitors
    ]


def _rectangle_inside_monitor(
    rectangle: Mapping[str, int],
    monitor: Mapping[str, int],
) -> bool:
    return (
        monitor["left"] <= rectangle["left"]
        and monitor["top"] <= rectangle["top"]
        and rectangle["left"] + rectangle["width"] <= monitor["left"] + monitor["width"]
        and rectangle["top"] + rectangle["height"] <= monitor["top"] + monitor["height"]
    )


def _button_point(value: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, dict)
        or set(value) != {"x", "y"}
        or any(
            isinstance(value.get(key), bool) or not isinstance(value.get(key), int)
            for key in ("x", "y")
        )
    ):
        raise ValueError(f"La coordenada guardada para {name} no es válida; recalibra.")
    return value["x"], value["y"]


def _validated_button_positions(
    click_call: tuple[int, int] | None,
    click_put: tuple[int, int] | None,
    *,
    desktop: Mapping[str, int],
    monitors: list[dict[str, int]],
) -> dict[str, dict[str, int]]:
    if (click_call is None) != (click_put is None):
        raise ValueError("Calibra CALL y PUT juntos o deja ambos sin configurar.")
    if click_call is None or click_put is None:
        return {}

    buttons: dict[str, dict[str, int]] = {}
    for name, point in (("CALL", click_call), ("PUT", click_put)):
        if (
            not isinstance(point, (tuple, list))
            or len(point) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in point)
        ):
            raise ValueError(f"La coordenada {name} no es válida.")
        x, y = point
        if (
            not (desktop["left"] <= x < desktop["left"] + desktop["width"])
            or not (desktop["top"] <= y < desktop["top"] + desktop["height"])
        ):
            raise ValueError(f"La coordenada {name} queda fuera del escritorio.")
        if not any(
            monitor["left"] <= x < monitor["left"] + monitor["width"]
            and monitor["top"] <= y < monitor["top"] + monitor["height"]
            for monitor in monitors
        ):
            raise ValueError(f"La coordenada {name} no cae dentro de un monitor activo.")
        buttons[name] = {"x": x, "y": y}
    return buttons


def _same_monitor_layout(
    calibration: Mapping[str, Any],
    desktop: Mapping[str, int],
    monitors: list[dict[str, int]],
) -> bool:
    calibrated_monitors = calibration.get("monitors")
    if isinstance(calibrated_monitors, list):
        return calibrated_monitors == monitors
    return (
        desktop["width"] == calibration["screen_width"]
        and desktop["height"] == calibration["screen_height"]
        and desktop["left"] == calibration["screen_left"]
        and desktop["top"] == calibration["screen_top"]
        and _system_dpi() == calibration["dpi"]
    )


def _enable_per_monitor_dpi_awareness() -> str | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetProcessDpiAwarenessContext.restype = ctypes.c_bool
        enabled = user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        if enabled:
            return None
        error = ctypes.get_last_error()
        user32.GetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        user32.GetAwarenessFromDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.GetAwarenessFromDpiAwarenessContext.restype = ctypes.c_int
        context = user32.GetThreadDpiAwarenessContext()
        awareness = user32.GetAwarenessFromDpiAwarenessContext(context)
        if awareness == 2:
            return None
        return (
            "Per-monitor DPI awareness could not be enabled "
            f"(Windows error {error}, awareness {awareness})."
        )
    except (AttributeError, OSError, ValueError) as exc:
        return f"Per-monitor DPI awareness is unavailable: {type(exc).__name__}."


def _monitor_dpi(x: int, y: int) -> int:
    if os.name != "nt":
        return 96
    try:
        import ctypes
        from ctypes import wintypes

        class Point(ctypes.Structure):
            _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

        point = Point(x, y)
        user32 = ctypes.windll.user32
        user32.MonitorFromPoint.argtypes = [Point, wintypes.DWORD]
        user32.MonitorFromPoint.restype = wintypes.HMONITOR
        monitor = user32.MonitorFromPoint(point, 2)
        dpi_x = wintypes.UINT()
        dpi_y = wintypes.UINT()
        result = ctypes.windll.shcore.GetDpiForMonitor(
            monitor,
            0,
            ctypes.byref(dpi_x),
            ctypes.byref(dpi_y),
        )
        if result == 0 and dpi_x.value:
            return int(dpi_x.value)
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return _system_dpi()


def _column_groups(columns: np.ndarray, *, maximum_gap: int) -> list[tuple[int, int]]:
    if len(columns) == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = previous = int(columns[0])
    for value in columns[1:]:
        current = int(value)
        if current - previous > maximum_gap + 1:
            groups.append((start, previous + 1))
            start = current
        previous = current
    groups.append((start, previous + 1))
    return groups


def _pivot_levels(centers: np.ndarray) -> dict[str, list[float]]:
    highs: list[float] = []
    lows: list[float] = []
    for index in range(1, len(centers) - 1):
        if centers[index] > centers[index - 1] and centers[index] >= centers[index + 1]:
            highs.append(float(centers[index]))
        if centers[index] < centers[index - 1] and centers[index] <= centers[index + 1]:
            lows.append(float(centers[index]))
    return {
        "support": sorted(lows[-3:]),
        "resistance": sorted(highs[-3:]),
    }


def _system_dpi() -> int:
    if os.name != "nt":
        return 96
    try:
        import ctypes

        dpi = int(ctypes.windll.user32.GetDpiForSystem())
        return dpi if dpi > 0 else 96
    except (AttributeError, OSError, ValueError):
        return 96


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(payload, temporary, sort_keys=True, allow_nan=False)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
