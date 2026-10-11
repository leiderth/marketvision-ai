"""Offline Tesseract OCR for calibrated screen regions."""
from __future__ import annotations

import csv
import io
import math
import os
import re
import shutil
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class OCRReader:
    def __init__(self, executable: str | Path | None = None) -> None:
        configured = str(executable or os.environ.get("TESSERACT_CMD", "")).strip()
        if configured:
            self._executable = configured
        else:
            self._executable = shutil.which("tesseract") or ""
        tessdata_dir = os.environ.get("TESSDATA_DIR")
        if not tessdata_dir and configured:
            candidate = Path(configured).parent / "tessdata"
            tessdata_dir = str(candidate) if candidate.is_dir() else None
        self._tessdata_dir = tessdata_dir

    @property
    def available(self) -> bool:
        return bool(self._executable and Path(self._executable).is_file())

    def read(self, image: np.ndarray, *, whitelist: str | None = None) -> dict[str, Any]:
        if not self.available:
            return {
                "text": "",
                "confidence": 0.0,
                "available": False,
                "error": "Tesseract local no está disponible.",
            }
        if image.ndim not in (2, 3) or image.size == 0:
            raise ValueError("La región OCR está vacía o tiene un formato inválido.")

        prepared = _prepare_ocr_image(image)
        success, encoded = cv2.imencode(".png", prepared)
        if not success:
            raise RuntimeError("No se pudo codificar la región para OCR.")
        command = [
            self._executable,
            "stdin",
            "stdout",
            "-l",
            "eng",
            "--oem",
            "1",
            "--psm",
            "7",
        ]
        if self._tessdata_dir:
            command.extend(["--tessdata-dir", self._tessdata_dir])
        command.extend(["-c", "preserve_interword_spaces=1"])
        if whitelist:
            command.extend(["-c", f"tessedit_char_whitelist={whitelist}"])
        command.append("tsv")
        environment = os.environ.copy()
        try:
            result = subprocess.run(
                command,
                input=encoded.tobytes(),
                capture_output=True,
                check=False,
                timeout=3,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "text": "",
                "confidence": 0.0,
                "available": False,
                "error": f"No se pudo ejecutar Tesseract: {type(exc).__name__}.",
            }
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            return {
                "text": "",
                "confidence": 0.0,
                "available": True,
                "error": detail or f"Tesseract terminó con código {result.returncode}.",
            }
        return _parse_tsv(result.stdout.decode("utf-8", errors="replace"))


def _prepare_ocr_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    scale = max(2, min(5, 96 // max(1, min(height, width))))
    enlarged = cv2.resize(
        gray,
        (width * scale, height * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    return cv2.threshold(
        cv2.GaussianBlur(enlarged, (3, 3), 0),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]


def _parse_tsv(content: str) -> dict[str, Any]:
    rows = csv.DictReader(io.StringIO(content), delimiter="\t")
    words: list[str] = []
    confidences: list[float] = []
    for row in rows:
        text = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf") or -1)
        except ValueError:
            continue
        if text and math.isfinite(confidence) and confidence >= 0:
            words.append(text)
            confidences.append(confidence / 100)
    return {
        "text": " ".join(words),
        "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "available": True,
        "error": None if words else "OCR sin texto reconocible.",
    }


def parse_asset(text: str, expected_asset: str) -> dict[str, Any]:
    observed = _normalize_asset(text)
    expected = _normalize_asset(expected_asset)
    similarity = SequenceMatcher(None, observed, expected).ratio() if observed and expected else 0.0
    matched = bool(observed and expected and similarity >= 0.82)
    return {
        "value": text.strip() or None,
        "normalized": observed or None,
        "matches_calibration": matched,
        "similarity": round(similarity, 3),
    }


def parse_timer(text: str) -> int | None:
    match = re.search(r"(?<!\d)(\d{1,2}):([0-5]\d)(?::([0-5]\d))?(?!\d)", text)
    if not match:
        return None
    first, second, third = match.groups()
    if third is not None:
        return int(first) * 3600 + int(second) * 60 + int(third)
    return int(first) * 60 + int(second)


def parse_percent(text: str) -> float | None:
    match = re.search(r"(?<!\d)(\d{1,3}(?:[.,]\d{1,2})?)\s*%", text)
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    return value if 0 < value <= 100 else None


def parse_amount(text: str) -> float | None:
    cleaned = text.replace(" ", "")
    match = re.search(r"(?<!\d)(?:[$€£])?(\d+(?:[.,]\d{1,2})?)(?!\d)", cleaned)
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    return value if math.isfinite(value) and value > 0 else None


def parse_trade_outcome(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text.strip().casefold())
    if re.search(r"\b(no\s+win|not\s+won|sin\s+ganancia)\b", normalized):
        return None
    if re.search(r"\b(win|won|profit|ganado|ganancia|beneficio)\b", normalized):
        return "win"
    if re.search(r"\b(loss|lost|perdido|perdida|pérdida)\b", normalized):
        return "loss"
    signed_result = re.search(
        r"(?<![\w.])([+-])\s*(?:[$€£])?\s*\d+(?:[.,]\d{1,2})?(?!\d)",
        normalized,
    )
    if signed_result:
        amount_text = (
            signed_result.group(0)
            .replace("$", "")
            .replace("€", "")
            .replace("£", "")
            .replace(",", ".")
            .replace("+", "")
            .replace("-", "")
            .strip()
        )
        amount = float(amount_text)
        if amount > 0:
            return "win" if signed_result.group(1) == "+" else "loss"
    return None


def _normalize_asset(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())
