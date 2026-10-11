"""Local storage and visual-only replay for saved chart captures."""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from vision.screen_capture import analyze_chart_image

_CAPTURE_ID = re.compile(r"capture-[a-f0-9]{32}\Z")


class VisionReplayError(RuntimeError):
    pass


class VisionReplayStore:
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._lock = threading.RLock()

    def save_capture(
        self,
        image_base64: str | None,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not image_base64 or not state.get("captured_at"):
            raise ValueError("No hay una captura reciente del gráfico para guardar.")
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise VisionReplayError("La captura del gráfico no contiene una imagen válida.") from exc

        try:
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise VisionReplayError("OpenCV no pudo decodificar la captura de replay.") from exc
        if image is None or image.shape[1] < 320 or image.shape[0] < 160:
            raise VisionReplayError("La captura guardada no tiene el tamaño mínimo de gráfico.")

        self._directory.mkdir(parents=True, exist_ok=True)
        capture_id = f"capture-{os.urandom(16).hex()}"
        image_path = self._directory / f"{capture_id}.jpg"
        metadata_path = self._directory / f"{capture_id}.json"
        metadata = {
            "capture_id": capture_id,
            "captured_at": str(state["captured_at"]),
            "saved_at": _utc_now(),
            "asset": state.get("asset") if isinstance(state.get("asset"), str) else None,
            "timeframe_seconds": (
                state.get("timeframe_seconds")
                if isinstance(state.get("timeframe_seconds"), int)
                and not isinstance(state.get("timeframe_seconds"), bool)
                else None
            ),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "visual_quality_score": state.get("visual_quality_score"),
        }
        with self._lock:
            try:
                _write_bytes_atomic(image_path, image_bytes)
                _write_json_atomic(metadata_path, metadata)
            except OSError:
                image_path.unlink(missing_ok=True)
                raise
        return {**metadata, "size_bytes": len(image_bytes)}

    def list_captures(self) -> list[dict[str, Any]]:
        if not self._directory.exists():
            return []
        captures: list[dict[str, Any]] = []
        with self._lock:
            for metadata_path in self._directory.glob("capture-*.json"):
                metadata = self._read_metadata(metadata_path)
                capture_id = metadata.get("capture_id")
                if (
                    not isinstance(capture_id, str)
                    or not _CAPTURE_ID.fullmatch(capture_id)
                    or metadata_path.stem != capture_id
                ):
                    raise VisionReplayError(
                        f"El archivo de replay {metadata_path.name} tiene un identificador inválido."
                    )
                image_path = self._directory / f"{capture_id}.jpg"
                if not image_path.is_file():
                    raise VisionReplayError(
                        f"Falta la imagen de la captura de replay {capture_id}."
                    )
                captures.append(
                    {
                        **metadata,
                        "size_bytes": image_path.stat().st_size,
                    }
                )
        return sorted(captures, key=lambda item: item["saved_at"], reverse=True)

    def replay_capture(self, capture_id: str) -> dict[str, Any]:
        if not isinstance(capture_id, str) or not _CAPTURE_ID.fullmatch(capture_id):
            raise ValueError("El identificador de replay no es válido.")
        metadata_path = self._directory / f"{capture_id}.json"
        image_path = self._directory / f"{capture_id}.jpg"
        with self._lock:
            if not metadata_path.is_file() or not image_path.is_file():
                raise FileNotFoundError(f"No existe la captura de replay {capture_id}.")
            metadata = self._read_metadata(metadata_path)
            if metadata["capture_id"] != capture_id:
                raise VisionReplayError(f"El índice de replay {capture_id} no coincide con su archivo.")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise VisionReplayError(f"No se pudo leer la imagen de replay {capture_id}.")

        analysis = analyze_chart_image(image)
        visual_quality = analysis["read_quality_score"]
        return {
            "source": "saved_chart_capture_replay",
            "capture_id": capture_id,
            "captured_at": metadata["captured_at"],
            "asset": metadata.get("asset"),
            "timeframe_seconds": metadata.get("timeframe_seconds"),
            "readable": visual_quality >= 55,
            "read_quality_score": visual_quality,
            "visual_quality_score": visual_quality,
            "candle_count": analysis["candle_count"],
            "trend": analysis["trend"],
            "momentum": analysis["momentum"],
            "pattern": analysis["pattern"],
            "support_levels": analysis["support_levels"],
            "resistance_levels": analysis["resistance_levels"],
            "directional_scores": analysis["directional_scores"],
            "decision": "NO_TRADE",
            "reason": (
                "Replay visual offline: no ejecuta operaciones ni reconstruye OCR, "
                "temporizador o probabilidad histórica."
            ),
        }

    def _read_metadata(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VisionReplayError(f"No se pudo leer el índice de replay {path.name}.") from exc
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("capture_id"), str)
            or not _CAPTURE_ID.fullmatch(value["capture_id"])
            or not isinstance(value.get("captured_at"), str)
            or not isinstance(value.get("saved_at"), str)
        ):
            raise VisionReplayError(f"El índice de replay {path.name} está incompleto.")
        return value


def _write_bytes_atomic(path: Path, contents: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    _write_bytes_atomic(path, encoded)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
