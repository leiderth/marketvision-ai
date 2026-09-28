"""Fail-closed, latched feature-distribution monitoring for active models."""
from __future__ import annotations

import json
import math
import os
import tempfile
from collections import deque
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.schema import FEATURE_NAMES


class FeatureDriftMonitor:
    def __init__(
        self,
        reference_means: tuple[float, ...],
        reference_scales: tuple[float, ...],
        *,
        window_size: int = 60,
        min_samples: int = 30,
        threshold: float = 2.5,
        state_path: str | Path | None = None,
    ) -> None:
        if (
            len(reference_means) != len(FEATURE_NAMES)
            or len(reference_scales) != len(FEATURE_NAMES)
            or any(not math.isfinite(value) for value in (*reference_means, *reference_scales))
            or any(value <= 0 for value in reference_scales)
            or window_size < min_samples
            or min_samples <= 1
            or not math.isfinite(threshold)
            or threshold <= 0
        ):
            raise ValueError("Configuración del detector de drift inválida.")
        self._means = reference_means
        self._scales = reference_scales
        self._window_size = window_size
        self._min_samples = min_samples
        self._threshold = threshold
        self._samples: deque[tuple[float, ...]] = deque(maxlen=window_size)
        self._last_timestamp: datetime | None = None
        self._drifted = False
        self._drift_features: tuple[str, ...] = ()
        self._state_path = Path(state_path) if state_path is not None else None
        self._load_state()

    def observe(
        self,
        features: Mapping[str, float],
        timestamp: str,
    ) -> dict[str, Any]:
        try:
            observed_at = datetime.fromisoformat(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("El timestamp de inferencia para drift es inválido.") from exc
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("El timestamp de inferencia para drift requiere zona horaria.")
        observed_at = observed_at.astimezone(timezone.utc)
        if self._last_timestamp is not None and observed_at <= self._last_timestamp:
            return self.get_status()
        values = tuple(float(features[name]) for name in FEATURE_NAMES)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("El vector de drift contiene valores no finitos.")
        self._samples.append(values)
        self._last_timestamp = observed_at

        drifted_features = []
        if len(self._samples) >= self._min_samples:
            for index, name in enumerate(FEATURE_NAMES):
                current_mean = sum(sample[index] for sample in self._samples) / len(self._samples)
                standardized_shift = abs(current_mean - self._means[index]) / self._scales[index]
                if standardized_shift >= self._threshold:
                    drifted_features.append(name)
        if drifted_features:
            self._drifted = True
            self._drift_features = tuple(drifted_features)
        try:
            self._save_state()
        except OSError:
            self._drifted = True
            self._drift_features = FEATURE_NAMES
            raise
        return self.get_status()

    def get_status(self) -> dict[str, Any]:
        return {
            "available": len(self._samples) >= self._min_samples,
            "drifted": self._drifted,
            "samples": len(self._samples),
            "minimum_samples": self._min_samples,
            "window_size": self._window_size,
            "threshold_standard_deviations": self._threshold,
            "drifted_features": list(self._drift_features),
            "last_sample_at": (
                self._last_timestamp.isoformat() if self._last_timestamp else None
            ),
            "reason": (
                "Drift detectado; el modelo permanece bloqueado hasta revisar y activar "
                "explícitamente una nueva versión."
                if self._drifted
                else "Muestras insuficientes para evaluar drift."
                if len(self._samples) < self._min_samples
                else "No se detecta desplazamiento medio fuera del umbral configurado."
            ),
        }

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            stored = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(stored, dict):
                raise TypeError("Estado de drift inválido.")
            samples = stored.get("samples")
            timestamp_value = stored.get("last_timestamp")
            timestamp = (
                datetime.fromisoformat(timestamp_value)
                if isinstance(timestamp_value, str)
                else None
            )
            if (
                not isinstance(samples, list)
                or len(samples) > self._window_size
                or any(
                    not isinstance(sample, list)
                    or len(sample) != len(FEATURE_NAMES)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in sample
                    )
                    for sample in samples
                )
                or timestamp is None
                or timestamp.tzinfo is None
                or not isinstance(stored.get("drifted"), bool)
            ):
                raise ValueError("Estado de drift inválido.")
            self._samples.extend(tuple(float(value) for value in sample) for sample in samples)
            self._last_timestamp = timestamp.astimezone(timezone.utc)
            self._drifted = stored["drifted"]
            drift_features = stored.get("drifted_features", [])
            if (
                not isinstance(drift_features, list)
                or any(feature not in FEATURE_NAMES for feature in drift_features)
            ):
                raise ValueError("Features persistidas de drift inválidas.")
            self._drift_features = tuple(drift_features)
            if self._drifted and not self._drift_features:
                raise ValueError("Falta el motivo del estado bloqueado por drift.")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            self._drifted = True
            self._drift_features = FEATURE_NAMES

    def _save_state(self) -> None:
        if self._state_path is None or self._last_timestamp is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "samples": [list(sample) for sample in self._samples],
            "last_timestamp": self._last_timestamp.isoformat(),
            "drifted": self._drifted,
            "drifted_features": list(self._drift_features),
        }
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._state_path.parent,
                prefix=f".{self._state_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                json.dump(state, temporary, sort_keys=True, allow_nan=False)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            temporary_path.replace(self._state_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
