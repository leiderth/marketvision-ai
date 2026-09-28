"""Local, JSON-based registry for reviewed sklearn logistic models.

Artifacts contain only numeric coefficients and metadata; no executable
pickle/joblib payloads are loaded from the registry.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from ml.drift import FeatureDriftMonitor
from ml.schema import (
    FEATURE_NAMES,
    MIN_BALANCED_ACCURACY,
    MIN_TRAINING_ROWS,
    ModelRegistryError,
)

ARTIFACT_SCHEMA_VERSION = 1
_LABELS = frozenset({"CALL", "PUT"})


@dataclass(frozen=True)
class RegisteredModel:
    version: str
    coefficients: tuple[float, ...]
    intercept: float
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    metrics: Mapping[str, float]
    trained_at: str

    def predict(self, features: Mapping[str, float]) -> dict[str, Any]:
        if set(features) != set(FEATURE_NAMES):
            raise ModelRegistryError("El conjunto de características no coincide con el modelo.")
        raw = np.asarray([features[name] for name in FEATURE_NAMES], dtype=float)
        if not np.isfinite(raw).all():
            raise ModelRegistryError("El vector de entrada contiene valores no finitos.")
        means = np.asarray(self.feature_means, dtype=float)
        scales = np.asarray(self.feature_scales, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        normalized = (raw - means) / scales
        if not np.isfinite(normalized).all():
            raise ModelRegistryError("El vector normalizado contiene valores no finitos.")
        logit = float(np.dot(normalized, coefficients) + self.intercept)
        if not math.isfinite(logit):
            raise ModelRegistryError("El resultado del modelo no es finito.")
        probability_call = _sigmoid(logit)
        contributions = [
            {
                "feature": name,
                "value": float(raw[index]),
                "baseline": float(means[index]),
                "contribution": float(coefficients[index] * normalized[index]),
            }
            for index, name in enumerate(FEATURE_NAMES)
        ]
        if any(not math.isfinite(item["contribution"]) for item in contributions):
            raise ModelRegistryError("La explicación del modelo contiene valores no finitos.")
        return {
            "available": True,
            "version": self.version,
            "prediction": "CALL" if probability_call >= 0.5 else "PUT",
            "class_score_call": probability_call,
            "class_score_put": 1.0 - probability_call,
            "logit": logit,
            "explainability": {
                "method": "linear_interventional_shap",
                "output_space": "log_odds",
                "expected_value": float(self.intercept),
                "contributions": contributions,
                "note": (
                    "Atribución Shapley exacta para modelo lineal bajo independencia "
                    "interventional; no representa probabilidad ni causalidad."
                ),
            },
        }


def feature_vector(
    values: Mapping[str, Any],
    latest_close: float | None,
) -> dict[str, float]:
    sma_20 = _finite_number(values.get("sma_20"))
    sma_50 = _finite_number(values.get("sma_50"))
    ema_20 = _finite_number(values.get("ema_20"))
    ema_50 = _finite_number(values.get("ema_50"))
    atr = _finite_number(values.get("atr_14"))
    rsi = _finite_number(values.get("rsi_14"))
    macd = values.get("macd")
    histogram = _finite_number(macd.get("histogram")) if isinstance(macd, Mapping) else None
    vwap = _finite_number(values.get("vwap_session"))
    if (
        None in (sma_20, sma_50, ema_20, ema_50, atr, rsi, histogram, vwap, latest_close)
        or atr <= 0
        or not 0 <= rsi <= 100
    ):
        raise ModelRegistryError("Indicadores insuficientes para inferencia del modelo.")
    return {
        "sma_gap_atr": (sma_20 - sma_50) / atr,
        "ema_gap_atr": (ema_20 - ema_50) / atr,
        "macd_histogram_atr": histogram / atr,
        "rsi_centered": (rsi - 50.0) / 50.0,
        "price_vwap_atr": (latest_close - vwap) / atr,
    }


def train_and_register(
    dataset_path: str | Path,
    registry_dir: str | Path,
    *,
    min_balanced_accuracy: float = MIN_BALANCED_ACCURACY,
) -> dict[str, Any]:
    """Train chronologically split data, validate holdout, and save an inactive model."""
    if (
        not math.isfinite(min_balanced_accuracy)
        or not MIN_BALANCED_ACCURACY <= min_balanced_accuracy <= 1
    ):
        raise ValueError(
            f"El umbral de balanced accuracy debe estar entre "
            f"{MIN_BALANCED_ACCURACY} y 1."
        )
    frame = pd.read_csv(dataset_path)
    required_columns = {"timestamp", "label_end_timestamp", "label", *FEATURE_NAMES}
    if not required_columns.issubset(frame.columns):
        missing = sorted(required_columns.difference(frame.columns))
        raise ModelRegistryError(f"Faltan columnas del dataset: {', '.join(missing)}.")
    frame = frame[["timestamp", "label_end_timestamp", *FEATURE_NAMES, "label"]].copy()
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    label_end_timestamps = pd.to_datetime(
        frame["label_end_timestamp"],
        utc=True,
        errors="coerce",
    )
    if (
        timestamps.isna().any()
        or timestamps.duplicated().any()
        or label_end_timestamps.isna().any()
        or (label_end_timestamps < timestamps).any()
    ):
        raise ModelRegistryError("El dataset contiene timestamps inválidos o duplicados.")
    frame["timestamp"] = timestamps
    frame["label_end_timestamp"] = label_end_timestamps
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    labels = frame["label"]
    if labels.isna().any() or not set(labels.unique()).issubset(_LABELS):
        raise ModelRegistryError("Las etiquetas deben ser exclusivamente CALL o PUT.")
    features = frame.loc[:, FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ModelRegistryError("El dataset contiene características no numéricas o no finitas.")
    row_count = len(frame)
    if row_count < MIN_TRAINING_ROWS:
        raise ModelRegistryError(
            f"Se requieren al menos {MIN_TRAINING_ROWS} filas etiquetadas; hay {row_count}."
        )

    train_end = int(row_count * 0.6)
    validation_end = int(row_count * 0.8)
    train_partition = frame.iloc[:train_end]
    validation_partition = frame.iloc[train_end:validation_end]
    test_partition = frame.iloc[validation_end:]
    validation_start = validation_partition["timestamp"].iloc[0]
    test_start = test_partition["timestamp"].iloc[0]
    train_partition = train_partition[
        train_partition["label_end_timestamp"] < validation_start
    ]
    validation_partition = validation_partition[
        validation_partition["label_end_timestamp"] < test_start
    ]
    if (
        len(train_partition) < MIN_TRAINING_ROWS * 0.5
        or len(validation_partition) < 2
        or len(test_partition) < 2
    ):
        raise ModelRegistryError(
            "El purgado por horizonte de etiqueta deja particiones insuficientes."
        )
    partitions = (
        train_partition["label"],
        validation_partition["label"],
        test_partition["label"],
    )
    if any(set(partition.unique()) != _LABELS for partition in partitions):
        raise ModelRegistryError(
            "Cada partición temporal train/validation/test debe contener CALL y PUT."
        )

    train_x = features.loc[train_partition.index].to_numpy(dtype=float)
    validation_x = features.loc[validation_partition.index].to_numpy(dtype=float)
    test_x = features.loc[test_partition.index].to_numpy(dtype=float)
    train_y = (train_partition["label"].to_numpy() == "CALL").astype(int)
    validation_y = (validation_partition["label"].to_numpy() == "CALL").astype(int)
    test_y = (test_partition["label"].to_numpy() == "CALL").astype(int)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x)
    scaled_validation = scaler.transform(validation_x)
    scaled_test = scaler.transform(test_x)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=0,
        solver="lbfgs",
    )
    classifier.fit(scaled_train, train_y)
    validation_prediction = classifier.predict(scaled_validation)
    test_prediction = classifier.predict(scaled_test)
    metrics = {
        "train_rows": float(len(train_y)),
        "validation_rows": float(len(validation_y)),
        "test_rows": float(len(test_y)),
        "validation_accuracy": float(accuracy_score(validation_y, validation_prediction)),
        "validation_balanced_accuracy": float(
            balanced_accuracy_score(validation_y, validation_prediction)
        ),
        "test_accuracy": float(accuracy_score(test_y, test_prediction)),
        "test_balanced_accuracy": float(balanced_accuracy_score(test_y, test_prediction)),
        "minimum_balanced_accuracy": min_balanced_accuracy,
    }
    if (
        metrics["validation_balanced_accuracy"] < min_balanced_accuracy
        or metrics["test_balanced_accuracy"] < min_balanced_accuracy
    ):
        raise ModelRegistryError(
            "El modelo no alcanza el balanced accuracy mínimo en validation y test; "
            "no se registró."
        )

    trained_at = datetime.now(timezone.utc).isoformat()
    model_body = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "algorithm": "sklearn.LogisticRegression",
        "feature_names": list(FEATURE_NAMES),
        "classes": ["PUT", "CALL"],
        "label_encoding": {"0": "PUT", "1": "CALL"},
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "feature_means": scaler.mean_.tolist(),
        "feature_scales": scaler.scale_.tolist(),
        "metrics": metrics,
        "trained_at": trained_at,
        "training_rows": row_count,
        "validation_policy": {
            "method": "chronological_60_20_20_label_horizon_purged",
            "minimum_balanced_accuracy": min_balanced_accuracy,
        },
    }
    canonical_body = json.dumps(model_body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    version = (
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-"
        f"{hashlib.sha256(canonical_body.encode('utf-8')).hexdigest()[:12]}"
    )
    artifact = {
        **model_body,
        "version": version,
        "sha256": hashlib.sha256(canonical_body.encode("utf-8")).hexdigest(),
    }
    destination = Path(registry_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(destination / f"{version}.json", artifact)
    return {
        "version": version,
        "registered": True,
        "active": False,
        "metrics": metrics,
        "artifact_path": str(destination / f"{version}.json"),
    }


class MLModelManager:
    def __init__(
        self,
        registry_dir: str | Path,
        active_version: str | None = None,
        *,
        min_confidence: float = 0.65,
        drift_state_dir: str | Path = "./data/model_drift",
    ) -> None:
        if not 0.5 < min_confidence <= 1:
            raise ValueError("El umbral de confianza ML debe estar entre 0.5 y 1.")
        self._registry_dir = Path(registry_dir)
        self._drift_state_dir = Path(drift_state_dir)
        self._configured_version = active_version.strip() if active_version else None
        self._min_confidence = min_confidence
        self._model: RegisteredModel | None = None
        self._drift_monitor: FeatureDriftMonitor | None = None
        self._error: str | None = None
        if self._configured_version:
            try:
                self._model = self._load_model(self._configured_version)
                self._drift_monitor = FeatureDriftMonitor(
                    self._model.feature_means,
                    self._model.feature_scales,
                    state_path=self._drift_state_dir / f"{self._model.version}.json",
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                detail = (
                    str(exc)
                    if isinstance(exc, ModelRegistryError)
                    else type(exc).__name__
                )
                self._error = f"Modelo activo inválido: {detail}"

    @property
    def has_configured_model(self) -> bool:
        return self._configured_version is not None

    def get_status(self) -> dict[str, Any]:
        return {
            "configured_model_version": self._configured_version,
            "active_model_version": self._model.version if self._model else None,
            "model_loaded": self._model is not None,
            "inference_enabled": bool(
                self._model
                and self._drift_monitor
                and self._drift_monitor.get_status()["available"]
                and not self._drift_monitor.get_status()["drifted"]
            ),
            "minimum_confidence": self._min_confidence,
            "metrics": dict(self._model.metrics) if self._model else None,
            "drift": self._drift_monitor.get_status() if self._drift_monitor else None,
            "error": self._error,
            "reason": (
                "Modelo ML no configurado; se mantiene el motor determinista."
                if self._configured_version is None
                else self._error or "Modelo cargado por selección explícita de versión."
            ),
        }

    def predict(
        self,
        values: Mapping[str, Any],
        latest_close: float | None,
        *,
        as_of: str | None = None,
    ) -> dict[str, Any]:
        if self._configured_version is None:
            return {"configured": False}
        if self._model is None:
            return {"configured": True, "available": False, "reason": self._error}
        try:
            features = feature_vector(values, latest_close)
            if as_of is None or self._drift_monitor is None:
                raise ModelRegistryError("Timestamp de observación necesario para inferencia ML.")
            drift = self._drift_monitor.observe(features, as_of)
            prediction = self._model.predict(features)
        except (ModelRegistryError, OSError, TypeError, ValueError, OverflowError) as exc:
            return {
                "configured": True,
                "available": False,
                "reason": f"No se pudo inferir con el modelo ({type(exc).__name__}).",
            }
        confidence = max(prediction["class_score_call"], prediction["class_score_put"])
        prediction["configured"] = True
        prediction["drift"] = drift
        prediction["confident"] = confidence >= self._min_confidence
        prediction["minimum_confidence"] = self._min_confidence
        return prediction

    def _load_model(self, version: str) -> RegisteredModel:
        if not version or Path(version).name != version or version in {".", ".."}:
            raise ModelRegistryError("Versión de modelo inválida.")
        artifact_path = self._registry_dir / f"{version}.json"
        raw = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != version:
            raise ModelRegistryError("El manifiesto de modelo no coincide con su versión.")
        if raw.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ModelRegistryError("Versión de esquema de artefacto no compatible.")
        if raw.get("algorithm") != "sklearn.LogisticRegression":
            raise ModelRegistryError("Algoritmo no permitido en el registro.")
        if (
            raw.get("feature_names") != list(FEATURE_NAMES)
            or raw.get("classes") != ["PUT", "CALL"]
            or raw.get("label_encoding") != {"0": "PUT", "1": "CALL"}
        ):
            raise ModelRegistryError("El esquema de características o clases no coincide.")
        digest = raw.pop("sha256", None)
        artifact_version = raw.pop("version", None)
        canonical_body = json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        computed_digest = hashlib.sha256(canonical_body.encode("utf-8")).hexdigest()
        if (
            not isinstance(digest, str)
            or not hmac.compare_digest(digest, computed_digest)
            or artifact_version != version
            or not version.endswith(f"-{computed_digest[:12]}")
        ):
            raise ModelRegistryError("La verificación de integridad del artefacto falló.")
        raw["version"] = artifact_version

        coefficients = _finite_sequence(raw.get("coefficients"), len(FEATURE_NAMES))
        means = _finite_sequence(raw.get("feature_means"), len(FEATURE_NAMES))
        scales = _finite_sequence(raw.get("feature_scales"), len(FEATURE_NAMES))
        intercept = _finite_number(raw.get("intercept"))
        metrics = raw.get("metrics")
        validation_balanced_accuracy = _finite_number(
            metrics.get("validation_balanced_accuracy")
        ) if isinstance(metrics, dict) else None
        test_balanced_accuracy = (
            _finite_number(metrics.get("test_balanced_accuracy"))
            if isinstance(metrics, dict)
            else None
        )
        policy = raw.get("validation_policy")
        policy_threshold = (
            _finite_number(policy.get("minimum_balanced_accuracy"))
            if isinstance(policy, dict)
            else None
        )
        if (
            intercept is None
            or any(scale <= 0 for scale in scales)
            or validation_balanced_accuracy is None
            or test_balanced_accuracy is None
            or policy_threshold is None
            or policy_threshold < MIN_BALANCED_ACCURACY
            or policy_threshold > 1
            or validation_balanced_accuracy < policy_threshold
            or validation_balanced_accuracy > 1
            or test_balanced_accuracy < policy_threshold
            or test_balanced_accuracy > 1
            or not isinstance(metrics, dict)
            or policy.get("method") != "chronological_60_20_20_label_horizon_purged"
            or raw.get("training_rows", 0) < MIN_TRAINING_ROWS
            or not isinstance(raw.get("trained_at"), str)
        ):
            raise ModelRegistryError("El artefacto no cumple la política de aprobación.")
        return RegisteredModel(
            version=version,
            coefficients=coefficients,
            intercept=intercept,
            feature_means=means,
            feature_scales=scales,
            metrics=metrics,
            trained_at=raw["trained_at"],
        )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _finite_sequence(value: Any, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ModelRegistryError("El artefacto tiene dimensiones inválidas.")
    parsed = tuple(_finite_number(item) for item in value)
    if any(item is None for item in parsed):
        raise ModelRegistryError("El artefacto contiene parámetros no finitos.")
    return tuple(float(item) for item in parsed)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
