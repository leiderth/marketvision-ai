"""Calibrated next-candle visual direction model with strict chronological holdout."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from vision.feature_pipeline import FEATURE_NAMES, extract_visual_features

MINIMUM_SAMPLES = 200
MIN_TRAIN_SAMPLES = 80
MIN_VALIDATION_SAMPLES = 30
MIN_TEST_SAMPLES = 30
MIN_PROMOTION_IMPROVEMENT = 0.002
ROLLBACK_DEGRADATION = 0.02


class VisionSeriesModel:
    def __init__(self, database_path: str | Path, registry_dir: str | Path) -> None:
        self._database_path = Path(database_path)
        self._registry_dir = Path(registry_dir)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vision_series_model_state (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vision_series_model_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    evaluated_at TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    validation_brier REAL,
                    brier_score REAL,
                    baseline_brier REAL,
                    log_loss REAL,
                    calibration_error REAL,
                    decision TEXT NOT NULL CHECK (
                        decision IN ('promoted', 'rejected', 'rollback')
                    ),
                    details_json TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def train(self, *, force: bool = False) -> dict[str, Any]:
        rows = self._load_samples()
        total = len(rows)
        if total < MINIMUM_SAMPLES:
            return {
                "trained": False,
                "sample_count": total,
                "minimum_samples": MINIMUM_SAMPLES,
                "reason": (
                    f"Se necesitan {MINIMUM_SAMPLES} etiquetas direccionales válidas "
                    "para entrenar y probar en ventanas temporales."
                ),
            }
        state = self._get_state()
        now = datetime.now(timezone.utc)
        if not force and not self._is_due(state, total, now):
            return {
                "trained": False,
                "sample_count": total,
                "reason": "El entrenamiento se ejecuta cada 50 muestras nuevas o una vez al día.",
            }

        frame = self._make_frame(rows)
        if frame is None:
            self._record_deferred_attempt(
                total,
                now,
                "El dataset contiene fechas, features o etiquetas inválidas.",
            )
            return {
                "trained": False,
                "sample_count": total,
                "reason": "El dataset contiene fechas, features o etiquetas inválidas.",
            }
        split = self._purged_splits(frame)
        if split is None:
            self._record_deferred_attempt(
                total,
                now,
                "El split walk-forward purgado no deja suficientes muestras.",
            )
            return {
                "trained": False,
                "sample_count": len(frame),
                "reason": "El split walk-forward purgado no deja suficientes muestras.",
            }
        train, validation, test = split
        if (
            train["target"].nunique() != 2
            or validation["target"].nunique() != 2
            or test["target"].nunique() != 2
        ):
            self._record_deferred_attempt(
                total,
                now,
                "Cada ventana temporal requiere etiquetas CALL y PUT observadas.",
            )
            return {
                "trained": False,
                "sample_count": len(frame),
                "reason": "Cada ventana temporal requiere etiquetas CALL y PUT observadas.",
            }

        scaler = StandardScaler()
        x_train = scaler.fit_transform(_matrix(train))
        x_validation = scaler.transform(_matrix(validation))
        x_test = scaler.transform(_matrix(test))
        classifier = LogisticRegression(
            max_iter=1500,
            random_state=0,
            solver="lbfgs",
        )
        classifier.fit(x_train, train["target"].to_numpy(dtype=int))

        validation_raw = classifier.predict_proba(x_validation)[:, 1]
        calibrator = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=0,
            solver="lbfgs",
        )
        calibrator.fit(_logit_matrix(validation_raw), validation["target"].to_numpy(dtype=int))
        validation_calibrated = calibrator.predict_proba(_logit_matrix(validation_raw))[:, 1]
        test_raw = classifier.predict_proba(x_test)[:, 1]
        test_calibrated = calibrator.predict_proba(_logit_matrix(test_raw))[:, 1]

        y_validation = validation["target"].to_numpy(dtype=int)
        y_test = test["target"].to_numpy(dtype=int)
        prior_probability = float(train["target"].mean())
        validation_brier = _brier(y_validation, validation_calibrated)
        brier = _brier(y_test, test_calibrated)
        baseline_brier = _brier(y_test, np.full(len(test), prior_probability))
        validation_baseline_brier = _brier(
            y_validation,
            np.full(len(validation), prior_probability),
        )
        loss = float(log_loss(y_test, test_calibrated, labels=[0, 1]))
        calibration_error = _expected_calibration_error(y_test, test_calibrated)
        reliability_bins = _reliability_bins(y_test, test_calibrated)
        champion = self._load_champion()
        champion_brier = (
            _artifact_brier(champion, _matrix(test), y_test)
            if champion
            else baseline_brier
        )
        rollback = False
        if champion and champion_brier > baseline_brier + ROLLBACK_DEGRADATION:
            rollback = self._rollback(champion, test, y_test, baseline_brier)
            if rollback:
                champion = self._load_champion()
                champion_brier = (
                    _artifact_brier(champion, _matrix(test), y_test)
                    if champion
                    else baseline_brier
                )

        version, artifact = _make_artifact(
            classifier=classifier,
            scaler=scaler,
            calibrator=calibrator,
            train_samples=len(train),
            validation_samples=len(validation),
            test_samples=len(test),
            validation_brier=validation_brier,
            validation_baseline_brier=validation_baseline_brier,
            test_brier=brier,
            baseline_brier=baseline_brier,
            test_log_loss=loss,
            calibration_error=calibration_error,
            reliability_bins=reliability_bins,
            prior_probability=prior_probability,
        )
        self._write_artifact(version, artifact)
        promoted = (
            validation_brier + MIN_PROMOTION_IMPROVEMENT < validation_baseline_brier
            and brier + MIN_PROMOTION_IMPROVEMENT < baseline_brier
            and brier + MIN_PROMOTION_IMPROVEMENT < champion_brier
        )
        decision = "promoted" if promoted else "rejected"
        now_text = now.isoformat()
        with self._connect() as connection:
            current_state = self._read_state(connection)
            if promoted:
                current_state = {
                    "active_version": version,
                    "previous_version": current_state.get("active_version"),
                }
            current_state.update(
                last_sample_count=total,
                last_evaluation_date=now.date().isoformat(),
                last_attempt_sample_count=total,
                last_attempt_date=now.date().isoformat(),
                last_attempt_reason=None,
                last_result={
                    "trained": True,
                    "version": version,
                    "decision": decision,
                    "sample_count": total,
                    "test_samples": len(test),
                    "brier_score": brier,
                    "baseline_brier_score": baseline_brier,
                    "log_loss": loss,
                    "calibration_error": calibration_error,
                    "rollback_performed": rollback,
                },
            )
            connection.execute(
                """
                INSERT INTO vision_series_model_state(key, value_json, updated_at)
                VALUES ('model', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (json.dumps(current_state, sort_keys=True, allow_nan=False), now_text),
            )
            details = {
                "validation_method": "chronological_60_20_20_label_horizon_purged",
                "feature_names": list(FEATURE_NAMES),
                "train_samples": len(train),
                "validation_samples": len(validation),
                "test_samples": len(test),
                "validation_brier": validation_brier,
                "validation_baseline_brier": validation_baseline_brier,
                "test_brier": brier,
                "test_baseline_brier": baseline_brier,
                "champion_brier_before_promotion": champion_brier,
                "test_log_loss": loss,
                "calibration_error": calibration_error,
                "test_predictions": [
                    {"probability": round(float(p), 6), "target": int(y)}
                    for p, y in zip(test_calibrated, y_test, strict=True)
                ],
            }
            connection.execute(
                """
                INSERT INTO vision_series_model_evaluations(
                    model_version, evaluated_at, sample_count, validation_brier,
                    brier_score, baseline_brier, log_loss, calibration_error,
                    decision, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version,
                    now_text,
                    len(test),
                    validation_brier,
                    brier,
                    baseline_brier,
                    loss,
                    calibration_error,
                    decision,
                    json.dumps(details, sort_keys=True, allow_nan=False),
                ),
            )
        return current_state["last_result"] | {
            "reason": (
                "Promovido tras superar el baseline y champion en validación temporal."
                if promoted
                else "El candidato no superó baseline y champion por el margen requerido."
            )
        }

    def maybe_train(self, *, now: datetime | None = None) -> dict[str, Any]:
        timestamp = now or datetime.now(timezone.utc)
        count = len(self._load_samples())
        state = self._get_state()
        if count < MINIMUM_SAMPLES:
            return {
                "trained": False,
                "sample_count": count,
                "minimum_samples": MINIMUM_SAMPLES,
                "reason": f"Se requieren al menos {MINIMUM_SAMPLES} etiquetas de próxima vela.",
            }
        if not self._is_due(state, count, timestamp):
            return {
                "trained": False,
                "sample_count": count,
                "reason": "El modelo sigue dentro del intervalo de reentrenamiento.",
            }
        return self.train(force=True)

    def get_status(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        model_state = self._get_state()
        rows = self._load_samples()
        active_version = model_state.get("active_version")
        active = self._load_artifact(active_version) if active_version else None
        prediction = None
        if state is not None and active is not None:
            snapshot = extract_visual_features(state)
            probability = _predict_artifact(active, snapshot["features"])
            prediction = {
                "probability_correct": round(probability, 6),
                "interval_95": _probability_interval(
                    probability,
                    active.get("reliability_bins", []),
                ),
                "calibrated": True,
                "price_basis": "normalized_chart_pixels",
                "interpretation": (
                    "Estimación de coincidencia de dirección visual, no probabilidad "
                    "de ganancia ni de precio real."
                ),
            }
        with self._connect() as connection:
            evaluations = connection.execute(
                """
                SELECT model_version, evaluated_at, sample_count,
                       validation_brier, brier_score, baseline_brier,
                       log_loss, calibration_error, decision
                FROM vision_series_model_evaluations
                ORDER BY id DESC LIMIT 10
                """
            ).fetchall()
        return {
            "source": "pixel_series",
            "price_basis": "normalized_chart_pixels",
            "model_training_enabled": True,
            "training_source": "visual_next_candle_labels_only",
            "total_samples": len(rows),
            "minimum_samples": MINIMUM_SAMPLES,
            "active_version": active_version,
            "active_metrics": active.get("metrics") if active else None,
            "last_result": model_state.get("last_result"),
            "last_attempt_reason": model_state.get("last_attempt_reason"),
            "prediction": prediction,
            "recent_evaluations": [dict(row) for row in evaluations],
            "feature_names": list(FEATURE_NAMES),
            "note": (
                "Probabilidad calibrada para coincidencia de dirección visual de "
                "la siguiente vela; no estima ganancia de una operación ni precio real."
            ),
        }

    def _load_samples(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT decision_id, asset, timeframe_seconds, market_regime,
                       target_direction, target, target_candle_open_at,
                       created_at, labeled_at, price_basis, features_json
                FROM vision_series_samples
                WHERE target IN (0, 1) AND source = 'pixel_series'
                ORDER BY created_at, decision_id
                """
            ).fetchall()

    def _make_frame(self, rows: list[sqlite3.Row]):
        import pandas as pd

        payload = []
        for row in rows:
            try:
                snapshot = json.loads(row["features_json"])
                features = snapshot["features"]
                values = [float(features[name]) for name in FEATURE_NAMES]
                created = datetime.fromisoformat(row["created_at"])
                candle_open = datetime.fromisoformat(row["target_candle_open_at"])
                if (
                    row["price_basis"] != "normalized_chart_pixels"
                    or snapshot.get("price_basis") != "normalized_chart_pixels"
                    or not all(math.isfinite(value) for value in values)
                    or created.tzinfo is None
                    or candle_open.tzinfo is None
                ):
                    return None
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return None
            payload.append(
                {
                    "decision_id": row["decision_id"],
                    "created_at": created.astimezone(timezone.utc),
                    "label_end": (
                        candle_open.astimezone(timezone.utc)
                        + timedelta(seconds=int(row["timeframe_seconds"]))
                    ),
                    "target": int(row["target"]),
                    **dict(zip(FEATURE_NAMES, values, strict=True)),
                }
            )
        return pd.DataFrame(payload).sort_values(
            ["created_at", "decision_id"],
            kind="stable",
        ).reset_index(drop=True)

    @staticmethod
    def _purged_splits(frame):
        train_end = int(len(frame) * 0.6)
        validation_end = int(len(frame) * 0.8)
        validation_start = frame.iloc[train_end]["created_at"]
        test_start = frame.iloc[validation_end]["created_at"]
        train = frame.iloc[:train_end]
        validation = frame.iloc[train_end:validation_end]
        test = frame.iloc[validation_end:]
        train = train[train["label_end"] < validation_start]
        validation = validation[
            validation["label_end"] < test_start
        ]
        if (
            len(train) < MIN_TRAIN_SAMPLES
            or len(validation) < MIN_VALIDATION_SAMPLES
            or len(test) < MIN_TEST_SAMPLES
        ):
            return None
        return train, validation, test

    def _is_due(self, state: dict[str, Any], sample_count: int, now: datetime) -> bool:
        last_count = int(
            state.get("last_attempt_sample_count", state.get("last_sample_count", 0))
        )
        last_date = state.get("last_attempt_date", state.get("last_evaluation_date"))
        return (
            sample_count - last_count >= 50
            or (
                last_date != now.date().isoformat()
                and sample_count > last_count
            )
        )

    def _record_deferred_attempt(
        self,
        sample_count: int,
        now: datetime,
        reason: str,
    ) -> None:
        with self._connect() as connection:
            state = self._read_state(connection)
            state.update(
                last_attempt_sample_count=sample_count,
                last_attempt_date=now.date().isoformat(),
                last_attempt_reason=reason,
            )
            connection.execute(
                """
                INSERT INTO vision_series_model_state(key, value_json, updated_at)
                VALUES ('model', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (
                    json.dumps(state, sort_keys=True, allow_nan=False),
                    now.isoformat(),
                ),
            )

    def _get_state(self) -> dict[str, Any]:
        with self._connect() as connection:
            return self._read_state(connection)

    @staticmethod
    def _read_state(connection: sqlite3.Connection) -> dict[str, Any]:
        row = connection.execute(
            "SELECT value_json FROM vision_series_model_state WHERE key = 'model'"
        ).fetchone()
        return json.loads(row["value_json"]) if row else {}

    def _load_champion(self) -> dict[str, Any] | None:
        version = self._get_state().get("active_version")
        return self._load_artifact(version) if version else None

    def _load_artifact(self, version: str | None) -> dict[str, Any] | None:
        if not isinstance(version, str) or not version.startswith("visual-"):
            return None
        path = self._registry_dir / f"{version}.json"
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if artifact.get("version") != version or artifact.get("feature_names") != list(FEATURE_NAMES):
            raise RuntimeError(f"El artefacto visual {version} no coincide con el registro.")
        return artifact

    def _write_artifact(self, version: str, artifact: dict[str, Any]) -> None:
        path = self._registry_dir / f"{version}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(artifact, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _rollback(self, champion, test, targets, baseline_brier) -> bool:
        state = self._get_state()
        previous_version = state.get("previous_version")
        previous = self._load_artifact(previous_version) if previous_version else None
        active_brier = _artifact_brier(champion, _matrix(test), targets)
        previous_brier = (
            _artifact_brier(previous, _matrix(test), targets)
            if previous
            else baseline_brier
        )
        if active_brier <= baseline_brier + ROLLBACK_DEGRADATION:
            return False
        state["active_version"] = previous_version
        state["previous_version"] = None
        state["last_rollback"] = {
            "from": champion["version"],
            "to": previous_version,
            "active_brier": active_brier,
            "previous_brier": previous_brier,
            "at": _utc_now(),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO vision_series_model_state(key, value_json, updated_at)
                VALUES ('model', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (json.dumps(state, sort_keys=True), _utc_now()),
            )
            connection.execute(
                """
                INSERT INTO vision_series_model_evaluations(
                    model_version, evaluated_at, sample_count, validation_brier,
                    brier_score, baseline_brier, log_loss, calibration_error,
                    decision, details_json
                ) VALUES (?, ?, ?, NULL, ?, ?, NULL, NULL, 'rollback', ?)
                """,
                (
                    champion["version"],
                    _utc_now(),
                    len(targets),
                    active_brier,
                    baseline_brier,
                    json.dumps(state["last_rollback"], sort_keys=True),
                ),
            )
        return True


def _matrix(frame) -> np.ndarray:
    return frame.loc[:, FEATURE_NAMES].to_numpy(dtype=float)


def _logit_matrix(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def _brier(targets: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((targets - probabilities) ** 2))


def _expected_calibration_error(targets: np.ndarray, probabilities: np.ndarray) -> float:
    edges = np.linspace(0, 1, 6)
    error = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (
            (probabilities >= lower)
            & (probabilities <= upper if index == 4 else probabilities < upper)
        )
        if np.any(mask):
            error += float(np.mean(mask)) * abs(
                float(np.mean(probabilities[mask]))
                - float(np.mean(targets[mask]))
            )
    return error


def _reliability_bins(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    edges = np.linspace(0, 1, 6)
    bins: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (
            (probabilities >= lower)
            & (probabilities <= upper if index == 4 else probabilities < upper)
        )
        if not np.any(mask):
            continue
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "samples": int(np.sum(mask)),
                "mean_probability": float(np.mean(probabilities[mask])),
                "observed_rate": float(np.mean(targets[mask])),
            }
        )
    return bins


def _probability_interval(
    probability: float,
    bins: list[dict[str, Any]],
) -> dict[str, Any] | None:
    selected = next(
        (
            item
            for item in bins
            if item["lower"] <= probability < item["upper"]
            or (probability == 1 and item["upper"] == 1)
        ),
        None,
    )
    if selected is None or selected["samples"] < 10:
        return None
    lower, upper = _wilson_interval(
        selected["observed_rate"],
        selected["samples"],
    )
    return {
        "lower": round(lower, 6),
        "upper": round(upper, 6),
        "samples": selected["samples"],
        "source": "chronological_out_of_sample_bin",
    }


def _wilson_interval(rate: float, samples: int) -> tuple[float, float]:
    z = 1.959963984540054
    denominator = 1 + z**2 / samples
    center = (rate + z**2 / (2 * samples)) / denominator
    margin = z * math.sqrt(
        rate * (1 - rate) / samples + z**2 / (4 * samples**2)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _artifact_brier(
    artifact: dict[str, Any],
    features: np.ndarray,
    targets: np.ndarray,
) -> float:
    probabilities = np.asarray(
        [_predict_artifact(artifact, dict(zip(FEATURE_NAMES, row, strict=True))) for row in features]
    )
    return _brier(targets, probabilities)


def _predict_artifact(artifact: dict[str, Any], feature_values: dict[str, Any]) -> float:
    vector = np.asarray(
        [[float(feature_values[name]) for name in FEATURE_NAMES]],
        dtype=float,
    )
    means = np.asarray(artifact["feature_means"], dtype=float)
    scales = np.asarray(artifact["feature_scales"], dtype=float)
    coefficients = np.asarray(artifact["coefficients"], dtype=float)
    linear = ((vector - means) / scales) @ coefficients + float(artifact["intercept"])
    raw = 1 / (1 + np.exp(-np.clip(linear, -40, 40)))
    raw_logit = _logit_matrix(raw)
    calibrated_linear = (
        raw_logit * float(artifact["calibration_coefficient"])
        + float(artifact["calibration_intercept"])
    )
    return float((1 / (1 + np.exp(-np.clip(calibrated_linear, -40, 40)))).item())


def _make_artifact(
    *,
    classifier: LogisticRegression,
    scaler: StandardScaler,
    calibrator: LogisticRegression,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    validation_brier: float,
    validation_baseline_brier: float,
    test_brier: float,
    baseline_brier: float,
    test_log_loss: float,
    calibration_error: float,
    reliability_bins: list[dict[str, Any]],
    prior_probability: float,
) -> tuple[str, dict[str, Any]]:
    body = {
        "schema_version": 1,
        "algorithm": "logistic_regression_platt_calibrated",
        "feature_names": list(FEATURE_NAMES),
        "feature_means": scaler.mean_.tolist(),
        "feature_scales": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "calibration_coefficient": float(calibrator.coef_[0][0]),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "prior_probability": prior_probability,
        "reliability_bins": reliability_bins,
        "metrics": {
            "validation_brier": validation_brier,
            "validation_baseline_brier": validation_baseline_brier,
            "test_brier": test_brier,
            "baseline_brier": baseline_brier,
            "test_log_loss": test_log_loss,
            "calibration_error": calibration_error,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
        },
        "validation_policy": "chronological_60_20_20_label_horizon_purged",
        "price_basis": "normalized_chart_pixels",
        "trained_at": _utc_now(),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    version = f"visual-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{digest[:12]}"
    return version, {**body, "version": version, "sha256": digest}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
