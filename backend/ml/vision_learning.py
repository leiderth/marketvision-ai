"""Incremental and walk-forward learning for labeled screenshot decisions."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = (
    "call_score",
    "put_score",
    "candle_count",
    "trend_up",
    "trend_down",
    "momentum_bullish",
    "momentum_bearish",
    "bullish_pattern",
    "bearish_pattern",
    "read_quality",
)
MIN_BATCH_SAMPLES = 100
RETRAIN_EVERY_SAMPLES = 50
MIN_BRIER_IMPROVEMENT = 0.002
ROLLBACK_BRIER_DEGRADATION = 0.02


class VisionLearningManager:
    def __init__(
        self,
        database_path: str | Path,
        dataset_path: str | Path,
        registry_dir: str | Path,
    ) -> None:
        self._database_path = Path(database_path)
        self._dataset_path = Path(dataset_path)
        self._registry_dir = Path(registry_dir)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS vision_learning_state (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vision_model_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    evaluated_at TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    brier_score REAL,
                    baseline_brier_score REAL,
                    decision TEXT NOT NULL CHECK (
                        decision IN ('candidate', 'promoted', 'rejected', 'rollback')
                    ),
                    details_json TEXT NOT NULL
                );
                """
            )
        self.export_dataset()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def on_outcome(self, decision_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT decision_id, created_at, asset, timeframe_seconds,
                       market_regime, direction, call_score, put_score,
                       confidence_pct, features_json, outcome, outcome_at
                FROM vision_decisions WHERE decision_id = ? AND outcome IS NOT NULL
                """,
                (decision_id,),
            ).fetchone()
            if row is None:
                raise KeyError("La decisión etiquetada no existe.")
            features = _features_from_row(row)
            state = self._read_state(connection, "online", _default_online_state())
            target = 1.0 if row["outcome"] == "win" else 0.0
            prediction = _sigmoid(
                state["bias"] + sum(state["weights"][name] * features[name] for name in FEATURE_NAMES)
            )
            error = target - prediction
            for name in FEATURE_NAMES:
                state["weights"][name] = max(
                    -2.0,
                    min(2.0, state["weights"][name] + 0.05 * error * features[name]),
                )
            state["bias"] = max(-4.0, min(4.0, state["bias"] + 0.05 * error))
            confidence = row["confidence_pct"]
            threshold = state["confidence_threshold"]
            if row["outcome"] == "loss":
                threshold = min(0.95, threshold + 0.005)
            elif confidence is not None and confidence / 100 >= threshold:
                threshold = max(0.65, threshold - 0.001)
            state["confidence_threshold"] = threshold
            state["updates"] += 1
            state["updated_at"] = _utc_now()

            context = _context_key(row)
            edge_margins = self._read_state(connection, "context_edges", {})
            current_margin = float(edge_margins.get(context, 5.0))
            edge_margins[context] = (
                min(15.0, current_margin + 0.25)
                if row["outcome"] == "loss"
                else max(5.0, current_margin - 0.05)
            )
            self._write_state(connection, "online", state)
            self._write_state(connection, "context_edges", edge_margins)
        self.export_dataset()

    def rebuild_online_state(self) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT created_at, asset, timeframe_seconds, market_regime,
                       direction, call_score, put_score, confidence_pct,
                       features_json, outcome
                FROM vision_decisions WHERE outcome IS NOT NULL
                ORDER BY outcome_at, decision_id
                """
            ).fetchall()
            state = _default_online_state()
            edge_margins: dict[str, float] = {}
            for row in rows:
                features = _features_from_row(row)
                target = 1.0 if row["outcome"] == "win" else 0.0
                prediction = _sigmoid(
                    state["bias"]
                    + sum(state["weights"][name] * features[name] for name in FEATURE_NAMES)
                )
                error = target - prediction
                for name in FEATURE_NAMES:
                    state["weights"][name] = max(
                        -2.0,
                        min(2.0, state["weights"][name] + 0.05 * error * features[name]),
                    )
                state["bias"] = max(-4.0, min(4.0, state["bias"] + 0.05 * error))
                confidence = row["confidence_pct"]
                if row["outcome"] == "loss":
                    state["confidence_threshold"] = min(
                        0.95, state["confidence_threshold"] + 0.005
                    )
                elif confidence is not None and confidence / 100 >= state["confidence_threshold"]:
                    state["confidence_threshold"] = max(
                        0.65, state["confidence_threshold"] - 0.001
                    )
                key = _context_key(row)
                edge_margins.setdefault(key, 5.0)
                edge_margins[key] = (
                    min(15.0, edge_margins[key] + 0.25)
                    if row["outcome"] == "loss"
                    else max(5.0, edge_margins[key] - 0.05)
                )
                state["updates"] += 1
            state["updated_at"] = _utc_now() if rows else None
            self._write_state(connection, "online", state)
            self._write_state(connection, "context_edges", edge_margins)
            batch_state = self._read_state(connection, "batch", {})
            batch_state["last_sample_count"] = max(
                0, len(rows) - RETRAIN_EVERY_SAMPLES
            )
            batch_state["last_evaluation_date"] = None
            self._write_state(connection, "batch", batch_state)
        self.export_dataset()

    def get_context_edge_margin(
        self,
        *,
        asset: str | None,
        timeframe_seconds: int | None,
        market_regime: str | None,
        hour_utc: int | None,
        configured_margin_pct: float,
    ) -> float:
        if (
            not asset
            or timeframe_seconds is None
            or not market_regime
            or hour_utc is None
        ):
            return configured_margin_pct
        key = f"{asset.strip()}|{timeframe_seconds}|{market_regime.strip()}|{hour_utc}"
        with self._connect() as connection:
            margins = self._read_state(connection, "context_edges", {})
        return max(configured_margin_pct, float(margins.get(key, configured_margin_pct)))

    def get_confidence_threshold_pct(self) -> float:
        with self._connect() as connection:
            state = self._read_state(connection, "online", _default_online_state())
        return round(float(state["confidence_threshold"]) * 100, 2)

    def export_dataset(self) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT decision_id, created_at, outcome_at, asset, timeframe_seconds,
                       market_regime, direction, call_score, put_score, confidence_pct,
                       features_json, outcome, outcome_source
                FROM vision_decisions WHERE outcome IS NOT NULL
                ORDER BY created_at, decision_id
                """
            ).fetchall()
        temporary = self._dataset_path.with_suffix(self._dataset_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "decision_id", "timestamp", "label_end_timestamp", "asset",
                    "timeframe_seconds", "market_regime", "direction",
                    "confidence_pct", "outcome", "outcome_source", *FEATURE_NAMES,
                ],
            )
            writer.writeheader()
            for row in rows:
                features = _features_from_row(row)
                writer.writerow(
                    {
                        "decision_id": row["decision_id"],
                        "timestamp": row["created_at"],
                        "label_end_timestamp": row["outcome_at"],
                        "asset": row["asset"],
                        "timeframe_seconds": row["timeframe_seconds"],
                        "market_regime": row["market_regime"],
                        "direction": row["direction"],
                        "confidence_pct": row["confidence_pct"],
                        "outcome": row["outcome"],
                        "outcome_source": row["outcome_source"],
                        **features,
                    }
                )
        temporary.replace(self._dataset_path)
        return len(rows)

    def maybe_retrain(self, *, force: bool = False) -> dict[str, Any]:
        rows = self._get_training_rows()
        sample_count = len(rows)
        with self._connect() as connection:
            online = self._read_state(connection, "online", _default_online_state())
        if int(online.get("updates", 0)) < sample_count:
            self.rebuild_online_state()
        if sample_count < MIN_BATCH_SAMPLES:
            return {
                "trained": False,
                "sample_count": sample_count,
                "reason": f"Se requieren al menos {MIN_BATCH_SAMPLES} resultados etiquetados.",
            }
        with self._connect() as connection:
            run_state = self._read_state(connection, "batch", {})
        now = datetime.now(timezone.utc)
        last_count = int(run_state.get("last_sample_count", 0))
        last_date = run_state.get("last_evaluation_date")
        due = (
            force
            or sample_count - last_count >= RETRAIN_EVERY_SAMPLES
            or (last_date != now.date().isoformat() and sample_count > last_count)
        )
        if not due:
            return {
                "trained": False,
                "sample_count": sample_count,
                "reason": "El reentrenamiento se ejecuta cada 50 etiquetas nuevas o una vez al día.",
            }
        result = self.train_batch(rows)
        run_state.update(
            last_sample_count=sample_count,
            last_evaluation_date=now.date().isoformat(),
            last_result=result,
        )
        with self._connect() as connection:
            self._write_state(connection, "batch", run_state)
        return result

    def train_batch(self, rows: list[sqlite3.Row] | None = None) -> dict[str, Any]:
        rows = rows if rows is not None else self._get_training_rows()
        if len(rows) < MIN_BATCH_SAMPLES:
            return {
                "trained": False,
                "sample_count": len(rows),
                "reason": f"Se requieren al menos {MIN_BATCH_SAMPLES} resultados etiquetados.",
            }
        frame = pd.DataFrame(
            [
                {
                    "timestamp": row["created_at"],
                    "label_end_timestamp": row["outcome_at"],
                    "target": 1 if row["outcome"] == "win" else 0,
                    **_features_from_row(row),
                }
                for row in rows
            ]
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["label_end_timestamp"] = pd.to_datetime(
            frame["label_end_timestamp"], utc=True, errors="coerce"
        )
        frame = frame.sort_values(["timestamp"], kind="stable").reset_index(drop=True)
        if frame[["timestamp", "label_end_timestamp"]].isna().any().any():
            return {"trained": False, "sample_count": len(frame), "reason": "Fechas inválidas en el historial."}
        train_end = int(len(frame) * 0.6)
        validation_end = int(len(frame) * 0.8)
        validation_start = frame.iloc[train_end]["timestamp"]
        test_start = frame.iloc[validation_end]["timestamp"]
        train = frame.iloc[:train_end]
        validation = frame.iloc[train_end:validation_end]
        test = frame.iloc[validation_end:]
        train = train[train["label_end_timestamp"] < validation_start]
        validation = validation[validation["label_end_timestamp"] < test_start]
        if (
            len(train) < 40
            or len(validation) < 10
            or len(test) < 10
            or train["target"].nunique() < 2
        ):
            return {
                "trained": False,
                "sample_count": len(frame),
                "reason": "El split temporal purgado no deja muestras suficientes.",
            }

        scaler = StandardScaler()
        train_x = scaler.fit_transform(train.loc[:, FEATURE_NAMES].to_numpy(dtype=float))
        validation_x = scaler.transform(validation.loc[:, FEATURE_NAMES].to_numpy(dtype=float))
        test_x = scaler.transform(test.loc[:, FEATURE_NAMES].to_numpy(dtype=float))
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=0,
            solver="lbfgs",
        )
        classifier.fit(train_x, train["target"].to_numpy(dtype=int))
        validation_predictions = classifier.predict_proba(validation_x)[:, 1]
        candidate_predictions = classifier.predict_proba(test_x)[:, 1]
        baseline_probability = float(train["target"].mean())
        baseline_brier = float(np.mean((test["target"].to_numpy() - baseline_probability) ** 2))
        candidate_brier = float(
            np.mean((test["target"].to_numpy() - candidate_predictions) ** 2)
        )
        validation_brier = float(
            np.mean((validation["target"].to_numpy() - validation_predictions) ** 2)
        )
        baseline_validation_brier = float(
            np.mean(
                (
                    validation["target"].to_numpy()
                    - float(train["target"].mean())
                )
                ** 2
            )
        )

        current_model = self._load_active_model()
        previous_brier = (
            _brier_score(
                current_model,
                test.loc[:, FEATURE_NAMES].to_numpy(dtype=float),
                test["target"].to_numpy(dtype=int),
            )
            if current_model
            else baseline_brier
        )
        if current_model and previous_brier > baseline_brier + ROLLBACK_BRIER_DEGRADATION:
            rollback = self._rollback_active_model(
                current_model,
                sample_count=len(test),
                brier_score=previous_brier,
                baseline_brier=baseline_brier,
            )
            if rollback:
                current_model = self._load_active_model()
                previous_brier = (
                    _brier_score(
                        current_model,
                        test.loc[:, FEATURE_NAMES].to_numpy(dtype=float),
                        test["target"].to_numpy(dtype=int),
                    )
                    if current_model
                    else baseline_brier
                )

        version, artifact = _build_artifact(
            classifier,
            scaler,
            validation_brier=validation_brier,
            baseline_validation_brier=baseline_validation_brier,
            test_brier=candidate_brier,
            baseline_brier=baseline_brier,
            training_rows=len(train),
            validation_rows=len(validation),
            test_rows=len(test),
        )
        self._write_artifact(version, artifact)
        promoted = (
            validation_brier + MIN_BRIER_IMPROVEMENT < baseline_validation_brier
            and candidate_brier + MIN_BRIER_IMPROVEMENT < baseline_brier
            and candidate_brier + MIN_BRIER_IMPROVEMENT < previous_brier
        )
        decision = "promoted" if promoted else "rejected"
        details = {
            "reason": (
                "El candidato mejora el Brier del modelo vigente y del baseline."
                if promoted
                else "El candidato no superó el modelo vigente y el baseline por el margen exigido."
            ),
            "validation_method": "chronological_60_20_20_label_end_purged",
            "validation_brier_score": validation_brier,
            "baseline_validation_brier_score": baseline_validation_brier,
            "previous_model_brier_score": previous_brier,
            "test_predictions": [
                {"predicted": round(float(prediction), 6), "observed": int(target)}
                for prediction, target in zip(
                    candidate_predictions,
                    test["target"].to_numpy(dtype=int),
                    strict=True,
                )
            ],
        }
        now_text = _utc_now()
        with self._connect() as connection:
            state = self._read_state(connection, "model", {})
            if promoted:
                state = {
                    "active_version": version,
                    "previous_version": state.get("active_version"),
                }
                self._write_state(connection, "model", state)
            connection.execute(
                """
                INSERT INTO vision_model_evaluations (
                    model_version, evaluated_at, sample_count, brier_score,
                    baseline_brier_score, decision, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version,
                    now_text,
                    len(test),
                    candidate_brier,
                    baseline_brier,
                    decision,
                    json.dumps(details, sort_keys=True, allow_nan=False),
                ),
            )
        return {
            "trained": True,
            "version": version,
            "decision": decision,
            "sample_count": len(rows),
            "test_samples": len(test),
            "brier_score": candidate_brier,
            "baseline_brier_score": baseline_brier,
            "previous_model_brier_score": previous_brier,
            "reason": details["reason"],
        }

    def get_dashboard(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT decision_id, created_at, outcome_at, asset, timeframe_seconds,
                       market_regime, direction, outcome, confidence_pct, call_score,
                       put_score, features_json, outcome_source
                FROM vision_decisions WHERE outcome IS NOT NULL
                ORDER BY created_at
                """
            ).fetchall()
            online = self._read_state(connection, "online", _default_online_state())
            model_state = self._read_state(connection, "model", {})
            evaluations = connection.execute(
                """
                SELECT model_version, evaluated_at, sample_count, brier_score,
                       baseline_brier_score, decision, details_json
                FROM vision_model_evaluations ORDER BY evaluated_at DESC LIMIT 20
                """
            ).fetchall()
            batch_state = self._read_state(connection, "batch", {})
        grouped: dict[str, dict[str, list[sqlite3.Row]]] = {
            "by_asset": {},
            "by_direction": {},
            "by_regime": {},
            "by_hour_utc": {},
            "by_context": {},
        }
        for row in rows:
            hour = _parse_utc(row["created_at"]).strftime("%H:00 UTC")
            for key, name in (
                ("asset", "by_asset"),
                ("direction", "by_direction"),
                ("market_regime", "by_regime"),
            ):
                grouped[name].setdefault(str(row[key]), []).append(row)
            grouped["by_hour_utc"].setdefault(hour, []).append(row)
            context = (
                f"{row['asset']}|{row['timeframe_seconds']}s|"
                f"{row['market_regime']}|{hour}"
            )
            grouped["by_context"].setdefault(context, []).append(row)
        latest_details: dict[str, Any] = {}
        active_version = model_state.get("active_version")
        promoted_evaluation = next(
            (
                row for row in evaluations
                if row["model_version"] == active_version
                and row["decision"] == "promoted"
            ),
            None,
        )
        if promoted_evaluation:
            latest_details = json.loads(promoted_evaluation["details_json"])
        active_artifact = self._load_active_model()
        model_importance = (
            [
                {"feature": name, "importance": abs(coefficient), "weight": coefficient}
                for name, coefficient in sorted(
                    zip(FEATURE_NAMES, active_artifact["coefficients"], strict=True),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )
            ]
            if active_artifact
            else []
        )
        calibration: dict[str, list[dict[str, float]]] = {}
        for sample in latest_details.get("test_predictions", []):
            key = f"{min(9, int(sample['predicted'] * 10)) * 10:02d}-{min(100, (min(9, int(sample['predicted'] * 10)) + 1) * 10):03d}%"
            calibration.setdefault(key, []).append(sample)
        return {
            "labeled_decisions": len(rows),
            "dataset_samples": len(rows),
            "by_asset": _outcome_rates(grouped["by_asset"]),
            "by_direction": _outcome_rates(grouped["by_direction"]),
            "by_regime": _outcome_rates(grouped["by_regime"]),
            "by_hour_utc": _outcome_rates(grouped["by_hour_utc"]),
            "by_context": _outcome_rates(grouped["by_context"]),
            "online_learning": {
                **online,
                "feature_importance": sorted(
                    (
                        {
                            "feature": name,
                            "importance": abs(weight),
                            "weight": weight,
                        }
                        for name, weight in online["weights"].items()
                    ),
                    key=lambda value: value["importance"],
                    reverse=True,
                ),
            },
            "active_model_version": active_version,
            "model_feature_importance": model_importance,
            "model_evolution": [
                {
                    "model_version": row["model_version"],
                    "evaluated_at": row["evaluated_at"],
                    "samples": row["sample_count"],
                    "brier_score": row["brier_score"],
                    "baseline_brier_score": row["baseline_brier_score"],
                    "decision": row["decision"],
                    "reason": json.loads(row["details_json"]).get("reason"),
                }
                for row in evaluations
            ],
            "calibration": [
                {
                    "probability_bucket": bucket,
                    "samples": len(samples),
                    "predicted_win_rate": sum(item["predicted"] for item in samples) / len(samples),
                    "observed_win_rate": sum(item["observed"] for item in samples) / len(samples),
                }
                for bucket, samples in sorted(calibration.items())
            ],
            "batch_training": {
                "enabled": True,
                "minimum_samples": MIN_BATCH_SAMPLES,
                "retrain_every_samples": RETRAIN_EVERY_SAMPLES,
                "last_result": batch_state.get("last_result"),
                "note": (
                    "Promoción únicamente si mejora el Brier fuera de muestra del modelo "
                    "vigente y del baseline; se conserva la versión anterior para rollback."
                ),
            },
        }

    def _get_training_rows(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT created_at, outcome_at, asset, timeframe_seconds, market_regime,
                       direction, call_score, put_score, confidence_pct, features_json,
                       outcome, outcome_source
                FROM vision_decisions WHERE outcome IS NOT NULL
                ORDER BY created_at, decision_id
                """
            ).fetchall()

    def _load_active_model(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            state = self._read_state(connection, "model", {})
        version = state.get("active_version")
        if not version:
            return None
        path = self._registry_dir / f"{version}.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        body = {key: value for key, value in artifact.items() if key not in {"version", "sha256"}}
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        if (
            artifact.get("version") != version
            or artifact.get("sha256") != digest
            or artifact.get("feature_names") != list(FEATURE_NAMES)
        ):
            raise ValueError("La integridad del modelo visual activo no es válida.")
        return artifact

    def _write_artifact(self, version: str, artifact: dict[str, Any]) -> None:
        destination = self._registry_dir / f"{version}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(destination)

    def _rollback_active_model(
        self,
        active: dict[str, Any],
        *,
        sample_count: int,
        brier_score: float,
        baseline_brier: float,
    ) -> bool:
        now_text = _utc_now()
        with self._connect() as connection:
            state = self._read_state(connection, "model", {})
            previous = state.get("previous_version")
            if previous:
                state = {"active_version": previous, "previous_version": None}
                reason = "Rollback automático: el modelo activo degradó frente al baseline."
            else:
                state = {"active_version": None, "previous_version": None}
                reason = "Rollback automático: sin versión previa; queda activo el baseline."
            self._write_state(connection, "model", state)
            connection.execute(
                """
                INSERT INTO vision_model_evaluations (
                    model_version, evaluated_at, sample_count, brier_score,
                    baseline_brier_score, decision, details_json
                ) VALUES (?, ?, ?, ?, ?, 'rollback', ?)
                """,
                (
                    active["version"],
                    now_text,
                    sample_count,
                    brier_score,
                    baseline_brier,
                    json.dumps({"reason": reason}, sort_keys=True),
                ),
            )
        return True

    def _read_state(
        self,
        connection: sqlite3.Connection,
        key: str,
        default: Any,
    ) -> Any:
        row = connection.execute(
            "SELECT value_json FROM vision_learning_state WHERE key = ?",
            (key,),
        ).fetchone()
        return json.loads(row["value_json"]) if row else default

    def _write_state(
        self,
        connection: sqlite3.Connection,
        key: str,
        value: Any,
    ) -> None:
        connection.execute(
            """
            INSERT INTO vision_learning_state(key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            (
                key,
                json.dumps(value, sort_keys=True, allow_nan=False),
                _utc_now(),
            ),
        )


def _default_online_state() -> dict[str, Any]:
    return {
        "weights": dict.fromkeys(FEATURE_NAMES, 0.0),
        "bias": 0.0,
        "confidence_threshold": 0.70,
        "updates": 0,
        "updated_at": None,
    }


def _features_from_row(row: sqlite3.Row) -> dict[str, float]:
    try:
        snapshot = json.loads(row["features_json"])
    except (TypeError, json.JSONDecodeError):
        snapshot = {}
    if not isinstance(snapshot, dict):
        snapshot = {}
    trend = snapshot.get("trend")
    momentum = snapshot.get("momentum")
    pattern = snapshot.get("pattern")
    quality = snapshot.get("read_quality_score", 50)
    candle_count = snapshot.get("candle_count", 0)
    return {
        "call_score": _bounded(row["call_score"]),
        "put_score": _bounded(row["put_score"]),
        "candle_count": _bounded(candle_count, maximum=100) / 100,
        "trend_up": float(trend == "up"),
        "trend_down": float(trend == "down"),
        "momentum_bullish": float(momentum == "bullish"),
        "momentum_bearish": float(momentum == "bearish"),
        "bullish_pattern": float(pattern == "bullish_engulfing_shape"),
        "bearish_pattern": float(pattern == "bearish_engulfing_shape"),
        "read_quality": _bounded(quality, maximum=100) / 100,
    }


def _bounded(value: Any, *, maximum: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(maximum, number)) if math.isfinite(number) else 0.0


def _context_key(row: sqlite3.Row) -> str:
    hour = _parse_utc(row["created_at"]).hour
    return (
        f"{str(row['asset']).strip()}|{row['timeframe_seconds']}|"
        f"{str(row['market_regime']).strip()}|{hour}"
    )


def _outcome_rates(groups: dict[str, list[sqlite3.Row]]) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "trades": len(items),
            "wins": sum(row["outcome"] == "win" for row in items),
            "win_rate": sum(row["outcome"] == "win" for row in items) / len(items),
        }
        for name, items in sorted(groups.items())
    }


def _brier_score(model: dict[str, Any], features: np.ndarray, targets: np.ndarray) -> float:
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    probabilities = 1 / (
        1 + np.exp(-np.clip(((features - means) / scales) @ coefficients + model["intercept"], -40, 40))
    )
    return float(np.mean((targets - probabilities) ** 2))


def _build_artifact(
    classifier: LogisticRegression,
    scaler: StandardScaler,
    *,
    validation_brier: float,
    baseline_validation_brier: float,
    test_brier: float,
    baseline_brier: float,
    training_rows: int,
    validation_rows: int,
    test_rows: int,
) -> tuple[str, dict[str, Any]]:
    body = {
        "schema_version": 1,
        "algorithm": "sklearn.LogisticRegression",
        "feature_names": list(FEATURE_NAMES),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "feature_means": scaler.mean_.tolist(),
        "feature_scales": scaler.scale_.tolist(),
        "metrics": {
            "validation_brier_score": validation_brier,
            "baseline_validation_brier_score": baseline_validation_brier,
            "test_brier_score": test_brier,
            "baseline_brier_score": baseline_brier,
            "train_rows": training_rows,
            "validation_rows": validation_rows,
            "test_rows": test_rows,
        },
        "trained_at": _utc_now(),
        "validation_policy": "chronological_60_20_20_label_end_purged",
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    version = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{digest[:12]}"
    return version, {**body, "version": version, "sha256": digest}


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-40.0, min(40.0, value))))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value)
    return (
        timestamp.replace(tzinfo=timezone.utc)
        if timestamp.tzinfo is None
        else timestamp.astimezone(timezone.utc)
    )
