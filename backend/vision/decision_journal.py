"""Outcome-backed directional calibration and decision history."""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from vision.feature_pipeline import extract_visual_features


class VisionDecisionJournal:
    MIN_SIDE_SAMPLES = 100
    MIN_BIN_SAMPLES = 10
    MIN_VALIDATION_SAMPLES = 20
    MIN_CONTEXT_SAMPLES = 30
    MIN_READ_QUALITY = 70

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS vision_decisions (
                    decision_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    timeframe_seconds INTEGER,
                    signal_window TEXT,
                    direction TEXT NOT NULL CHECK (direction IN ('CALL', 'PUT')),
                    call_score REAL NOT NULL,
                    put_score REAL NOT NULL,
                    confidence_pct REAL,
                    market_regime TEXT NOT NULL DEFAULT 'unknown',
                    features_json TEXT NOT NULL,
                    outcome TEXT CHECK (outcome IN ('win', 'loss') OR outcome IS NULL),
                    outcome_at TEXT,
                    payout_pct REAL,
                    duration_seconds INTEGER,
                    source TEXT NOT NULL CHECK (source IN ('paper', 'demo_manual'))
                );
                CREATE INDEX IF NOT EXISTS idx_vision_decisions_side_time
                    ON vision_decisions(direction, created_at);
                CREATE TABLE IF NOT EXISTS vision_outcome_watch (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    decision_id TEXT NOT NULL UNIQUE,
                    armed_at TEXT NOT NULL,
                    expected_at TEXT NOT NULL,
                    last_outcome TEXT,
                    last_text TEXT,
                    stable_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS vision_outcome_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL,
                    previous_outcome TEXT,
                    new_outcome TEXT NOT NULL CHECK (new_outcome IN ('win', 'loss')),
                    source TEXT NOT NULL CHECK (
                        source IN ('manual', 'screen_ocr', 'manual_correction')
                    ),
                    changed_at TEXT NOT NULL,
                    note TEXT
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(vision_decisions)")
            }
            for name, declaration in (
                ("timeframe_seconds", "INTEGER"),
                ("signal_window", "TEXT"),
                ("confidence_pct", "REAL"),
                ("market_regime", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("payout_pct", "REAL"),
                ("duration_seconds", "INTEGER"),
                ("outcome_source", "TEXT NOT NULL DEFAULT 'manual'"),
            ):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE vision_decisions ADD COLUMN {name} {declaration}"
                    )
            watch_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(vision_outcome_watch)")
            }
            if "last_observed_at" not in watch_columns:
                connection.execute(
                    "ALTER TABLE vision_outcome_watch ADD COLUMN last_observed_at TEXT"
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def record_decision(self, state: dict[str, Any]) -> dict[str, Any]:
        scores = state.get("directional_scores")
        if not state.get("readable") or not isinstance(scores, dict):
            raise ValueError("Se requiere una lectura legible para registrar una decisión.")
        call_score, put_score = scores.get("CALL"), scores.get("PUT")
        if not _unit_interval(call_score) or not _unit_interval(put_score):
            raise ValueError("Las puntuaciones visuales no son válidas.")
        direction = "CALL" if call_score >= put_score else "PUT"
        timeframe = state.get("timeframe_seconds")
        if (
            isinstance(timeframe, bool)
            or not isinstance(timeframe, int)
            or timeframe < 1
        ):
            raise ValueError("Se requiere un timeframe calibrado para evitar duplicar señales.")
        timestamp = datetime.now(timezone.utc)
        captured_at = state.get("captured_at")
        if captured_at:
            try:
                timestamp = datetime.fromisoformat(captured_at)
            except (TypeError, ValueError) as exc:
                raise ValueError("La marca temporal de captura no es válida.") from exc
            if timestamp.tzinfo is None:
                raise ValueError("La marca temporal de captura debe incluir zona horaria.")
            timestamp = timestamp.astimezone(timezone.utc)
        signal_window = str(int(timestamp.timestamp()) // timeframe)
        confidence = (
            (state.get("calibration_status") or {})
            .get("estimated_accuracy_pct", {})
            .get(direction)
        )
        decision_id = str(uuid4())
        features = {
            "trend": state.get("trend"),
            "momentum": state.get("momentum"),
            "pattern": state.get("pattern"),
            "candle_count": state.get("candle_count"),
            "read_quality_score": state.get("read_quality_score"),
            "support_levels": state.get("support_levels", []),
            "resistance_levels": state.get("resistance_levels", []),
            "visual_snapshot": extract_visual_features(state),
        }
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT decision_id, direction FROM vision_decisions
                WHERE asset = ? AND timeframe_seconds = ? AND signal_window = ?
                ORDER BY created_at LIMIT 1
                """,
                (
                    str(state.get("asset") or "asset-unconfirmed"),
                    timeframe,
                    signal_window,
                ),
            ).fetchone()
            if existing is not None:
                return {
                    "decision_id": existing["decision_id"],
                    "candidate_direction": existing["direction"],
                    "duplicate": True,
                }
            connection.execute(
                """
                INSERT INTO vision_decisions (
                    decision_id, created_at, asset, timeframe_seconds, signal_window, direction,
                    call_score, put_score, confidence_pct, market_regime,
                    features_json, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'demo_manual')
                """,
                (
                    decision_id,
                    timestamp.isoformat(),
                    str(state.get("asset") or "asset-unconfirmed"),
                    timeframe,
                    signal_window,
                    direction,
                    call_score,
                    put_score,
                    confidence,
                    str(state.get("market_regime") or state.get("regime") or "unknown"),
                    json.dumps(features, sort_keys=True, allow_nan=False),
                ),
            )
        return {"decision_id": decision_id, "candidate_direction": direction}

    def record_outcome(
        self,
        decision_id: str,
        outcome: str,
        *,
        payout_pct: float | None = None,
        duration_seconds: int | None = None,
        source: str = "manual",
    ) -> dict[str, Any]:
        if outcome not in {"win", "loss"}:
            raise ValueError("El resultado debe ser win o loss.")
        if source not in {"manual", "screen_ocr"}:
            raise ValueError("El origen del resultado no es válido.")
        if payout_pct is not None and (
            not isinstance(payout_pct, (int, float))
            or isinstance(payout_pct, bool)
            or not math.isfinite(payout_pct)
            or not 0 <= payout_pct <= 100
        ):
            raise ValueError("El payout debe estar entre 0 y 100 por ciento.")
        if duration_seconds is not None and (
            isinstance(duration_seconds, bool)
            or not isinstance(duration_seconds, int)
            or not 1 <= duration_seconds <= 86_400
        ):
            raise ValueError("La duración debe estar entre 1 y 86400 segundos.")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE vision_decisions SET outcome = ?, outcome_at = ?,
                    payout_pct = ?, duration_seconds = ?, outcome_source = ?
                WHERE decision_id = ? AND outcome IS NULL
                """,
                (
                    outcome,
                    now,
                    payout_pct,
                    duration_seconds,
                    source,
                    decision_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError("Decisión inexistente o ya etiquetada.")
            connection.execute(
                """
                INSERT INTO vision_outcome_audit (
                    decision_id, previous_outcome, new_outcome, source, changed_at
                ) VALUES (?, NULL, ?, ?, ?)
                """,
                (decision_id, outcome, source, now),
            )
            connection.execute(
                "DELETE FROM vision_outcome_watch WHERE decision_id = ?",
                (decision_id,),
            )
        return self.get_calibration()

    def correct_outcome(
        self,
        decision_id: str,
        outcome: str,
        *,
        note: str,
    ) -> dict[str, Any]:
        if outcome not in {"win", "loss"} or not note.strip():
            raise ValueError("Indica win/loss y el motivo de la corrección.")
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT outcome FROM vision_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
            if row is None or row["outcome"] is None:
                raise KeyError("La decisión no tiene una etiqueta que corregir.")
            previous = row["outcome"]
            if previous == outcome:
                raise ValueError("La etiqueta ya tiene ese resultado.")
            connection.execute(
                """
                UPDATE vision_decisions
                SET outcome = ?, outcome_at = ?, outcome_source = 'manual_correction'
                WHERE decision_id = ?
                """,
                (outcome, now, decision_id),
            )
            connection.execute(
                """
                INSERT INTO vision_outcome_audit (
                    decision_id, previous_outcome, new_outcome, source, changed_at, note
                ) VALUES (?, ?, ?, 'manual_correction', ?, ?)
                """,
                (decision_id, previous, outcome, now, note.strip()[:500]),
            )
            connection.execute(
                "DELETE FROM vision_outcome_watch WHERE decision_id = ?",
                (decision_id,),
            )
        return self.get_calibration()

    def arm_auto_outcome(
        self,
        decision_id: str,
        duration_seconds: int,
        outcome_field: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            isinstance(duration_seconds, bool)
            or not isinstance(duration_seconds, int)
            or not 1 <= duration_seconds <= 86_400
        ):
            raise ValueError("La duración esperada debe estar entre 1 y 86400 segundos.")
        if not outcome_field.get("configured") or not outcome_field.get("available"):
            raise ValueError("Calibra la región OCR del resultado y verifica que Tesseract esté disponible.")
        observed_at = outcome_field.get("captured_at")
        if not isinstance(observed_at, str) or not _aware_timestamp(observed_at):
            raise ValueError("Espera una lectura OCR reciente del resultado antes de iniciar el seguimiento.")
        if outcome_field.get("result") in {"win", "loss"}:
            raise ValueError("El área de resultado no está vacía; limpia/abre la operación DEMO primero.")
        now = datetime.now(timezone.utc)
        expected_at = now.timestamp() + duration_seconds
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            decision = connection.execute(
                "SELECT outcome FROM vision_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
            if decision is None or decision["outcome"] is not None:
                raise KeyError("La decisión no existe o ya está etiquetada.")
            active = connection.execute(
                "SELECT decision_id FROM vision_outcome_watch WHERE id = 1"
            ).fetchone()
            if active:
                raise ValueError("Ya hay una operación DEMO bajo seguimiento.")
            connection.execute(
                """
                INSERT INTO vision_outcome_watch (
                    id, decision_id, armed_at, expected_at, last_observed_at
                ) VALUES (1, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    now.isoformat(),
                    datetime.fromtimestamp(expected_at, timezone.utc).isoformat(),
                    observed_at,
                ),
            )
        return self.get_auto_outcome_watch()

    def observe_auto_outcome(
        self,
        outcome_field: dict[str, Any],
        *,
        payout_pct: float | None = None,
    ) -> dict[str, Any] | None:
        outcome = outcome_field.get("result")
        confidence = outcome_field.get("confidence", 0)
        text = str(outcome_field.get("text") or "")
        observed_at = outcome_field.get("captured_at")
        if not isinstance(observed_at, str) or not _aware_timestamp(observed_at):
            return None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            watch = connection.execute(
                "SELECT * FROM vision_outcome_watch WHERE id = 1"
            ).fetchone()
            if watch is None:
                return None
            expected_at = datetime.fromisoformat(watch["expected_at"])
            observation_time = datetime.fromisoformat(observed_at)
            if observation_time.tzinfo is None:
                return None
            observation_time = observation_time.astimezone(timezone.utc)
            if (
                datetime.now(timezone.utc) < expected_at
                or observation_time < expected_at
                or watch["last_observed_at"] == observed_at
            ):
                return None
            if outcome not in {"win", "loss"} or not _unit_interval(confidence) or confidence < 0.8:
                connection.execute(
                    """
                    UPDATE vision_outcome_watch
                    SET last_outcome = NULL, last_text = ?,
                        last_observed_at = ?, stable_count = 0
                    WHERE id = 1
                    """,
                    (text[:250], observed_at),
                )
                return None
            count = watch["stable_count"] + 1 if watch["last_outcome"] == outcome else 1
            connection.execute(
                """
                UPDATE vision_outcome_watch
                SET last_outcome = ?, last_text = ?, last_observed_at = ?,
                    stable_count = ?
                WHERE id = 1
                """,
                (outcome, text[:250], observed_at, count),
            )
            if count < 2:
                return None
            now = datetime.now(timezone.utc).isoformat()
            duration = min(
                86_400,
                max(
                    1,
                    int(
                        (
                            datetime.fromisoformat(now)
                            - datetime.fromisoformat(watch["armed_at"])
                        ).total_seconds()
                    ),
                ),
            )
            cursor = connection.execute(
                """
                UPDATE vision_decisions
                SET outcome = ?, outcome_at = ?, duration_seconds = ?,
                    payout_pct = ?, outcome_source = 'screen_ocr'
                WHERE decision_id = ? AND outcome IS NULL
                """,
                (outcome, now, duration, payout_pct, watch["decision_id"]),
            )
            if cursor.rowcount != 1:
                connection.execute("DELETE FROM vision_outcome_watch WHERE id = 1")
                raise KeyError("La decisión vigilada ya no está pendiente.")
            connection.execute(
                """
                INSERT INTO vision_outcome_audit (
                    decision_id, previous_outcome, new_outcome, source, changed_at, note
                ) VALUES (?, NULL, ?, 'screen_ocr', ?, ?)
                """,
                (watch["decision_id"], outcome, now, text[:250]),
            )
            connection.execute("DELETE FROM vision_outcome_watch WHERE id = 1")
            return {
                "decision_id": watch["decision_id"],
                "outcome": outcome,
                "duration_seconds": duration,
                "source": "screen_ocr",
            }

    def get_auto_outcome_watch(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT decision_id, armed_at, expected_at FROM vision_outcome_watch WHERE id = 1"
            ).fetchone()
        return dict(row) if row else None

    def cancel_auto_outcome_watch(self, decision_id: str) -> bool:
        with self._connect() as connection:
            active = connection.execute(
                "SELECT decision_id FROM vision_outcome_watch WHERE id = 1"
            ).fetchone()
            if active is None:
                return False
            if active["decision_id"] != decision_id:
                raise KeyError("La decisión indicada no tiene un seguimiento activo.")
            connection.execute("DELETE FROM vision_outcome_watch WHERE id = 1")
        return True

    def get_recent(self, *, limit: int = 25) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT decision_id, created_at, asset, direction, call_score,
                       put_score, timeframe_seconds, signal_window, confidence_pct, market_regime,
                       outcome, payout_pct, duration_seconds, outcome_source, source
                FROM vision_decisions ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_calibration(
        self,
        call_score: float | None = None,
        put_score: float | None = None,
        *,
        asset: str | None = None,
        timeframe_seconds: int | None = None,
        market_regime: str | None = None,
        hour_utc: int | None = None,
        payout_pct: float | None = None,
        read_quality_score: int = 0,
        edge_margin_pct: float = 5.0,
        minimum_margin_pct: float = 15.0,
        minimum_probability_pct: float = 0.0,
    ) -> dict[str, Any]:
        if (
            not math.isfinite(edge_margin_pct)
            or not 0 <= edge_margin_pct <= 25
            or not math.isfinite(minimum_margin_pct)
            or not 0 <= minimum_margin_pct <= 49
            or not math.isfinite(minimum_probability_pct)
            or not 0 <= minimum_probability_pct <= 100
        ):
            raise ValueError("Los umbrales de calibración están fuera de rango.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT created_at, asset, timeframe_seconds, market_regime,
                       direction, call_score, put_score, outcome
                FROM vision_decisions WHERE outcome IS NOT NULL
                ORDER BY created_at, decision_id
                """
            ).fetchall()
        by_side = {
            side: [row for row in rows if row["direction"] == side]
            for side in ("CALL", "PUT")
        }
        side_report: dict[str, Any] = {}
        for side, examples in by_side.items():
            side_report[side] = self._validate_side(side, examples)
        accepted = all(report["validated"] for report in side_report.values())
        probabilities = {"CALL": None, "PUT": None}
        contextual: dict[str, dict[str, Any]] = {}
        context_ready = (
            isinstance(asset, str)
            and bool(asset.strip())
            and isinstance(timeframe_seconds, int)
            and not isinstance(timeframe_seconds, bool)
            and timeframe_seconds > 0
            and isinstance(market_regime, str)
            and bool(market_regime.strip())
            and isinstance(hour_utc, int)
            and not isinstance(hour_utc, bool)
            and 0 <= hour_utc <= 23
        )
        if context_ready:
            for side, examples in by_side.items():
                training = examples[: math.floor(len(examples) * 0.8)]
                selected = [
                    row for row in training
                    if _matches_context(
                        row,
                        asset=asset.strip(),
                        timeframe_seconds=timeframe_seconds,
                        market_regime=market_regime.strip(),
                        hour_utc=hour_utc,
                    )
                ]
                wins = sum(row["outcome"] == "win" for row in selected)
                probability = wins / len(selected) if selected else None
                lower, upper = _wilson_interval(wins, len(selected))
                contextual[side] = {
                    "samples": len(selected),
                    "wins": wins,
                    "probability_pct": round(probability * 100, 2) if probability is not None else None,
                    "lower_95_pct": round(lower * 100, 2) if lower is not None else None,
                    "upper_95_pct": round(upper * 100, 2) if upper is not None else None,
                    "minimum_samples": self.MIN_CONTEXT_SAMPLES,
                    "sufficient": len(selected) >= self.MIN_CONTEXT_SAMPLES,
                }

        payout_valid = (
            isinstance(payout_pct, (int, float))
            and not isinstance(payout_pct, bool)
            and math.isfinite(payout_pct)
            and 0 < payout_pct <= 100
        )
        break_even_pct = 100 / (100 + float(payout_pct)) * 100 if payout_valid else None
        required_probability_pct = (
            max(break_even_pct + edge_margin_pct, minimum_probability_pct)
            if break_even_pct is not None
            else None
        )
        decision = "NO_TRADE"
        reason = "El historial global aún no supera la validación temporal."
        directional_margin_pct = None
        conservative_margin_pct = None
        visual_direction = (
            "CALL"
            if _unit_interval(call_score) and _unit_interval(put_score) and call_score > put_score
            else "PUT"
            if _unit_interval(call_score) and _unit_interval(put_score) and put_score > call_score
            else None
        )
        visual_margin_pct = (
            round(abs(call_score - put_score) * 100, 2)
            if _unit_interval(call_score) and _unit_interval(put_score)
            else None
        )
        if not context_ready:
            reason = "Falta el contexto completo de activo, timeframe, régimen u hora UTC."
        elif not accepted:
            reason = "Se requiere historial suficiente y validado fuera de muestra en ambas direcciones."
        elif not payout_valid:
            reason = "No se pudo leer un payout válido; no es posible calcular el punto de equilibrio."
        elif read_quality_score < self.MIN_READ_QUALITY:
            reason = (
                f"Calidad visual/OCR insuficiente ({read_quality_score}/100; "
                f"se requiere {self.MIN_READ_QUALITY})."
            )
        elif not all(report["sufficient"] for report in contextual.values()):
            reason = (
                "Faltan resultados etiquetados para este activo, timeframe, régimen y hora; "
                "se requieren al menos 30 por dirección."
            )
        elif visual_direction is None:
            reason = "La lectura visual no presenta una dirección dominante válida."
        else:
            call_probability = contextual["CALL"]["probability_pct"]
            put_probability = contextual["PUT"]["probability_pct"]
            directional_margin_pct = call_probability - put_probability
            candidate = "CALL" if call_probability > put_probability else "PUT" if put_probability > call_probability else None
            if candidate is None:
                reason = "CALL y PUT tienen la misma tasa histórica en este contexto."
            else:
                other = "PUT" if candidate == "CALL" else "CALL"
                candidate_report = contextual[candidate]
                other_report = contextual[other]
                conservative_margin_pct = (
                    candidate_report["lower_95_pct"] - other_report["upper_95_pct"]
                )
                probabilities = {
                    side: report["probability_pct"] / 100
                    for side, report in contextual.items()
                }
                if visual_direction != candidate:
                    reason = (
                        f"El score visual favorece {visual_direction}, pero el historial "
                        f"contextual favorece {candidate}; se requiere concordancia."
                    )
                elif candidate_report["lower_95_pct"] < required_probability_pct:
                    reason = (
                        f"El límite inferior IC95% de {candidate} no supera el umbral requerido "
                        f"({required_probability_pct:.2f}%, incluyendo breakeven y calibración adaptativa)."
                    )
                elif conservative_margin_pct < minimum_margin_pct:
                    reason = (
                        f"Los intervalos 95% se solapan o no separan {candidate} de {other} "
                        f"por {minimum_margin_pct:.2f} puntos."
                    )
                else:
                    decision = candidate
                    reason = (
                        f"{candidate} supera el umbral de probabilidad requerido y separa "
                        "su intervalo IC95% del contrario."
                    )

        return {
            "calibrated": accepted,
            "minimum_samples_per_direction": self.MIN_SIDE_SAMPLES,
            "minimum_context_samples_per_direction": self.MIN_CONTEXT_SAMPLES,
            "minimum_read_quality": self.MIN_READ_QUALITY,
            "read_quality_score": read_quality_score,
            "edge_margin_pct": edge_margin_pct,
            "break_even_pct": round(break_even_pct, 2) if break_even_pct is not None else None,
            "required_probability_pct": (
                round(required_probability_pct, 2)
                if required_probability_pct is not None
                else None
            ),
            "minimum_margin_pct": minimum_margin_pct,
            "adaptive_confidence_threshold_pct": minimum_probability_pct,
            "directional_margin_pct": (
                round(directional_margin_pct, 2)
                if directional_margin_pct is not None
                else None
            ),
            "visual_direction": visual_direction,
            "visual_margin_pct": visual_margin_pct,
            "conservative_margin_pct": (
                round(conservative_margin_pct, 2)
                if conservative_margin_pct is not None
                else None
            ),
            "context": {
                "asset": asset,
                "timeframe_seconds": timeframe_seconds,
                "market_regime": market_regime,
                "hour_utc": hour_utc,
                "payout_pct": float(payout_pct) if payout_valid else None,
            },
            "contextual_estimates": contextual,
            "directional_win_rates": {
                side: round(report["win_rate"] * 100, 2)
                if report["win_rate"] is not None
                else None
                for side, report in side_report.items()
            },
            "estimated_accuracy_pct": {
                side: (
                    round(probability * 100, 2)
                    if probability is not None
                    else None
                )
                for side, probability in probabilities.items()
            },
            "validation": side_report,
            "decision": decision,
            "reason": reason,
        }

    def _validate_side(self, side: str, examples: list[sqlite3.Row]) -> dict[str, Any]:
        count = len(examples)
        wins = sum(row["outcome"] == "win" for row in examples)
        result: dict[str, Any] = {
            "samples": count,
            "wins": wins,
            "win_rate": wins / count if count else None,
            "validated": False,
            "validation_samples": 0,
            "brier_score": None,
            "baseline_brier_score": None,
            "reason": None,
        }
        if count < self.MIN_SIDE_SAMPLES:
            result["reason"] = (
                f"Se requieren {self.MIN_SIDE_SAMPLES} resultados etiquetados; hay {count}."
            )
            return result
        split = math.floor(count * 0.8)
        training = examples[:split]
        validation = examples[split:]
        if len(training) < 80 or len(validation) < self.MIN_VALIDATION_SAMPLES:
            result["reason"] = "Particiones temporales insuficientes para validar."
            return result
        base_rate = sum(row["outcome"] == "win" for row in training) / len(training)
        squared_errors = []
        accepted_validation = 0
        for row in validation:
            score = row["call_score"] if side == "CALL" else row["put_score"]
            probability, bin_count = self._estimate_probability(training, score)
            if probability is None or bin_count < self.MIN_BIN_SAMPLES:
                continue
            actual = 1.0 if row["outcome"] == "win" else 0.0
            squared_errors.append((probability - actual) ** 2)
            accepted_validation += 1
        result["validation_samples"] = accepted_validation
        if accepted_validation < self.MIN_VALIDATION_SAMPLES:
            result["reason"] = (
                "El holdout temporal no contiene suficientes muestras en rangos calibrables."
            )
            return result
        brier = sum(squared_errors) / len(squared_errors)
        baseline = base_rate * (1 - base_rate)
        result["brier_score"] = brier
        result["baseline_brier_score"] = baseline
        result["validated"] = brier < baseline and brier <= 0.25
        result["reason"] = (
            "Calibración supera al baseline en holdout temporal."
            if result["validated"]
            else "Calibración no supera al baseline en holdout temporal."
        )
        return result

    def _estimate_probability(
        self,
        examples: list[sqlite3.Row],
        score: float,
    ) -> tuple[float | None, int]:
        bucket = min(9, max(0, int(score * 10)))
        selected = [
            row
            for row in examples
            if min(9, max(0, int(
                (row["call_score"] if row["direction"] == "CALL" else row["put_score"]) * 10
            ))) == bucket
        ]
        if len(selected) < self.MIN_BIN_SAMPLES:
            return None, len(selected)
        wins = sum(row["outcome"] == "win" for row in selected)
        return wins / len(selected), len(selected)


def _unit_interval(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _aware_timestamp(value: str) -> bool:
    try:
        return datetime.fromisoformat(value).tzinfo is not None
    except ValueError:
        return False


def _matches_context(
    row: sqlite3.Row,
    *,
    asset: str,
    timeframe_seconds: int,
    market_regime: str,
    hour_utc: int,
) -> bool:
    try:
        timestamp = datetime.fromisoformat(row["created_at"])
    except (TypeError, ValueError):
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return (
        str(row["asset"]).strip().casefold() == asset.casefold()
        and row["timeframe_seconds"] == timeframe_seconds
        and str(row["market_regime"] or "unknown").strip().casefold()
        == market_regime.casefold()
        and timestamp.astimezone(timezone.utc).hour == hour_utc
    )


def _wilson_interval(wins: int, samples: int, *, z_score: float = 1.96) -> tuple[float | None, float | None]:
    if samples <= 0:
        return None, None
    rate = wins / samples
    z_squared = z_score**2
    denominator = 1 + z_squared / samples
    center = (rate + z_squared / (2 * samples)) / denominator
    radius = (
        z_score
        * math.sqrt(rate * (1 - rate) / samples + z_squared / (4 * samples**2))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)
