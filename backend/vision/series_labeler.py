"""Persist and cautiously label next-candle visual prediction samples."""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from vision.feature_pipeline import extract_visual_features
from vision.ohlc_store import VisionCandleStore

MIN_LABEL_QUALITY = 70
MIN_BODY_RATIO = 0.012
MIN_CONFIRM_SECONDS = 1.0


class VisionSeriesLabeler:
    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS vision_series_outcome_watch (
                    decision_id TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    timeframe_seconds INTEGER NOT NULL,
                    target_candle_open_at TEXT NOT NULL,
                    prediction_direction TEXT NOT NULL CHECK (
                        prediction_direction IN ('CALL', 'PUT')
                    ),
                    market_regime TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_observed_direction TEXT CHECK (
                        last_observed_direction IN ('CALL', 'PUT', 'FLAT') OR
                        last_observed_direction IS NULL
                    ),
                    last_observed_at TEXT,
                    stable_since TEXT,
                    stable_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending' CHECK (
                        status IN ('pending', 'labeled', 'expired')
                    ),
                    completion_reason TEXT
                );
                CREATE TABLE IF NOT EXISTS vision_series_samples (
                    decision_id TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    timeframe_seconds INTEGER NOT NULL,
                    market_regime TEXT NOT NULL,
                    prediction_direction TEXT NOT NULL CHECK (
                        prediction_direction IN ('CALL', 'PUT')
                    ),
                    target_direction TEXT NOT NULL CHECK (
                        target_direction IN ('CALL', 'PUT', 'FLAT')
                    ),
                    target INTEGER CHECK (target IN (0, 1) OR target IS NULL),
                    target_candle_open_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    labeled_at TEXT NOT NULL,
                    source TEXT NOT NULL CHECK (source = 'pixel_series'),
                    price_basis TEXT NOT NULL CHECK (
                        price_basis = 'normalized_chart_pixels'
                    ),
                    features_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_vision_series_samples_context
                    ON vision_series_samples(asset, timeframe_seconds, labeled_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def watch_decision(self, decision_id: str, state: dict[str, Any]) -> dict[str, Any]:
        if not state.get("readable"):
            raise ValueError("Solo se puede observar una decisión con lectura legible.")
        timeframe = state.get("timeframe_seconds")
        timer = (
            (state.get("ocr") or {})
            .get("fields", {})
            .get("timer", {})
            .get("seconds_remaining")
        )
        timestamp_text = state.get("captured_at")
        direction = _prediction_direction(state)
        if direction is None:
            return {
                "decision_id": decision_id,
                "watching": False,
                "reason": "La puntuación visual está empatada; no hay dirección que etiquetar.",
                "label_source": "pixel_series",
            }
        if (
            isinstance(timeframe, bool)
            or not isinstance(timeframe, int)
            or timeframe < 1
            or isinstance(timer, bool)
            or not isinstance(timer, int)
            or timer < 0
            or not isinstance(timestamp_text, str)
        ):
            raise ValueError("Falta timeframe, timer estable, marca temporal o dirección válida.")
        timestamp = datetime.fromisoformat(timestamp_text)
        if timestamp.tzinfo is None:
            raise ValueError("La marca temporal de captura debe incluir zona horaria.")
        target_open = _next_candle_open(
            timestamp.astimezone(timezone.utc),
            timer,
            timeframe,
        )
        snapshot = extract_visual_features(state)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO vision_series_outcome_watch (
                    decision_id, asset, timeframe_seconds, target_candle_open_at,
                    prediction_direction, market_regime, features_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    str(state.get("asset") or "unknown"),
                    timeframe,
                    target_open.isoformat(),
                    direction,
                    str(state.get("market_regime") or "unknown"),
                    json.dumps(snapshot, sort_keys=True, allow_nan=False),
                    timestamp.astimezone(timezone.utc).isoformat(),
                ),
            )
            return {
                "decision_id": decision_id,
                "watching": True,
                "duplicate": cursor.rowcount == 0,
                "target_candle_open_at": target_open.isoformat(),
                "label_source": "pixel_series",
            }

    def observe_completed_candles(
        self,
        candle_store: VisionCandleStore,
        *,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        observation_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        with self._connect() as connection:
            watches = connection.execute(
                "SELECT * FROM vision_series_outcome_watch WHERE status = 'pending' "
                "ORDER BY target_candle_open_at"
            ).fetchall()
        expired = [
            watch
            for watch in watches
            if observation_time.timestamp()
            > (
                datetime.fromisoformat(watch["target_candle_open_at"]).timestamp()
                + watch["timeframe_seconds"] * 2
            )
        ]
        if expired:
            with self._connect() as connection:
                connection.executemany(
                    """
                    UPDATE vision_series_outcome_watch
                    SET status = 'expired', completion_reason = ?
                    WHERE decision_id = ? AND status = 'pending'
                    """,
                    [
                        (
                            "No se observó el cierre con muestras visuales recientes y de calidad.",
                            watch["decision_id"],
                        )
                        for watch in expired
                    ],
                )
            expired_ids = {watch["decision_id"] for watch in expired}
            watches = [
                watch for watch in watches if watch["decision_id"] not in expired_ids
            ]
        labeled: list[dict[str, Any]] = []
        for watch in watches:
            candle = candle_store.get_completed_candle(
                asset=watch["asset"],
                timeframe_seconds=watch["timeframe_seconds"],
                candle_open_at=watch["target_candle_open_at"],
                minimum_quality=MIN_LABEL_QUALITY,
            )
            if candle is None:
                continue
            target_direction = _candle_direction(candle)
            observed_at = str(candle["latest_observed_at"])
            if observed_at == watch["last_observed_at"]:
                continue
            if target_direction is None:
                continue
            same_direction = watch["last_observed_direction"] == target_direction
            stable_count = watch["stable_count"] + 1 if same_direction else 1
            if stable_count < 2:
                with self._connect() as connection:
                    connection.execute(
                        """
                        UPDATE vision_series_outcome_watch
                        SET last_observed_direction = ?, last_observed_at = ?,
                            stable_since = ?, stable_count = ?
                        WHERE decision_id = ?
                        """,
                        (
                            target_direction,
                            observed_at,
                            watch["stable_since"] if same_direction else observed_at,
                            stable_count,
                            watch["decision_id"],
                        ),
                    )
                continue
            first_observed = (
                datetime.fromisoformat(watch["stable_since"])
                if same_direction and watch["stable_since"]
                else None
            )
            latest_observed = datetime.fromisoformat(observed_at)
            if first_observed and first_observed.tzinfo is None:
                first_observed = first_observed.replace(tzinfo=timezone.utc)
            if latest_observed.tzinfo is None:
                latest_observed = latest_observed.replace(tzinfo=timezone.utc)
            stable_duration = (
                (latest_observed - first_observed).total_seconds()
                if first_observed is not None
                else 0.0
            )
            if not same_direction or stable_duration < MIN_CONFIRM_SECONDS:
                with self._connect() as connection:
                    connection.execute(
                        """
                        UPDATE vision_series_outcome_watch
                        SET last_observed_direction = ?, last_observed_at = ?,
                            stable_since = ?, stable_count = ?
                        WHERE decision_id = ?
                        """,
                        (
                            target_direction,
                            observed_at,
                            watch["stable_since"] if same_direction else observed_at,
                            stable_count,
                            watch["decision_id"],
                        ),
                    )
                continue

            prediction = watch["prediction_direction"]
            target = (
                1
                if target_direction == prediction
                else 0
                if target_direction in {"CALL", "PUT"}
                else None
            )
            labeled_at = observation_time.isoformat()
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT OR IGNORE INTO vision_series_samples (
                        decision_id, asset, timeframe_seconds, market_regime,
                        prediction_direction, target_direction, target,
                        target_candle_open_at, created_at, labeled_at,
                        source, price_basis, features_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              'pixel_series', 'normalized_chart_pixels', ?)
                    """,
                    (
                        watch["decision_id"],
                        watch["asset"],
                        watch["timeframe_seconds"],
                        watch["market_regime"],
                        prediction,
                        target_direction,
                        target,
                        watch["target_candle_open_at"],
                        watch["created_at"],
                        labeled_at,
                        watch["features_json"],
                    ),
                )
                connection.execute(
                    """
                    UPDATE vision_series_outcome_watch
                    SET status = 'labeled', completion_reason = NULL
                    WHERE decision_id = ?
                    """,
                    (watch["decision_id"],),
                )
            labeled.append(
                {
                    "decision_id": watch["decision_id"],
                    "prediction_direction": prediction,
                    "target_direction": target_direction,
                    "target": target,
                    "source": "pixel_series",
                    "price_basis": "normalized_chart_pixels",
                }
            )
        return labeled

    def get_dataset_status(self) -> dict[str, Any]:
        with self._connect() as connection:
            counts = connection.execute(
                """
                SELECT target_direction, COUNT(*) AS count
                FROM vision_series_samples
                GROUP BY target_direction
                """
            ).fetchall()
            pending = connection.execute(
                "SELECT COUNT(*) AS count FROM vision_series_outcome_watch "
                "WHERE status = 'pending'"
            ).fetchone()["count"]
            expired = connection.execute(
                "SELECT COUNT(*) AS count FROM vision_series_outcome_watch "
                "WHERE status = 'expired'"
            ).fetchone()["count"]
            total = connection.execute(
                "SELECT COUNT(*) AS count FROM vision_series_samples"
            ).fetchone()["count"]
        by_direction = {row["target_direction"]: int(row["count"]) for row in counts}
        return {
            "source": "pixel_series",
            "price_basis": "normalized_chart_pixels",
            "total_labeled": int(total),
            "pending": int(pending),
            "expired": int(expired),
            "by_direction": {
                "CALL": by_direction.get("CALL", 0),
                "PUT": by_direction.get("PUT", 0),
                "FLAT": by_direction.get("FLAT", 0),
            },
            "model_training_enabled": False,
            "note": (
                "Etiquetas direccionales inferidas del color/cuerpo de velas visibles; "
                "no son resultados de bróker ni precios OHLC reales."
            ),
        }

    def get_samples(self, *, limit: int = 5000) -> list[dict[str, Any]]:
        if not 1 <= limit <= 50_000:
            raise ValueError("El límite de muestras debe estar entre 1 y 50000.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT decision_id, asset, timeframe_seconds, market_regime,
                       prediction_direction, target_direction, target,
                       target_candle_open_at, created_at, labeled_at, source,
                       price_basis, features_json
                FROM vision_series_samples
                ORDER BY labeled_at, decision_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                **dict(row),
                "features": json.loads(row["features_json"]),
            }
            for row in rows
        ]


def _prediction_direction(state: dict[str, Any]) -> str | None:
    scores = state.get("directional_scores")
    if not isinstance(scores, dict):
        return None
    call_score, put_score = scores.get("CALL"), scores.get("PUT")
    if not (
        _finite_unit(call_score)
        and _finite_unit(put_score)
        and call_score != put_score
    ):
        return None
    return "CALL" if call_score > put_score else "PUT"


def _finite_unit(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _next_candle_open(
    captured_at: datetime,
    timer_seconds_remaining: int,
    timeframe_seconds: int,
) -> datetime:
    boundary = captured_at.timestamp() + timer_seconds_remaining
    timestamp = math.ceil(boundary / timeframe_seconds) * timeframe_seconds
    return datetime.fromtimestamp(timestamp, timezone.utc)


def _candle_direction(candle: dict[str, Any]) -> str | None:
    width = candle["chart_width"]
    height = candle["chart_height"]
    pixel_body_size = abs(candle["close"] - candle["open"]) * height
    if width < 320 or height < 160 or pixel_body_size / height < MIN_BODY_RATIO:
        return "FLAT"
    if candle["close"] > candle["open"]:
        return "CALL"
    if candle["close"] < candle["open"]:
        return "PUT"
    return "FLAT"
