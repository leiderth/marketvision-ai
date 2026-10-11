"""Persistent paper-trade journal and outcome-driven learning summaries."""
from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.model_registry import feature_vector
from ml.schema import FEATURE_NAMES


class ContinuousLearningManager:
    def __init__(
        self,
        database_path: str | Path,
        dataset_path: str | Path,
    ) -> None:
        self._database_path = Path(database_path)
        self._dataset_path = Path(dataset_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()
        self.export_dataset()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _migrate(self) -> None:
        frozen_root = getattr(sys, "_MEIPASS", None)
        migration_root = (
            Path(frozen_root) / "database" / "migrations"
            if frozen_root
            else Path(__file__).parents[1] / "database" / "migrations"
        )
        migration = migration_root / "001_trade_learning.sql"
        restore_migration = migration_root / "002_restore_open_paper_trades.sql"
        status_migration = migration_root / "003_trade_status.sql"
        vision_learning_migration = migration_root / "004_vision_learning.sql"
        with self._connect() as connection:
            connection.executescript(migration.read_text(encoding="utf-8"))
            connection.executescript(
                vision_learning_migration.read_text(encoding="utf-8")
            )
            trade_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(trades)").fetchall()
            }
            for statement in restore_migration.read_text(encoding="utf-8").split(";"):
                match = re.search(r"ADD COLUMN\s+([a-z_]+)", statement, re.IGNORECASE)
                if match and match.group(1) not in trade_columns:
                    connection.execute(statement)
                    trade_columns.add(match.group(1))
            for statement in status_migration.read_text(encoding="utf-8").split(";"):
                match = re.search(r"ADD COLUMN\s+([a-z_]+)", statement, re.IGNORECASE)
                if match and match.group(1) not in trade_columns:
                    connection.execute(statement)
                    trade_columns.add(match.group(1))

    def record_open_trade(
        self,
        order_id: str,
        order: dict[str, Any],
        opened_at: str,
    ) -> None:
        snapshot = order.get("indicator_snapshot")
        if not isinstance(snapshot, dict):
            raise TypeError("La orden paper requiere un snapshot de indicadores.")
        values = snapshot.get("values")
        latest_close = snapshot.get("latest_close")
        if not isinstance(values, dict) or not _finite_number(latest_close):
            raise ValueError("El snapshot paper no contiene indicadores y cierre válidos.")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO trades (
                    order_id, opened_at, asset, timeframe, direction, amount,
                    expiration_seconds, reasons_json, indicators_json, market_regime,
                    confidence, strategy, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    opened_at,
                    order["asset"],
                    order["timeframe"],
                    order["direction"],
                    order["amount"],
                    order["expiration_seconds"],
                    json.dumps(order["reasons"], allow_nan=False),
                    json.dumps(snapshot, sort_keys=True, allow_nan=False),
                    order["market_regime"],
                    order["signal_confidence"],
                    order["strategy"],
                    order.get("model_version"),
                ),
            )

    def get_open_trades(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT order_id, opened_at, asset, timeframe, direction, amount,
                       expiration_seconds, reasons_json, indicators_json,
                       market_regime, confidence, strategy, model_version
                FROM trades WHERE status = 'open' ORDER BY opened_at
                """
            ).fetchall()
        return [
            {
                "order_id": row["order_id"],
                "opened_at": row["opened_at"],
                "order": {
                    "asset": row["asset"],
                    "timeframe": row["timeframe"],
                    "direction": row["direction"],
                    "amount": row["amount"],
                    "expiration_seconds": row["expiration_seconds"],
                    "reasons": json.loads(row["reasons_json"]),
                    "indicator_snapshot": json.loads(row["indicators_json"]),
                    "market_regime": row["market_regime"],
                    "signal_confidence": row["confidence"],
                    "strategy": row["strategy"],
                    "model_version": row["model_version"],
                },
            }
            for row in rows
        ]

    def record_closed_trade(self, trade: dict[str, Any]) -> dict[str, Any]:
        pnl_pct = trade["pnl_pct"]
        outcome = "win" if pnl_pct > 0 else "loss"
        snapshot = trade["indicator_snapshot"]
        features = feature_vector(snapshot["values"], snapshot["latest_close"])
        label = trade["direction"] if outcome == "win" else _opposite(trade["direction"])
        closed_at = trade["settled_at"]

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE trades
                SET closed_at = ?, outcome = ?, payout_pct = ?, duration_seconds = ?,
                    pnl_pct = ?, status = 'closed'
                WHERE order_id = ? AND status = 'open'
                """,
                (
                    closed_at,
                    outcome,
                    trade.get("payout_pct"),
                    trade["duration_seconds"],
                    pnl_pct,
                    trade["order_id"],
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Operación paper no abierta en el diario: {trade['order_id']}")
            connection.execute(
                """
                INSERT INTO training_samples (
                    order_id, timestamp, label_end_timestamp, label, features_json,
                    realized_return_pct
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trade["order_id"],
                    trade["opened_at"],
                    closed_at,
                    label,
                    json.dumps(features, sort_keys=True, allow_nan=False),
                    _directional_return(label, outcome, trade.get("payout_pct")),
                ),
            )
            self._update_online_state(
                connection,
                features,
                label,
                outcome,
                trade["signal_confidence"],
            )
        self.export_dataset()
        return {"order_id": trade["order_id"], "outcome": outcome, "label": label}

    def cancel_open_trade(self, order_id: str, cancelled_at: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE trades SET status = 'cancelled', closed_at = ?
                WHERE order_id = ? AND status = 'open'
                """,
                (cancelled_at, order_id),
            )
        return cursor.rowcount == 1

    def _update_online_state(
        self,
        connection: sqlite3.Connection,
        features: dict[str, float],
        label: str,
        outcome: str,
        confidence: float,
    ) -> None:
        row = connection.execute(
            "SELECT value_json FROM learning_state WHERE key = 'online'"
        ).fetchone()
        state = json.loads(row["value_json"]) if row else {
            "weights": dict.fromkeys(FEATURE_NAMES, 0.0),
            "confidence_threshold": 0.65,
            "updates": 0,
        }
        direction = 1.0 if label == "CALL" else -1.0
        for name in FEATURE_NAMES:
            delta = 0.05 * direction * math.tanh(features[name])
            state["weights"][name] = max(
                -2.0,
                min(2.0, state["weights"][name] + delta),
            )
        threshold = float(state["confidence_threshold"])
        state["confidence_threshold"] = max(
            0.5,
            min(
                0.95,
                threshold + (0.01 if outcome == "loss" else -0.002 if confidence >= threshold else 0),
            ),
        )
        state["updates"] = int(state["updates"]) + 1
        connection.execute(
            """
            INSERT INTO learning_state(key, value_json, updated_at) VALUES ('online', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json,
                                           updated_at = excluded.updated_at
            """,
            (
                json.dumps(state, sort_keys=True, allow_nan=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def export_dataset(self) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.timestamp, s.label_end_timestamp, s.label, s.features_json
                    , s.realized_return_pct
                FROM training_samples s
                JOIN trades t ON t.order_id = s.order_id
                ORDER BY s.timestamp, s.order_id
                """
            ).fetchall()
        self._dataset_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._dataset_path.with_suffix(self._dataset_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8", newline="") as dataset:
            writer = csv.DictWriter(
                dataset,
                fieldnames=[
                    "timestamp",
                    "label_end_timestamp",
                    *FEATURE_NAMES,
                    "label",
                    "realized_return_pct",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "timestamp": row["timestamp"],
                        "label_end_timestamp": row["label_end_timestamp"],
                        **json.loads(row["features_json"]),
                        "label": row["label"],
                        "realized_return_pct": row["realized_return_pct"],
                    }
                )
        temporary_path.replace(self._dataset_path)
        return len(rows)

    def get_dashboard(self) -> dict[str, Any]:
        with self._connect() as connection:
            closed_trades = connection.execute(
                """
                SELECT asset, strategy, opened_at, outcome, pnl_pct
                FROM trades WHERE status = 'closed'
                ORDER BY opened_at
                """
            ).fetchall()
            online = connection.execute(
                "SELECT value_json, updated_at FROM learning_state WHERE key = 'online'"
            ).fetchone()
            models = connection.execute(
                """
                SELECT model_version, evaluated_at, metric_name, metric_value, decision
                FROM model_evaluations ORDER BY evaluated_at DESC LIMIT 20
                """
            ).fetchall()
            sample_count = connection.execute(
                "SELECT COUNT(*) FROM training_samples"
            ).fetchone()[0]
        by_asset: dict[str, list[sqlite3.Row]] = {}
        by_strategy: dict[str, list[sqlite3.Row]] = {}
        by_hour: dict[str, list[sqlite3.Row]] = {}
        for trade in closed_trades:
            by_asset.setdefault(trade["asset"], []).append(trade)
            by_strategy.setdefault(trade["strategy"], []).append(trade)
            hour = datetime.fromisoformat(trade["opened_at"]).astimezone(timezone.utc).strftime("%H:00 UTC")
            by_hour.setdefault(hour, []).append(trade)
        state = json.loads(online["value_json"]) if online else {
            "weights": dict.fromkeys(FEATURE_NAMES, 0.0),
            "confidence_threshold": 0.65,
            "updates": 0,
        }
        return {
            "closed_trades": len(closed_trades),
            "dataset_samples": sample_count,
            "by_asset": _win_rates(by_asset),
            "by_strategy": _win_rates(by_strategy),
            "by_hour_utc": _win_rates(by_hour),
            "online_learning": {
                **state,
                "updated_at": online["updated_at"] if online else None,
                "feature_importance": sorted(
                    (
                        {"feature": name, "importance": abs(weight), "weight": weight}
                        for name, weight in state["weights"].items()
                    ),
                    key=lambda feature: feature["importance"],
                    reverse=True,
                ),
            },
            "model_evolution": [dict(row) for row in models],
            "batch_training": {
                "enabled": False,
                "minimum_samples": 130,
                "note": (
                    "La recolección está activa. El reentrenamiento automático y "
                    "la promoción walk-forward aún requieren un ciclo de revisión."
                ),
            },
        }


def _win_rates(groups: dict[str, list[sqlite3.Row]]) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "trades": len(trades),
            "wins": sum(trade["outcome"] == "win" for trade in trades),
            "win_rate": sum(trade["outcome"] == "win" for trade in trades) / len(trades),
            "pnl_pct": sum(float(trade["pnl_pct"]) for trade in trades),
        }
        for name, trades in sorted(groups.items())
    }


def _opposite(direction: str) -> str:
    if direction == "CALL":
        return "PUT"
    if direction == "PUT":
        return "CALL"
    raise ValueError("La dirección debe ser CALL o PUT.")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _directional_return(label: str, outcome: str, payout_pct: float | None) -> float:
    magnitude = payout_pct if outcome == "win" and payout_pct is not None else 100.0
    return magnitude if label == "CALL" else -magnitude
