"""Local persistence for OHLC values reconstructed from chart pixels."""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


class VisionCandleStore:
    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS vision_pixel_candles (
                    asset TEXT NOT NULL,
                    timeframe_seconds INTEGER NOT NULL CHECK (timeframe_seconds > 0),
                    candle_open_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    candle_index INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    price_basis TEXT NOT NULL CHECK (
                        price_basis = 'normalized_chart_pixels'
                    ),
                    chart_width INTEGER NOT NULL,
                    chart_height INTEGER NOT NULL,
                    read_quality_score INTEGER NOT NULL,
                    source TEXT NOT NULL CHECK (source = 'local_screen_capture'),
                    PRIMARY KEY (asset, timeframe_seconds, candle_open_at)
                );
                CREATE INDEX IF NOT EXISTS idx_vision_pixel_candles_context
                    ON vision_pixel_candles(asset, timeframe_seconds, candle_open_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def record_frame(
        self,
        *,
        asset: str,
        timeframe_seconds: int,
        candles: list[Mapping[str, Any]],
        observed_at: str,
        timer_seconds_remaining: int,
        chart_width: int,
        chart_height: int,
        read_quality_score: int,
    ) -> list[dict[str, Any]]:
        if (
            not asset.strip()
            or isinstance(timeframe_seconds, bool)
            or timeframe_seconds < 1
            or isinstance(timer_seconds_remaining, bool)
            or timer_seconds_remaining < 0
            or chart_width < 1
            or chart_height < 1
        ):
            raise ValueError("El contexto OHLC reconstruido no es válido.")
        captured = datetime.fromisoformat(observed_at)
        if captured.tzinfo is None:
            raise ValueError("La captura OHLC debe tener una marca temporal con zona horaria.")
        captured = captured.astimezone(timezone.utc)
        timeframe = timedelta(seconds=timeframe_seconds)
        estimated_close_timestamp = captured.timestamp() + timer_seconds_remaining
        next_close_timestamp = (
            math.ceil(estimated_close_timestamp / timeframe_seconds) * timeframe_seconds
        )
        last_open_at = datetime.fromtimestamp(
            next_close_timestamp - timeframe_seconds,
            tz=timezone.utc,
        )

        rows: list[dict[str, Any]] = []
        candle_count = len(candles)
        for index, candle in enumerate(candles):
            values = _normalized_ohlc(candle)
            candle_open_at = last_open_at - timeframe * (candle_count - index - 1)
            rows.append(
                {
                    "asset": asset.strip(),
                    "timeframe_seconds": timeframe_seconds,
                    "candle_open_at": candle_open_at.isoformat(),
                    "observed_at": captured.isoformat(),
                    "candle_index": index,
                    **values,
                    "price_basis": "normalized_chart_pixels",
                    "chart_width": chart_width,
                    "chart_height": chart_height,
                    "read_quality_score": read_quality_score,
                    "source": "local_screen_capture",
                }
            )

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO vision_pixel_candles (
                    asset, timeframe_seconds, candle_open_at, observed_at,
                    candle_index, open, high, low, close, price_basis,
                    chart_width, chart_height, read_quality_score, source
                ) VALUES (
                    :asset, :timeframe_seconds, :candle_open_at, :observed_at,
                    :candle_index, :open, :high, :low, :close, :price_basis,
                    :chart_width, :chart_height, :read_quality_score, :source
                )
                ON CONFLICT(asset, timeframe_seconds, candle_open_at) DO UPDATE SET
                    observed_at = excluded.observed_at,
                    candle_index = excluded.candle_index,
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    chart_width = excluded.chart_width,
                    chart_height = excluded.chart_height,
                    read_quality_score = excluded.read_quality_score
                """,
                rows,
            )
        return rows

    def list_candles(
        self,
        *,
        asset: str | None = None,
        timeframe_seconds: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 5000:
            raise ValueError("El límite de consulta debe estar entre 1 y 5000.")
        clauses: list[str] = []
        parameters: list[Any] = []
        if asset:
            clauses.append("asset = ?")
            parameters.append(asset)
        if timeframe_seconds is not None:
            if isinstance(timeframe_seconds, bool) or timeframe_seconds < 1:
                raise ValueError("El timeframe consultado no es válido.")
            clauses.append("timeframe_seconds = ?")
            parameters.append(timeframe_seconds)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT asset, timeframe_seconds, candle_open_at, observed_at,
                       candle_index, open, high, low, close, price_basis,
                       chart_width, chart_height, read_quality_score, source
                FROM vision_pixel_candles
                {where}
                ORDER BY candle_open_at DESC
                LIMIT ?
                """,
                (*parameters, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def count(self, *, asset: str | None = None) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM vision_pixel_candles"
                + (" WHERE asset = ?" if asset else ""),
                (asset,) if asset else (),
            ).fetchone()
        return int(row["count"])

    def get_completed_candle(
        self,
        *,
        asset: str,
        timeframe_seconds: int,
        candle_open_at: str,
        minimum_quality: int,
    ) -> dict[str, Any] | None:
        target_open = datetime.fromisoformat(candle_open_at)
        if target_open.tzinfo is None:
            raise ValueError("La vela destino debe tener una marca temporal con zona horaria.")
        target_open = target_open.astimezone(timezone.utc)
        end_of_candle = target_open + timedelta(seconds=timeframe_seconds)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT target.*,
                       (
                           SELECT MAX(observed_at)
                           FROM vision_pixel_candles AS latest
                           WHERE latest.asset = target.asset
                             AND latest.timeframe_seconds = target.timeframe_seconds
                       ) AS latest_observed_at,
                       (
                           SELECT MAX(candle_open_at)
                           FROM vision_pixel_candles AS latest
                           WHERE latest.asset = target.asset
                             AND latest.timeframe_seconds = target.timeframe_seconds
                       ) AS latest_candle_open_at,
                       (
                           SELECT read_quality_score
                           FROM vision_pixel_candles AS latest
                           WHERE latest.asset = target.asset
                             AND latest.timeframe_seconds = target.timeframe_seconds
                           ORDER BY observed_at DESC
                           LIMIT 1
                       ) AS latest_quality,
                       (
                           SELECT chart_width
                           FROM vision_pixel_candles AS latest
                           WHERE latest.asset = target.asset
                             AND latest.timeframe_seconds = target.timeframe_seconds
                           ORDER BY observed_at DESC
                           LIMIT 1
                       ) AS latest_chart_width,
                       (
                           SELECT chart_height
                           FROM vision_pixel_candles AS latest
                           WHERE latest.asset = target.asset
                             AND latest.timeframe_seconds = target.timeframe_seconds
                           ORDER BY observed_at DESC
                           LIMIT 1
                       ) AS latest_chart_height
                FROM vision_pixel_candles AS target
                WHERE target.asset = ?
                  AND target.timeframe_seconds = ?
                  AND target.candle_open_at = ?
                  AND target.read_quality_score >= ?
                """,
                (
                    asset,
                    timeframe_seconds,
                    target_open.isoformat(),
                    minimum_quality,
                ),
            ).fetchone()
        if row is None or not row["latest_observed_at"] or not row["latest_candle_open_at"]:
            return None
        latest_observed_at = datetime.fromisoformat(row["latest_observed_at"])
        latest_candle_open_at = datetime.fromisoformat(row["latest_candle_open_at"])
        if latest_observed_at.tzinfo is None:
            latest_observed_at = latest_observed_at.replace(tzinfo=timezone.utc)
        if latest_candle_open_at.tzinfo is None:
            latest_candle_open_at = latest_candle_open_at.replace(tzinfo=timezone.utc)
        if (
            latest_observed_at.astimezone(timezone.utc) < end_of_candle
            or latest_candle_open_at.astimezone(timezone.utc) <= target_open
            or row["latest_quality"] is None
            or row["latest_quality"] < minimum_quality
            or row["chart_width"] != row["latest_chart_width"]
            or row["chart_height"] != row["latest_chart_height"]
            or latest_observed_at.astimezone(timezone.utc) - end_of_candle
            > timedelta(seconds=max(5, timeframe_seconds * 2))
        ):
            return None
        return dict(row)


def _normalized_ohlc(candle: Mapping[str, Any]) -> dict[str, float]:
    try:
        top = float(candle["body_top"])
        bottom = float(candle["body_bottom"])
        wick_top = float(candle["wick_top"])
        wick_bottom = float(candle["wick_bottom"])
        direction = candle["direction"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("La vela visual carece de geometría OHLC válida.") from exc
    if (
        not all(math.isfinite(value) and 0 <= value <= 1 for value in (top, bottom, wick_top, wick_bottom))
        or not wick_top <= top <= bottom <= wick_bottom
        or direction not in {"CALL", "PUT"}
    ):
        raise ValueError("La geometría OHLC de la vela visual no es válida.")

    body_high = 1 - top
    body_low = 1 - bottom
    return {
        "open": body_low if direction == "CALL" else body_high,
        "high": 1 - wick_top,
        "low": 1 - wick_bottom,
        "close": body_high if direction == "CALL" else body_low,
    }
