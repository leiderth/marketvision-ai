"""Opt-in paper-only signal execution; this module never sends OS/browser input."""
from __future__ import annotations

import logging
import math
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from execution.base import OrderRequest
from ml.continuous_learning import ContinuousLearningManager
from risk.risk_manager import RiskManager

from paper_trading.paper_execution import PaperExecutionManager

logger = logging.getLogger(__name__)


class PaperAutopilot:
    def __init__(
        self,
        execution_manager: PaperExecutionManager,
        risk_manager: RiskManager,
        learning_manager: ContinuousLearningManager,
        signal_generator: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._execution_manager = execution_manager
        self._risk_manager = risk_manager
        self._learning_manager = learning_manager
        self._signal_generator = signal_generator
        self._enabled = False
        self._amount = 0.0
        self._payout_pct = 0.0
        self._last_candle_key: tuple[str, str] | None = None

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "mode": "PAPER_ONLY",
            "amount": self._amount if self._enabled else None,
            "payout_pct": self._payout_pct if self._enabled else None,
            "max_open_positions": 1,
            "last_candle": self._last_candle_key[1] if self._last_candle_key else None,
        }

    def configure(self, *, enabled: bool, amount: float, payout_pct: float) -> dict[str, Any]:
        if not isinstance(enabled, bool):
            raise TypeError("El estado del autopilot debe ser booleano.")
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(amount)
            or amount <= 0
        ):
            raise ValueError("El importe paper debe ser positivo y finito.")
        if (
            isinstance(payout_pct, bool)
            or not isinstance(payout_pct, (int, float))
            or not math.isfinite(payout_pct)
            or not 0 <= payout_pct <= 1000
        ):
            raise ValueError("El payout paper debe estar entre 0 y 1000.")
        if enabled:
            allowed, reason = self._risk_manager.authorize_trade(float(amount))
            if not allowed:
                raise PermissionError(reason or "RiskManager bloqueó la operación.")
            positions = self._execution_manager.get_open_positions()
            if positions:
                raise PermissionError("Cierra las operaciones paper abiertas antes de activar el autopilot.")
        self._enabled = enabled
        self._amount = float(amount)
        self._payout_pct = float(payout_pct)
        self._last_candle_key = None
        logger.info(
            "paper_autopilot_configured",
            extra={
                "enabled": enabled,
                "amount": self._amount,
                "payout_pct": self._payout_pct,
                "mode": "PAPER_ONLY",
            },
        )
        return self.get_status()

    def prime(self, market_snapshot: dict[str, Any]) -> None:
        candles = market_snapshot.get("candles", [])
        if not candles:
            self._last_candle_key = None
            return
        latest = candles[-1]
        timestamp = _parse_timestamp(latest.get("timestamp"))
        self._last_candle_key = (
            (str(latest["symbol"]), timestamp.isoformat())
            if timestamp is not None
            else None
        )

    def stop(self, *, reason: str) -> None:
        was_enabled = self._enabled
        self._enabled = False
        self._last_candle_key = None
        if was_enabled:
            logger.warning(
                "paper_autopilot_stopped",
                extra={"reason": reason, "mode": "PAPER_ONLY"},
            )

    def process_once(self, *, mode: str, market_snapshot: dict[str, Any]) -> None:
        if not self._enabled:
            return
        if mode != "PAPER_TRADING_MODE":
            self.stop(reason="paper_mode_disabled")
            return
        candles = market_snapshot.get("candles", [])
        status = market_snapshot.get("status", {})
        if not candles or not status.get("connected") or not status.get("fresh") or not status.get(
            "quality_accepted"
        ):
            return

        latest = candles[-1]
        timestamp = _parse_timestamp(latest.get("timestamp"))
        if timestamp is None:
            return

        positions = self._execution_manager.get_open_positions()
        self._settle_due_positions(positions, candles, timestamp)
        candle_key = (str(latest["symbol"]), timestamp.isoformat())
        if candle_key == self._last_candle_key:
            return
        self._last_candle_key = candle_key

        signal = self._signal_generator(market_snapshot)
        direction = signal.get("signal")
        decision = {
            "candle": timestamp.isoformat(),
            "asset": latest["symbol"],
            "signal": direction,
            "confidence": signal.get("confidence"),
            "reasons": signal.get("reasons", []),
            "paper_only": True,
        }
        current_positions = self._execution_manager.get_open_positions()
        if not signal.get("should_trade") or direction not in {"CALL", "PUT"}:
            logger.info("paper_autopilot_decision", extra={**decision, "action": "NO_TRADE"})
            return
        if current_positions:
            logger.info(
                "paper_autopilot_decision",
                extra={**decision, "action": "BLOCKED_OPEN_POSITION"},
            )
            return
        if not self._risk_manager.is_trading_allowed():
            logger.warning(
                "paper_autopilot_decision",
                extra={
                    **decision,
                    "action": "BLOCKED_RISK",
                    "risk": self._risk_manager.get_state(),
                },
            )
            self.stop(reason="risk_manager_blocked")
            return

        indicator_snapshot = market_snapshot.get("indicators", {})
        indicator_values = indicator_snapshot.get("values", {})
        explainability = signal.get("explainability")
        model_explanation = (
            explainability.get("ml_model")
            if isinstance(explainability, dict)
            else None
        )
        model_version = (
            model_explanation.get("version")
            if isinstance(model_explanation, dict)
            else None
        )
        order = OrderRequest(
            asset=latest["symbol"],
            direction=direction,
            amount=self._amount,
            expiration_seconds=int(latest["interval_seconds"]),
            signal_confidence=float(signal["confidence"]),
            reasons=list(signal.get("reasons", [])),
            timeframe=f"{latest['interval_seconds']}s",
            indicator_snapshot={
                "values": indicator_values,
                "latest_close": float(latest["close"]),
            },
            market_regime=market_snapshot.get("market_regime", {}).get("regime"),
            strategy="paper-autopilot",
            model_version=model_version,
        )
        result = self._execution_manager.submit_order(order)
        if result.status != "SIMULATED":
            logger.warning(
                "paper_autopilot_decision",
                extra={**decision, "action": result.status, "detail": result.detail},
            )
            if result.status == "REJECTED_BY_RISK_MANAGER":
                self.stop(reason="risk_manager_rejected_order")
            return

        opened_at = next(
            position["opened_at"]
            for position in self._execution_manager.get_open_positions()
            if position["order_id"] == result.order_id
        )
        try:
            self._learning_manager.record_open_trade(
                result.order_id,
                vars(order),
                opened_at,
            )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            self._execution_manager.cancel_order(result.order_id)
            raise
        logger.info(
            "paper_autopilot_decision",
            extra={**decision, "action": "SIMULATED", "order_id": result.order_id},
        )

    def _settle_due_positions(
        self,
        positions: list[dict[str, Any]],
        candles: list[dict[str, Any]],
        latest_timestamp: datetime,
    ) -> None:
        for position in positions:
            if position.get("strategy") != "paper-autopilot":
                continue
            opened_at = _parse_timestamp(position.get("opened_at"))
            if opened_at is None:
                raise ValueError("Orden autopilot paper con timestamp de apertura inválido.")
            expiration = opened_at + timedelta(seconds=position["expiration_seconds"])
            expiry_candle = next(
                (
                    candle
                    for candle in candles
                    if (candle_time := _parse_timestamp(candle.get("timestamp")))
                    and candle_time >= expiration
                ),
                None,
            )
            if expiry_candle is None or latest_timestamp < expiration:
                continue
            entry_price = float(position["indicator_snapshot"]["latest_close"])
            expiry_price = float(expiry_candle["open"])
            if not math.isfinite(entry_price) or not math.isfinite(expiry_price) or entry_price <= 0:
                raise ValueError("Precios inválidos al liquidar la operación paper automática.")
            won = (
                expiry_price > entry_price
                if position["direction"] == "CALL"
                else expiry_price < entry_price
            )
            balance = float(self._risk_manager.get_limits()["paper_account_balance"])
            pnl_pct = (
                float(position["amount"]) / balance * self._payout_pct
                if won
                else -float(position["amount"]) / balance * 100
            )
            settled = self._execution_manager.settle_order(
                position["order_id"],
                pnl_pct,
                payout_pct=self._payout_pct,
            )
            settled["settled_at"] = expiration.isoformat()
            settled["duration_seconds"] = float(position["expiration_seconds"])
            self._learning_manager.record_closed_trade(settled)
            logger.info(
                "paper_autopilot_settled",
                extra={
                    "order_id": position["order_id"],
                    "outcome": "win" if won else "loss",
                    "entry_price": entry_price,
                    "expiry_price": expiry_price,
                    "pnl_pct": pnl_pct,
                    "paper_only": True,
                },
            )


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return None
    return timestamp.astimezone(timezone.utc)
