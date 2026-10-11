"""
PaperExecutionManager — implementación simulada de ExecutionManagerBase.

Este es el motor de ejecución que SÍ queda operativo en este repositorio.
No mueve dinero real bajo ninguna circunstancia. Ver backend/execution/README.md
para la razón por la que no existe un equivalente de ejecución real.
"""
import math
import uuid
from datetime import datetime, timezone
from numbers import Real

from execution.base import ExecutionManagerBase, OrderRequest, OrderResult
from risk.risk_manager import RiskManager


class PaperExecutionManager(ExecutionManagerBase):
    def __init__(self, risk_manager: RiskManager) -> None:
        self._risk_manager = risk_manager
        self._open_positions: dict[str, OrderRequest] = {}
        self._opened_at: dict[str, datetime] = {}
        self._closed_positions: list[dict] = []

    def submit_order(self, order: OrderRequest) -> OrderResult:
        if self._open_positions:
            return OrderResult(
                order_id="",
                status="REJECTED_OPEN_POSITION",
                detail="Debe cerrarse la operación paper abierta antes de enviar otra.",
            )
        if (
            not isinstance(order.asset, str)
            or not order.asset.strip()
            or order.direction not in {"CALL", "PUT"}
            or isinstance(order.signal_confidence, bool)
            or not isinstance(order.signal_confidence, Real)
            or not math.isfinite(order.signal_confidence)
            or order.signal_confidence < 0
            or order.signal_confidence > 1
            or isinstance(order.expiration_seconds, bool)
            or not isinstance(order.expiration_seconds, int)
            or order.expiration_seconds <= 0
        ):
            return OrderResult(
                order_id="",
                status="REJECTED_INVALID_ORDER",
                detail="Activo, dirección, consenso o expiración inválidos.",
            )
        allowed, reason = self._risk_manager.register_trade(order.amount)
        if not allowed:
            return OrderResult(
                order_id="",
                status="REJECTED_BY_RISK_MANAGER",
                detail=reason,
            )
        order_id = str(uuid.uuid4())
        self._open_positions[order_id] = order
        self._opened_at[order_id] = datetime.now(timezone.utc)
        return OrderResult(order_id=order_id, status="SIMULATED")

    def cancel_order(self, order_id: str) -> bool:
        cancelled = self._open_positions.pop(order_id, None) is not None
        self._opened_at.pop(order_id, None)
        return cancelled

    def restore_open_position(
        self,
        order_id: str,
        order: OrderRequest,
        opened_at: str,
    ) -> None:
        restored_at = datetime.fromisoformat(opened_at)
        if (
            not order_id
            or order_id in self._open_positions
            or restored_at.tzinfo is None
            or restored_at.utcoffset() is None
            or order.direction not in {"CALL", "PUT"}
            or order.expiration_seconds <= 0
            or order.amount <= 0
        ):
            raise ValueError("La operación paper persistida no es válida para restaurar.")
        self._open_positions[order_id] = order
        self._opened_at[order_id] = restored_at.astimezone(timezone.utc)

    def get_open_positions(self) -> list[dict]:
        return [
            {
                "order_id": order_id,
                **vars(order),
                "opened_at": self._opened_at[order_id].isoformat(),
            }
            for order_id, order in self._open_positions.items()
        ]

    def get_closed_positions(self) -> list[dict]:
        return list(self._closed_positions)

    def settle_order(
        self,
        order_id: str,
        pnl_pct: float,
        *,
        payout_pct: float | None = None,
    ) -> dict:
        if (
            isinstance(pnl_pct, bool)
            or not isinstance(pnl_pct, Real)
            or not math.isfinite(pnl_pct)
        ):
            raise ValueError("El PnL paper debe ser un porcentaje finito.")
        order = self._open_positions.get(order_id)
        if order is None:
            raise KeyError("Orden paper abierta no encontrada.")
        maximum_loss_pct = (
            order.amount
            / self._risk_manager.get_limits()["paper_account_balance"]
            * 100
        )
        if pnl_pct < -maximum_loss_pct:
            raise ValueError("El PnL excede el importe máximo arriesgado en la orden paper.")
        if payout_pct is not None and (
            isinstance(payout_pct, bool)
            or not isinstance(payout_pct, Real)
            or not math.isfinite(payout_pct)
            or payout_pct < 0
            or payout_pct > 1000
        ):
            raise ValueError("El payout paper debe ser un porcentaje entre 0 y 1000.")
        self._risk_manager.record_trade_result(pnl_pct)
        settled_at = datetime.now(timezone.utc)
        opened_at = self._opened_at[order_id]
        settled = {
            "order_id": order_id,
            **vars(order),
            "pnl_pct": float(pnl_pct),
            "payout_pct": float(payout_pct) if payout_pct is not None else None,
            "settled_at": settled_at.isoformat(),
            "opened_at": opened_at.isoformat(),
            "duration_seconds": max(0.0, (settled_at - opened_at).total_seconds()),
            "outcome": "win" if pnl_pct > 0 else "loss",
            "status": "SETTLED",
        }
        del self._open_positions[order_id]
        del self._opened_at[order_id]
        self._closed_positions.append(settled)
        return settled

    def kill_switch(self) -> None:
        self._open_positions.clear()
        self._opened_at.clear()
        self._risk_manager.lock_trading(reason="Kill switch activado por el usuario.")
