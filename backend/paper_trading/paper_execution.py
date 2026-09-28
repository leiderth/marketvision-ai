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
        self._closed_positions: list[dict] = []

    def submit_order(self, order: OrderRequest) -> OrderResult:
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
        return OrderResult(order_id=order_id, status="SIMULATED")

    def cancel_order(self, order_id: str) -> bool:
        return self._open_positions.pop(order_id, None) is not None

    def get_open_positions(self) -> list[dict]:
        return [{"order_id": oid, **vars(o)} for oid, o in self._open_positions.items()]

    def get_closed_positions(self) -> list[dict]:
        return list(self._closed_positions)

    def settle_order(self, order_id: str, pnl_pct: float) -> dict:
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
        self._risk_manager.record_trade_result(pnl_pct)
        settled = {
            "order_id": order_id,
            **vars(order),
            "pnl_pct": float(pnl_pct),
            "settled_at": datetime.now(timezone.utc).isoformat(),
            "status": "SETTLED",
        }
        del self._open_positions[order_id]
        self._closed_positions.append(settled)
        return settled

    def kill_switch(self) -> None:
        self._open_positions.clear()
        self._risk_manager.lock_trading(reason="Kill switch activado por el usuario.")
