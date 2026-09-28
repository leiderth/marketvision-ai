"""Manual order management for simulation only; no broker integration."""
import logging

from api.routes.risk import _require_local
from api.routes.signals import get_latest_signal
from execution.base import OrderRequest
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


class PaperOrderRequest(BaseModel):
    amount: float = Field(gt=0)
    expiration_seconds: int = Field(ge=1, le=86_400)
    confirmation: str


class PaperSettlementRequest(BaseModel):
    pnl_pct: float
    confirmation: str


@router.get("/orders")
async def get_paper_orders(request: Request):
    manager = request.app.state.paper_execution_manager
    return {
        "open": manager.get_open_positions(),
        "closed": manager.get_closed_positions(),
    }


@router.post("/orders")
async def submit_paper_order(body: PaperOrderRequest, request: Request):
    _require_local(request)
    if request.app.state.current_mode != "PAPER_TRADING_MODE":
        raise HTTPException(status_code=409, detail="Activa explícitamente PAPER_TRADING_MODE.")
    if body.confirmation != "SIMULATE_ONLY":
        raise HTTPException(status_code=400, detail="Confirmación de simulación incorrecta.")

    signal = await get_latest_signal(request)
    if not signal["should_trade"] or signal["signal"] not in {"CALL", "PUT"}:
        raise HTTPException(
            status_code=409,
            detail={"message": "SignalEngine mantiene NO_TRADE.", "reasons": signal["reasons"]},
        )
    snapshot = request.app.state.market_data_manager.get_snapshot()
    candles = snapshot.get("candles", [])
    if not candles:
        raise HTTPException(status_code=409, detail="No hay precio para simular la orden.")
    order = OrderRequest(
        asset=candles[-1]["symbol"],
        direction=signal["signal"],
        amount=body.amount,
        expiration_seconds=body.expiration_seconds,
        signal_confidence=signal["confidence"],
        reasons=signal["reasons"],
    )
    result = request.app.state.paper_execution_manager.submit_order(order)
    if result.status != "SIMULATED":
        raise HTTPException(
            status_code=409,
            detail={"message": result.detail, "status": result.status},
        )
    logger.info("paper_order_simulated", order_id=result.order_id, asset=order.asset)
    return {
        "order_id": result.order_id,
        "status": result.status,
        "detail": "Orden simulada sin dinero real.",
        "order": vars(order),
    }


@router.post("/orders/{order_id}/settle")
async def settle_paper_order(
    order_id: str,
    body: PaperSettlementRequest,
    request: Request,
):
    _require_local(request)
    if request.app.state.current_mode != "PAPER_TRADING_MODE":
        raise HTTPException(status_code=409, detail="Solo se liquidan órdenes en modo paper.")
    if body.confirmation != "SIMULATE_ONLY":
        raise HTTPException(status_code=400, detail="Confirmación de simulación incorrecta.")
    try:
        return request.app.state.paper_execution_manager.settle_order(order_id, body.pnl_pct)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
