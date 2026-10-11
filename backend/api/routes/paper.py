"""Manual order management for simulation only; no broker integration."""
import logging
import sqlite3

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
    payout_pct: float | None = Field(default=None, ge=0, le=1000)
    confirmation: str


class PaperAutopilotRequest(BaseModel):
    enabled: bool
    amount: float = Field(gt=0)
    payout_pct: float = Field(gt=0, le=1000)
    confirmation: str


@router.get("/orders")
async def get_paper_orders(request: Request):
    manager = request.app.state.paper_execution_manager
    return {
        "open": manager.get_open_positions(),
        "closed": manager.get_closed_positions(),
    }


@router.get("/autopilot")
async def get_paper_autopilot(request: Request):
    return request.app.state.paper_autopilot.get_status()


@router.post("/autopilot")
async def configure_paper_autopilot(body: PaperAutopilotRequest, request: Request):
    _require_local(request)
    if body.confirmation != "PAPER_AUTOPILOT_ONLY":
        raise HTTPException(status_code=400, detail="Confirmación de autopilot paper incorrecta.")
    if body.enabled and request.app.state.current_mode != "PAPER_TRADING_MODE":
        raise HTTPException(
            status_code=409,
            detail="Activa explícitamente PAPER_TRADING_MODE antes del autopilot.",
        )
    try:
        result = request.app.state.paper_autopilot.configure(
            enabled=body.enabled,
            amount=body.amount,
            payout_pct=body.payout_pct,
        )
        if body.enabled:
            request.app.state.paper_autopilot.prime(
                request.app.state.market_data_manager.get_snapshot()
            )
        return result
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
        timeframe=f"{candles[-1]['interval_seconds']}s",
        indicator_snapshot={
            "values": snapshot.get("indicators", {}).get("values", {}),
            "latest_close": candles[-1]["close"],
        },
        market_regime=snapshot.get("market_regime", {}).get("regime"),
        model_version=request.app.state.ml_model_manager.get_status()["active_model_version"],
    )
    result = request.app.state.paper_execution_manager.submit_order(order)
    if result.status != "SIMULATED":
        raise HTTPException(
            status_code=409,
            detail={"message": result.detail, "status": result.status},
        )
    opened_at = next(
        position["opened_at"]
        for position in request.app.state.paper_execution_manager.get_open_positions()
        if position["order_id"] == result.order_id
    )
    try:
        request.app.state.continuous_learning_manager.record_open_trade(
            result.order_id,
            vars(order),
            opened_at,
        )
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        request.app.state.paper_execution_manager.cancel_order(result.order_id)
        logger.exception(
            "paper_trade_journal_open_failed",
            extra={"order_id": result.order_id},
        )
        raise HTTPException(
            status_code=500,
            detail="No se pudo guardar la operación paper; revisa el log local.",
        ) from exc
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
        settled = request.app.state.paper_execution_manager.settle_order(
            order_id,
            body.pnl_pct,
            payout_pct=body.payout_pct,
        )
        try:
            settled["learning"] = (
                request.app.state.continuous_learning_manager.record_closed_trade(settled)
            )
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            logger.exception(
                "paper_trade_learning_persistence_failed",
                extra={"order_id": order_id},
            )
            raise HTTPException(
                status_code=500,
                detail="La operación se liquidó en paper, pero no se guardó su muestra; revisa el log.",
            ) from exc
        return settled
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
