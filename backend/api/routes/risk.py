"""Risk state, explicit paper-mode selection, kill switch and manual unlock."""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


@router.get("/limits")
async def get_limits(request: Request):
    return request.app.state.risk_manager.get_limits()


@router.get("/state")
async def get_state(request: Request):
    return request.app.state.risk_manager.get_state()


class PaperModeRequest(BaseModel):
    enabled: bool
    confirmation: str


class UnlockRequest(BaseModel):
    confirmation: str


def _require_local(request: Request) -> None:
    client = request.client
    if client is None:
        raise HTTPException(status_code=403, detail="Se requiere una conexión local.")
    try:
        import ipaddress

        if not ipaddress.ip_address(client.host).is_loopback:
            raise HTTPException(status_code=403, detail="Operación disponible solo en localhost.")
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Dirección cliente inválida.") from exc


@router.post("/paper-mode")
async def set_paper_mode(body: PaperModeRequest, request: Request):
    _require_local(request)
    if body.confirmation != "PAPER_TRADING_ONLY":
        raise HTTPException(status_code=400, detail="Confirmación de paper trading incorrecta.")
    if body.enabled and not request.app.state.risk_manager.is_trading_allowed():
        raise HTTPException(
            status_code=409,
            detail="RiskManager mantiene el bloqueo. Requiere desbloqueo explícito por separado.",
        )
    if not body.enabled:
        request.app.state.paper_execution_manager.kill_switch()
    request.app.state.current_mode = (
        "PAPER_TRADING_MODE" if body.enabled else "ANALYSIS_MODE"
    )
    return {
        "mode": request.app.state.current_mode,
        "paper_only": True,
        "risk_state": request.app.state.risk_manager.get_state(),
    }


@router.post("/kill-switch")
async def activate_kill_switch(request: Request):
    _require_local(request)
    request.app.state.paper_execution_manager.kill_switch()
    request.app.state.current_mode = "ANALYSIS_MODE"
    return {
        "mode": request.app.state.current_mode,
        "risk_state": request.app.state.risk_manager.get_state(),
        "open_orders_cancelled": True,
    }


@router.post("/unlock")
async def unlock_trading(body: UnlockRequest, request: Request):
    _require_local(request)
    if body.confirmation != "UNLOCK_PAPER_TRADING":
        raise HTTPException(status_code=400, detail="Confirmación de desbloqueo incorrecta.")
    request.app.state.risk_manager.unlock_trading()
    return request.app.state.risk_manager.get_state()
