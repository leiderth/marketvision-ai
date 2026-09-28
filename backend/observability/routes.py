"""Read-only health and Prometheus-compatible operational gauges."""
from datetime import datetime, timezone

from fastapi import APIRouter, Request, Response

router = APIRouter()


@router.get("/health")
async def get_health(request: Request) -> dict:
    market = request.app.state.market_data_manager.get_snapshot()
    ml = request.app.state.ml_model_manager.get_status()
    risk = request.app.state.risk_manager.get_state()
    components = {
        "market_feed": (
            "ok"
            if market["status"]["connected"] and market["status"]["fresh"]
            else "degraded"
        ),
        "data_quality": "ok" if market["status"]["quality_accepted"] else "blocked",
        "economic_calendar": (
            "ok" if market["economic_calendar"]["available"] else "blocked"
        ),
        "risk_manager": "blocked" if risk["trading_locked"] else "ok",
        "ml_model": "ok" if ml["model_loaded"] else "inactive",
    }
    return {
        "status": (
            "ok"
            if components["market_feed"] == "ok"
            and components["data_quality"] == "ok"
            and components["economic_calendar"] == "ok"
            and components["risk_manager"] == "ok"
            else "degraded"
        ),
        "as_of": datetime.now(timezone.utc).isoformat(),
        "mode": request.app.state.current_mode,
        "components": components,
    }


@router.get("/metrics", response_class=Response)
async def get_metrics(request: Request) -> Response:
    snapshot = request.app.state.market_data_manager.get_snapshot()
    ml = request.app.state.ml_model_manager.get_status()
    risk = request.app.state.risk_manager.get_state()
    values = {
        "marketvision_feed_connected": int(snapshot["status"]["connected"]),
        "marketvision_feed_fresh": int(snapshot["status"]["fresh"]),
        "marketvision_data_quality_accepted": int(snapshot["status"]["quality_accepted"]),
        "marketvision_calendar_available": int(snapshot["economic_calendar"]["available"]),
        "marketvision_model_loaded": int(ml["model_loaded"]),
        "marketvision_risk_locked": int(risk["trading_locked"]),
        "marketvision_paper_mode": int(
            request.app.state.current_mode == "PAPER_TRADING_MODE"
        ),
        "marketvision_open_paper_orders": len(
            request.app.state.paper_execution_manager.get_open_positions()
        ),
    }
    body = "\n".join(
        f"# TYPE {name} gauge\n{name} {value}"
        for name, value in values.items()
    ) + "\n"
    return Response(content=body, media_type="text/plain; version=0.0.4")
