"""Estado del sistema: modo actual, salud del feed de datos, versión de modelo activo."""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/")
async def get_status(request: Request):
    feed_status = request.app.state.market_data_manager.get_status()
    model_status = request.app.state.ml_model_manager.get_status()
    return {
        "mode": request.app.state.current_mode,
        "data_feed": {
            **feed_status,
            "note": (
                "Estado de feed y calidad disponibles."
                if feed_status["quality_accepted"]
                else "Feed no disponible o no actualizado; NO_TRADE."
            ),
        },
        "active_model_version": model_status["active_model_version"],
        "ml_model": model_status,
        "risk": request.app.state.risk_manager.get_state(),
        "risk_limits": request.app.state.risk_manager.get_limits(),
    }
