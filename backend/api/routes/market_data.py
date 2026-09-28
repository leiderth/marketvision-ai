"""Flujo WebSocket de velas aprobadas por el DataQualityGate."""
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.get("/regime")
async def get_market_regime(request: Request) -> dict:
    return request.app.state.market_data_manager.get_snapshot()["market_regime"]


@router.get("/calendar")
async def get_economic_calendar(request: Request) -> dict:
    return request.app.state.market_data_manager.get_snapshot()["economic_calendar"]


@router.websocket("/ws")
async def stream_market_data(websocket: WebSocket) -> None:
    await websocket.accept()
    manager = websocket.app.state.market_data_manager
    queue = manager.subscribe()
    try:
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(queue)
