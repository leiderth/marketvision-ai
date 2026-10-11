"""Read-only continuous-learning dashboard backed by closed paper trades."""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/dashboard")
async def get_learning_dashboard(request: Request) -> dict:
    return request.app.state.continuous_learning_manager.get_dashboard()
