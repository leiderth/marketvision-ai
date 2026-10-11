"""Read-only analytical candidate endpoint; it never authorizes execution."""
from analysis.signal_engine import SignalEngine
from fastapi import APIRouter, Request
from ml.model_registry import MLModelManager
from pydantic import BaseModel, Field

router = APIRouter()
_signal_engine = SignalEngine()


class FeatureContribution(BaseModel):
    feature: str
    direction: str
    weight: float
    contribution: float
    description: str
    value: object


class ExplainabilityResponse(BaseModel):
    method: str
    baseline_score: float
    score: float
    contributions: list[FeatureContribution] = Field(default_factory=list)
    shap_available: bool
    shap_note: str
    summary: str
    ml_model: dict[str, object] | None = None


class SignalResponse(BaseModel):
    signal: str = "NO_TRADE"
    confidence: float = 0.0
    market_regime: str | None = None
    risk_level: str | None = None
    should_trade: bool = False
    reasons: list[str] = Field(default_factory=list)
    as_of: str | None = None
    explainability: ExplainabilityResponse


@router.get("/latest", response_model=SignalResponse)
async def get_latest_signal(request: Request) -> dict:
    snapshot = request.app.state.market_data_manager.get_snapshot()
    return generate_signal_from_snapshot(snapshot, request.app.state.ml_model_manager)


def generate_signal_from_snapshot(
    snapshot: dict,
    model_manager: MLModelManager,
) -> dict:
    candles = snapshot.get("candles", [])
    latest_close = candles[-1].get("close") if candles else None
    values = snapshot.get("indicators", {}).get("values", {})
    model_prediction = model_manager.predict(
        values,
        latest_close,
        as_of=snapshot.get("indicators", {}).get("as_of"),
    )
    return _signal_engine.generate(snapshot, ml_prediction=model_prediction)
