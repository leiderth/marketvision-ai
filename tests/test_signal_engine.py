import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from analysis.signal_engine import SignalEngine
from api.main import app
from fastapi.testclient import TestClient


def ready_snapshot(*, bearish: bool = False) -> dict:
    sign = -1 if bearish else 1
    return {
        "status": {
            "connected": True,
            "fresh": True,
            "quality_accepted": True,
        },
        "indicators": {
            "ready": True,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "reasons": [],
            "values": {
                "sma_20": 101 if not bearish else 99,
                "sma_50": 100,
                "ema_20": 101 if not bearish else 99,
                "ema_50": 100,
                "macd": {
                    "line": 0.1 * sign,
                    "signal": 0.05 * sign,
                    "histogram": 0.05 * sign,
                },
                "atr_14": 1.0,
                "rsi_14": 60 if not bearish else 40,
                "price_action": {
                    "structure": (
                        "higher_highs_and_lows" if not bearish else "lower_highs_and_lows"
                    )
                },
                "vwap_session": 99 if not bearish else 101,
            },
        },
        "market_regime": {
            "ready": True,
            "trade_blocked": False,
            "regime": "trending_up" if not bearish else "trending_down",
            "reasons": [],
        },
        "candles": [{"close": 100}],
    }


def test_signal_engine_emits_direction_only_with_aligned_multi_factor_evidence():
    result = SignalEngine().generate(ready_snapshot())

    assert result["signal"] == "CALL"
    assert result["should_trade"] is True
    assert result["confidence"] == 1.0
    assert result["explainability"]["method"] == "deterministic_additive_attribution"
    assert result["explainability"]["shap_available"] is False
    assert len(result["explainability"]["contributions"]) == 5
    assert sum(item["contribution"] for item in result["explainability"]["contributions"]) == (
        result["explainability"]["score"]
    )


def test_signal_engine_supports_put_when_all_evidence_aligns_down():
    result = SignalEngine().generate(ready_snapshot(bearish=True))

    assert result["signal"] == "PUT"
    assert result["should_trade"] is True


def test_active_ml_model_must_agree_with_deterministic_candidate():
    result = SignalEngine().generate(
        ready_snapshot(),
        ml_prediction={
            "configured": True,
            "available": True,
            "prediction": "PUT",
            "class_score_call": 0.1,
            "class_score_put": 0.9,
            "confident": True,
            "minimum_confidence": 0.65,
            "version": "reviewed-version",
            "explainability": {
                "method": "linear_interventional_shap",
                "note": "Atribuciones lineales en log-odds.",
                "contributions": [],
            },
        },
    )

    assert result["signal"] == "NO_TRADE"
    assert result["should_trade"] is False
    assert "El modelo ML contradice el candidato determinista; NO_TRADE." in result["reasons"]
    assert result["explainability"]["shap_available"] is True


def test_signal_engine_blocks_opposing_evidence():
    snapshot = ready_snapshot()
    snapshot["indicators"]["values"]["rsi_14"] = 40

    result = SignalEngine().generate(snapshot)

    assert result["signal"] == "NO_TRADE"
    assert result["should_trade"] is False
    assert any("contradictoria" in reason for reason in result["reasons"])


def test_signal_engine_fails_closed_when_regime_gate_is_blocked():
    snapshot = ready_snapshot()
    snapshot["market_regime"]["trade_blocked"] = True
    snapshot["market_regime"]["ready"] = False
    snapshot["market_regime"]["reasons"] = ["Calendario económico no disponible."]

    result = SignalEngine().generate(snapshot)

    assert result["signal"] == "NO_TRADE"
    assert result["should_trade"] is False
    assert "Calendario económico no disponible." in result["reasons"]
    assert result["explainability"]["contributions"]


def test_signal_engine_blocks_missing_and_non_finite_features():
    snapshot = ready_snapshot()
    snapshot["indicators"]["values"]["rsi_14"] = float("nan")
    snapshot["indicators"]["values"]["macd"] = {"histogram": float("inf")}

    result = SignalEngine().generate(snapshot)

    assert result["signal"] == "NO_TRADE"
    assert result["should_trade"] is False
    assert any("ausente o inválido" in reason for reason in result["reasons"])


def test_latest_signal_api_returns_explanation_and_keeps_unavailable_market_no_trade():
    with TestClient(app) as client:
        response = client.get("/api/signals/latest")

    assert response.status_code == 200
    body = response.json()
    assert body["signal"] == "NO_TRADE"
    assert body["should_trade"] is False
    assert body["explainability"]["shap_available"] is False
    assert body["as_of"] is None
