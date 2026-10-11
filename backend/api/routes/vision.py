"""Local screenshot-vision state and demo-outcome calibration endpoints."""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, StrictInt

from vision.desktop_input import move_cursor_to
from vision.replay import VisionReplayError
from vision.screen_capture import ScreenVisionError

logger = logging.getLogger(__name__)
router = APIRouter()


def _calibration_for_state(request: Request, state: dict) -> dict:
    captured_at = state.get("captured_at")
    hour_utc = None
    if isinstance(captured_at, str):
        try:
            timestamp = datetime.fromisoformat(captured_at)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            hour_utc = timestamp.astimezone(timezone.utc).hour
        except ValueError:
            logger.warning("vision_capture_timestamp_invalid")
    payout = (state.get("ocr") or {}).get("fields", {}).get("payout", {}).get("percent")
    configured_edge = request.app.state.vision_edge_margin_pct
    edge_margin = request.app.state.vision_learning_manager.get_context_edge_margin(
        asset=state.get("asset"),
        timeframe_seconds=state.get("timeframe_seconds"),
        market_regime=state.get("market_regime"),
        hour_utc=hour_utc,
        configured_margin_pct=configured_edge,
    )
    confidence_threshold = (
        request.app.state.vision_learning_manager.get_confidence_threshold_pct()
    )
    calibration = request.app.state.vision_decisions.get_calibration(
        state.get("directional_scores", {}).get("CALL"),
        state.get("directional_scores", {}).get("PUT"),
        asset=state.get("asset"),
        timeframe_seconds=state.get("timeframe_seconds"),
        market_regime=state.get("market_regime"),
        hour_utc=hour_utc,
        payout_pct=payout,
        read_quality_score=state.get("read_quality_score", 0),
        edge_margin_pct=edge_margin,
        minimum_margin_pct=request.app.state.vision_minimum_margin_pct,
        minimum_probability_pct=confidence_threshold,
    )
    calibration["configured_edge_margin_pct"] = configured_edge
    calibration["adaptive_edge_margin_pct"] = edge_margin
    return calibration


class ScreenRect(BaseModel):
    left: int
    top: int
    width: int = Field(ge=12, le=1600)
    height: int = Field(ge=10, le=500)


class RegionCalibrationRequest(BaseModel):
    left: int
    top: int
    width: int = Field(ge=320, le=8192)
    height: int = Field(ge=160, le=8192)
    asset: str = Field(min_length=1, max_length=80)
    timeframe_seconds: int = Field(ge=1, le=86_400)
    ocr_regions: dict[Literal["asset", "timer", "payout", "amount", "outcome"], "ScreenRect"]
    click_call: tuple[int, int] | None = None
    click_put: tuple[int, int] | None = None


class ButtonCalibrationRequest(BaseModel):
    click_call: tuple[StrictInt, StrictInt]
    click_put: tuple[StrictInt, StrictInt]


class ButtonProbeRequest(BaseModel):
    side: Literal["CALL", "PUT"]


class DecisionOutcomeRequest(BaseModel):
    outcome: Literal["win", "loss"]
    payout_pct: float | None = Field(default=None, ge=0, le=100)
    duration_seconds: int | None = Field(default=None, ge=1, le=86_400)


class OutcomeWatchRequest(BaseModel):
    duration_seconds: int = Field(ge=1, le=86_400)


class CorrectOutcomeRequest(BaseModel):
    outcome: Literal["win", "loss"]
    note: str = Field(min_length=1, max_length=500)


class ThresholdRequest(BaseModel):
    edge_margin_pct: float = Field(ge=0, le=25)
    minimum_margin_pct: float = Field(ge=0, le=49)


class CaptureStateRequest(BaseModel):
    paused: bool


class TrainVisionSeriesModelRequest(BaseModel):
    confirmation: str


@router.get("/state")
async def get_vision_state(request: Request) -> dict:
    state = request.app.state.screen_vision.get_state()
    state["capture_fps_target"] = request.app.state.vision_capture_fps
    calibration = _calibration_for_state(request, state)
    if not state.get("readable"):
        calibration["decision"] = "NO_TRADE"
        calibration["reason"] = (
            state.get("reason") or "La calidad de lectura es insuficiente; NO OPERAR."
        )
    state["calibration_status"] = calibration
    state["live_execution"] = {
        "enabled": False,
        "available": False,
        "reason": (
            "Clic real bloqueado: no hay identificación verificada de la ventana objetivo, "
            "detector fiable de operación abierta, confirmación visual posterior al clic "
            "ni validación histórica suficiente."
        ),
    }
    state["auto_outcome_watch"] = (
        request.app.state.vision_decisions.get_auto_outcome_watch()
    )
    return state


@router.get("/desktop-preview")
async def get_desktop_preview(request: Request) -> dict:
    try:
        return await asyncio.to_thread(
            request.app.state.screen_vision.capture_desktop_preview
        )
    except (OSError, ScreenVisionError) as exc:
        logger.exception("desktop_preview_failed")
        raise HTTPException(
            status_code=503,
            detail="No se pudo capturar el escritorio local.",
        ) from exc


@router.get("/detect-chart")
async def detect_chart(request: Request) -> dict:
    try:
        return await asyncio.to_thread(
            request.app.state.screen_vision.detect_chart_candidates
        )
    except (OSError, ScreenVisionError, ValueError) as exc:
        logger.exception("chart_detection_failed")
        raise HTTPException(
            status_code=503,
            detail="No se pudo analizar la captura para localizar posibles gráficos.",
        ) from exc


@router.get("/candles")
async def get_vision_candles(
    request: Request,
    asset: str | None = None,
    timeframe_seconds: int | None = Query(default=None, ge=1, le=86_400),
    limit: int = Query(default=200, ge=1, le=5000),
) -> dict:
    try:
        candles = await asyncio.to_thread(
            request.app.state.vision_candle_store.list_candles,
            asset=asset,
            timeframe_seconds=timeframe_seconds,
            limit=limit,
        )
        return {
            "source": "local_screen_capture",
            "price_basis": "normalized_chart_pixels",
            "candles": candles,
        }
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/chart-preview")
async def get_chart_preview(request: Request) -> dict:
    preview = request.app.state.screen_vision.get_chart_preview()
    if preview is None:
        raise HTTPException(status_code=404, detail="Todavía no hay captura del área calibrada.")
    return {"image": preview}

@router.put("/capture-state")
async def set_capture_state(body: CaptureStateRequest, request: Request) -> dict:
    state = request.app.state.screen_vision.set_paused(body.paused)
    return {"paused": body.paused, "state": state}


@router.get("/calibration")
async def get_vision_calibration(request: Request) -> dict:
    return request.app.state.screen_vision.get_calibration()


@router.post("/calibration")
async def calibrate_vision_region(body: RegionCalibrationRequest, request: Request) -> dict:
    try:
        return await asyncio.to_thread(
            request.app.state.screen_vision.calibrate_region,
            left=body.left,
            top=body.top,
            width=body.width,
            height=body.height,
            asset=body.asset,
            timeframe_seconds=body.timeframe_seconds,
            ocr_regions={
                name: region.model_dump()
                for name, region in body.ocr_regions.items()
            },
            click_call=body.click_call,
            click_put=body.click_put,
        )
    except (OSError, ScreenVisionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/calibration/buttons")
async def calibrate_vision_buttons(
    body: ButtonCalibrationRequest,
    request: Request,
) -> dict:
    try:
        result = await asyncio.to_thread(
            request.app.state.screen_vision.calibrate_buttons,
            click_call=body.click_call,
            click_put=body.click_put,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, ScreenVisionError) as exc:
        logger.exception("vision_button_calibration_failed")
        raise HTTPException(
            status_code=503,
            detail="No se pudo validar el escritorio para calibrar los botones.",
        ) from exc
    logger.info("vision_button_calibration_saved", extra=result)
    return result


@router.post("/calibration/buttons/probe")
async def probe_vision_button(body: ButtonProbeRequest, request: Request) -> dict:
    try:
        x, y = await asyncio.to_thread(
            request.app.state.screen_vision.get_button_position,
            body.side,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, ScreenVisionError) as exc:
        logger.exception("vision_button_probe_validation_failed")
        raise HTTPException(
            status_code=503,
            detail="No se pudo validar la calibración de pantalla.",
        ) from exc
    try:
        await asyncio.to_thread(move_cursor_to, x, y)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.exception(
            "vision_button_cursor_probe_failed",
            extra={"side": body.side, "x": x, "y": y},
        )
        raise HTTPException(
            status_code=503,
            detail=f"No se pudo mover el cursor a la posición calibrada: {exc}",
        ) from exc
    logger.info(
        "vision_button_cursor_probe",
        extra={"side": body.side, "x": x, "y": y, "clicked": False},
    )
    return {
        "side": body.side,
        "positioned": True,
        "clicked": False,
        "x": x,
        "y": y,
        "message": "Cursor movido; no se hizo clic.",
    }


@router.post("/replay/captures")
async def save_vision_replay_capture(request: Request) -> dict:
    state = request.app.state.screen_vision.get_state()
    image = request.app.state.screen_vision.get_chart_preview()
    try:
        capture = await asyncio.to_thread(
            request.app.state.vision_replay_store.save_capture,
            image,
            state,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, VisionReplayError) as exc:
        logger.exception("vision_replay_capture_save_failed")
        raise HTTPException(
            status_code=503,
            detail="No se pudo guardar la captura local de replay.",
        ) from exc
    logger.info("vision_replay_capture_saved", extra=capture)
    return capture


@router.get("/replay/captures")
async def list_vision_replay_captures(request: Request) -> dict:
    try:
        captures = await asyncio.to_thread(
            request.app.state.vision_replay_store.list_captures
        )
    except (OSError, VisionReplayError) as exc:
        logger.exception("vision_replay_capture_list_failed")
        raise HTTPException(
            status_code=503,
            detail="No se pudo leer el índice local de replay.",
        ) from exc
    return {"captures": captures}


@router.post("/replay/captures/{capture_id}/run")
async def run_vision_replay_capture(capture_id: str, request: Request) -> dict:
    try:
        return await asyncio.to_thread(
            request.app.state.vision_replay_store.replay_capture,
            capture_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, VisionReplayError) as exc:
        logger.exception("vision_replay_capture_failed")
        raise HTTPException(
            status_code=503,
            detail="No se pudo reproducir la captura local.",
        ) from exc


@router.put("/thresholds")
async def set_thresholds(body: ThresholdRequest, request: Request) -> dict:
    request.app.state.vision_edge_margin_pct = body.edge_margin_pct
    request.app.state.vision_minimum_margin_pct = body.minimum_margin_pct
    return {
        "edge_margin_pct": request.app.state.vision_edge_margin_pct,
        "minimum_margin_pct": request.app.state.vision_minimum_margin_pct,
    }


@router.post("/decisions")
async def record_vision_decision(request: Request) -> dict:
    state = request.app.state.screen_vision.get_state()
    state["calibration_status"] = _calibration_for_state(request, state)
    try:
        result = request.app.state.vision_decisions.record_decision(state)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if result.get("duplicate"):
        result["series_labeling"] = {
            "watching": False,
            "label_source": "pixel_series",
            "reason": "La muestra de esta ventana ya se había registrado.",
        }
    else:
        try:
            result["series_labeling"] = await asyncio.to_thread(
                request.app.state.vision_series_labeler.watch_decision,
                result["decision_id"],
                state,
            )
        except (OSError, ValueError, sqlite3.Error) as exc:
            logger.exception(
                "vision_pixel_series_watch_registration_failed",
                extra={"decision_id": result.get("decision_id")},
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "El candidato se registró, pero no se pudo preparar la etiqueta "
                    "visual de la siguiente vela."
                ),
            ) from exc
    logger.info("vision_decision_recorded", extra=result)
    return result


@router.get("/series/dataset-status")
async def get_vision_series_dataset_status(request: Request) -> dict:
    try:
        return await asyncio.to_thread(
            request.app.state.vision_series_labeler.get_dataset_status
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        logger.exception("vision_series_dataset_status_failed")
        raise HTTPException(
            status_code=503,
            detail="No se pudo leer el estado del dataset visual local.",
        ) from exc


@router.get("/series/model")
async def get_vision_series_model(request: Request) -> dict:
    try:
        return await asyncio.to_thread(
            request.app.state.vision_series_model.get_status,
            request.app.state.screen_vision.get_state(),
        )
    except (OSError, ValueError, sqlite3.Error, RuntimeError) as exc:
        logger.exception("vision_series_model_status_failed")
        raise HTTPException(
            status_code=503,
            detail="No se pudo leer el estado del modelo visual fuera de muestra.",
        ) from exc


@router.post("/series/model/train")
async def train_vision_series_model(
    body: TrainVisionSeriesModelRequest,
    request: Request,
) -> dict:
    if body.confirmation != "EVALUATE_VISUAL_SERIES_MODEL":
        raise HTTPException(
            status_code=400,
            detail="Confirmación para evaluar el modelo visual incorrecta.",
        )
    try:
        return await asyncio.to_thread(
            request.app.state.vision_series_model.train,
            force=True,
        )
    except (OSError, ValueError, sqlite3.Error, RuntimeError) as exc:
        logger.exception("vision_series_model_training_failed")
        raise HTTPException(
            status_code=503,
            detail="No se pudo completar la evaluación temporal del modelo visual.",
        ) from exc


@router.get("/series/samples")
async def get_vision_series_samples(
    request: Request,
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    try:
        samples = await asyncio.to_thread(
            request.app.state.vision_series_labeler.get_samples,
            limit=limit,
        )
        return {
            "source": "pixel_series",
            "price_basis": "normalized_chart_pixels",
            "model_training_enabled": False,
            "samples": samples,
        }
    except (OSError, ValueError, sqlite3.Error) as exc:
        logger.exception("vision_series_samples_read_failed")
        raise HTTPException(
            status_code=503,
            detail="No se pudieron leer las muestras de velas visuales.",
        ) from exc


@router.get("/decisions")
async def get_vision_decisions(request: Request) -> dict:
    state = request.app.state.screen_vision.get_state()
    return {
        "items": request.app.state.vision_decisions.get_recent(),
        "calibration": _calibration_for_state(request, state),
        "auto_outcome_watch": request.app.state.vision_decisions.get_auto_outcome_watch(),
    }


@router.post("/decisions/{decision_id}/outcome")
async def record_vision_outcome(
    decision_id: str,
    body: DecisionOutcomeRequest,
    request: Request,
) -> dict:
    try:
        calibration = request.app.state.vision_decisions.record_outcome(
            decision_id,
            body.outcome,
            payout_pct=body.payout_pct,
            duration_seconds=body.duration_seconds,
        )
        request.app.state.vision_learning_manager.on_outcome(decision_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info(
        "vision_demo_outcome_recorded",
        extra={"decision_id": decision_id, "outcome": body.outcome},
    )
    return calibration


@router.post("/decisions/{decision_id}/watch-outcome")
async def watch_demo_outcome(
    decision_id: str,
    body: OutcomeWatchRequest,
    request: Request,
) -> dict:
    state = request.app.state.screen_vision.get_state()
    outcome_field = (state.get("ocr") or {}).get("fields", {}).get("outcome", {})
    try:
        return request.app.state.vision_decisions.arm_auto_outcome(
            decision_id,
            body.duration_seconds,
            outcome_field,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/decisions/{decision_id}/watch-outcome")
async def cancel_demo_outcome_watch(decision_id: str, request: Request) -> dict:
    try:
        cancelled = request.app.state.vision_decisions.cancel_auto_outcome_watch(
            decision_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"cancelled": cancelled}


@router.put("/decisions/{decision_id}/outcome")
async def correct_vision_outcome(
    decision_id: str,
    body: CorrectOutcomeRequest,
    request: Request,
) -> dict:
    try:
        result = request.app.state.vision_decisions.correct_outcome(
            decision_id,
            body.outcome,
            note=body.note,
        )
        request.app.state.vision_learning_manager.rebuild_online_state()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


@router.get("/learning")
async def get_vision_learning(request: Request) -> dict:
    return request.app.state.vision_learning_manager.get_dashboard()
