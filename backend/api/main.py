"""
Punto de entrada de la API de MarketVision AI.

Expone REST + WebSocket. La app arranca siempre en modo ANALYSIS_MODE,
sin importar lo que diga la configuración, hasta que el usuario lo cambie
explícitamente desde la UI (ver onboarding obligatorio, sección 2 de la spec).
"""
import asyncio
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import structlog
from numpy.linalg import LinAlgError
from analysis.economic_calendar import EconomicCalendarManager
from analysis.market_regime import MarketRegimeDetector
from api.routes import learning, market_data, paper, risk, signals, status, vision
from config import OperationMode, settings
from execution.base import OrderRequest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from market_data.manager import MarketDataManager
from market_data.quality_gate import DataQualityGate
from ml.continuous_learning import ContinuousLearningManager
from ml.model_registry import MLModelManager
from ml.vision_learning import VisionLearningManager
from ml.vision_series_model import VisionSeriesModel
from observability import routes as observability
from paper_trading.autopilot import PaperAutopilot
from paper_trading.paper_execution import PaperExecutionManager
from risk.risk_manager import RiskManager
from vision.decision_journal import VisionDecisionJournal
from vision.ohlc_store import VisionCandleStore
from vision.replay import VisionReplayStore
from vision.screen_capture import ScreenVisionService
from vision.series_labeler import VisionSeriesLabeler

logger = structlog.get_logger(__name__)


async def _run_paper_autopilot(application: FastAPI) -> None:
    try:
        while True:
            application.state.paper_autopilot.process_once(
                mode=application.state.current_mode,
                market_snapshot=application.state.market_data_manager.get_snapshot(),
            )
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        application.state.paper_autopilot.stop(reason="autopilot_error")
        logger.exception("paper_autopilot_stopped_after_error", error_type=type(exc).__name__)
        raise
    finally:
        application.state.paper_autopilot.stop(reason="application_shutdown")


async def _run_screen_vision(application: FastAPI) -> None:
    next_training_check = 0.0
    while True:
        try:
            state = await asyncio.to_thread(
                application.state.screen_vision.capture_and_analyze
            )
            series_labels = await asyncio.to_thread(
                application.state.vision_series_labeler.observe_completed_candles,
                application.state.vision_candle_store,
            )
            if series_labels:
                logger.info(
                    "vision_pixel_series_outcomes_labeled",
                    count=len(series_labels),
                    labels=series_labels,
                )
            ocr_fields = (state.get("ocr") or {}).get("fields", {})
            outcome = await asyncio.to_thread(
                application.state.vision_decisions.observe_auto_outcome,
                ocr_fields.get("outcome", {}),
                payout_pct=ocr_fields.get("payout", {}).get("percent"),
            )
            if outcome:
                await asyncio.to_thread(
                    application.state.vision_learning_manager.on_outcome,
                    outcome["decision_id"],
                )
                logger.info("vision_outcome_labeled_from_screen", **outcome)
            now = asyncio.get_running_loop().time()
            if now >= next_training_check:
                next_training_check = now + 60
                result = await asyncio.to_thread(
                    application.state.vision_learning_manager.maybe_retrain
                )
                if result.get("trained"):
                    logger.info("vision_batch_training_evaluated", **result)
                visual_model_result = await asyncio.to_thread(
                    application.state.vision_series_model.maybe_train
                )
                if visual_model_result.get("trained"):
                    logger.info(
                        "vision_series_model_evaluated",
                        **visual_model_result,
                    )
        except asyncio.CancelledError:
            raise
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            sqlite3.Error,
            LinAlgError,
        ) as exc:
            logger.exception(
                "screen_vision_learning_cycle_failed",
                error_type=type(exc).__name__,
            )
        await asyncio.sleep(1 / settings.vision_capture_fps)


def _build_market_data_manager() -> MarketDataManager:
    return MarketDataManager(
        provider=None,
        quality_gate=DataQualityGate(settings.market_data_max_price_change_pct),
        regime_detector=MarketRegimeDetector(
            extreme_atr_pct=settings.market_regime_extreme_atr_pct,
            max_spread_pct=settings.market_regime_max_spread_pct,
        ),
        economic_calendar=EconomicCalendarManager(),
        setup_error="El feed externo está desactivado; selecciona un área de captura visual.",
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    tasks = [
        asyncio.create_task(_run_screen_vision(application), name="screen-vision"),
    ]

    def report_failure(completed_task: asyncio.Task[None]) -> None:
        if completed_task.cancelled():
            return
        error = completed_task.exception()
        if error is not None:
            logger.error(
                "background_task_stopped",
                task_name=completed_task.get_name(),
                error_type=type(error).__name__,
            )

    for task in tasks:
        task.add_done_callback(report_failure)

    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task

app = FastAPI(
    title="MarketVision AI API",
    version="0.1.0",
    description="Copiloto de análisis de mercado asistido por IA (análisis y paper trading).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "tauri://localhost",
        "http://tauri.localhost",
        "http://localhost:1420",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.current_mode = OperationMode.ANALYSIS
app.state.vision_capture_fps = settings.vision_capture_fps
app.state.market_data_manager = _build_market_data_manager()
app.state.ml_model_manager = MLModelManager(
    settings.ml_model_registry_dir,
    settings.ml_active_model_version,
    min_confidence=settings.ml_min_confidence,
    drift_state_dir=settings.ml_drift_state_dir,
)
app.state.risk_manager = RiskManager(state_path=settings.risk_state_path)
app.state.continuous_learning_manager = ContinuousLearningManager(
    settings.trade_database_path,
    settings.trade_dataset_path,
)
app.state.paper_execution_manager = PaperExecutionManager(app.state.risk_manager)
for persisted_trade in app.state.continuous_learning_manager.get_open_trades():
    app.state.paper_execution_manager.restore_open_position(
        persisted_trade["order_id"],
        OrderRequest(**persisted_trade["order"]),
        persisted_trade["opened_at"],
    )
app.state.paper_autopilot = PaperAutopilot(
    app.state.paper_execution_manager,
    app.state.risk_manager,
    app.state.continuous_learning_manager,
    lambda snapshot: signals.generate_signal_from_snapshot(
        snapshot,
        app.state.ml_model_manager,
    ),
)
vision_config_path = settings.trade_database_path
app.state.vision_candle_store = VisionCandleStore(settings.trade_database_path)
app.state.vision_series_labeler = VisionSeriesLabeler(settings.trade_database_path)
app.state.vision_series_model = VisionSeriesModel(
    settings.trade_database_path,
    Path(settings.ml_model_registry_dir).with_name("vision_series_registry"),
)
app.state.screen_vision = ScreenVisionService(
    Path(vision_config_path).with_name("screen_vision.json"),
    candle_store=app.state.vision_candle_store,
)
app.state.vision_replay_store = VisionReplayStore(
    Path(settings.trade_dataset_path).with_name("vision_replay"),
)
app.state.vision_decisions = VisionDecisionJournal(settings.trade_database_path)
app.state.vision_learning_manager = VisionLearningManager(
    settings.trade_database_path,
    Path(settings.trade_dataset_path).with_name("vision_training.csv"),
    Path(settings.ml_model_registry_dir).with_name("vision_registry"),
)
app.state.vision_edge_margin_pct = 5.0
app.state.vision_minimum_margin_pct = 15.0

app.include_router(status.router, prefix="/api/status", tags=["status"])
app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(risk.router, prefix="/api/risk", tags=["risk"])
app.include_router(market_data.router, prefix="/api/market-data", tags=["market-data"])
app.include_router(paper.router, prefix="/api/paper", tags=["paper-trading"])
app.include_router(learning.router, prefix="/api/learning", tags=["learning"])
app.include_router(observability.router, prefix="/api/observability", tags=["observability"])
app.include_router(vision.router, prefix="/api/vision", tags=["screen-vision"])


@app.get("/")
async def root():
    return {
        "app": "MarketVision AI",
        "mode": app.state.current_mode,
        "note": "Visión de pantalla local. Clic real desactivado hasta validación explícita.",
    }
