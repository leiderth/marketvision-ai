"""
Punto de entrada de la API de MarketVision AI.

Expone REST + WebSocket. La app arranca siempre en modo ANALYSIS_MODE,
sin importar lo que diga la configuración, hasta que el usuario lo cambie
explícitamente desde la UI (ver onboarding obligatorio, sección 2 de la spec).
"""
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

import structlog
from analysis.economic_calendar import EconomicCalendarManager
from analysis.market_regime import MarketRegimeDetector
from analysis.trading_economics_calendar import TradingEconomicsCalendarProvider
from api.routes import market_data, paper, risk, signals, status
from config import OperationMode, settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import HTTPError
from market_data.manager import MarketDataManager
from market_data.quality_gate import DataQualityGate
from market_data.twelve_data import TwelveDataProvider
from ml.model_registry import MLModelManager
from observability import routes as observability
from paper_trading.paper_execution import PaperExecutionManager
from risk.risk_manager import RiskManager
from websockets.exceptions import WebSocketException

logger = structlog.get_logger(__name__)


async def _run_market_data(manager: MarketDataManager) -> None:
    retry_delay = 5
    while True:
        try:
            await manager.run()
        except asyncio.CancelledError:
            raise
        except (
            HTTPError,
            WebSocketException,
            TimeoutError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
            OSError,
        ) as exc:
            logger.error(
                "market_data_reconnect_scheduled",
                error_type=type(exc).__name__,
                retry_delay_seconds=retry_delay,
            )
        else:
            logger.warning(
                "market_data_stream_ended",
                retry_delay_seconds=retry_delay,
            )
        await asyncio.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 60)


async def _run_economic_calendar(manager: MarketDataManager) -> None:
    retry_delay = 5
    while True:
        try:
            await manager.refresh_economic_calendar()
        except asyncio.CancelledError:
            raise
        except (HTTPError, TimeoutError, RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.error(
                "economic_calendar_refresh_failed",
                error_type=type(exc).__name__,
                retry_delay_seconds=retry_delay,
            )
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 300)
        else:
            retry_delay = 5
            await asyncio.sleep(settings.economic_calendar_refresh_minutes * 60)


def _build_market_data_manager() -> MarketDataManager:
    provider_name = (settings.market_data_provider or "").strip().casefold()
    provider = None
    setup_error = None

    if not provider_name:
        setup_error = "Configura MARKET_DATA_PROVIDER=twelve_data para iniciar el feed."
    elif provider_name != "twelve_data":
        setup_error = "Proveedor no compatible; configura MARKET_DATA_PROVIDER=twelve_data."
    elif not settings.market_data_api_key or not settings.market_data_api_key.strip():
        setup_error = "Configura MARKET_DATA_API_KEY en el entorno local."
    elif not settings.market_data_symbol or not settings.market_data_symbol.strip():
        setup_error = "Configura MARKET_DATA_SYMBOL en el entorno local."
    else:
        try:
            provider = TwelveDataProvider(
                api_key=settings.market_data_api_key,
                symbol=settings.market_data_symbol,
                interval=settings.market_data_interval,
                history_size=settings.market_data_history_size,
                websocket_url=(
                    settings.market_data_ws_url
                    or "wss://ws.twelvedata.com/v1/quotes/price"
                ),
            )
        except ValueError as exc:
            setup_error = str(exc)

    calendar_provider = None
    calendar_configuration_error = None
    if settings.economic_calendar_api_key and settings.economic_calendar_api_key.strip():
        try:
            currencies = [
                currency.strip()
                for currency in settings.economic_calendar_currencies.split(",")
                if currency.strip()
            ]
            calendar_provider = TradingEconomicsCalendarProvider(
                settings.economic_calendar_api_key,
                currencies=currencies,
            )
        except ValueError as exc:
            calendar_configuration_error = str(exc)
            logger.error("economic_calendar_configuration_invalid", error_type=type(exc).__name__)

    return MarketDataManager(
        provider=provider,
        quality_gate=DataQualityGate(settings.market_data_max_price_change_pct),
        regime_detector=MarketRegimeDetector(
            extreme_atr_pct=settings.market_regime_extreme_atr_pct,
            max_spread_pct=settings.market_regime_max_spread_pct,
        ),
        economic_calendar=EconomicCalendarManager(
            provider=calendar_provider,
            configuration_error=calendar_configuration_error,
            event_window_minutes=settings.economic_calendar_event_window_minutes,
            max_age_hours=settings.economic_calendar_max_age_hours,
        ),
        setup_error=setup_error,
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    manager = application.state.market_data_manager
    tasks = []
    if manager.has_provider:
        tasks.append(asyncio.create_task(_run_market_data(manager), name="market-data-provider"))
    if manager.has_calendar_provider:
        tasks.append(
            asyncio.create_task(
                _run_economic_calendar(manager),
                name="economic-calendar-provider",
            )
        )

    def report_failure(completed_task: asyncio.Task[None]) -> None:
        if completed_task.cancelled():
            return
        error = completed_task.exception()
        if error is not None:
            logger.error(
                "background_market_data_task_stopped",
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
    allow_origins=["tauri://localhost", "http://localhost:1420", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado de sesión en memoria (fase 1). En fases posteriores esto se mueve
# a DatabaseManager / un store persistente real.
app.state.current_mode = OperationMode.ANALYSIS
app.state.market_data_manager = _build_market_data_manager()
app.state.ml_model_manager = MLModelManager(
    settings.ml_model_registry_dir,
    settings.ml_active_model_version,
    min_confidence=settings.ml_min_confidence,
    drift_state_dir=settings.ml_drift_state_dir,
)
app.state.risk_manager = RiskManager(state_path=settings.risk_state_path)
app.state.paper_execution_manager = PaperExecutionManager(app.state.risk_manager)

app.include_router(status.router, prefix="/api/status", tags=["status"])
app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(risk.router, prefix="/api/risk", tags=["risk"])
app.include_router(market_data.router, prefix="/api/market-data", tags=["market-data"])
app.include_router(paper.router, prefix="/api/paper", tags=["paper-trading"])
app.include_router(observability.router, prefix="/api/observability", tags=["observability"])


@app.get("/")
async def root():
    return {
        "app": "MarketVision AI",
        "mode": app.state.current_mode,
        "note": "Análisis y paper trading únicamente; ejecución real no implementada.",
    }
