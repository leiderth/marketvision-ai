"""
Configuración central de MarketVision AI.

Todas las claves y URLs sensibles se leen desde variables de entorno (.env),
nunca hardcodeadas. Ver .env.example en la raíz del proyecto.
"""
import os
from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OperationMode(str, Enum):
    ANALYSIS = "ANALYSIS_MODE"
    PAPER_TRADING = "PAPER_TRADING_MODE"
    # AUTONOMOUS_EXECUTION existe como valor del enum para que el resto del
    # sistema (Risk Manager, UI, logging) pueda razonar sobre él, pero no hay
    # ningún ExecutionManager concreto que lo implemente en este repositorio.
    # Ver backend/execution/README.md.
    AUTONOMOUS_EXECUTION = "AUTONOMOUS_EXECUTION_MODE"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.environ.get("MARKETVISION_ENV_FILE", "../.env"),
        env_file_encoding="utf-8",
    )

    app_env: str = "development"
    log_level: str = "info"
    default_mode: OperationMode = OperationMode.ANALYSIS

    database_url: str = "sqlite:///./marketvision_dev.db"

    market_data_provider: str | None = None
    market_data_api_key: str | None = None
    market_data_ws_url: str | None = None
    market_data_symbol: str | None = None
    market_data_interval: str = "1min"
    market_data_history_size: int = 500
    market_data_max_price_change_pct: float = 20.0
    market_regime_extreme_atr_pct: float = 1.0
    market_regime_max_spread_pct: float = 0.1

    economic_calendar_api_key: str | None = None
    economic_calendar_currencies: str = "AUD,CAD,CHF,EUR,GBP,JPY,NZD,USD"
    economic_calendar_refresh_minutes: int = Field(default=15, ge=1)
    economic_calendar_event_window_minutes: int = 30
    economic_calendar_max_age_hours: int = 6

    mlflow_tracking_uri: str = "./models/mlruns"
    ml_model_registry_dir: str = "./models/registry"
    ml_active_model_version: str | None = None
    ml_min_confidence: float = Field(default=0.65, gt=0.5, le=1.0)
    ml_drift_state_dir: str = "./data/model_drift"

    notifications_enabled: bool = True

    secret_key: str = "changeme"
    local_api_auth_token: str = "changeme"

    risk_max_loss_per_trade_pct: float = 1.0
    risk_max_daily_loss_pct: float = 5.0
    risk_max_consecutive_losses: int = 3
    risk_max_trades_per_day: int = 20
    risk_paper_account_balance: float = Field(default=10_000.0, gt=0)
    risk_state_path: str = "./data/risk_state.json"


settings = Settings()
