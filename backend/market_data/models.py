"""Tipos comunes para datos de mercado recibidos de adaptadores autorizados."""
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MarketDataCandle:
    symbol: str
    timestamp: datetime
    interval_seconds: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class DataQualityResult:
    accepted: bool
    status: str
    checked_at: datetime
    reasons: tuple[str, ...]
    candles_checked: int

