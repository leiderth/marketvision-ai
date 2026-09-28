"""Shared, versioned dataset and registry contract for local ML workflows."""

FEATURE_NAMES = (
    "sma_gap_atr",
    "ema_gap_atr",
    "macd_histogram_atr",
    "rsi_centered",
    "price_vwap_atr",
)
MIN_TRAINING_ROWS = 100
MIN_BALANCED_ACCURACY = 0.55


class ModelRegistryError(ValueError):
    """Raised when model data is incomplete, invalid, or not review-approved."""
