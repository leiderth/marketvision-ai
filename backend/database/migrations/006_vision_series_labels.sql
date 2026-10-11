CREATE TABLE IF NOT EXISTS vision_series_outcome_watch (
    decision_id TEXT PRIMARY KEY,
    asset TEXT NOT NULL,
    timeframe_seconds INTEGER NOT NULL,
    target_candle_open_at TEXT NOT NULL,
    prediction_direction TEXT NOT NULL CHECK (
        prediction_direction IN ('CALL', 'PUT')
    ),
    market_regime TEXT NOT NULL,
    features_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_observed_direction TEXT CHECK (
        last_observed_direction IN ('CALL', 'PUT', 'FLAT') OR
        last_observed_direction IS NULL
    ),
    last_observed_at TEXT,
    stable_since TEXT,
    stable_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'labeled', 'expired')
    ),
    completion_reason TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'labeled', 'expired')
    ),
    completion_reason TEXT
);

CREATE TABLE IF NOT EXISTS vision_series_samples (
    decision_id TEXT PRIMARY KEY,
    asset TEXT NOT NULL,
    timeframe_seconds INTEGER NOT NULL,
    market_regime TEXT NOT NULL,
    prediction_direction TEXT NOT NULL CHECK (
        prediction_direction IN ('CALL', 'PUT')
    ),
    target_direction TEXT NOT NULL CHECK (
        target_direction IN ('CALL', 'PUT', 'FLAT')
    ),
    target INTEGER CHECK (target IN (0, 1) OR target IS NULL),
    target_candle_open_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    labeled_at TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source = 'pixel_series'),
    price_basis TEXT NOT NULL CHECK (
        price_basis = 'normalized_chart_pixels'
    ),
    features_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vision_series_samples_context
    ON vision_series_samples(asset, timeframe_seconds, labeled_at);
