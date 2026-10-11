CREATE TABLE IF NOT EXISTS vision_pixel_candles (
    asset TEXT NOT NULL,
    timeframe_seconds INTEGER NOT NULL CHECK (timeframe_seconds > 0),
    candle_open_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    candle_index INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    price_basis TEXT NOT NULL CHECK (price_basis = 'normalized_chart_pixels'),
    chart_width INTEGER NOT NULL,
    chart_height INTEGER NOT NULL,
    read_quality_score INTEGER NOT NULL,
    source TEXT NOT NULL CHECK (source = 'local_screen_capture'),
    PRIMARY KEY (asset, timeframe_seconds, candle_open_at)
);

CREATE INDEX IF NOT EXISTS idx_vision_pixel_candles_context
    ON vision_pixel_candles(asset, timeframe_seconds, candle_open_at);
