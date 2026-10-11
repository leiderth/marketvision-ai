CREATE TABLE IF NOT EXISTS trades (
    order_id TEXT PRIMARY KEY,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('CALL', 'PUT')),
    indicators_json TEXT NOT NULL,
    market_regime TEXT,
    confidence REAL NOT NULL,
    outcome TEXT CHECK (outcome IN ('win', 'loss') OR outcome IS NULL),
    payout_pct REAL,
    duration_seconds REAL,
    pnl_pct REAL,
    strategy TEXT NOT NULL,
    model_version TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades(closed_at);
CREATE INDEX IF NOT EXISTS idx_trades_asset_strategy ON trades(asset, strategy);

CREATE TABLE IF NOT EXISTS training_samples (
    order_id TEXT PRIMARY KEY REFERENCES trades(order_id),
    timestamp TEXT NOT NULL,
    label_end_timestamp TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('CALL', 'PUT')),
    features_json TEXT NOT NULL,
    realized_return_pct REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('candidate', 'promoted', 'rejected', 'rollback'))
);
