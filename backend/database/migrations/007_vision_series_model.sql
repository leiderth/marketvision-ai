CREATE TABLE IF NOT EXISTS vision_series_model_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vision_series_model_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    validation_brier REAL,
    brier_score REAL,
    baseline_brier REAL,
    log_loss REAL,
    calibration_error REAL,
    decision TEXT NOT NULL CHECK (
        decision IN ('promoted', 'rejected', 'rollback')
    ),
    details_json TEXT NOT NULL
);
