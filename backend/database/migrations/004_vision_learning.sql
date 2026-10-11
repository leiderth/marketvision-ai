CREATE TABLE IF NOT EXISTS vision_learning_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vision_model_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    brier_score REAL,
    baseline_brier_score REAL,
    decision TEXT NOT NULL CHECK (
        decision IN ('candidate', 'promoted', 'rejected', 'rollback')
    ),
    details_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vision_outcome_watch (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    decision_id TEXT NOT NULL UNIQUE,
    armed_at TEXT NOT NULL,
    expected_at TEXT NOT NULL,
    last_outcome TEXT,
    last_text TEXT,
    last_observed_at TEXT,
    stable_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS vision_outcome_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT NOT NULL,
    previous_outcome TEXT,
    new_outcome TEXT NOT NULL CHECK (new_outcome IN ('win', 'loss')),
    source TEXT NOT NULL CHECK (source IN ('manual', 'screen_ocr', 'manual_correction')),
    changed_at TEXT NOT NULL,
    note TEXT
);
