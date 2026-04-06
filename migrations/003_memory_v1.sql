-- migrations/003_memory_v1.sql
-- Nexus Agent — Memory V1: UI fingerprints and correction records
-- Idempotent: every CREATE uses IF NOT EXISTS.

-- ---------------------------------------------------------------------------
-- ui_fingerprints
-- Rich per-UI-context memory: transport preferences, strategy outcomes.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ui_fingerprints (
    id                   TEXT    PRIMARY KEY,
    app_name             TEXT    NOT NULL DEFAULT '',
    layout_hash          TEXT    NOT NULL,        -- hash of element positions/sizes
    element_signature    TEXT    NOT NULL,        -- hash of element types+roles (sorted)
    successful_strategies TEXT   NOT NULL DEFAULT '[]',  -- JSON list[str]
    failure_patterns     TEXT    NOT NULL DEFAULT '[]',  -- JSON list[str]
    last_seen            TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    seen_count           INTEGER NOT NULL DEFAULT 1,
    confidence_boost     REAL    NOT NULL DEFAULT 0.0,
    preferred_transport  TEXT                             -- NULL = not yet learned
);

CREATE INDEX IF NOT EXISTS idx_ui_fingerprints_layout
    ON ui_fingerprints (layout_hash);

CREATE INDEX IF NOT EXISTS idx_ui_fingerprints_last_seen
    ON ui_fingerprints (last_seen);

-- ---------------------------------------------------------------------------
-- correction_records
-- Per-context action corrections (wrong→correct) accumulated from HITL.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS correction_records (
    id                   TEXT    PRIMARY KEY,
    fingerprint_id       TEXT    REFERENCES ui_fingerprints (id) ON DELETE SET NULL,
    context_hash         TEXT    NOT NULL,        -- hash of the decision context
    wrong_action         TEXT    NOT NULL,        -- JSON (action_type + params)
    correct_action       TEXT    NOT NULL,        -- JSON (action_type + params)
    transport_correction TEXT,                    -- NULL = transport was correct
    apply_count          INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_correction_records_context
    ON correction_records (context_hash);

CREATE INDEX IF NOT EXISTS idx_correction_records_fingerprint
    ON correction_records (fingerprint_id);

-- Mark migration applied
INSERT OR IGNORE INTO schema_migrations (version) VALUES (3);
