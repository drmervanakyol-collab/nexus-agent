-- migrations/001_initial.sql
-- Nexus Agent — initial schema
-- Idempotent: every CREATE uses IF NOT EXISTS.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

-- ---------------------------------------------------------------------------
-- Schema version tracking
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- ---------------------------------------------------------------------------
-- tasks
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT    PRIMARY KEY,
    goal        TEXT    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','running','success','failed','aborted')),
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    metadata    TEXT             -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);

-- ---------------------------------------------------------------------------
-- actions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS actions (
    id          TEXT    PRIMARY KEY,
    task_id     TEXT    NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
    type        TEXT    NOT NULL,
    payload     TEXT             NOT NULL DEFAULT '{}',  -- JSON
    status      TEXT    NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','success','failed')),
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_actions_task_id ON actions (task_id);

-- ---------------------------------------------------------------------------
-- cost_ledger
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cost_ledger (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT    NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
    provider    TEXT    NOT NULL,
    tokens      INTEGER NOT NULL DEFAULT 0,
    cost_usd    REAL    NOT NULL DEFAULT 0.0,
    recorded_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_cost_ledger_task_id ON cost_ledger (task_id);

-- ---------------------------------------------------------------------------
-- memory_fingerprints
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_fingerprints (
    id              TEXT    PRIMARY KEY,
    task_id         TEXT    REFERENCES tasks (id) ON DELETE SET NULL,
    fingerprint     TEXT    NOT NULL UNIQUE,
    label           TEXT    NOT NULL DEFAULT '',
    confidence      REAL    NOT NULL DEFAULT 1.0,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- ---------------------------------------------------------------------------
-- corrections
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT    REFERENCES tasks (id) ON DELETE SET NULL,
    action_id       TEXT    REFERENCES actions (id) ON DELETE SET NULL,
    original        TEXT    NOT NULL,  -- JSON
    corrected       TEXT    NOT NULL,  -- JSON
    source          TEXT    NOT NULL DEFAULT 'human'
                            CHECK (source IN ('human','policy','automated')),
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- ---------------------------------------------------------------------------
-- user_consent
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_consent (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT    NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
    scope       TEXT    NOT NULL,
    granted     INTEGER NOT NULL DEFAULT 0 CHECK (granted IN (0, 1)),
    expires_at  TEXT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_user_consent_task_id ON user_consent (task_id);

-- ---------------------------------------------------------------------------
-- transport_audit
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transport_audit (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT    NOT NULL REFERENCES tasks (id) ON DELETE CASCADE,
    action_id           TEXT    REFERENCES actions (id) ON DELETE SET NULL,
    attempted_transport TEXT    NOT NULL
                                CHECK (attempted_transport IN ('uia','dom','file','mouse','keyboard')),
    fallback_used       INTEGER NOT NULL DEFAULT 0 CHECK (fallback_used IN (0, 1)),
    success             INTEGER NOT NULL DEFAULT 0 CHECK (success IN (0, 1)),
    latency_ms          REAL    NOT NULL DEFAULT 0.0,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_transport_audit_task_id  ON transport_audit (task_id);
CREATE INDEX IF NOT EXISTS idx_transport_audit_action_id ON transport_audit (action_id);

-- ---------------------------------------------------------------------------
-- Mark migration as applied (idempotent via INSERT OR IGNORE)
-- ---------------------------------------------------------------------------
INSERT OR IGNORE INTO schema_migrations (version) VALUES (1);
