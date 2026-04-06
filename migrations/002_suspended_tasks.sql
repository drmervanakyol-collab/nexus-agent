-- migrations/002_suspended_tasks.sql
-- Nexus Agent — suspended_tasks table
-- Idempotent: CREATE TABLE IF NOT EXISTS

-- ---------------------------------------------------------------------------
-- suspended_tasks
-- Tracks tasks that have been paused pending human review.
-- A row exists here IFF the task is currently suspended.
-- On resume() the row is deleted; on abort the row is also deleted.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS suspended_tasks (
    task_id      TEXT    PRIMARY KEY,
    reason       TEXT    NOT NULL,
    context      TEXT    NOT NULL DEFAULT '{}',   -- JSON blob (caller-supplied)
    fingerprint  TEXT,                            -- screen fingerprint at suspend time
    suspended_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- ---------------------------------------------------------------------------
-- hitl_log
-- Audit trail for every HITL request / response pair.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hitl_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT    NOT NULL,
    question    TEXT    NOT NULL,
    options     TEXT    NOT NULL DEFAULT '[]',  -- JSON array of strings
    chosen      TEXT,                           -- NULL if timed out
    timed_out   INTEGER NOT NULL DEFAULT 0 CHECK (timed_out IN (0, 1)),
    elapsed_s   REAL    NOT NULL DEFAULT 0.0,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_hitl_log_task_id ON hitl_log (task_id);

-- Mark migration applied
INSERT OR IGNORE INTO schema_migrations (version) VALUES (2);
