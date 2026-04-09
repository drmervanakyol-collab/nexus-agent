-- migrations/004_onboarding_consent.sql
-- Nexus Agent — Onboarding consent tracking
-- Idempotent: uses CREATE TABLE IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS onboarding_consent (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scope       TEXT    NOT NULL UNIQUE,   -- 'privacy' | 'terms'
    granted     INTEGER NOT NULL DEFAULT 0 CHECK (granted IN (0, 1)),
    granted_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Mark migration applied
INSERT OR IGNORE INTO schema_migrations (version) VALUES (4);
