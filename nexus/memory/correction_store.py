"""
nexus/memory/correction_store.py
CorrectionStore — persistent record of action corrections accumulated from HITL.

CorrectionRecord
----------------
  id                 : str           opaque UUID
  fingerprint_id     : str | None    owning UIFingerprint (optional)
  context_hash       : str           SHA-1 of the decision context blob
  wrong_action       : dict          the action that was originally taken
  correct_action     : dict          the action that should have been taken
  transport_correction: str | None   if the transport was also wrong,
                                     the correct transport to use
  apply_count        : int           how many times this correction was applied
  created_at         : str           ISO-8601 UTC

CorrectionStore
---------------
  save(record)                       Persist a new correction.
  find_applicable(context_hash,      Return all corrections that apply to
                  fingerprint_id)    this context.
  increment_apply_count(record_id)   Bump the counter after applying.

Context hash
------------
Use CorrectionRecord.hash_context() to compute a stable SHA-1 from an
arbitrary dict (keys are sorted; nested values are JSON-serialised).

Integration with DecisionEngine
--------------------------------
Before generating an action, DecisionEngine should:
  1. Compute context_hash from the current PerceptionResult.
  2. Call find_applicable(context_hash, fingerprint_id) on the store.
  3. If a correction exists, use correct_action (and correct transport) instead.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any

from nexus.infra.database import Database
from nexus.infra.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# CorrectionRecord
# ---------------------------------------------------------------------------


@dataclass
class CorrectionRecord:
    """A single recorded correction (wrong → correct action mapping)."""

    id: str
    context_hash: str
    wrong_action: dict[str, Any]
    correct_action: dict[str, Any]
    fingerprint_id: str | None = None
    transport_correction: str | None = None
    apply_count: int = 0
    created_at: str = ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def hash_context(context: dict[str, Any]) -> str:
        """
        Compute a stable SHA-1 hash from a context dict.

        Keys are sorted recursively; values are JSON-serialised.  This
        ensures that two dicts with the same key-value pairs (regardless
        of insertion order) produce the same hash.
        """
        payload = json.dumps(context, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()


# ---------------------------------------------------------------------------
# CorrectionStore
# ---------------------------------------------------------------------------


class CorrectionStore:
    """
    Persistent store for action correction records.

    Parameters
    ----------
    db:
        Database instance.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save(self, record: CorrectionRecord) -> None:
        """
        Persist *record* to the database.

        If a record with the same (context_hash, wrong_action JSON) already
        exists for the same fingerprint_id, it is replaced to avoid
        duplicates.
        """
        async with self._db.connection() as conn:
            await conn.execute(
                """
                INSERT INTO correction_records
                    (id, fingerprint_id, context_hash,
                     wrong_action, correct_action,
                     transport_correction, apply_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    fingerprint_id       = excluded.fingerprint_id,
                    context_hash         = excluded.context_hash,
                    wrong_action         = excluded.wrong_action,
                    correct_action       = excluded.correct_action,
                    transport_correction = excluded.transport_correction,
                    apply_count          = excluded.apply_count
                """,
                (
                    record.id,
                    record.fingerprint_id,
                    record.context_hash,
                    json.dumps(record.wrong_action, sort_keys=True),
                    json.dumps(record.correct_action, sort_keys=True),
                    record.transport_correction,
                    record.apply_count,
                ),
            )
        _log.debug(
            "correction_store.saved",
            record_id=record.id,
            context_hash=record.context_hash,
        )

    async def find_applicable(
        self,
        context_hash: str,
        fingerprint_id: str | None = None,
    ) -> list[CorrectionRecord]:
        """
        Return all corrections that match *context_hash*.

        When *fingerprint_id* is provided, fingerprint-specific corrections
        are returned first (higher priority), followed by global ones
        (fingerprint_id IS NULL).
        """
        async with self._db.connection() as conn:
            if fingerprint_id is not None:
                async with conn.execute(
                    """
                    SELECT id, fingerprint_id, context_hash,
                           wrong_action, correct_action,
                           transport_correction, apply_count, created_at
                    FROM correction_records
                    WHERE context_hash = ?
                      AND (fingerprint_id = ? OR fingerprint_id IS NULL)
                    ORDER BY
                        CASE WHEN fingerprint_id = ? THEN 0 ELSE 1 END,
                        apply_count DESC
                    """,
                    (context_hash, fingerprint_id, fingerprint_id),
                ) as cur:
                    rows = await cur.fetchall()
            else:
                async with conn.execute(
                    """
                    SELECT id, fingerprint_id, context_hash,
                           wrong_action, correct_action,
                           transport_correction, apply_count, created_at
                    FROM correction_records
                    WHERE context_hash = ?
                    ORDER BY apply_count DESC
                    """,
                    (context_hash,),
                ) as cur:
                    rows = await cur.fetchall()

        return [self._row_to_record(r) for r in rows]

    async def increment_apply_count(self, record_id: str) -> None:
        """Increment the apply_count for *record_id* by one."""
        async with self._db.connection() as conn:
            await conn.execute(
                "UPDATE correction_records "
                "SET apply_count = apply_count + 1 WHERE id = ?",
                (record_id,),
            )
        _log.debug("correction_store.apply_count_incremented", record_id=record_id)

    async def count(self) -> int:
        """Return the total number of stored correction records."""
        async with self._db.connection() as conn, conn.execute(
            "SELECT COUNT(*) FROM correction_records"
        ) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: Any) -> CorrectionRecord:
        return CorrectionRecord(
            id=row["id"],
            fingerprint_id=row["fingerprint_id"],
            context_hash=row["context_hash"],
            wrong_action=json.loads(row["wrong_action"]),
            correct_action=json.loads(row["correct_action"]),
            transport_correction=row["transport_correction"],
            apply_count=int(row["apply_count"]),
            created_at=row["created_at"] or "",
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def new_correction(
    context_hash: str,
    wrong_action: dict[str, Any],
    correct_action: dict[str, Any],
    *,
    fingerprint_id: str | None = None,
    transport_correction: str | None = None,
) -> CorrectionRecord:
    """Create a new CorrectionRecord with a fresh UUID."""
    return CorrectionRecord(
        id=str(uuid.uuid4()),
        context_hash=context_hash,
        wrong_action=wrong_action,
        correct_action=correct_action,
        fingerprint_id=fingerprint_id,
        transport_correction=transport_correction,
    )
