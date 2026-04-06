"""
nexus/memory/fingerprint_store.py
FingerprintStore — persistent UI context memory with transport learning.

UIFingerprint
-------------
Represents a recognised UI state.  Two fingerprints are "similar" when
their layout_hash matches exactly (same element geometry) OR when both
layout_hash and element_signature match within a configurable threshold.

Fields
------
  id                   : str           opaque UUID
  app_name             : str           process / window title hint
  layout_hash          : str           SHA-1 of element bounding-box set
  element_signature    : str           SHA-1 of sorted (type, role) pairs
  successful_strategies: list[str]     transport + action combos that worked
  failure_patterns     : list[str]     combos that have consistently failed
  last_seen            : str           ISO-8601 UTC — updated on every hit
  seen_count           : int           total times this fingerprint was matched
  confidence_boost     : float         cumulative success/failure signal [0,1]
  preferred_transport  : str | None    highest-success transport (learned)

FingerprintStore
----------------
  save(fp)                             Insert or replace.
  find_similar(layout_hash,            Return best match or None.
               element_signature)
  record_outcome(fp_id, success,       Update success/failure stats and
                 transport_used,       learn preferred_transport.
                 strategy)
  evict_stale(days)                    Remove rows not seen for *days* days;
                                       also enforces max_rows LRU cap.

Transport learning
------------------
Each record_outcome() call:
  success=True  → strategy appended to successful_strategies (capped at 20).
                  Transport win-count incremented in a JSON tally stored in
                  successful_strategies as the last element ``__tally__:{...}``.
  success=False → strategy appended to failure_patterns (capped at 20).
  After every call, preferred_transport is set to the transport with the
  highest win count from the tally (ties broken by lexicographic order).

Max-size / LRU eviction
-----------------------
  After save() and record_outcome(), if the row count exceeds *max_rows*,
  the least-recently-seen rows are deleted until the count is at or below
  the cap.  Default max_rows = settings.storage.memory_max_size_mb * 500
  (rough estimate: ~2 KB per row → 500 rows/MB).
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from nexus.core.settings import NexusSettings
from nexus.infra.database import Database
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

_MAX_LIST_LEN = 20          # cap on successful_strategies / failure_patterns
_TALLY_KEY = "__tally__"   # sentinel key for transport win-count tally


# ---------------------------------------------------------------------------
# UIFingerprint
# ---------------------------------------------------------------------------


@dataclass
class UIFingerprint:
    """
    A recognised UI state with learned execution preferences.

    All fields have sensible defaults so callers can construct a minimal
    fingerprint and let record_outcome() fill in the learned fields over time.
    """

    id: str
    layout_hash: str
    element_signature: str
    app_name: str = ""
    successful_strategies: list[str] = field(default_factory=list)
    failure_patterns: list[str] = field(default_factory=list)
    last_seen: str = ""
    seen_count: int = 1
    confidence_boost: float = 0.0
    preferred_transport: str | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_layout_hash(elements: list[dict[str, Any]]) -> str:
        """
        Compute a stable layout hash from a list of element dicts.

        Each element dict should contain ``x``, ``y``, ``width``, ``height``
        (integers) and optionally ``role`` (string).  The hash is the SHA-1
        of the sorted, serialised representation.
        """
        normalised = sorted(
            
                (
                    int(e.get("x", 0)),
                    int(e.get("y", 0)),
                    int(e.get("width", 0)),
                    int(e.get("height", 0)),
                )
                for e in elements
            
        )
        payload = json.dumps(normalised, separators=(",", ":"))
        return hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()

    @staticmethod
    def compute_element_signature(elements: list[dict[str, Any]]) -> str:
        """
        Compute a stable element signature from type+role pairs.

        Sorts pairs so that element order does not matter.
        """
        pairs = sorted(
            (str(e.get("type", "")), str(e.get("role", "")))
            for e in elements
        )
        payload = json.dumps(pairs, separators=(",", ":"))
        return hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()


# ---------------------------------------------------------------------------
# FingerprintStore
# ---------------------------------------------------------------------------


class FingerprintStore:
    """
    Persistent store for UIFingerprint objects.

    Parameters
    ----------
    db:
        Database instance.
    settings:
        NexusSettings — used for storage.memory_max_size_mb and
        storage.memory_evict_after_days.
    max_rows:
        Override for the row cap (primarily for tests).  When None,
        derived from settings.storage.memory_max_size_mb * 500.
    """

    def __init__(
        self,
        db: Database,
        settings: NexusSettings | None = None,
        *,
        max_rows: int | None = None,
    ) -> None:
        self._db = db
        cfg = settings.storage if settings else None
        self._evict_after_days: int = (
            cfg.memory_evict_after_days if cfg else 30
        )
        self._max_rows: int = (
            max_rows
            if max_rows is not None
            else ((cfg.memory_max_size_mb * 500) if cfg else 128_000)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save(self, fp: UIFingerprint) -> None:
        """Insert or replace *fp* in the database."""
        async with self._db.connection() as conn:
            await conn.execute(
                """
                INSERT INTO ui_fingerprints
                    (id, app_name, layout_hash, element_signature,
                     successful_strategies, failure_patterns,
                     last_seen, seen_count, confidence_boost, preferred_transport)
                VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                        ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    app_name              = excluded.app_name,
                    layout_hash           = excluded.layout_hash,
                    element_signature     = excluded.element_signature,
                    successful_strategies = excluded.successful_strategies,
                    failure_patterns      = excluded.failure_patterns,
                    last_seen             = excluded.last_seen,
                    seen_count            = excluded.seen_count,
                    confidence_boost      = excluded.confidence_boost,
                    preferred_transport   = excluded.preferred_transport
                """,
                (
                    fp.id,
                    fp.app_name,
                    fp.layout_hash,
                    fp.element_signature,
                    json.dumps(fp.successful_strategies),
                    json.dumps(fp.failure_patterns),
                    fp.seen_count,
                    fp.confidence_boost,
                    fp.preferred_transport,
                ),
            )
        await self._enforce_max_rows()
        _log.debug("fingerprint_store.saved", fp_id=fp.id)

    async def find_similar(
        self,
        layout_hash: str,
        element_signature: str,
    ) -> UIFingerprint | None:
        """
        Return the best-matching fingerprint or None.

        Similarity strategy (V1):
          1. Exact layout_hash match — pick highest seen_count.
          2. No exact match — return None (V1 keeps it simple).

        When a match is found, its last_seen and seen_count are updated.
        """
        async with self._db.connection() as conn:
            async with conn.execute(
                """
                SELECT id, app_name, layout_hash, element_signature,
                       successful_strategies, failure_patterns,
                       last_seen, seen_count, confidence_boost, preferred_transport
                FROM ui_fingerprints
                WHERE layout_hash = ?
                ORDER BY seen_count DESC
                LIMIT 1
                """,
                (layout_hash,),
            ) as cur:
                row = await cur.fetchone()

            if row is None:
                return None

            # Bump last_seen and seen_count
            await conn.execute(
                """
                UPDATE ui_fingerprints
                   SET last_seen  = strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                       seen_count = seen_count + 1
                 WHERE id = ?
                """,
                (row["id"],),
            )

        # seen_count in the SELECT row is the pre-bump value; we already issued
        # UPDATE seen_count = seen_count + 1, so add 1 to reflect the true DB state.
        fp = self._row_to_fp(row)
        return UIFingerprint(
            id=fp.id,
            layout_hash=fp.layout_hash,
            element_signature=fp.element_signature,
            app_name=fp.app_name,
            successful_strategies=fp.successful_strategies,
            failure_patterns=fp.failure_patterns,
            last_seen=fp.last_seen,
            seen_count=fp.seen_count + 1,
            confidence_boost=fp.confidence_boost,
            preferred_transport=fp.preferred_transport,
        )

    async def record_outcome(
        self,
        fingerprint_id: str,
        success: bool,
        transport_used: str,
        strategy: str,
    ) -> None:
        """
        Record an execution outcome for *fingerprint_id*.

        Updates successful_strategies / failure_patterns, transport tally,
        confidence_boost, and preferred_transport.
        """
        async with self._db.connection() as conn:
            async with conn.execute(
                "SELECT successful_strategies, failure_patterns, "
                "confidence_boost FROM ui_fingerprints WHERE id = ?",
                (fingerprint_id,),
            ) as cur:
                row = await cur.fetchone()

            if row is None:
                _log.warning(
                    "fingerprint_store.record_outcome.not_found",
                    fp_id=fingerprint_id,
                )
                return

            successes: list[str] = json.loads(row["successful_strategies"] or "[]")
            failures: list[str] = json.loads(row["failure_patterns"] or "[]")
            boost: float = float(row["confidence_boost"])

            # Update tally (last element if it is the tally sentinel)
            tally: dict[str, int] = {}
            if successes and successes[-1].startswith(_TALLY_KEY):
                tally = json.loads(successes[-1][len(_TALLY_KEY):])
                successes = successes[:-1]

            if success:
                successes.append(strategy)
                if len(successes) > _MAX_LIST_LEN:
                    successes = successes[-_MAX_LIST_LEN:]
                tally[transport_used] = tally.get(transport_used, 0) + 1
                boost = min(boost + 0.05, 1.0)
            else:
                failures.append(strategy)
                if len(failures) > _MAX_LIST_LEN:
                    failures = failures[-_MAX_LIST_LEN:]
                tally[transport_used] = max(tally.get(transport_used, 0) - 1, 0)
                boost = max(boost - 0.02, 0.0)

            # Re-append tally sentinel
            successes.append(_TALLY_KEY + json.dumps(tally, separators=(",", ":")))

            # Determine preferred_transport (highest tally; ties → lexicographic)
            preferred: str | None = None
            if tally:
                preferred = max(tally, key=lambda t: (tally[t], t))
                if tally[preferred] <= 0:
                    preferred = None

            await conn.execute(
                """
                UPDATE ui_fingerprints
                   SET successful_strategies = ?,
                       failure_patterns      = ?,
                       confidence_boost      = ?,
                       preferred_transport   = ?
                 WHERE id = ?
                """,
                (
                    json.dumps(successes),
                    json.dumps(failures),
                    round(boost, 4),
                    preferred,
                    fingerprint_id,
                ),
            )

        _log.debug(
            "fingerprint_store.outcome_recorded",
            fp_id=fingerprint_id,
            success=success,
            transport=transport_used,
            preferred=preferred,
        )

    async def evict_stale(self, days: int | None = None) -> int:
        """
        Delete fingerprints not seen for *days* days.

        Returns the number of rows deleted.
        """
        evict_days = days if days is not None else self._evict_after_days
        async with self._db.connection() as conn, conn.execute(
            """
                DELETE FROM ui_fingerprints
                 WHERE last_seen < strftime('%Y-%m-%dT%H:%M:%fZ',
                                           'now', ? )
                """,
            (f"-{evict_days} days",),
        ) as cur:
            deleted = cur.rowcount if cur.rowcount >= 0 else 0
        _log.info("fingerprint_store.evict_stale", days=evict_days, deleted=deleted)
        return deleted

    async def count(self) -> int:
        """Return the current number of stored fingerprints."""
        async with self._db.connection() as conn, conn.execute(
            "SELECT COUNT(*) FROM ui_fingerprints"
        ) as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _enforce_max_rows(self) -> None:
        """Delete oldest (LRU) rows when count exceeds max_rows."""
        async with self._db.connection() as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM ui_fingerprints"
            ) as cur:
                row = await cur.fetchone()
            current = int(row[0]) if row else 0

            if current <= self._max_rows:
                return

            excess = current - self._max_rows
            await conn.execute(
                """
                DELETE FROM ui_fingerprints
                 WHERE id IN (
                     SELECT id FROM ui_fingerprints
                      ORDER BY last_seen ASC
                      LIMIT ?
                 )
                """,
                (excess,),
            )
            _log.info(
                "fingerprint_store.lru_evict",
                evicted=excess,
                remaining=self._max_rows,
            )

    @staticmethod
    def _row_to_fp(row: Any) -> UIFingerprint:
        return UIFingerprint(
            id=row["id"],
            app_name=row["app_name"] or "",
            layout_hash=row["layout_hash"],
            element_signature=row["element_signature"],
            successful_strategies=json.loads(row["successful_strategies"] or "[]"),
            failure_patterns=json.loads(row["failure_patterns"] or "[]"),
            last_seen=row["last_seen"] or "",
            seen_count=int(row["seen_count"]),
            confidence_boost=float(row["confidence_boost"]),
            preferred_transport=row["preferred_transport"],
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def new_fingerprint(
    layout_hash: str,
    element_signature: str,
    *,
    app_name: str = "",
    preferred_transport: str | None = None,
) -> UIFingerprint:
    """Create a new UIFingerprint with a fresh UUID."""
    return UIFingerprint(
        id=str(uuid.uuid4()),
        layout_hash=layout_hash,
        element_signature=element_signature,
        app_name=app_name,
        preferred_transport=preferred_transport,
    )
