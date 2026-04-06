"""
nexus/core/suspend_manager.py
SuspendManager — pause and resume tasks with DB persistence and drift detection.

SuspendedTask
-------------
Value object representing a task that is currently suspended.  Stored as a
row in the ``suspended_tasks`` table.

ResumeResult
------------
Outcome of a resume() call.

  success         — True when the task was found and removed from suspended state.
  drift_detected  — True when the screen fingerprint has changed significantly
                    since the task was suspended.
  drift_score     — 0.0 (no drift) → 1.0 (completely different fingerprint).
  detail          — Human-readable summary for logs / HITL prompts.

SuspendManager
--------------
  suspend(task_id, reason, context) → SuspendedTask
    Inserts a row into suspended_tasks; if fingerprint_fn is set, also stores
    the current screen fingerprint for later drift comparison.

  resume(task_id) → ResumeResult
    Removes the row from suspended_tasks.  If fingerprint_fn is set, computes
    a new fingerprint and compares with the stored one (drift check).

  list_suspended() → list[SuspendedTask]
    Returns all currently suspended tasks ordered by suspended_at.

Injectability
-------------
All DB I/O is channelled through the Database instance passed at construction.
The optional fingerprint_fn: Callable[[], str] | None captures a screen
fingerprint as a short string; when None all drift detection is skipped and
drift_detected is always False.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nexus.infra.database import Database
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# Fraction of characters that must differ to declare drift.
_DRIFT_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass
class SuspendedTask:
    """A task that is currently suspended."""

    task_id: str
    reason: str
    context: dict[str, Any]
    fingerprint: str | None
    suspended_at: str


@dataclass
class ResumeResult:
    """Outcome of SuspendManager.resume()."""

    task_id: str
    success: bool
    drift_detected: bool = False
    drift_score: float = 0.0
    detail: str = ""


# ---------------------------------------------------------------------------
# SuspendManager
# ---------------------------------------------------------------------------


class SuspendManager:
    """
    Manages task suspension and resumption with persistent storage.

    Parameters
    ----------
    db:
        The Database instance used for all persistence operations.
    fingerprint_fn:
        Optional callable that returns the current screen fingerprint as a
        string.  Injected for testability; pass ``None`` to disable drift
        detection.
    """

    def __init__(
        self,
        db: Database,
        fingerprint_fn: Callable[[], str] | None = None,
    ) -> None:
        self._db = db
        self._fingerprint_fn = fingerprint_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def suspend(
        self,
        task_id: str,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> SuspendedTask:
        """
        Suspend *task_id* and persist the suspension to the database.

        Parameters
        ----------
        task_id:
            Unique identifier of the task to suspend.
        reason:
            Short human-readable explanation (shown in HITL prompts).
        context:
            Arbitrary key/value dict with pipeline state at suspension time.
            Stored as JSON; defaults to an empty dict.
        """
        ctx = context or {}
        ctx_json = json.dumps(ctx)
        fingerprint = self._fingerprint_fn() if self._fingerprint_fn else None

        async with self._db.connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO suspended_tasks
                    (task_id, reason, context, fingerprint)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, reason, ctx_json, fingerprint),
            )
            row = await conn.execute(
                "SELECT suspended_at FROM suspended_tasks WHERE task_id = ?",
                (task_id,),
            )
            record = await row.fetchone()
            suspended_at = record["suspended_at"] if record else ""

        suspended = SuspendedTask(
            task_id=task_id,
            reason=reason,
            context=ctx,
            fingerprint=fingerprint,
            suspended_at=suspended_at,
        )

        _log.info(
            "suspend_manager.suspended",
            task_id=task_id,
            reason=reason,
            has_fingerprint=fingerprint is not None,
        )
        return suspended

    async def resume(self, task_id: str) -> ResumeResult:
        """
        Resume a suspended task.

        Removes the suspension record from the database, performs a drift
        check (if fingerprint_fn is set), and returns a ResumeResult.

        Returns a failed ResumeResult if the task is not currently suspended.
        """
        async with self._db.connection() as conn:
            row = await conn.execute(
                "SELECT task_id, reason, context, fingerprint, suspended_at "
                "FROM suspended_tasks WHERE task_id = ?",
                (task_id,),
            )
            record = await row.fetchone()

            if record is None:
                _log.warning("suspend_manager.not_suspended", task_id=task_id)
                return ResumeResult(
                    task_id=task_id,
                    success=False,
                    detail="task is not currently suspended",
                )

            stored_fingerprint: str | None = record["fingerprint"]
            await conn.execute(
                "DELETE FROM suspended_tasks WHERE task_id = ?",
                (task_id,),
            )

        drift_detected, drift_score = self._check_drift(stored_fingerprint)

        detail = self._build_detail(drift_detected, drift_score)

        _log.info(
            "suspend_manager.resumed",
            task_id=task_id,
            drift_detected=drift_detected,
            drift_score=round(drift_score, 3),
        )

        return ResumeResult(
            task_id=task_id,
            success=True,
            drift_detected=drift_detected,
            drift_score=drift_score,
            detail=detail,
        )

    async def list_suspended(self) -> list[SuspendedTask]:
        """Return all currently suspended tasks ordered by suspension time."""
        async with self._db.connection() as conn:
            rows = await conn.execute(
                "SELECT task_id, reason, context, fingerprint, suspended_at "
                "FROM suspended_tasks ORDER BY suspended_at"
            )
            records = await rows.fetchall()

        result = []
        for r in records:
            try:
                ctx = json.loads(r["context"] or "{}")
            except json.JSONDecodeError:
                ctx = {}
            result.append(
                SuspendedTask(
                    task_id=r["task_id"],
                    reason=r["reason"],
                    context=ctx,
                    fingerprint=r["fingerprint"],
                    suspended_at=r["suspended_at"],
                )
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_drift(
        self, stored_fingerprint: str | None
    ) -> tuple[bool, float]:
        """
        Compare the stored fingerprint with a freshly captured one.

        Returns (drift_detected, drift_score).
        drift_score is in [0.0, 1.0]; values > _DRIFT_THRESHOLD trigger drift.
        """
        if self._fingerprint_fn is None or stored_fingerprint is None:
            return False, 0.0

        current = self._fingerprint_fn()
        score = _character_diff_ratio(stored_fingerprint, current)
        return score > _DRIFT_THRESHOLD, score

    @staticmethod
    def _build_detail(drift_detected: bool, drift_score: float) -> str:
        if not drift_detected:
            return f"resume ok; drift_score={drift_score:.3f}"
        return (
            f"drift detected; drift_score={drift_score:.3f} "
            f"(threshold={_DRIFT_THRESHOLD})"
        )


# ---------------------------------------------------------------------------
# Drift metric helper
# ---------------------------------------------------------------------------


def _character_diff_ratio(a: str, b: str) -> float:
    """
    Simple character-level difference ratio between two strings.

    Uses the longer string as the denominator so that an empty reference
    never produces 0/0.  Returns 0.0 when both strings are empty.
    """
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    # Count positions where characters differ (zip truncates to shorter).
    diff = sum(ca != cb for ca, cb in zip(a, b, strict=False))
    # Penalise length difference.
    diff += abs(len(a) - len(b))
    return min(diff / max_len, 1.0)
