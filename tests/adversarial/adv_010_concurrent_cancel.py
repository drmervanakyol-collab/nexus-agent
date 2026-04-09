"""
tests/adversarial/adv_010_concurrent_cancel.py
Adversarial Test 010 — Concurrent cancel during execution → DB consistent

Scenario:
  TaskExecutor.execute() is running.  A concurrent task calls cancel() mid-loop.

Success criteria:
  - TaskResult.status == "cancelled" (not "running" / "completed").
  - TaskResult.success is False.
  - At least one step ran before cancellation.
  - DB task row status is "cancelled" (not left as "running").
  - No exception escapes the executor.

DB is a real in-memory SQLite instance for consistency verification.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nexus.core.settings import NexusSettings
from nexus.core.task_executor import TaskExecutor
from nexus.decision.engine import Decision, TargetSpec
from nexus.infra.database import Database
from nexus.source.resolver import SourceResult
from nexus.source.transport.resolver import TransportResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "adv010.db"))
    await database.init()
    return database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings() -> NexusSettings:
    return NexusSettings(safety={"max_actions_per_task": 100})


def _decision(action_type: str) -> Decision:
    return Decision(
        source="local",  # type: ignore[arg-type]
        action_type=action_type,
        target=TargetSpec(
            element_id=None,
            coordinates=(100, 200),
            description=f"target-{action_type}",
            preferred_transport=None,
        ),
        value=None,
        confidence=0.9,
        reasoning="adv010",
        cost_incurred=0.0,
        transport_hint="uia",
    )


def _uia_source() -> SourceResult:
    return SourceResult(source_type="uia", data={}, confidence=1.0, latency_ms=0.0)


def _transport(method: str = "uia") -> TransportResult:
    return TransportResult(method_used=method, success=True, latency_ms=5.0, fallback_used=False)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.adversarial
class TestConcurrentCancel:
    """ADV-010: cancel() during execution → status 'cancelled', DB consistent."""

    @pytest.mark.asyncio
    async def test_cancel_sets_status_cancelled(self, db: Database):
        """
        Executor is running slow steps.  cancel() is called after 0.03 s.
        Result must be status='cancelled', success=False.
        """
        steps_run = [0]

        async def _slow_decide(*_a: Any, **_k: Any) -> Decision:
            steps_run[0] += 1
            await asyncio.sleep(0.015)
            return _decision("click")

        dec_engine = MagicMock()
        dec_engine.decide = _slow_decide

        executor = TaskExecutor(
            db=db,
            settings=_settings(),
            source_fn=AsyncMock(return_value=_uia_source()),
            decision_engine=dec_engine,
            max_steps=100,
        )

        async def _cancel_after_delay() -> None:
            await asyncio.sleep(0.04)
            executor.cancel()

        result, _ = await asyncio.gather(
            executor.execute("adversarial goal"),
            _cancel_after_delay(),
        )

        assert result.status == "cancelled", (
            f"Expected status='cancelled'; got {result.status!r}"
        )
        assert result.success is False
        assert steps_run[0] >= 1, "At least one step must have run before cancel"

    @pytest.mark.asyncio
    async def test_cancel_db_row_is_cancelled(self, db: Database):
        """
        After cancel, the DB task row status must be 'cancelled'
        (not left as 'running').
        """
        async def _slow_decide(*_a: Any, **_k: Any) -> Decision:
            await asyncio.sleep(0.015)
            return _decision("click")

        dec_engine = MagicMock()
        dec_engine.decide = _slow_decide

        executor = TaskExecutor(
            db=db,
            settings=_settings(),
            source_fn=AsyncMock(return_value=_uia_source()),
            decision_engine=dec_engine,
            max_steps=100,
        )

        async def _cancel_after_delay() -> None:
            await asyncio.sleep(0.04)
            executor.cancel()

        result, _ = await asyncio.gather(
            executor.execute("db consistency check"),
            _cancel_after_delay(),
        )

        assert result.status == "cancelled"

        # Verify DB row directly
        async with db.connection() as conn:
            cursor = await conn.execute(
                "SELECT status FROM tasks WHERE id = ?", (result.task_id,)
            )
            row = await cursor.fetchone()

        assert row is not None, "Task row must exist in DB"
        db_status = row[0]
        # _map_status maps "cancelled" → "aborted" in the DB
        assert db_status == "aborted", (
            f"DB task row must be 'aborted' (mapped from cancelled); got {db_status!r}"
        )

    @pytest.mark.asyncio
    async def test_cancel_property_resets_on_new_execute(self, db: Database):
        """
        TaskExecutor.execute() resets _cancelled to False at start,
        so a new execution can run even after a prior cancel().
        This verifies that cancel state does not leak between runs.
        """
        dec_engine = MagicMock()
        # First run: done immediately
        dec_engine.decide = AsyncMock(return_value=_decision("done"))

        executor = TaskExecutor(
            db=db,
            settings=_settings(),
            source_fn=AsyncMock(return_value=_uia_source()),
            decision_engine=dec_engine,
            max_steps=100,
        )

        # First run — completes normally
        r1 = await executor.execute("first run")
        assert r1.status == "completed"

        # Cancel the executor
        executor.cancel()

        # Second run: execute() resets _cancelled=False, should complete again
        dec_engine.decide = AsyncMock(return_value=_decision("done"))
        r2 = await executor.execute("second run after cancel reset")
        assert r2.status == "completed", (
            f"execute() must reset cancel flag; got status={r2.status!r}"
        )

    @pytest.mark.asyncio
    async def test_no_exception_escapes_on_cancel(self, db: Database):
        """
        cancel() must not cause any unhandled exception to escape execute().
        """
        call_count = [0]

        async def _decide(*_a: Any, **_k: Any) -> Decision:
            call_count[0] += 1
            await asyncio.sleep(0.01)
            return _decision("click")

        dec_engine = MagicMock()
        dec_engine.decide = _decide

        executor = TaskExecutor(
            db=db,
            settings=_settings(),
            source_fn=AsyncMock(return_value=_uia_source()),
            decision_engine=dec_engine,
            max_steps=100,
        )

        try:
            async def _cancel() -> None:
                await asyncio.sleep(0.03)
                executor.cancel()

            result, _ = await asyncio.gather(
                executor.execute("exception safety test"),
                _cancel(),
            )
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"execute() must not raise on cancel; got: {exc!r}")
        else:
            assert result.status in {"cancelled", "failed", "completed"}
