"""
tests/unit/test_hitl_suspend.py
Unit tests for SuspendManager and HITLManager.

Coverage
--------
  SuspendManager
    - suspend() → row inserted in suspended_tasks
    - suspend() stores fingerprint when fingerprint_fn provided
    - list_suspended() returns all suspended tasks
    - resume() → row deleted, success=True
    - resume() on unknown task → success=False
    - resume() drift_detected when fingerprints differ beyond threshold
    - resume() drift_detected=False when fingerprints identical
    - resume() drift_detected=False when no fingerprint_fn

  HITLManager
    - headless=True → immediate default response, timed_out=False
    - option selection by number (1-based)
    - empty input → default option
    - invalid number (out-of-range) → stored as free-form
    - free-form text stored as chosen_option
    - timeout → default response, timed_out=True
    - audit log written to hitl_log table
    - default_index respected (options + default_index)
    - headless=True logs to hitl_log
"""
from __future__ import annotations

import json
import time

import pytest
import pytest_asyncio

from nexus.core.hitl_manager import HITLManager, HITLRequest
from nexus.core.settings import HITLSettings
from nexus.core.suspend_manager import SuspendManager, _character_diff_ratio
from nexus.infra.database import Database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(tmp_path):
    """In-memory-equivalent: fresh SQLite in a temp file."""
    db = Database(str(tmp_path / "test.db"))
    await db.init()
    return db


# ---------------------------------------------------------------------------
# _character_diff_ratio helper
# ---------------------------------------------------------------------------


class TestCharacterDiffRatio:
    def test_identical_strings(self):
        assert _character_diff_ratio("abc", "abc") == 0.0

    def test_completely_different(self):
        score = _character_diff_ratio("aaa", "zzz")
        assert score == 1.0

    def test_one_char_diff(self):
        score = _character_diff_ratio("abc", "axc")
        assert 0.0 < score < 1.0

    def test_both_empty(self):
        assert _character_diff_ratio("", "") == 0.0

    def test_length_difference(self):
        score = _character_diff_ratio("a", "aaaa")
        assert score > 0.0

    def test_score_capped_at_1(self):
        assert _character_diff_ratio("a", "z" * 100) <= 1.0


# ---------------------------------------------------------------------------
# SuspendManager
# ---------------------------------------------------------------------------


class TestSuspendManager:
    async def _count_suspended(self, db: Database, task_id: str) -> int:
        async with db.connection() as conn:
            row = await conn.execute(
                "SELECT COUNT(*) FROM suspended_tasks WHERE task_id = ?",
                (task_id,),
            )
            result = await row.fetchone()
        return result[0]

    @pytest.mark.asyncio
    async def test_suspend_inserts_row(self, db):
        mgr = SuspendManager(db)
        task = await mgr.suspend("task-1", "ambiguous action")
        assert task.task_id == "task-1"
        assert task.reason == "ambiguous action"
        assert await self._count_suspended(db, "task-1") == 1

    @pytest.mark.asyncio
    async def test_suspend_stores_context(self, db):
        mgr = SuspendManager(db)
        ctx = {"step": 3, "action": "click"}
        task = await mgr.suspend("task-2", "test", context=ctx)
        assert task.context == ctx

    @pytest.mark.asyncio
    async def test_suspend_stores_fingerprint(self, db):
        fp_calls = iter(["fp-before", "fp-after"])
        mgr = SuspendManager(db, fingerprint_fn=lambda: next(fp_calls))
        task = await mgr.suspend("task-3", "test")
        assert task.fingerprint == "fp-before"

    @pytest.mark.asyncio
    async def test_list_suspended_empty(self, db):
        mgr = SuspendManager(db)
        result = await mgr.list_suspended()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_suspended_multiple(self, db):
        mgr = SuspendManager(db)
        await mgr.suspend("t1", "reason A")
        await mgr.suspend("t2", "reason B")
        tasks = await mgr.list_suspended()
        task_ids = {t.task_id for t in tasks}
        assert {"t1", "t2"} == task_ids

    @pytest.mark.asyncio
    async def test_resume_removes_row(self, db):
        mgr = SuspendManager(db)
        await mgr.suspend("task-4", "test")
        result = await mgr.resume("task-4")
        assert result.success is True
        assert await self._count_suspended(db, "task-4") == 0

    @pytest.mark.asyncio
    async def test_resume_unknown_task_fails(self, db):
        mgr = SuspendManager(db)
        result = await mgr.resume("nonexistent")
        assert result.success is False
        assert "not currently suspended" in result.detail

    @pytest.mark.asyncio
    async def test_resume_no_drift_when_same_fingerprint(self, db):
        fp_calls = iter(["same", "same"])
        mgr = SuspendManager(db, fingerprint_fn=lambda: next(fp_calls))
        await mgr.suspend("task-5", "test")
        result = await mgr.resume("task-5")
        assert result.success is True
        assert result.drift_detected is False
        assert result.drift_score == 0.0

    @pytest.mark.asyncio
    async def test_resume_drift_detected_on_large_change(self, db):
        fp_iter = iter(["aaaaaaaaaa", "zzzzzzzzzz"])
        mgr = SuspendManager(db, fingerprint_fn=lambda: next(fp_iter))
        await mgr.suspend("task-6", "test")
        result = await mgr.resume("task-6")
        assert result.success is True
        assert result.drift_detected is True
        assert result.drift_score > 0.10

    @pytest.mark.asyncio
    async def test_resume_no_drift_check_without_fn(self, db):
        mgr = SuspendManager(db, fingerprint_fn=None)
        await mgr.suspend("task-7", "test")
        result = await mgr.resume("task-7")
        assert result.drift_detected is False
        assert result.drift_score == 0.0

    @pytest.mark.asyncio
    async def test_suspend_idempotent_via_replace(self, db):
        """Second suspend on same task_id replaces the first row."""
        mgr = SuspendManager(db)
        await mgr.suspend("task-8", "first")
        await mgr.suspend("task-8", "second")
        tasks = await mgr.list_suspended()
        matching = [t for t in tasks if t.task_id == "task-8"]
        assert len(matching) == 1
        assert matching[0].reason == "second"


# ---------------------------------------------------------------------------
# HITLManager
# ---------------------------------------------------------------------------


class TestHITLManager:
    def _mgr(self, db, input_fn=None, headless=False, timeout_s=5.0):
        settings = HITLSettings(headless=headless, timeout_s=timeout_s)
        return HITLManager(db, settings=settings, input_fn=input_fn)

    async def _hitl_log_count(self, db: Database, task_id: str) -> int:
        async with db.connection() as conn:
            row = await conn.execute(
                "SELECT COUNT(*) FROM hitl_log WHERE task_id = ?", (task_id,)
            )
            r = await row.fetchone()
        return r[0]

    @pytest.mark.asyncio
    async def test_headless_returns_default_immediately(self, db):
        mgr = self._mgr(db, headless=True)
        req = HITLRequest(
            task_id="t1",
            question="Continue?",
            options=["yes", "no", "skip"],
            default_index=2,
        )
        resp = await mgr.request(req)
        assert resp.chosen_option == "skip"
        assert resp.timed_out is False
        assert resp.elapsed_s >= 0.0

    @pytest.mark.asyncio
    async def test_headless_logs_to_db(self, db):
        mgr = self._mgr(db, headless=True)
        req = HITLRequest(task_id="t2", question="Q?", options=["a", "b"], default_index=0)
        await mgr.request(req)
        assert await self._hitl_log_count(db, "t2") == 1

    @pytest.mark.asyncio
    async def test_option_selection_by_number(self, db):
        mgr = self._mgr(db, input_fn=lambda _: "2")
        req = HITLRequest(
            task_id="t3",
            question="Choose:",
            options=["abort", "retry", "skip"],
            default_index=0,
        )
        resp = await mgr.request(req)
        assert resp.chosen_option == "retry"
        assert resp.chosen_index == 1
        assert resp.timed_out is False

    @pytest.mark.asyncio
    async def test_empty_input_selects_default(self, db):
        mgr = self._mgr(db, input_fn=lambda _: "")
        req = HITLRequest(
            task_id="t4",
            question="Q?",
            options=["abort", "continue"],
            default_index=1,
        )
        resp = await mgr.request(req)
        assert resp.chosen_option == "continue"
        assert resp.chosen_index == 1

    @pytest.mark.asyncio
    async def test_out_of_range_number_stored_as_freeform(self, db):
        mgr = self._mgr(db, input_fn=lambda _: "99")
        req = HITLRequest(
            task_id="t5",
            question="Q?",
            options=["a", "b"],
            default_index=0,
        )
        resp = await mgr.request(req)
        assert resp.chosen_option == "99"
        assert resp.chosen_index is None

    @pytest.mark.asyncio
    async def test_freeform_text_response(self, db):
        mgr = self._mgr(db, input_fn=lambda _: "custom answer")
        req = HITLRequest(task_id="t6", question="What now?")
        resp = await mgr.request(req)
        assert resp.chosen_option == "custom answer"
        assert resp.chosen_index is None

    @pytest.mark.asyncio
    async def test_timeout_returns_default(self, db):
        """
        Simulate timeout by using an input_fn that blocks longer than timeout_s.
        We use a very short timeout (0.05 s) so the test completes fast.
        """
        def slow_input(_prompt: str) -> str:
            time.sleep(10)  # longer than timeout
            return "never"

        settings = HITLSettings(timeout_s=0.05, headless=False)
        mgr = HITLManager(db, settings=settings, input_fn=slow_input)
        req = HITLRequest(
            task_id="t7",
            question="Hurry?",
            options=["yes", "no"],
            default_index=1,
        )
        resp = await mgr.request(req)
        assert resp.timed_out is True
        assert resp.chosen_option == "no"

    @pytest.mark.asyncio
    async def test_timeout_logs_to_db(self, db):
        def slow_input(_prompt: str) -> str:
            time.sleep(10)
            return "never"

        settings = HITLSettings(timeout_s=0.05, headless=False)
        mgr = HITLManager(db, settings=settings, input_fn=slow_input)
        req = HITLRequest(task_id="t8", question="Q?", options=["a"], default_index=0)
        await mgr.request(req)
        assert await self._hitl_log_count(db, "t8") == 1

    @pytest.mark.asyncio
    async def test_audit_log_content(self, db):
        mgr = self._mgr(db, input_fn=lambda _: "1")
        req = HITLRequest(
            task_id="t9",
            question="Proceed?",
            options=["yes", "no"],
            default_index=0,
        )
        await mgr.request(req)
        async with db.connection() as conn:
            row = await conn.execute(
                "SELECT question, options, chosen, timed_out FROM hitl_log WHERE task_id = ?",
                ("t9",),
            )
            r = await row.fetchone()
        assert r["question"] == "Proceed?"
        assert json.loads(r["options"]) == ["yes", "no"]
        assert r["chosen"] == "yes"
        assert r["timed_out"] == 0

    @pytest.mark.asyncio
    async def test_default_index_out_of_bounds_clamped(self, db):
        mgr = self._mgr(db, input_fn=lambda _: "")
        req = HITLRequest(
            task_id="t10",
            question="Q?",
            options=["only"],
            default_index=99,  # out of bounds
        )
        resp = await mgr.request(req)
        assert resp.chosen_option == "only"

    @pytest.mark.asyncio
    async def test_no_options_freeform_default(self, db):
        """No options list → default_action from settings used on empty input."""
        settings = HITLSettings(default_action="abort", headless=False)
        mgr = HITLManager(db, settings=settings, input_fn=lambda _: "")
        req = HITLRequest(task_id="t11", question="Q?")
        resp = await mgr.request(req)
        assert resp.chosen_option == "abort"
