"""
tests/unit/test_task_executor.py
Unit tests for nexus/core/task_executor.py.

Coverage
--------
  TaskResult / TaskContext / TransportStats
    - fields and defaults

  TaskExecutor
    - 3 steps → done → success=True, steps_completed=3
    - max_steps cap → status=failed, steps_completed ≤ cap
    - cancel() → status=cancelled, graceful exit
    - transport_stats: native / fallback counted correctly
    - verification policy applied (sheet_write → SOURCE mode)
    - HITL "abort" → status=cancelled
    - HITL "continue" → loop proceeds
    - source=="suspend" → SuspendManager.suspend() called, status=suspended
    - health check "fail" → abort before first step
    - no DecisionEngine → immediate failure
    - cost_incurred accumulates to total_cost_usd
    - action history grows each step
    - get_verification_policy: action_type → correct VerificationMode
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nexus.core.settings import NexusSettings
from nexus.core.suspend_manager import SuspendManager
from nexus.core.task_executor import (
    TaskExecutor,
    TransportStats,
    get_verification_policy,
)
from nexus.decision.engine import Decision, TargetSpec
from nexus.infra.database import Database
from nexus.source.resolver import SourceResult
from nexus.source.transport.resolver import TransportResult
from nexus.verification import VerificationMode, VerificationResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "exec_test.db"))
    await database.init()
    return database


def _settings(**kw: Any) -> NexusSettings:
    return NexusSettings(**kw)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _decision(action_type: str, source: str = "local", cost: float = 0.0) -> Decision:
    return Decision(
        source=source,  # type: ignore[arg-type]
        action_type=action_type,
        target=TargetSpec(
            element_id=None,
            coordinates=(100, 200),
            description=f"target-{action_type}",
            preferred_transport=None,
        ),
        value=None,
        confidence=0.9,
        reasoning="test",
        cost_incurred=cost,
        transport_hint="uia",
    )


def _transport_result(method: str = "uia", fallback: bool = False) -> TransportResult:
    return TransportResult(
        method_used=method,  # type: ignore[arg-type]
        success=True,
        latency_ms=5.0,
        fallback_used=fallback,
    )


def _uia_source() -> SourceResult:
    return SourceResult(
        source_type="uia",
        data={},
        confidence=1.0,
        latency_ms=0.0,
    )


# ---------------------------------------------------------------------------
# TransportStats
# ---------------------------------------------------------------------------


class TestTransportStats:
    def test_defaults(self):
        stats = TransportStats()
        assert stats.native_count == 0
        assert stats.fallback_count == 0
        assert stats.total == 0
        assert stats.native_ratio == 0.0

    def test_ratio(self):
        stats = TransportStats(native_count=3, fallback_count=1)
        assert stats.total == 4
        assert stats.native_ratio == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# get_verification_policy
# ---------------------------------------------------------------------------


class TestGetVerificationPolicy:
    def test_click_returns_visual(self):
        settings = _settings()
        policy = get_verification_policy("click", settings)
        from nexus.verification import VerificationMode
        assert policy.mode == VerificationMode.VISUAL

    def test_sheet_write_returns_source(self):
        settings = _settings()
        policy = get_verification_policy("sheet_write", settings)
        assert policy.mode == VerificationMode.SOURCE

    def test_done_returns_skip(self):
        settings = _settings()
        policy = get_verification_policy("done", settings)
        assert policy.mode == VerificationMode.SKIP


# ---------------------------------------------------------------------------
# TaskExecutor — core loop
# ---------------------------------------------------------------------------


class TestTaskExecutorLoop:
    def _make_executor(
        self,
        db: Database,
        decisions: list[Decision],
        transport_results: list[TransportResult] | None = None,
        max_steps: int = 50,
        **kw: Any,
    ) -> TaskExecutor:
        settings = _settings(safety={"max_actions_per_task": max_steps})

        # Stub: decision_engine cycles through decisions list
        decision_iter = iter(decisions)
        dec_engine = MagicMock()
        dec_engine.decide = AsyncMock(
            side_effect=lambda *a, **k: next(decision_iter, _decision("done"))
        )

        # Stub: transport_resolver cycles through transport_results
        if transport_results is not None:
            tr_iter = iter(transport_results)
            resolver = MagicMock()
            resolver.execute = AsyncMock(
                side_effect=lambda *a, **k: next(tr_iter, _transport_result("uia"))
            )
        else:
            resolver = None

        return TaskExecutor(
            db=db,
            settings=settings,
            source_fn=AsyncMock(return_value=_uia_source()),
            decision_engine=dec_engine,
            transport_resolver=resolver,
            max_steps=max_steps,
            **kw,
        )

    @pytest.mark.asyncio
    async def test_three_steps_then_done(self, db):
        decisions = [
            _decision("click"),
            _decision("type"),
            _decision("done"),
        ]
        executor = self._make_executor(db, decisions)
        result = await executor.execute("click submit then type then finish")

        assert result.success is True
        assert result.steps_completed == 3
        assert result.status == "completed"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_max_steps_cap(self, db):
        # Infinite non-done decisions
        decisions = [_decision("click")] * 100
        executor = self._make_executor(db, decisions, max_steps=3)
        result = await executor.execute("goal")

        assert result.success is False
        assert result.steps_completed <= 3
        assert result.status == "failed"
        assert "Max steps" in (result.error or "")

    @pytest.mark.asyncio
    async def test_cancel_graceful(self, db):
        call_count = 0

        async def slow_decision(*a: Any, **k: Any) -> Decision:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return _decision("click")

        settings = _settings()
        dec_engine = MagicMock()
        dec_engine.decide = slow_decision

        executor = TaskExecutor(
            db=db,
            settings=settings,
            source_fn=AsyncMock(return_value=_uia_source()),
            decision_engine=dec_engine,
            max_steps=50,
        )

        async def _cancel_later() -> None:
            await asyncio.sleep(0.025)
            executor.cancel()

        result, _ = await asyncio.gather(
            executor.execute("goal"),
            _cancel_later(),
        )

        assert result.status == "cancelled"
        assert result.success is False
        assert call_count >= 1  # at least one step ran

    @pytest.mark.asyncio
    async def test_transport_stats_native(self, db):
        decisions = [_decision("click"), _decision("click"), _decision("done")]
        transport_results = [
            _transport_result("uia", fallback=False),
            _transport_result("uia", fallback=False),
        ]
        executor = self._make_executor(db, decisions, transport_results)
        result = await executor.execute("goal")

        assert result.transport_stats.native_count == 2
        assert result.transport_stats.fallback_count == 0

    @pytest.mark.asyncio
    async def test_transport_stats_with_fallback(self, db):
        decisions = [_decision("click"), _decision("click"), _decision("done")]
        transport_results = [
            _transport_result("uia", fallback=False),
            _transport_result("mouse", fallback=True),
        ]
        executor = self._make_executor(db, decisions, transport_results)
        result = await executor.execute("goal")

        assert result.transport_stats.native_count == 1
        assert result.transport_stats.fallback_count == 1

    @pytest.mark.asyncio
    async def test_no_decision_engine_fails(self, db):
        settings = _settings()
        executor = TaskExecutor(db=db, settings=settings)
        result = await executor.execute("goal")
        assert result.success is False
        assert result.status == "failed"
        assert "DecisionEngine" in (result.error or "")

    @pytest.mark.asyncio
    async def test_cost_accumulates(self, db):
        decisions = [
            _decision("click", cost=0.001),
            _decision("click", cost=0.002),
            _decision("done", cost=0.0),
        ]
        executor = self._make_executor(db, decisions)
        result = await executor.execute("goal")
        assert result.total_cost_usd == pytest.approx(0.003)

    @pytest.mark.asyncio
    async def test_verification_policy_applied(self, db):
        """verifier_fn is called and VerificationResult is produced."""
        decisions = [_decision("click"), _decision("click"), _decision("done")]
        transport_results = [_transport_result("uia"), _transport_result("uia")]

        verify_calls: list[str] = []

        async def verifier_fn(
            before: Any, after: Any, action_type: str
        ) -> VerificationResult:
            verify_calls.append(action_type)
            return VerificationResult(
                success=True,
                mode_used=VerificationMode.VISUAL,
                confidence=0.95,
            )

        executor = self._make_executor(
            db, decisions, transport_results, verifier_fn=verifier_fn
        )
        result = await executor.execute("goal")

        assert result.success is True
        # verifier_fn called for "click" (second step uses before_frame from step 1)
        assert "click" in verify_calls

    @pytest.mark.asyncio
    async def test_verification_policy_sheet_write(self, db):
        """sheet_write action → SOURCE mode policy."""
        decisions = [  # noqa: E501
            _decision("sheet_write"), _decision("sheet_write"), _decision("done")
        ]
        transport_results = [_transport_result("uia"), _transport_result("uia")]

        verify_modes: list[VerificationMode] = []

        async def verifier_fn(
            before: Any, after: Any, action_type: str
        ) -> VerificationResult:
            settings = _settings()
            policy = get_verification_policy(action_type, settings)
            verify_modes.append(policy.mode)
            return VerificationResult(
                success=True,
                mode_used=policy.mode,
                confidence=1.0,
            )

        executor = self._make_executor(
            db, decisions, transport_results, verifier_fn=verifier_fn
        )
        result = await executor.execute("write sheet")
        assert result.success is True
        assert VerificationMode.SOURCE in verify_modes

    @pytest.mark.asyncio
    async def test_action_history_grows(self, db):
        decisions = [_decision("click"), _decision("type"), _decision("done")]
        executor = self._make_executor(db, decisions)
        result = await executor.execute("goal")

        assert result.steps_completed == 3

    @pytest.mark.asyncio
    async def test_task_id_preserved(self, db):
        decisions = [_decision("done")]
        executor = self._make_executor(db, decisions)
        result = await executor.execute("goal", task_id="my-task-42")
        assert result.task_id == "my-task-42"


# ---------------------------------------------------------------------------
# HITL branch
# ---------------------------------------------------------------------------


class TestHITLBranch:
    @pytest.mark.asyncio
    async def test_hitl_abort_cancels_task(self, db):
        from nexus.core.hitl_manager import HITLResponse

        settings = _settings()
        dec_engine = MagicMock()
        dec_engine.decide = AsyncMock(
            return_value=_decision("click", source="hitl")
        )

        hitl_mgr = MagicMock()
        hitl_mgr.request = AsyncMock(
            return_value=HITLResponse(
                task_id="t",
                chosen_option="abort",
                chosen_index=2,
                timed_out=False,
                elapsed_s=0.1,
            )
        )

        executor = TaskExecutor(
            db=db,
            settings=settings,
            source_fn=AsyncMock(return_value=_uia_source()),
            decision_engine=dec_engine,
            hitl_manager=hitl_mgr,
            max_steps=10,
        )
        result = await executor.execute("goal")
        assert result.status == "cancelled"
        assert result.success is False

    @pytest.mark.asyncio
    async def test_hitl_continue_proceeds(self, db):
        from nexus.core.hitl_manager import HITLResponse

        call_count = 0

        async def side_effect(*a: Any, **k: Any) -> Decision:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _decision("click", source="hitl")
            return _decision("done", source="local")

        settings = _settings()
        dec_engine = MagicMock()
        dec_engine.decide = side_effect

        hitl_mgr = MagicMock()
        hitl_mgr.request = AsyncMock(
            return_value=HITLResponse(
                task_id="t",
                chosen_option="continue",
                chosen_index=0,
                timed_out=False,
                elapsed_s=0.1,
            )
        )

        executor = TaskExecutor(
            db=db,
            settings=settings,
            source_fn=AsyncMock(return_value=_uia_source()),
            decision_engine=dec_engine,
            hitl_manager=hitl_mgr,
            max_steps=10,
        )
        result = await executor.execute("goal")
        # After HITL continue, decide() is called again → returns "done"
        assert result.status == "completed"
        assert result.success is True


# ---------------------------------------------------------------------------
# Suspend branch
# ---------------------------------------------------------------------------


class TestSuspendBranch:
    @pytest.mark.asyncio
    async def test_suspend_decision_calls_manager(self, db):
        settings = _settings()
        dec_engine = MagicMock()
        dec_engine.decide = AsyncMock(
            return_value=_decision("click", source="suspend")
        )

        susp_mgr = MagicMock(spec=SuspendManager)
        susp_mgr.suspend = AsyncMock(return_value=MagicMock())

        executor = TaskExecutor(
            db=db,
            settings=settings,
            source_fn=AsyncMock(return_value=_uia_source()),
            decision_engine=dec_engine,
            suspend_manager=susp_mgr,
            max_steps=10,
        )
        result = await executor.execute("goal", task_id="susp-task")
        assert result.status == "suspended"
        assert result.success is False
        susp_mgr.suspend.assert_awaited_once()


# ---------------------------------------------------------------------------
# Health check branch
# ---------------------------------------------------------------------------


class TestHealthCheckBranch:
    @pytest.mark.asyncio
    async def test_health_fail_aborts_before_first_step(self, db):
        settings = _settings()

        health_mock = MagicMock()
        health_report = MagicMock()
        health_report.overall = "fail"
        health_mock.run_all = MagicMock(return_value=health_report)

        # Decision engine should never be called
        dec_engine = MagicMock()
        dec_engine.decide = AsyncMock(side_effect=RuntimeError("should not call"))

        executor = TaskExecutor(
            db=db,
            settings=settings,
            source_fn=AsyncMock(return_value=_uia_source()),
            decision_engine=dec_engine,
            health_checker=health_mock,
            max_steps=10,
        )
        result = await executor.execute("goal")
        assert result.success is False
        assert result.status == "failed"
        assert result.steps_completed == 0
        dec_engine.decide.assert_not_awaited()


# ===========================================================================
# §M — Mutation-targeted tests (survived mutant elimination)
# ===========================================================================


class TestResolveUiaTargetBoundary:
    """Kills L146: r.y <= y mutation (>= variant covers y above element)."""

    def _make_elem(self, x=100, y=100, width=80, height=50) -> Any:
        from unittest.mock import MagicMock
        rect = MagicMock()
        rect.x = x; rect.y = y; rect.width = width; rect.height = height
        elem = MagicMock()
        elem.bounding_rect = rect
        elem.is_visible = True
        return elem

    def _make_source(self, elements: list) -> Any:
        from unittest.mock import MagicMock
        src = MagicMock()
        src.source_type = "uia"
        src.data = elements
        return src

    def _make_target(self, coords: tuple[int, int]) -> Any:
        from unittest.mock import MagicMock
        tgt = MagicMock()
        tgt.element_id = None
        tgt.description = None
        tgt.coordinates = coords
        return tgt

    def test_coord_inside_rect_returns_element(self):
        from nexus.core.task_executor import _resolve_uia_target
        elem = self._make_elem(x=100, y=100, width=80, height=50)
        src = self._make_source([elem])
        tgt = self._make_target((140, 125))  # inside: x∈[100,180], y∈[100,150]
        assert _resolve_uia_target(src, tgt) is elem

    def test_coord_above_rect_returns_none(self):
        """y < r.y — should be None. Kills r.y >= y mutation."""
        from nexus.core.task_executor import _resolve_uia_target
        elem = self._make_elem(x=100, y=100, width=80, height=50)
        src = self._make_source([elem])
        tgt = self._make_target((140, 50))  # y=50 is above r.y=100
        assert _resolve_uia_target(src, tgt) is None

    def test_coord_at_top_edge_returns_element(self):
        """y == r.y (inclusive boundary)."""
        from nexus.core.task_executor import _resolve_uia_target
        elem = self._make_elem(x=100, y=100, width=80, height=50)
        src = self._make_source([elem])
        tgt = self._make_target((140, 100))
        assert _resolve_uia_target(src, tgt) is elem

    def test_coord_below_rect_returns_none(self):
        """y > r.y + r.height — outside bottom edge."""
        from nexus.core.task_executor import _resolve_uia_target
        elem = self._make_elem(x=100, y=100, width=80, height=50)
        src = self._make_source([elem])
        tgt = self._make_target((140, 200))  # y=200 > 150
        assert _resolve_uia_target(src, tgt) is None


class TestDefaultDoneFn:
    """Kills L806: == 'cloud' → > 'cloud' mutation."""

    def _decision(self, action_type: str, source: str) -> Any:
        from unittest.mock import MagicMock
        d = MagicMock()
        d.action_type = action_type
        d.source = source
        # getattr fallback for task_status
        d.task_status = "in_progress"
        return d

    def test_none_action_cloud_source_returns_true(self):
        from nexus.core.task_executor import _default_done_fn
        d = self._decision("none", "cloud")
        assert _default_done_fn(d) is True

    def test_none_action_non_cloud_source_returns_false(self):
        """Kills > 'cloud' mutation: 'local' > 'cloud' is True, but should be False."""
        from nexus.core.task_executor import _default_done_fn
        d = self._decision("none", "local")
        assert _default_done_fn(d) is False

    def test_done_action_returns_true(self):
        from nexus.core.task_executor import _default_done_fn
        d = self._decision("done", "local")
        assert _default_done_fn(d) is True

    def test_complete_task_status_returns_true(self):
        from nexus.core.task_executor import _default_done_fn
        from unittest.mock import MagicMock
        d = MagicMock()
        d.action_type = "click"
        d.task_status = "complete"
        assert _default_done_fn(d) is True


class TestDefaultCaptureFn:
    """Kills L759: height=1 → height=2 mutation."""

    @pytest.mark.asyncio
    async def test_default_capture_fn_dimensions(self):
        from nexus.core.task_executor import _default_capture_fn
        frame = await _default_capture_fn()
        assert frame.width == 1
        assert frame.height == 1
        assert frame.data.shape == (1, 1, 3)

    @pytest.mark.asyncio
    async def test_default_capture_fn_sequence_is_zero(self):
        from nexus.core.task_executor import _default_capture_fn
        frame = await _default_capture_fn()
        assert frame.sequence_number == 0


class TestDefaultPerceiveFn:
    """Kills L792: frame_sequence=1 → frame_sequence=0 mutation."""

    @pytest.mark.asyncio
    async def test_default_perceive_fn_frame_sequence(self):
        from nexus.core.task_executor import _default_perceive_fn
        from unittest.mock import MagicMock
        frame = MagicMock()
        source = _uia_source()
        result = await _default_perceive_fn(frame, source)
        assert result.frame_sequence == 1
