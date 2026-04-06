"""
tests/integration/test_block6.py
Blok 6 Integration Tests — Faz 46

Full TaskExecutor pipeline wired with real DB and lightweight stubs.
No real UIA, DOM, or mouse hardware is exercised.

TEST 1 — Happy path end-to-end
  3 decisions (click, click, done) → TaskResult.success=True,
  transport_stats.native_count > 0.

TEST 2 — HITL path
  Decision source="hitl" → HITLManager returns "continue" → loop continues
  → next decision="done" → status="completed".

TEST 3 — Suspend path
  Decision source="suspend" → SuspendManager.suspend() called once
  → status="suspended".

TEST 4 — Memory learning
  Successful transport → FingerprintStore.record_outcome() called →
  preferred_transport updated for that fingerprint.

TEST 5 — Cost cap
  Decision with cost_incurred >= max_cost_per_task_usd → status="failed",
  error contains "Cost cap".

TEST 6 — Cancel
  executor.cancel() called mid-loop → status="cancelled".

TEST 7 — Full transport chain
  UIA element → Decision transport_hint="uia" → TransportResolver.invoke() →
  VerificationResult SOURCE success → audit record written.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio

from nexus.capture.frame import Frame
from nexus.core.hitl_manager import HITLResponse
from nexus.core.settings import NexusSettings
from nexus.core.suspend_manager import SuspendManager
from nexus.core.task_executor import TaskExecutor
from nexus.decision.engine import Decision, TargetSpec
from nexus.infra.database import Database
from nexus.memory.fingerprint_store import FingerprintStore, new_fingerprint
from nexus.perception.arbitration.arbitrator import ArbitrationResult
from nexus.perception.orchestrator import PerceptionResult
from nexus.perception.spatial_graph import SpatialGraph
from nexus.perception.temporal.temporal_expert import ScreenState, StateType
from nexus.source.resolver import SourceResult
from nexus.source.transport.resolver import (
    ActionSpec as TransportActionSpec,
)
from nexus.source.transport.resolver import (
    TransportResolver,
    TransportResult,
)
from nexus.verification import (
    SourceVerifier,
    VerificationMode,
    VerificationPolicy,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UTC = "2026-04-07T00:00:00+00:00"
_TASK_ID = "blok6-task"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _frame(pixel_value: int = 128, width: int = 8, height: int = 8) -> Frame:
    data = np.full((height, width, 3), pixel_value, dtype=np.uint8)
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=time.monotonic(),
        captured_at_utc=_UTC,
        sequence_number=1,
    )


def _uia_source() -> SourceResult:
    return SourceResult(
        source_type="uia",
        data={"elements": []},
        confidence=1.0,
        latency_ms=0.0,
    )


def _minimal_perception(source: SourceResult | None = None) -> PerceptionResult:
    stable_state = ScreenState(
        state_type=StateType.STABLE,
        confidence=1.0,
        blocks_perception=False,
        reason="stable",
        retry_after_ms=0,
    )
    arbitration = ArbitrationResult(
        resolved_elements=(),
        resolved_labels=(),
        conflicts_detected=0,
        conflicts_resolved=0,
        temporal_blocked=False,
        overall_confidence=1.0,
    )
    return PerceptionResult(
        spatial_graph=SpatialGraph([], [], {}),
        screen_state=stable_state,
        arbitration=arbitration,
        source_result=source or _uia_source(),
        perception_ms=0.0,
        frame_sequence=1,
        timestamp=_UTC,
    )


def _decision(
    action_type: str,
    source: str = "local",
    cost: float = 0.0,
    transport_hint: str = "uia",
) -> Decision:
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
        transport_hint=transport_hint,
    )


def _transport_result(method: str = "uia", fallback: bool = False) -> TransportResult:
    return TransportResult(
        method_used=method,  # type: ignore[arg-type]
        success=True,
        latency_ms=5.0,
        fallback_used=fallback,
    )


def _settings(**kw: Any) -> NexusSettings:
    return NexusSettings(**kw)


# ---------------------------------------------------------------------------
# Fixture: Database
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "blok6.db"))
    await database.init()
    return database


# ---------------------------------------------------------------------------
# Helpers to build minimal TaskExecutor
# ---------------------------------------------------------------------------


def _make_executor(
    db: Database,
    decisions: list[Decision],
    transport_results: list[TransportResult] | None = None,
    *,
    settings: NexusSettings | None = None,
    fingerprint_store: FingerprintStore | None = None,
    hitl_manager: Any = None,
    suspend_manager: Any = None,
    max_steps: int = 50,
) -> TaskExecutor:
    cfg = settings or _settings()

    decision_iter = iter(decisions)
    dec_engine = MagicMock()
    dec_engine.decide = AsyncMock(
        side_effect=lambda *a, **k: next(decision_iter, _decision("done"))
    )

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
        settings=cfg,
        source_fn=AsyncMock(return_value=_uia_source()),
        capture_fn=AsyncMock(return_value=_frame()),
        perceive_fn=AsyncMock(return_value=_minimal_perception()),
        decision_engine=dec_engine,
        transport_resolver=resolver,
        fingerprint_store=fingerprint_store,
        hitl_manager=hitl_manager,
        suspend_manager=suspend_manager,
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# TEST 1 — Happy path end-to-end
# ---------------------------------------------------------------------------


class TestHappyPath:
    """
    3 decisions (click, click, done) → TaskResult.success=True,
    transport_stats.native_count > 0.
    """

    @pytest.mark.asyncio
    async def test_three_steps_success(self, db):
        decisions = [_decision("click"), _decision("click"), _decision("done")]
        transport_results = [_transport_result("uia"), _transport_result("uia")]
        executor = _make_executor(db, decisions, transport_results)

        result = await executor.execute("complete three steps", task_id=_TASK_ID)

        assert result.success is True
        assert result.status == "completed"
        assert result.steps_completed == 3
        assert result.transport_stats.native_count > 0
        assert result.error is None


# ---------------------------------------------------------------------------
# TEST 2 — HITL path
# ---------------------------------------------------------------------------


class TestHITLPath:
    """
    source="hitl" → HITLManager.request() returns "continue"
    → next decision="done" → status="completed".
    """

    @pytest.mark.asyncio
    async def test_hitl_continue_completes(self, db):
        call_count = 0

        async def _decide(*a: Any, **k: Any) -> Decision:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _decision("click", source="hitl")
            return _decision("done", source="local")

        hitl_mgr = MagicMock()
        hitl_mgr.request = AsyncMock(
            return_value=HITLResponse(
                task_id=_TASK_ID,
                chosen_option="continue",
                chosen_index=0,
                timed_out=False,
                elapsed_s=0.05,
            )
        )

        settings = _settings()
        dec_engine = MagicMock()
        dec_engine.decide = _decide

        executor = TaskExecutor(
            db=db,
            settings=settings,
            source_fn=AsyncMock(return_value=_uia_source()),
            decision_engine=dec_engine,
            hitl_manager=hitl_mgr,
            max_steps=10,
        )
        result = await executor.execute("hitl goal", task_id=_TASK_ID)

        assert result.status == "completed"
        assert result.success is True
        hitl_mgr.request.assert_awaited_once()


# ---------------------------------------------------------------------------
# TEST 3 — Suspend path
# ---------------------------------------------------------------------------


class TestSuspendPath:
    """
    source="suspend" → SuspendManager.suspend() called once
    → status="suspended".
    """

    @pytest.mark.asyncio
    async def test_suspend_decision_suspends_task(self, db):
        susp_mgr = MagicMock(spec=SuspendManager)
        susp_mgr.suspend = AsyncMock(return_value=MagicMock())

        decisions = [_decision("click", source="suspend")]
        executor = _make_executor(
            db, decisions, suspend_manager=susp_mgr, max_steps=5
        )
        result = await executor.execute("goal", task_id=_TASK_ID + "-s")

        assert result.status == "suspended"
        assert result.success is False
        susp_mgr.suspend.assert_awaited_once()


# ---------------------------------------------------------------------------
# TEST 4 — Memory learning
# ---------------------------------------------------------------------------


class TestMemoryLearning:
    """
    Successful transport → FingerprintStore.record_outcome() called →
    preferred_transport updated for the matched fingerprint.
    """

    @pytest.mark.asyncio
    async def test_preferred_transport_updated_after_success(self, db):
        fp_store = FingerprintStore(db, max_rows=100)

        # Pre-seed a fingerprint with layout_hash matching frame_sequence=1
        # (task_executor uses str(perception.frame_sequence) as layout_hash)
        fp = new_fingerprint("1", "")
        await fp_store.save(fp)

        decisions = [_decision("click"), _decision("done")]
        transport_results = [_transport_result("uia")]

        executor = _make_executor(
            db,
            decisions,
            transport_results,
            fingerprint_store=fp_store,
        )
        result = await executor.execute("learn transport", task_id=_TASK_ID + "-m")

        assert result.success is True

        # preferred_transport should be "uia" after one successful uia transport
        updated = await fp_store.find_similar("1", "")
        assert updated is not None
        assert updated.preferred_transport == "uia"


# ---------------------------------------------------------------------------
# TEST 5 — Cost cap
# ---------------------------------------------------------------------------


class TestCostCap:
    """
    Decision with cost_incurred >= max_cost_per_task_usd
    → status="failed", error contains "Cost cap".
    """

    @pytest.mark.asyncio
    async def test_cost_cap_stops_execution(self, db):
        # Set a very low cost cap
        settings = _settings(budget={"max_cost_per_task_usd": 0.001})

        decisions = [
            _decision("click", cost=0.002),  # immediately exceeds cap
            _decision("done"),
        ]
        transport_results = [_transport_result("uia")]
        executor = _make_executor(
            db, decisions, transport_results, settings=settings, max_steps=10
        )
        result = await executor.execute("cost goal", task_id=_TASK_ID + "-c")

        assert result.success is False
        assert result.status == "failed"
        assert "Cost cap" in (result.error or "")
        assert result.steps_completed <= 2


# ---------------------------------------------------------------------------
# TEST 6 — Cancel
# ---------------------------------------------------------------------------


class TestCancel:
    """
    executor.cancel() called mid-loop → status="cancelled", success=False.
    """

    @pytest.mark.asyncio
    async def test_cancel_sets_cancelled_status(self, db):
        call_count = 0

        async def _slow_decide(*a: Any, **k: Any) -> Decision:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return _decision("click")

        settings = _settings()
        dec_engine = MagicMock()
        dec_engine.decide = _slow_decide

        executor = TaskExecutor(
            db=db,
            settings=settings,
            source_fn=AsyncMock(return_value=_uia_source()),
            decision_engine=dec_engine,
            max_steps=50,
        )

        async def _cancel_after() -> None:
            await asyncio.sleep(0.025)
            executor.cancel()

        result, _ = await asyncio.gather(
            executor.execute("cancel goal", task_id=_TASK_ID + "-x"),
            _cancel_after(),
        )

        assert result.status == "cancelled"
        assert result.success is False
        assert call_count >= 1


# ---------------------------------------------------------------------------
# TEST 7 — Full transport chain
# ---------------------------------------------------------------------------


class TestFullTransportChain:
    """
    UIA element → TransportResolver (uia invoker) → SourceVerifier
    → audit record written → VerificationResult.success=True.
    """

    @pytest.mark.asyncio
    async def test_uia_transport_source_verify_audit(self, db):
        settings = _settings()

        audit_records: list[TransportResult] = []

        async def _audit(result: TransportResult, spec: TransportActionSpec) -> None:
            audit_records.append(result)

        # UIA invoker always succeeds
        resolver = TransportResolver(
            settings,
            _uia_invoker=lambda el: True,
            _audit_writer=_audit,
        )

        transport_spec = TransportActionSpec(
            action_type="click",
            task_id=_TASK_ID + "-t",
        )
        tr = await resolver.execute(
            transport_spec, _uia_source(), target_element=object()
        )

        assert tr.method_used == "uia"
        assert tr.success is True
        assert tr.fallback_used is False

        # Audit written
        assert len(audit_records) == 1
        assert audit_records[0].method_used == "uia"

        # Source-level verification
        probe = lambda ctx: "clicked"  # noqa: E731
        verifier = SourceVerifier(source_probe=probe)
        v_result = verifier.verify("clicked", VerificationPolicy.source())

        assert v_result.success is True
        assert v_result.mode_used == VerificationMode.SOURCE
        assert v_result.confidence == 1.0

        # Executor with transport resolver: native_count tracked
        decision_iter = iter(
            [_decision("click", transport_hint="uia"), _decision("done")]
        )
        dec_engine = MagicMock()
        dec_engine.decide = AsyncMock(
            side_effect=lambda *a, **k: next(decision_iter, _decision("done"))
        )

        tr_iter = iter([tr])
        mock_resolver = MagicMock()
        mock_resolver.execute = AsyncMock(
            side_effect=lambda *a, **k: next(tr_iter, _transport_result("uia"))
        )

        executor = TaskExecutor(
            db=db,
            settings=settings,
            source_fn=AsyncMock(return_value=_uia_source()),
            decision_engine=dec_engine,
            transport_resolver=mock_resolver,
            max_steps=10,
        )
        exec_result = await executor.execute(
            "uia transport chain", task_id=_TASK_ID + "-t2"
        )

        assert exec_result.success is True
        assert exec_result.transport_stats.native_count >= 1
