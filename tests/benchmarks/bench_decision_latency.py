"""
tests/benchmarks/bench_decision_latency.py
Decision Latency Benchmark — FAZ 64

Runs DecisionEngine.decide() 100 times on the local-resolution path (mock
LocalResolver always returns a Decision immediately, no cloud call) and
measures average latency.

Target: average decision latency < 200 ms over 100 iterations.
"""
from __future__ import annotations

import asyncio
import datetime
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from tests.benchmarks.conftest import BenchmarkRecord, make_frame, make_source_result, register_result

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_DECISIONS: int = 100
_TARGET_AVG_MS: float = 200.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perception_result(seq: int = 1) -> Any:
    """Build a minimal PerceptionResult with an empty SpatialGraph."""
    from nexus.perception.arbitration.arbitrator import ArbitrationResult
    from nexus.perception.orchestrator import PerceptionResult
    from nexus.perception.spatial_graph import SpatialGraph
    from nexus.perception.temporal.temporal_expert import ScreenState, StateType
    from nexus.source.resolver import SourceResult

    return PerceptionResult(
        spatial_graph=SpatialGraph([], [], {}),
        screen_state=ScreenState(
            state_type=StateType.STABLE,
            confidence=1.0,
            blocks_perception=False,
            reason="bench",
            retry_after_ms=0,
        ),
        arbitration=ArbitrationResult(
            resolved_elements=(),
            resolved_labels=(),
            conflicts_detected=0,
            conflicts_resolved=0,
            temporal_blocked=False,
            overall_confidence=1.0,
        ),
        source_result=SourceResult(
            source_type="uia",
            data=[],
            confidence=1.0,
            latency_ms=0.5,
        ),
        perception_ms=0.1,
        frame_sequence=seq,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def _make_local_decision() -> Any:
    """Build a minimal local Decision."""
    from nexus.decision.engine import Decision, TargetSpec

    return Decision(
        source="local",
        action_type="click",
        target=TargetSpec(
            element_id="bench-el-1",
            coordinates=(100, 100),
            description="Bench button",
            preferred_transport="uia",
        ),
        value=None,
        confidence=0.95,
        reasoning="Benchmark local resolution.",
        cost_incurred=0.0,
        transport_hint="uia",
    )


def _make_engine() -> Any:
    """Build a DecisionEngine where LocalResolver always succeeds immediately."""
    from nexus.cloud.planner import CloudPlanner
    from nexus.core.policy import PolicyEngine
    from nexus.core.settings import NexusSettings
    from nexus.decision.ambiguity_scorer import AmbiguityScorer
    from nexus.decision.engine import DecisionEngine, LocalResolver

    settings = NexusSettings()
    local_decision = _make_local_decision()

    # Policy: always allow
    policy = MagicMock(spec=PolicyEngine)
    policy.check_action.return_value = MagicMock(verdict="allow", reason="bench")

    # Scorer: always recommend "local"
    scorer = MagicMock(spec=AmbiguityScorer)
    scorer.score.return_value = MagicMock(recommendation="local", score=0.2)

    # LocalResolver: always returns the canned decision
    resolver = MagicMock(spec=LocalResolver)
    resolver.resolve.return_value = local_decision

    # CloudPlanner: should never be called; mock as safety net
    planner = AsyncMock(spec=CloudPlanner)

    engine = DecisionEngine(
        policy=policy,
        scorer=scorer,
        resolver=resolver,
        planner=planner,
        cost_before_fn=lambda _tid: 0.0,
    )
    return engine


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bench_decision_latency() -> None:
    """
    Benchmark: DecisionEngine.decide() averages < 200 ms over 100 iterations.

    LocalResolver always returns immediately, so this measures the full
    decision pipeline overhead (policy check, scorer, resolver wiring)
    without any cloud calls.
    """
    from nexus.cloud.prompt_builder import ActionRecord
    from nexus.decision.engine import DecisionContext

    engine = _make_engine()
    context = DecisionContext(
        task_id="bench-task",
        candidate_is_destructive=False,
        actions_so_far=0,
        elapsed_seconds=0.0,
        task_cost_usd=0.0,
        daily_cost_usd=0.0,
    )

    latencies_ms: list[float] = []

    for i in range(_N_DECISIONS):
        perception = _make_perception_result(seq=i + 1)
        t0 = time.perf_counter()
        decision = await engine.decide(
            goal="click the benchmark button",
            perception=perception,
            action_history=[],
            context=context,
        )
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1_000)

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    p95_ms = sorted(latencies_ms)[int(_N_DECISIONS * 0.95)]

    record = BenchmarkRecord(
        name="decision_latency",
        target_label=f"avg < {_TARGET_AVG_MS} ms over {_N_DECISIONS} decisions",
        unit="ms",
        target_value=_TARGET_AVG_MS,
        higher_is_better=False,
        samples=latencies_ms,
        extra={
            "avg_ms": round(avg_ms, 3),
            "p95_ms": round(p95_ms, 3),
            "min_ms": round(min(latencies_ms), 3),
            "max_ms": round(max(latencies_ms), 3),
            "decisions": _N_DECISIONS,
        },
    )
    record.finish(avg_ms)
    register_result(record)

    assert record.passed, (
        f"decision latency benchmark failed: {avg_ms:.3f} ms avg > {_TARGET_AVG_MS} ms target"
    )
