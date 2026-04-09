"""
tests/benchmarks/bench_end_to_end.py
End-to-End Benchmark — FAZ 64

Runs a simple one-step task through TaskExecutor 10 times using a mock system
(no real DB disk I/O — in-memory SQLite, no hardware, no cloud calls).
Each run: source → capture → perceive → decide(click) → decide(done).

Target: >80% success rate across 10 runs.
"""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio

from tests.benchmarks.conftest import BenchmarkRecord, make_frame, register_result

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_RUNS: int = 10
_TARGET_SUCCESS_RATE: float = 0.80
_UTC = "2026-04-09T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Helpers (mirror integration test_block6 pattern)
# ---------------------------------------------------------------------------


def _make_source_result() -> Any:
    from nexus.source.resolver import SourceResult

    return SourceResult(
        source_type="uia",
        data={"elements": []},
        confidence=1.0,
        latency_ms=0.0,
    )


def _make_perception() -> Any:
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
            latency_ms=0.0,
        ),
        perception_ms=0.1,
        frame_sequence=1,
        timestamp=_UTC,
    )


def _make_decision(action_type: str) -> Any:
    from nexus.decision.engine import Decision, TargetSpec

    return Decision(
        source="local",
        action_type=action_type,
        target=TargetSpec(
            element_id="bench-btn",
            coordinates=(100, 100),
            description="bench button",
            preferred_transport="uia",
        ),
        value=None,
        confidence=0.95,
        reasoning="benchmark decision",
        cost_incurred=0.0,
        transport_hint="uia",
    )


def _make_transport_result() -> Any:
    from nexus.source.transport.resolver import TransportResult

    return TransportResult(
        method_used="uia",
        success=True,
        latency_ms=1.0,
        fallback_used=False,
    )


async def _make_executor(db: Any) -> Any:
    """Build a TaskExecutor with mocked sub-systems."""
    from nexus.core.settings import NexusSettings
    from nexus.core.task_executor import TaskExecutor

    settings = NexusSettings()

    # Decision sequence: click → done
    decisions = [_make_decision("click"), _make_decision("done")]
    decision_iter = iter(decisions)

    dec_engine = MagicMock()
    dec_engine.decide = AsyncMock(
        side_effect=lambda *a, **k: next(decision_iter, _make_decision("done"))
    )

    transport_result = _make_transport_result()
    resolver = MagicMock()
    resolver.execute = AsyncMock(return_value=transport_result)

    frame = make_frame(seq=1)
    perception = _make_perception()

    return TaskExecutor(
        db=db,
        settings=settings,
        source_fn=AsyncMock(return_value=_make_source_result()),
        capture_fn=AsyncMock(return_value=frame),
        perceive_fn=AsyncMock(return_value=perception),
        decision_engine=dec_engine,
        transport_resolver=resolver,
        max_steps=10,
    )


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bench_end_to_end(tmp_path) -> None:
    """
    Benchmark: TaskExecutor completes a simple task successfully in >80% of 10 runs.

    Each run uses a fresh in-memory SQLite DB and fresh mock components.
    """
    from nexus.infra.database import Database

    successes = 0
    run_times_ms: list[float] = []

    for run_idx in range(_N_RUNS):
        db_path = str(tmp_path / f"bench_e2e_{run_idx}.db")
        db = Database(db_path)
        await db.init()

        executor = await _make_executor(db)

        t0 = time.perf_counter()
        try:
            result = await executor.execute(
                goal="click the benchmark button",
                task_id=f"bench-e2e-{run_idx}",
            )
            t1 = time.perf_counter()
            run_times_ms.append((t1 - t0) * 1_000)
            if result.success:
                successes += 1
        except Exception:
            t1 = time.perf_counter()
            run_times_ms.append((t1 - t0) * 1_000)
        finally:
            await db.close()

    success_rate = successes / _N_RUNS
    avg_run_ms = sum(run_times_ms) / len(run_times_ms)

    record = BenchmarkRecord(
        name="end_to_end",
        target_label=f">{_TARGET_SUCCESS_RATE * 100:.0f}% success rate over {_N_RUNS} runs",
        unit="success_rate",
        target_value=_TARGET_SUCCESS_RATE,
        higher_is_better=True,
        samples=[float(successes)],
        extra={
            "successes": successes,
            "total_runs": _N_RUNS,
            "success_rate_pct": round(success_rate * 100, 1),
            "avg_run_ms": round(avg_run_ms, 2),
            "min_run_ms": round(min(run_times_ms), 2),
            "max_run_ms": round(max(run_times_ms), 2),
        },
    )
    record.finish(success_rate)
    register_result(record)

    assert record.passed, (
        f"End-to-end benchmark failed: "
        f"{successes}/{_N_RUNS} = {success_rate * 100:.1f}% < {_TARGET_SUCCESS_RATE * 100:.0f}% target"
    )
