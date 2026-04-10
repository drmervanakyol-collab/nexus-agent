"""
tests/benchmarks/bench_memory_soak.py
Memory Soak Benchmark — FAZ 64

Runs a tight mock loop equivalent to 30 minutes of agent operation
(compressed: ~5 400 iterations at ~200ms simulated cycle time) and
measures memory growth using tracemalloc.

Target: heap growth < 50 MB over the soak period.

The loop exercises the same objects allocated during a real agent cycle:
  Frame → PerceptionResult → Decision → ActionRecord
These are created and discarded on each iteration to exercise the GC.
No real I/O, DB, or hardware.
"""
from __future__ import annotations

import gc
import time
import tracemalloc

import numpy as np

from tests.benchmarks.conftest import (
    BenchmarkRecord,
    register_result,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 30 minutes at ~200ms/step = 9 000 steps; use a representative sample
_SOAK_ITERATIONS: int = 9_000
_TARGET_MAX_GROWTH_MB: float = 50.0


# ---------------------------------------------------------------------------
# Per-iteration object factory (mirrors real agent cycle allocations)
# ---------------------------------------------------------------------------


def _one_cycle(seq: int) -> None:
    """Allocate and immediately release the objects produced in one agent step."""
    import datetime

    from nexus.capture.frame import Frame
    from nexus.decision.engine import Decision, TargetSpec
    from nexus.perception.arbitration.arbitrator import ArbitrationResult
    from nexus.perception.orchestrator import PerceptionResult
    from nexus.perception.spatial_graph import SpatialGraph
    from nexus.perception.temporal.temporal_expert import ScreenState, StateType
    from nexus.source.resolver import SourceResult

    # Frame (8×8 pixel thumbnail — minimal memory)
    data = np.zeros((8, 8, 3), dtype=np.uint8)
    _frame = Frame(
        data=data,
        width=8,
        height=8,
        captured_at_monotonic=time.monotonic(),
        captured_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        sequence_number=seq,
    )

    # PerceptionResult
    _source = SourceResult(source_type="uia", data=[], confidence=1.0, latency_ms=0.1)
    _perception = PerceptionResult(
        spatial_graph=SpatialGraph([], [], {}),
        screen_state=ScreenState(
            state_type=StateType.STABLE,
            confidence=1.0,
            blocks_perception=False,
            reason="soak",
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
        source_result=_source,
        perception_ms=0.05,
        frame_sequence=seq,
        timestamp="2026-04-10T00:00:00+00:00",
    )

    # Decision
    _decision = Decision(
        source="local",
        action_type="click",
        target=TargetSpec(
            element_id=f"el-{seq}",
            coordinates=(seq % 1920, seq % 1080),
            description="soak element",
            preferred_transport="uia",
        ),
        value=None,
        confidence=0.95,
        reasoning="soak benchmark",
        cost_incurred=0.0,
        transport_hint="uia",
    )

    # Explicitly delete to give GC a chance (matches real loop discard)
    del _frame, _perception, _decision, _source, data


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


def test_bench_memory_soak() -> None:
    """
    Benchmark: heap growth over {_SOAK_ITERATIONS} mock agent cycles < 50 MB.

    tracemalloc is used to measure Python-heap allocations; resident set
    size growth is an additional informational metric.
    """
    # Warm-up: let import caches and class initialisation settle
    for i in range(100):
        _one_cycle(i)
    gc.collect()

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    t_start = time.perf_counter()

    for i in range(_SOAK_ITERATIONS):
        _one_cycle(i)
        # Periodic GC to replicate real agent behaviour (Python GC threshold)
        if i % 1000 == 999:
            gc.collect()

    gc.collect()
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    elapsed_s = time.perf_counter() - t_start

    # Compute net heap growth
    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    total_growth_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
    growth_mb = total_growth_bytes / (1024 * 1024)

    # Top leaking sites (informational)
    top_leaks = [
        {
            "location": str(s.traceback),
            "size_kb": round(s.size_diff / 1024, 1),
        }
        for s in sorted(stats, key=lambda x: x.size_diff, reverse=True)[:5]
        if s.size_diff > 0
    ]

    record = BenchmarkRecord(
        name="memory_soak",
        target_label=f"heap growth < {_TARGET_MAX_GROWTH_MB} MB over {_SOAK_ITERATIONS} iterations",
        unit="MB",
        target_value=_TARGET_MAX_GROWTH_MB,
        higher_is_better=False,
        samples=[growth_mb],
        extra={
            "growth_mb": round(growth_mb, 3),
            "iterations": _SOAK_ITERATIONS,
            "elapsed_s": round(elapsed_s, 2),
            "simulated_minutes": round(_SOAK_ITERATIONS * 0.2 / 60, 1),
            "top_leaks": top_leaks,
        },
    )
    record.finish(growth_mb)
    register_result(record)

    assert record.passed, (
        f"Memory soak benchmark failed: heap grew {growth_mb:.2f} MB "
        f"> {_TARGET_MAX_GROWTH_MB} MB target"
    )
