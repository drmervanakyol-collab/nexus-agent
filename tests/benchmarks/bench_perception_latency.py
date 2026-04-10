"""
tests/benchmarks/bench_perception_latency.py
Perception Latency Benchmark — FAZ 64

Runs 50 frames through PerceptionOrchestrator with a UIA source (structured
path — skips Locator + OCR for minimal overhead) and measures average latency.

Target: average perception latency < 500 ms over 50 frames.
"""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.benchmarks.conftest import (
    BenchmarkRecord,
    make_frame,
    make_source_result,
    register_result,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_FRAMES: int = 50
_TARGET_AVG_MS: float = 500.0


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------


def _make_orchestrator() -> Any:
    """Build a PerceptionOrchestrator with fully mocked sub-components."""
    from nexus.perception.arbitration.arbitrator import (
        PerceptionArbitrator,
    )
    from nexus.perception.locator.locator import Locator
    from nexus.perception.matcher.matcher import Matcher
    from nexus.perception.orchestrator import PerceptionOrchestrator
    from nexus.perception.reader.ocr_engine import OCREngine
    from nexus.perception.temporal.temporal_expert import (
        TemporalExpert,
    )

    # Minimal stubs — structured source path never calls these
    temporal = MagicMock(spec=TemporalExpert)
    locator = MagicMock(spec=Locator)
    matcher = MagicMock(spec=Matcher)
    arbitrator = MagicMock(spec=PerceptionArbitrator)
    ocr = MagicMock(spec=OCREngine)

    return PerceptionOrchestrator(
        temporal_expert=temporal,
        locator=locator,
        matcher=matcher,
        arbitrator=arbitrator,
        ocr_engine=ocr,
        cache_ttl_s=0.0,   # disable cache so every frame is processed
    )


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bench_perception_latency() -> None:
    """
    Benchmark: perception pipeline averages < 500 ms per frame over 50 frames.

    Uses the structured UIA path which bypasses Locator/OCR; it measures
    pure framework overhead (dataclass construction, cache lookup, graph init).
    """
    orchestrator = _make_orchestrator()
    source = make_source_result(source_type="uia")   # structured path

    latencies_ms: list[float] = []

    for i in range(_N_FRAMES):
        frame = make_frame(seq=i + 1)
        t0 = time.perf_counter()
        await orchestrator.perceive(frame, source)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1_000)

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    p95_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]

    record = BenchmarkRecord(
        name="perception_latency",
        target_label=f"avg < {_TARGET_AVG_MS} ms over {_N_FRAMES} frames",
        unit="ms",
        target_value=_TARGET_AVG_MS,
        higher_is_better=False,
        samples=latencies_ms,
        extra={
            "avg_ms": round(avg_ms, 3),
            "p95_ms": round(p95_ms, 3),
            "min_ms": round(min(latencies_ms), 3),
            "max_ms": round(max(latencies_ms), 3),
            "frames": _N_FRAMES,
        },
    )
    record.finish(avg_ms)
    register_result(record)

    assert record.passed, (
        f"perception latency benchmark failed: {avg_ms:.2f} ms avg > {_TARGET_AVG_MS} ms target"
    )
