"""
tests/benchmarks/conftest.py
Shared fixtures and helpers for performance benchmarks.

All benchmarks run without real hardware (no dxcam, no Tesseract, no UIA).
Injectable fns provide deterministic, fast mock behaviour so that benchmarks
measure framework overhead rather than external I/O.

Benchmark results are written to tests/benchmarks/results/ as both JSON and
Markdown.  A ``NEXUS_BENCH_BASELINE`` environment variable can point at a
prior JSON report to enable regression detection.
"""
from __future__ import annotations

import datetime
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pytest session-finish hook — write JSON + Markdown report
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session: Any, exitstatus: object) -> None:
    """Write benchmark report at the end of any session that ran benchmarks."""
    if _BENCH_REGISTRY:
        from tests.benchmarks.bench_report import write_report
        write_report()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Minimal NexusSettings factory
# ---------------------------------------------------------------------------


def make_settings(**overrides: Any):  # noqa: ANN201
    """Return a NexusSettings instance suitable for benchmarks."""
    from nexus.core.settings import NexusSettings

    return NexusSettings(**overrides)


# ---------------------------------------------------------------------------
# Frame factory
# ---------------------------------------------------------------------------


def make_frame(
    width: int = 1920,
    height: int = 1080,
    seq: int = 1,
) -> Any:
    """Return a minimal Frame with synthetic pixel data."""
    from nexus.capture.frame import Frame

    data = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=time.monotonic(),
        captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sequence_number=seq,
    )


# ---------------------------------------------------------------------------
# SourceResult factory
# ---------------------------------------------------------------------------


def make_source_result(source_type: str = "uia") -> Any:
    """Return a minimal SourceResult."""
    from nexus.source.resolver import SourceResult

    return SourceResult(
        source_type=source_type,  # type: ignore[arg-type]
        data=[],
        confidence=1.0,
        latency_ms=1.0,
    )


# ---------------------------------------------------------------------------
# BenchmarkRecord — collects timings for a single benchmark run
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRecord:
    """Accumulates per-benchmark timing and pass/fail result."""

    name: str
    target_label: str
    unit: str
    target_value: float
    higher_is_better: bool  # True = measured >= target; False = measured <= target
    measured_value: float = 0.0
    passed: bool = False
    samples: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def finish(self, measured: float) -> None:
        self.measured_value = measured
        if self.higher_is_better:
            self.passed = measured >= self.target_value
        else:
            self.passed = measured <= self.target_value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_label": self.target_label,
            "unit": self.unit,
            "target_value": self.target_value,
            "higher_is_better": self.higher_is_better,
            "measured_value": round(self.measured_value, 4),
            "passed": self.passed,
            "samples_count": len(self.samples),
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Global registry (populated by each bench module's fixture)
# ---------------------------------------------------------------------------

_BENCH_REGISTRY: list[BenchmarkRecord] = []


def register_result(record: BenchmarkRecord) -> None:
    _BENCH_REGISTRY.append(record)


def get_all_results() -> list[BenchmarkRecord]:
    return list(_BENCH_REGISTRY)
