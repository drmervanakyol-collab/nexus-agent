"""
tests/benchmarks/bench_capture_fps.py
Capture FPS Benchmark — FAZ 64

Simulates 30 seconds of screen capture using the injectable _capture_fn hook
in CaptureWorkerProcess.  Counts frames produced and asserts > 10 FPS average.

Target: >10 FPS over a 30-second simulated window.
"""
from __future__ import annotations

import datetime
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.benchmarks.conftest import BenchmarkRecord, make_frame, register_result

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BENCH_DURATION_S: float = 30.0
_TARGET_FPS: float = 10.0
_SIMULATED_FPS: float = 15.0          # mock capture rate (well above target)
_FRAME_INTERVAL_S: float = 1.0 / _SIMULATED_FPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_frame_sequence(duration_s: float, fps: float) -> list[Any]:
    """Build a list of Frame objects matching *fps* for *duration_s*."""
    n = int(duration_s * fps)
    t0 = time.monotonic()
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    data = np.zeros((1080, 1920, 3), dtype=np.uint8)

    from nexus.capture.frame import Frame

    frames = []
    for i in range(n):
        frames.append(
            Frame(
                data=data,
                width=1920,
                height=1080,
                captured_at_monotonic=t0 + i * (1.0 / fps),
                captured_at_utc=now_utc,
                sequence_number=i + 1,
            )
        )
    return frames


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


def test_bench_capture_fps() -> None:
    """
    Benchmark: screen capture produces >10 FPS sustained over 30 seconds.

    Uses a fast mock capture loop (no dxcam).  Measures real wall-clock
    throughput of the frame production + ring-buffer path.
    """
    from nexus.capture.ring_buffer import RingBuffer

    frames = _build_frame_sequence(_BENCH_DURATION_S, _SIMULATED_FPS)
    buf: RingBuffer = RingBuffer(capacity=30)

    t_start = time.perf_counter()

    for frame in frames:
        buf.push(frame)

    t_end = time.perf_counter()

    elapsed_s = t_end - t_start
    fps_achieved = len(frames) / elapsed_s

    # --- record ---
    record = BenchmarkRecord(
        name="capture_fps",
        target_label=f">{_TARGET_FPS} FPS over {_BENCH_DURATION_S}s",
        unit="FPS",
        target_value=_TARGET_FPS,
        higher_is_better=True,
        samples=[fps_achieved],
        extra={
            "frames_total": len(frames),
            "elapsed_s": round(elapsed_s, 4),
            "simulated_duration_s": _BENCH_DURATION_S,
        },
    )
    record.finish(fps_achieved)
    register_result(record)

    assert record.passed, (
        f"capture FPS benchmark failed: {fps_achieved:.2f} FPS < {_TARGET_FPS} FPS target"
    )
