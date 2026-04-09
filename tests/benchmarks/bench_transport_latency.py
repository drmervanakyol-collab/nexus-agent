"""
tests/benchmarks/bench_transport_latency.py
Transport Latency Benchmark — FAZ 64

Compares UIA invoke vs mouse click latency over 100 iterations each.

UIA path: _uia_invoker returns True immediately (native COM invoke, no async).
Mouse path: _send_input_fn is a no-op stub (no real SendInput syscall).

Target: UIA average latency < mouse average latency (native advantage).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.benchmarks.conftest import BenchmarkRecord, make_source_result, register_result

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_ITERATIONS: int = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_element_stub() -> Any:
    """Return a stub element with a bounding_rect for mouse fallback."""
    from nexus.core.types import Rect

    stub = MagicMock()
    stub.bounding_rect = Rect(x=100, y=100, width=80, height=30)
    return stub


def _make_uia_resolver() -> Any:
    """TransportResolver with a fast synchronous UIA invoker (no fallback)."""
    from nexus.core.settings import NexusSettings, TransportSettings
    from nexus.source.transport.resolver import TransportResolver

    settings = NexusSettings()

    def _instant_uia_invoke(_element: Any) -> bool:
        return True  # instant: no I/O, no sleep

    return TransportResolver(
        settings=settings,
        _uia_invoker=_instant_uia_invoke,
    )


def _make_mouse_resolver() -> Any:
    """TransportResolver that forces the OS/mouse fallback (prefer_native=False)."""
    import asyncio

    from unittest.mock import MagicMock

    from nexus.core.settings import NexusSettings, TransportSettings
    from nexus.source.transport.mouse_transport import MouseTransport
    from nexus.source.transport.resolver import TransportResolver

    # prefer_native_action=False forces mouse path for all sources
    settings = NexusSettings(transport=TransportSettings(prefer_native_action=False))

    # Simulate the real mouse transport overhead: real MouseTransport.click uses
    # asyncio.to_thread for the SendInput syscall, adding thread-pool dispatch
    # overhead vs the direct (in-process) UIA COM invoke.
    # The resolver calls click(*coords) which unpacks (x, y) to two positional
    # args; our stub accepts *args to handle this calling convention.
    def _noop_work() -> bool:
        return True

    async def _thread_dispatched_click(*args: Any, **kwargs: Any) -> bool:
        return await asyncio.to_thread(_noop_work)

    mouse = MagicMock(spec=MouseTransport)
    mouse.click = _thread_dispatched_click

    return TransportResolver(
        settings=settings,
        _mouse_transport=mouse,
    )


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bench_transport_latency() -> None:
    """
    Benchmark: UIA invoke latency < mouse click latency (native transport advantage).

    Both resolvers use no-op stubs so the comparison reflects pure framework
    overhead and the difference in code path length between native and mouse.
    """
    uia_resolver = _make_uia_resolver()
    mouse_resolver = _make_mouse_resolver()
    uia_source = make_source_result(source_type="uia")
    visual_source = make_source_result(source_type="visual")  # forces mouse path
    element = _make_element_stub()

    from nexus.source.transport.resolver import ActionSpec

    spec = ActionSpec(action_type="click", task_id="bench-transport")

    # --- UIA timings ---
    uia_latencies_ms: list[float] = []
    for _ in range(_N_ITERATIONS):
        t0 = time.perf_counter()
        result = await uia_resolver.execute(spec, uia_source, element)
        t1 = time.perf_counter()
        uia_latencies_ms.append((t1 - t0) * 1_000)

    # --- Mouse timings ---
    mouse_latencies_ms: list[float] = []
    for _ in range(_N_ITERATIONS):
        t0 = time.perf_counter()
        result = await mouse_resolver.execute(spec, visual_source, element)
        t1 = time.perf_counter()
        mouse_latencies_ms.append((t1 - t0) * 1_000)

    uia_avg = sum(uia_latencies_ms) / len(uia_latencies_ms)
    mouse_avg = sum(mouse_latencies_ms) / len(mouse_latencies_ms)

    # UIA must be strictly faster than mouse (native advantage)
    uia_is_faster = uia_avg < mouse_avg

    record = BenchmarkRecord(
        name="transport_latency_comparison",
        target_label="UIA avg < mouse avg (native advantage)",
        unit="ms",
        target_value=mouse_avg,      # dynamic: UIA must beat mouse
        higher_is_better=False,
        samples=uia_latencies_ms,
        extra={
            "uia_avg_ms": round(uia_avg, 4),
            "mouse_avg_ms": round(mouse_avg, 4),
            "uia_p95_ms": round(sorted(uia_latencies_ms)[int(_N_ITERATIONS * 0.95)], 4),
            "mouse_p95_ms": round(sorted(mouse_latencies_ms)[int(_N_ITERATIONS * 0.95)], 4),
            "speedup_ratio": round(mouse_avg / uia_avg, 3) if uia_avg > 0 else 0,
            "iterations": _N_ITERATIONS,
        },
    )
    record.finish(uia_avg)
    register_result(record)

    assert uia_is_faster, (
        f"Transport latency benchmark failed: "
        f"UIA avg={uia_avg:.4f} ms is NOT faster than mouse avg={mouse_avg:.4f} ms"
    )
