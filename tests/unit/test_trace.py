"""Unit tests for nexus/infra/trace.py — async-safe TraceContext."""
from __future__ import annotations

import asyncio
import uuid

import pytest

from nexus.core.types import TaskId, TraceId
from nexus.infra.trace import TraceContext, async_traced, traced


# ---------------------------------------------------------------------------
# TraceContext.start / stop
# ---------------------------------------------------------------------------


class TestTraceContextBasics:
    def test_start_sets_current(self) -> None:
        ctx = TraceContext.start("test")
        try:
            assert TraceContext.current() is ctx
        finally:
            ctx.stop()

    def test_stop_restores_none(self) -> None:
        ctx = TraceContext.start("test")
        ctx.stop()
        assert TraceContext.current() is None

    def test_start_generates_trace_id(self) -> None:
        ctx = TraceContext.start()
        try:
            assert ctx.trace_id
            uuid.UUID(ctx.trace_id)  # must be valid UUID
        finally:
            ctx.stop()

    def test_explicit_trace_id(self) -> None:
        tid = TraceId("fixed-id-123")
        ctx = TraceContext.start(trace_id=tid)
        try:
            assert ctx.trace_id == "fixed-id-123"
        finally:
            ctx.stop()

    def test_explicit_task_id(self) -> None:
        ctx = TraceContext.start(task_id=TaskId("task-99"))
        try:
            assert ctx.task_id == "task-99"
        finally:
            ctx.stop()

    def test_default_task_id_empty(self) -> None:
        ctx = TraceContext.start()
        try:
            assert ctx.task_id == ""
        finally:
            ctx.stop()

    def test_phase_stored(self) -> None:
        ctx = TraceContext.start("capture")
        try:
            assert ctx.phase == "capture"
        finally:
            ctx.stop()

    def test_stop_twice_is_safe(self) -> None:
        ctx = TraceContext.start()
        ctx.stop()
        ctx.stop()  # second stop must not raise

    def test_no_current_returns_none(self) -> None:
        assert TraceContext.current() is None


# ---------------------------------------------------------------------------
# traced() sync context manager
# ---------------------------------------------------------------------------


class TestTracedContextManager:
    def test_sets_and_restores(self) -> None:
        assert TraceContext.current() is None
        with traced("phase_a") as ctx:
            assert TraceContext.current() is ctx
            assert ctx.phase == "phase_a"
        assert TraceContext.current() is None

    def test_yields_context(self) -> None:
        with traced("p", task_id=TaskId("t1")) as ctx:
            assert isinstance(ctx, TraceContext)
            assert ctx.task_id == "t1"

    def test_restores_on_exception(self) -> None:
        with pytest.raises(RuntimeError):
            with traced("err_phase"):
                raise RuntimeError("boom")
        assert TraceContext.current() is None

    def test_nested_traces(self) -> None:
        with traced("outer", trace_id=TraceId("outer-id")) as outer:
            assert TraceContext.current() is outer
            with traced("inner", trace_id=TraceId("inner-id")) as inner:
                assert TraceContext.current() is inner
            # inner stopped → outer restored
            assert TraceContext.current() is outer
        assert TraceContext.current() is None

    def test_explicit_trace_id_in_cm(self) -> None:
        with traced(trace_id=TraceId("stable-123")) as ctx:
            assert ctx.trace_id == "stable-123"


# ---------------------------------------------------------------------------
# async_traced() context manager
# ---------------------------------------------------------------------------


class TestAsyncTraced:
    async def test_sets_and_restores(self) -> None:
        assert TraceContext.current() is None
        async with async_traced("async_phase") as ctx:
            assert TraceContext.current() is ctx
            assert ctx.phase == "async_phase"
        assert TraceContext.current() is None

    async def test_yields_context(self) -> None:
        async with async_traced("p", task_id=TaskId("at1")) as ctx:
            assert ctx.task_id == "at1"

    async def test_restores_on_exception(self) -> None:
        with pytest.raises(ValueError):
            async with async_traced("err"):
                raise ValueError("async boom")
        assert TraceContext.current() is None

    async def test_explicit_ids(self) -> None:
        async with async_traced(
            trace_id=TraceId("tr-1"), task_id=TaskId("tk-1")
        ) as ctx:
            assert ctx.trace_id == "tr-1"
            assert ctx.task_id == "tk-1"


# ---------------------------------------------------------------------------
# Async isolation between concurrent tasks
# ---------------------------------------------------------------------------


class TestAsyncIsolation:
    async def test_independent_contexts_per_task(self) -> None:
        """Two concurrent coroutines must each see their own TraceContext."""
        results: dict[str, str] = {}

        async def worker(name: str) -> None:
            async with async_traced(name, trace_id=TraceId(name)) as ctx:
                # Yield to the event loop so the other task can run.
                await asyncio.sleep(0)
                results[name] = ctx.trace_id

        await asyncio.gather(worker("alpha"), worker("beta"))

        assert results["alpha"] == "alpha"
        assert results["beta"] == "beta"

    async def test_no_leak_after_task(self) -> None:
        async def inner() -> None:
            async with async_traced("transient"):
                pass

        await inner()
        # After the coroutine finishes its own context should be gone.
        # The *current* context in THIS task is still None.
        assert TraceContext.current() is None

    async def test_many_concurrent_tasks(self) -> None:
        n = 50
        seen: set[str] = set()
        lock = asyncio.Lock()

        async def task(i: int) -> None:
            tid = TraceId(f"t-{i}")
            async with async_traced(trace_id=tid):
                await asyncio.sleep(0)
                current = TraceContext.current()
                assert current is not None
                assert current.trace_id == f"t-{i}"
                async with lock:
                    seen.add(current.trace_id)

        await asyncio.gather(*(task(i) for i in range(n)))
        assert len(seen) == n
