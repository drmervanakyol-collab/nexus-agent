"""
nexus/infra/trace.py
Async-safe trace context using contextvars.

Usage
-----
async def my_coroutine():
    async with traced("phase_name", task_id=TaskId("t-1")) as ctx:
        ...  # ctx.trace_id and ctx.task_id are set

# Or as a sync context manager:
with traced("phase_name") as ctx:
    ...
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from nexus.core.types import TaskId, TraceId

# ---------------------------------------------------------------------------
# Context variable — one slot per async task / thread
# ---------------------------------------------------------------------------

_CURRENT_TRACE: ContextVar[TraceContext | None] = ContextVar(
    "_CURRENT_TRACE", default=None
)


# ---------------------------------------------------------------------------
# TraceContext
# ---------------------------------------------------------------------------


@dataclass
class TraceContext:
    """Immutable snapshot of the current trace."""

    trace_id: TraceId
    task_id: TaskId
    phase: str = ""
    _token: object = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def start(
        cls,
        phase: str = "",
        *,
        task_id: TaskId | None = None,
        trace_id: TraceId | None = None,
    ) -> TraceContext:
        """Create and activate a new TraceContext."""
        ctx = cls(
            trace_id=trace_id or TraceId(str(uuid.uuid4())),
            task_id=task_id or TaskId(""),
            phase=phase,
        )
        ctx._token = _CURRENT_TRACE.set(ctx)
        return ctx

    def stop(self) -> None:
        """Restore the previous TraceContext."""
        if self._token is not None:
            _CURRENT_TRACE.reset(self._token)  # type: ignore[arg-type]
            self._token = None

    # ------------------------------------------------------------------
    # Current-context accessor
    # ------------------------------------------------------------------

    @staticmethod
    def current() -> TraceContext | None:
        """Return the active TraceContext for this task/thread, or None."""
        return _CURRENT_TRACE.get()


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


@contextmanager
def traced(
    phase: str = "",
    *,
    task_id: TaskId | None = None,
    trace_id: TraceId | None = None,
) -> Iterator[TraceContext]:
    """Sync context manager that activates a TraceContext."""
    ctx = TraceContext.start(phase, task_id=task_id, trace_id=trace_id)
    try:
        yield ctx
    finally:
        ctx.stop()


@asynccontextmanager
async def async_traced(
    phase: str = "",
    *,
    task_id: TaskId | None = None,
    trace_id: TraceId | None = None,
) -> AsyncIterator[TraceContext]:
    """Async context manager that activates a TraceContext."""
    ctx = TraceContext.start(phase, task_id=task_id, trace_id=trace_id)
    try:
        yield ctx
    finally:
        ctx.stop()
