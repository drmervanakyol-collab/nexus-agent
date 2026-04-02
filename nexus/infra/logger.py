"""
nexus/infra/logger.py
Structured JSON logger for Nexus Agent.

Every log record includes:
  timestamp  — ISO-8601 UTC
  level      — debug / info / warning / error / critical
  module     — dotted Python module name
  trace_id   — active TraceContext id (or "none")
  task_id    — active TraceContext task id (or "none")
  event      — human-readable message
  duration_ms — optional float, present when caller supplies it
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from nexus.infra.trace import TraceContext

# ---------------------------------------------------------------------------
# stdlib → structlog bridge
# ---------------------------------------------------------------------------


def _add_nexus_fields(
    logger: Any,  # noqa: ANN401
    method: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Inject trace_id / task_id from the active TraceContext."""
    ctx = TraceContext.current()
    event_dict.setdefault("trace_id", ctx.trace_id if ctx else "none")
    event_dict.setdefault("task_id", ctx.task_id if ctx else "none")
    return event_dict


def configure_logging(*, level: int = logging.DEBUG, stream: Any = None) -> None:
    """
    Configure structlog for JSON output.

    Call once at application startup.  Safe to call multiple times (idempotent
    because structlog replaces its own configuration each call).
    """
    shared_processors: list[Any] = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_nexus_fields,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog BoundLogger bound to *name* as the module field."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
