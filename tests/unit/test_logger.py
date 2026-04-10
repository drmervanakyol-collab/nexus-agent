"""Unit tests for nexus/infra/logger.py — JSON format and trace injection."""
from __future__ import annotations

import io
import json
import logging

import pytest

from nexus.core.types import TaskId, TraceId
from nexus.infra.logger import configure_logging, get_logger
from nexus.infra.trace import traced

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_stream() -> io.StringIO:
    """Return a fresh StringIO and configure logging to write into it."""
    stream = io.StringIO()
    configure_logging(level=logging.DEBUG, stream=stream)
    return stream


def _last_line(stream: io.StringIO) -> dict:  # type: ignore[type-arg]
    """Parse the last non-empty line of *stream* as JSON."""
    stream.seek(0)
    lines = [ln.strip() for ln in stream.read().splitlines() if ln.strip()]
    assert lines, "No log output captured"
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# JSON parseable
# ---------------------------------------------------------------------------


class TestLoggerJSONFormat:
    def test_output_is_json(self) -> None:
        stream = _capture_stream()
        log = get_logger("nexus.test")
        log.info("hello world")
        record = _last_line(stream)
        assert isinstance(record, dict)

    def test_event_field_present(self) -> None:
        stream = _capture_stream()
        get_logger("nexus.test").info("my_event")
        record = _last_line(stream)
        assert record.get("event") == "my_event"

    def test_level_field_present(self) -> None:
        stream = _capture_stream()
        get_logger("nexus.test").info("lvl_test")
        record = _last_line(stream)
        assert "level" in record

    def test_timestamp_field_present(self) -> None:
        stream = _capture_stream()
        get_logger("nexus.test").info("ts_test")
        record = _last_line(stream)
        assert "timestamp" in record

    def test_timestamp_is_iso8601(self) -> None:
        stream = _capture_stream()
        get_logger("nexus.test").info("iso")
        record = _last_line(stream)
        ts = record["timestamp"]
        assert isinstance(ts, str)
        # ISO-8601 UTC ends with Z or +00:00
        assert ts.endswith("Z") or "+00:00" in ts or ts.endswith("+0000")

    def test_module_field_present(self) -> None:
        stream = _capture_stream()
        get_logger("nexus.infra.capture").info("mod_test")
        record = _last_line(stream)
        assert "logger" in record or "module" in record  # structlog uses "logger"

    def test_extra_fields_included(self) -> None:
        stream = _capture_stream()
        get_logger("nexus.test").info("with_extra", duration_ms=42.5)
        record = _last_line(stream)
        assert record.get("duration_ms") == pytest.approx(42.5)

    def test_multiple_records_each_valid_json(self) -> None:
        stream = _capture_stream()
        log = get_logger("nexus.test")
        log.info("first")
        log.warning("second")
        log.error("third")
        stream.seek(0)
        lines = [ln.strip() for ln in stream.read().splitlines() if ln.strip()]
        assert len(lines) >= 3
        for line in lines:
            json.loads(line)  # must not raise


# ---------------------------------------------------------------------------
# Trace context injection
# ---------------------------------------------------------------------------


class TestLoggerTraceInjection:
    def test_trace_id_none_without_context(self) -> None:
        stream = _capture_stream()
        get_logger("nexus.test").info("no_trace")
        record = _last_line(stream)
        assert record.get("trace_id") == "none"

    def test_task_id_none_without_context(self) -> None:
        stream = _capture_stream()
        get_logger("nexus.test").info("no_task")
        record = _last_line(stream)
        assert record.get("task_id") == "none"

    def test_trace_id_injected_from_context(self) -> None:
        stream = _capture_stream()
        with traced("phase", trace_id=TraceId("trace-abc")):
            get_logger("nexus.test").info("in_trace")
        record = _last_line(stream)
        assert record.get("trace_id") == "trace-abc"

    def test_task_id_injected_from_context(self) -> None:
        stream = _capture_stream()
        with traced("phase", task_id=TaskId("task-xyz")):
            get_logger("nexus.test").info("in_task")
        record = _last_line(stream)
        assert record.get("task_id") == "task-xyz"

    def test_ids_reset_after_context_exits(self) -> None:
        stream = _capture_stream()
        with traced("phase", trace_id=TraceId("tr-1"), task_id=TaskId("tk-1")):
            pass
        get_logger("nexus.test").info("after_trace")
        record = _last_line(stream)
        assert record.get("trace_id") == "none"
        assert record.get("task_id") == "none"


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_returns_bound_logger(self) -> None:
        log = get_logger("nexus.test")
        assert hasattr(log, "info")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")

    def test_different_names_independent(self) -> None:
        a = get_logger("nexus.a")
        b = get_logger("nexus.b")
        assert a is not b
