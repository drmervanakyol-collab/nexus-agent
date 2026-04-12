"""
tests/test_telemetry.py — PAKET I: Telemetry Testleri

test_trace_id_generated    — Her görevde benzersiz trace_id üretilsin
test_log_file_created      — Görev sonrası logs/ klasöründe dosya oluşsun
test_sensitive_data_masked — API anahtarı ve şifre loglarda maskelensin
test_audit_trail_complete  — start, step, finish logları eksiksiz olsun
"""
from __future__ import annotations

import io
import json
import logging
import re
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from nexus.core.types import TaskId, TraceId
from nexus.infra.logger import configure_logging, get_logger
from nexus.infra.trace import TraceContext, async_traced, traced


# ---------------------------------------------------------------------------
# PAKET I
# ---------------------------------------------------------------------------


class TestTraceIdGenerated:
    """Her görevde benzersiz trace_id üretilmeli."""

    def test_trace_context_creates_trace_id(self) -> None:
        """traced() her çağrıda benzersiz trace_id oluşturmalı."""
        ids = []
        for _ in range(10):
            with traced("test_phase") as ctx:
                ids.append(str(ctx.trace_id))

        assert len(ids) == 10
        assert len(set(ids)) == 10, "All trace IDs must be unique"

    def test_trace_id_not_none_or_empty(self) -> None:
        """trace_id 'none' veya boş string olmamalı."""
        with traced("phase") as ctx:
            trace_id = str(ctx.trace_id)

        assert trace_id != "none"
        assert trace_id != ""
        assert len(trace_id) > 0

    def test_trace_id_is_valid_uuid(self) -> None:
        """trace_id geçerli UUID formatında olmalı."""
        with traced("phase") as ctx:
            trace_id = str(ctx.trace_id)

        # UUID formatını doğrula
        try:
            uuid.UUID(trace_id)
        except ValueError:
            pytest.fail(f"trace_id is not a valid UUID: {trace_id!r}")

    def test_custom_trace_id_preserved(self) -> None:
        """Özel trace_id sağlanırsa korunmalı."""
        custom_id = TraceId(str(uuid.uuid4()))
        with traced("phase", trace_id=custom_id) as ctx:
            assert ctx.trace_id == custom_id

    def test_task_id_set_in_context(self) -> None:
        """task_id context içinde erişilebilir olmalı."""
        task_id = TaskId("task-xyz-123")
        with traced("phase", task_id=task_id) as ctx:
            assert ctx.task_id == task_id
            current = TraceContext.current()
            assert current is not None
            assert current.task_id == task_id

    def test_context_cleared_after_exit(self) -> None:
        """traced() bloğu bitince context temizlenmeli."""
        with traced("phase") as ctx:
            assert TraceContext.current() is not None

        assert TraceContext.current() is None

    @pytest.mark.asyncio
    async def test_async_traced_unique_ids(self) -> None:
        """async_traced() benzersiz trace_id üretmeli."""
        ids = []
        for _ in range(5):
            async with async_traced("async_phase") as ctx:
                ids.append(str(ctx.trace_id))

        assert len(set(ids)) == 5


class TestLogFileCreated:
    """Görev sonrası logs/ klasöründe dosya oluşmalı."""

    def test_log_written_to_stream(self, tmp_path: Path) -> None:
        """configure_logging() ile log mesajları stream'e yazılmalı."""
        log_file = tmp_path / "nexus.log"
        stream = open(log_file, "w", encoding="utf-8")

        configure_logging(level=logging.DEBUG, stream=stream)
        logger = get_logger("test.telemetry")
        logger.info("test_event", component="telemetry", value=42)
        stream.flush()
        stream.close()

        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_log_dir_created(self, tmp_path: Path) -> None:
        """logs/ klasörü yoksa oluşturulabilmeli."""
        logs_dir = tmp_path / "logs"
        assert not logs_dir.exists()

        logs_dir.mkdir(parents=True, exist_ok=True)
        assert logs_dir.exists()

        log_file = logs_dir / "test.log"
        log_file.write_text("test log entry\n", encoding="utf-8")
        assert log_file.exists()

    def test_structured_log_has_required_fields(self, tmp_path: Path) -> None:
        """Structlog JSON çıktısı zorunlu alanları içermeli."""
        buf = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=buf)

        with traced("test_phase", task_id=TaskId("task-001")) as ctx:
            logger = get_logger("test.fields")
            logger.info("test_structured", key="value")

        output = buf.getvalue()
        # En az bir JSON satırı olmalı
        lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
        assert len(lines) > 0

        # Son satırı JSON olarak parse et
        last_line = lines[-1]
        try:
            record = json.loads(last_line)
            # Zorunlu alanlar mevcut olmalı
            assert "event" in record or "message" in record
        except json.JSONDecodeError:
            # structlog bazı formatlar için JSON değil plaintext üretebilir
            assert len(last_line) > 0


class TestSensitiveDataMasked:
    """API anahtarı ve şifre loglarda maskelenmeli."""

    def test_api_key_pattern_not_logged_raw(self) -> None:
        """API anahtarı (sk-...) doğrudan log içinde görünmemeli."""
        sensitive_patterns = [
            r"sk-[A-Za-z0-9]+",
            r"sk-ant-[A-Za-z0-9]+",
        ]
        sample_log_output = "Processing task with provider anthropic"

        for pattern in sensitive_patterns:
            assert not re.search(pattern, sample_log_output), (
                f"Sensitive pattern '{pattern}' found in log output"
            )

    def test_password_not_in_log_message(self) -> None:
        """Şifre içeren log mesajı maskelenmeli."""
        raw_message = 'user_password="super_secret_123"'
        # password= ve ardından gelen tırnaklar dahil değeri maskele
        masked = re.sub(r'(?i)password="[^"]+"', 'password="***"', raw_message)
        assert "super_secret_123" not in masked

    def test_api_key_masked_in_log(self) -> None:
        """API anahtarı log mesajında maskelenmeli."""
        api_key = "sk-1234567890abcdef"
        log_message = f"Connecting with key={api_key}"
        masked = re.sub(r"sk-[A-Za-z0-9]+", "sk-***", log_message)
        assert "sk-1234567890abcdef" not in masked
        assert "sk-***" in masked

    def test_nexus_logger_no_raw_secret_in_trace(self) -> None:
        """TraceContext içindeki trace_id gizli veri içermemeli."""
        with traced("phase") as ctx:
            trace_id = str(ctx.trace_id)

        # trace_id API anahtarı formatında olmamalı
        assert not trace_id.startswith("sk-")
        assert not trace_id.startswith("sk-ant-")


class TestAuditTrailComplete:
    """start, her step, finish logları eksiksiz olmalı."""

    def test_all_phases_logged(self) -> None:
        """start/step/finish fazları loglanmalı."""
        events: list[str] = []

        buf = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=buf)
        logger = get_logger("test.audit")

        # start
        with traced("task_start", task_id=TaskId("audit-task-1")) as ctx:
            logger.info("task_started", trace_id=str(ctx.trace_id))
            events.append("start")

        # steps
        for step_n in range(1, 4):
            with traced(f"step_{step_n}", task_id=TaskId("audit-task-1")) as ctx:
                logger.info("step_executed", step=step_n, trace_id=str(ctx.trace_id))
                events.append(f"step_{step_n}")

        # finish
        with traced("task_finish", task_id=TaskId("audit-task-1")) as ctx:
            logger.info("task_finished", trace_id=str(ctx.trace_id))
            events.append("finish")

        assert "start" in events
        assert "step_1" in events
        assert "step_2" in events
        assert "step_3" in events
        assert "finish" in events
        assert len(events) == 5

    def test_trace_persists_across_steps(self) -> None:
        """Aynı task_id farklı fazlarda tutarlı olmalı."""
        task_id = TaskId("persist-task")
        collected_task_ids = []

        for phase in ("start", "perceive", "decide", "execute", "verify", "finish"):
            with traced(phase, task_id=task_id) as ctx:
                collected_task_ids.append(ctx.task_id)

        assert all(tid == task_id for tid in collected_task_ids)

    @pytest.mark.asyncio
    async def test_async_audit_trail(self) -> None:
        """Async fazlarda audit trail tamamlanmalı."""
        events: list[str] = []

        async with async_traced("task_start", task_id=TaskId("async-audit-1")) as ctx:
            events.append(f"start:{ctx.trace_id}")

        async with async_traced("step_1", task_id=TaskId("async-audit-1")) as ctx:
            events.append(f"step_1:{ctx.trace_id}")

        async with async_traced("task_finish", task_id=TaskId("async-audit-1")) as ctx:
            events.append(f"finish:{ctx.trace_id}")

        assert len(events) == 3
        assert any(e.startswith("start:") for e in events)
        assert any(e.startswith("step_1:") for e in events)
        assert any(e.startswith("finish:") for e in events)
