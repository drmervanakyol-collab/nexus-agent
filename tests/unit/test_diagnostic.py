"""
tests/unit/test_diagnostic.py
Unit tests for nexus/infra/diagnostic.py — Faz 58.

TEST PLAN
---------
sanitize_settings:
  1.  Top-level sensitive key redacted.
  2.  Nested sensitive key redacted.
  3.  Non-sensitive key preserved.
  4.  Multiple sensitive substrings: "api_key", "secret", "token",
      "password", "credential" — all redacted.
  5.  Case-insensitive match ("API_KEY", "Secret").

DiagnosticReporter.generate:
  6.  health_report populated with overall + checks.
  7.  logs up to 50 entries collected.
  8.  tasks up to 10 entries collected.
  9.  settings_sanitized: sensitive fields removed.
  10. system_info present (has at least "platform" key).
  11. errors up to 5 entries collected.
  12. transport_audit up to 100 entries collected.

DiagnosticReporter.export_zip:
  13. ZIP is written via _write_zip_fn.
  14. ZIP contains expected filenames.
  15. API key does NOT appear in any ZIP file content.
  16. transport_audit_last_100.json is present in ZIP.
  17. last_5_errors_stacktrace.txt is present in ZIP.

CrashHandler.install / handle_crash:
  18. install() overrides sys.excepthook.
  19. handle_crash: _write_crash_fn called with formatted trace.
  20. handle_crash: _generate_diagnostic_fn called with zip path.
  21. handle_crash: print contains "Tanı dosyası".
  22. handle_crash: internal error in _write_crash_fn does not propagate.

build_zip_bytes:
  23. Returns bytes that form a valid ZIP archive.
  24. All provided files are accessible inside the archive.
"""
from __future__ import annotations

import io
import json
import sys
import traceback
import zipfile
from typing import Any

import pytest

from nexus.infra.diagnostic import (
    CrashHandler,
    DiagnosticReport,
    DiagnosticReporter,
    build_zip_bytes,
    sanitize_settings,
)
from nexus.infra.health import CheckResult, HealthReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_API_KEY = "sk-ant-secret-abc123"


def _ok_report() -> HealthReport:
    return HealthReport(
        checks=[
            CheckResult(
                name="CHECK_PYTHON_VERSION",
                status="ok",
                message="Python 3.14 — OK",
            )
        ]
    )


def _reporter(
    *,
    health: HealthReport | None = None,
    logs: list[str] | None = None,
    tasks: list[dict] | None = None,
    settings: dict[str, Any] | None = None,
    system_info: dict[str, Any] | None = None,
    errors: list[str] | None = None,
    transport: list[dict] | None = None,
    written: dict[str, tuple[str, dict[str, bytes]]] | None = None,
) -> DiagnosticReporter:
    """Build a DiagnosticReporter backed entirely by stubs."""
    _written = written if written is not None else {}

    def _write_zip(path: str, files: dict[str, bytes]) -> None:
        _written["last"] = (path, files)

    return DiagnosticReporter(
        _get_health_fn=lambda: health or _ok_report(),
        _get_logs_fn=lambda n: (logs or [])[:n],
        _get_tasks_fn=lambda n: (tasks or [])[:n],
        _get_settings_fn=lambda: settings or {},
        _get_system_info_fn=lambda: system_info or {"platform": "TestOS"},
        _get_errors_fn=lambda: errors or [],
        _get_transport_fn=lambda n: (transport or [])[:n],
        _write_zip_fn=_write_zip,
    )


# ---------------------------------------------------------------------------
# sanitize_settings
# ---------------------------------------------------------------------------


class TestSanitizeSettings:
    def test_top_level_key_redacted(self):
        raw = {"api_key": "sk-secret", "timeout": 30}
        result = sanitize_settings(raw)
        assert result["api_key"] == "***REDACTED***"
        assert result["timeout"] == 30

    def test_nested_sensitive_key_redacted(self):
        raw = {"cloud": {"openai_api_key": "sk-oai-123", "model": "gpt-4o"}}
        result = sanitize_settings(raw)
        assert result["cloud"]["openai_api_key"] == "***REDACTED***"
        assert result["cloud"]["model"] == "gpt-4o"

    def test_non_sensitive_key_preserved(self):
        raw = {"fps": 15, "dry_run_mode": False, "log_dir": "logs"}
        result = sanitize_settings(raw)
        assert result == raw

    @pytest.mark.parametrize(
        "field_name",
        ["api_key", "secret", "token", "password", "credential", "auth_header"],
    )
    def test_sensitive_substrings_all_redacted(self, field_name: str):
        result = sanitize_settings({field_name: "SENSITIVE_VALUE"})
        assert result[field_name] == "***REDACTED***"

    def test_case_insensitive(self):
        raw = {"API_KEY": "sk-upper", "Secret": "s3cr3t"}
        result = sanitize_settings(raw)
        assert result["API_KEY"] == "***REDACTED***"
        assert result["Secret"] == "***REDACTED***"


# ---------------------------------------------------------------------------
# DiagnosticReporter.generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_health_report_populated(self):
        rep = _reporter()
        report = rep.generate()
        assert "overall" in report.health_report
        assert "checks" in report.health_report

    def test_logs_collected(self):
        logs = [f"log_{i}" for i in range(60)]
        rep = _reporter(logs=logs)
        report = rep.generate()
        assert len(report.logs) == 50

    def test_tasks_collected(self):
        tasks = [{"id": f"t{i}"} for i in range(15)]
        rep = _reporter(tasks=tasks)
        report = rep.generate()
        assert len(report.tasks) == 10

    def test_settings_sanitized(self):
        settings = {"api_key": _FAKE_API_KEY, "fps": 15}
        rep = _reporter(settings=settings)
        report = rep.generate()
        assert report.settings_sanitized["api_key"] == "***REDACTED***"
        assert report.settings_sanitized["fps"] == 15

    def test_system_info_present(self):
        rep = _reporter(system_info={"platform": "Windows", "python_version": "3.14"})
        report = rep.generate()
        assert "platform" in report.system_info

    def test_errors_collected(self):
        errors = ["Traceback (most recent call last):\n  ...\nValueError: bad"] * 7
        rep = _reporter(errors=errors)
        report = rep.generate()
        assert len(report.errors) == 5

    def test_transport_audit_collected(self):
        transport = [{"id": i, "method": "uia"} for i in range(120)]
        rep = _reporter(transport=transport)
        report = rep.generate()
        assert len(report.transport_audit) == 100


# ---------------------------------------------------------------------------
# DiagnosticReporter.export_zip
# ---------------------------------------------------------------------------


_EXPECTED_FILES = {
    "health_report.json",
    "last_50_logs.jsonl",
    "last_10_tasks.json",
    "settings_sanitized.json",
    "system_info.json",
    "last_5_errors_stacktrace.txt",
    "transport_audit_last_100.json",
}


class TestExportZip:
    def test_write_zip_called(self):
        written: dict = {}
        rep = _reporter(written=written)
        rep.export_zip("output.zip")
        assert "last" in written
        assert written["last"][0] == "output.zip"

    def test_expected_filenames_present(self):
        written: dict = {}
        rep = _reporter(written=written)
        rep.export_zip("output.zip")
        _, files = written["last"]
        assert set(files.keys()) == _EXPECTED_FILES

    def test_api_key_not_in_zip_content(self):
        """No ZIP file content should contain the raw API key."""
        settings = {
            "api_key": _FAKE_API_KEY,
            "nested": {"token": _FAKE_API_KEY},
            "fps": 15,
        }
        written: dict = {}
        rep = _reporter(settings=settings, written=written)
        rep.export_zip("output.zip")
        _, files = written["last"]
        for filename, content_bytes in files.items():
            content = content_bytes.decode(errors="replace")
            assert _FAKE_API_KEY not in content, (
                f"API key leaked into {filename!r}"
            )

    def test_transport_audit_in_zip(self):
        transport = [{"id": i, "method": "dom"} for i in range(10)]
        written: dict = {}
        rep = _reporter(transport=transport, written=written)
        rep.export_zip("output.zip")
        _, files = written["last"]
        assert "transport_audit_last_100.json" in files
        rows = json.loads(files["transport_audit_last_100.json"])
        assert len(rows) == 10

    def test_errors_stacktrace_in_zip(self):
        errors = ["Traceback:\n  File x.py line 1\nRuntimeError: boom"]
        written: dict = {}
        rep = _reporter(errors=errors, written=written)
        rep.export_zip("output.zip")
        _, files = written["last"]
        assert "last_5_errors_stacktrace.txt" in files
        text = files["last_5_errors_stacktrace.txt"].decode()
        assert "RuntimeError" in text


# ---------------------------------------------------------------------------
# CrashHandler
# ---------------------------------------------------------------------------


class TestCrashHandler:
    def test_install_overrides_excepthook(self):
        original = sys.excepthook
        handler = CrashHandler(_print_fn=lambda _: None)
        try:
            handler.install()
            assert sys.excepthook is handler._installed_hook
        finally:
            handler.uninstall()
            assert sys.excepthook is original

    def test_handle_crash_calls_write_crash(self):
        traces: list[str] = []
        handler = CrashHandler(
            _write_crash_fn=lambda t: traces.append(t),
            _print_fn=lambda _: None,
        )
        try:
            raise ValueError("test crash")
        except ValueError:
            exc_type, exc_value, exc_tb = sys.exc_info()
            handler.handle_crash(exc_type, exc_value, exc_tb)

        assert len(traces) == 1
        assert "ValueError" in traces[0]
        assert "test crash" in traces[0]

    def test_handle_crash_calls_generate_diagnostic(self):
        diag_calls: list[str] = []
        handler = CrashHandler(
            _generate_diagnostic_fn=lambda path: diag_calls.append(path) or path,
            _print_fn=lambda _: None,
            diagnostic_zip_path="crash_diag.zip",
        )
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            handler.handle_crash(*sys.exc_info())

        assert diag_calls == ["crash_diag.zip"]

    def test_handle_crash_prints_diagnostic_path(self):
        printed: list[str] = []
        handler = CrashHandler(
            _print_fn=lambda t: printed.append(t),
            diagnostic_zip_path="nexus_diagnostic.zip",
        )
        try:
            raise OSError("io error")
        except OSError:
            handler.handle_crash(*sys.exc_info())

        assert any("Tanı dosyası" in line for line in printed)
        assert any("nexus_diagnostic.zip" in line for line in printed)

    def test_internal_error_in_write_does_not_propagate(self):
        """CrashHandler must never raise, even if its own callbacks fail."""

        def _bad_write(trace: str) -> None:
            raise RuntimeError("write failed")

        handler = CrashHandler(
            _write_crash_fn=_bad_write,
            _print_fn=lambda _: None,
        )
        try:
            raise ValueError("original")
        except ValueError:
            # Should complete silently
            handler.handle_crash(*sys.exc_info())


# ---------------------------------------------------------------------------
# build_zip_bytes
# ---------------------------------------------------------------------------


class TestBuildZipBytes:
    def test_returns_valid_zip(self):
        files = {
            "hello.txt": b"hello world",
            "data.json": b'{"key": "value"}',
        }
        raw = build_zip_bytes(files)
        assert isinstance(raw, bytes)
        assert len(raw) > 0
        # Must be readable as a ZIP
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            assert set(zf.namelist()) == {"hello.txt", "data.json"}

    def test_file_contents_accessible(self):
        files = {"readme.txt": b"content here"}
        raw = build_zip_bytes(files)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            assert zf.read("readme.txt") == b"content here"
