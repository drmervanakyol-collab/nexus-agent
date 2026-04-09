"""
nexus/infra/diagnostic.py
Diagnostic reporter and crash handler for Nexus Agent.

DiagnosticReport
----------------
Value object holding all collected diagnostic artefacts.

DiagnosticReporter
------------------
Collects and exports a ZIP archive containing:
  health_report.json          — HealthChecker.run_all() output
  last_50_logs.jsonl          — most recent 50 structured log lines
  last_10_tasks.json          — most recent 10 task records
  settings_sanitized.json     — NexusSettings minus any key/secret/token fields
  system_info.json            — platform, Python version, timestamp
  last_5_errors_stacktrace.txt— last 5 captured exception stack traces
  transport_audit_last_100.json— last 100 transport audit records

generate() -> DiagnosticReport
  Collect all artefacts into memory.

export_zip(output_path) -> str
  Write the ZIP archive to *output_path* and return the path as a string.

CrashHandler
------------
install() -> None
  Override sys.excepthook so all unhandled exceptions are captured.

handle_crash(exc_type, exc_value, exc_tb) -> None
  Format and store the crash:
    1. Format the stack trace.
    2. Call _write_crash_fn (e.g. DB write).
    3. Call _generate_diagnostic_fn (optional) to export a ZIP.
    4. Print "Tanı dosyası: nexus_diagnostic.zip" to stderr.

All external I/O is injectable for testability.

Injectable callables (DiagnosticReporter)
-----------------------------------------
_get_health_fn       : () -> HealthReport
_get_logs_fn         : (n: int) -> list[str]       — last n JSONL log lines
_get_tasks_fn        : (n: int) -> list[dict]      — last n task dicts
_get_settings_fn     : () -> dict                  — raw settings dict
_get_system_info_fn  : () -> dict                  — system info dict
_get_errors_fn       : () -> list[str]             — last 5 error traces
_get_transport_fn    : (n: int) -> list[dict]      — last n transport rows
_write_zip_fn        : (path: str, files: dict[str,bytes]) -> None
                        — write a ZIP archive; default uses zipfile

Injectable callables (CrashHandler)
-------------------------------------
_write_crash_fn      : (trace: str) -> None        — persist crash record
_generate_diagnostic_fn : (output_path: str) -> str | None — export ZIP
_print_fn            : (text: str) -> None         — stderr output
"""
from __future__ import annotations

import io
import json
import platform
import sys
import traceback
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from nexus.infra.health import HealthChecker, HealthReport
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Keys that must never appear in sanitized settings exports
# ---------------------------------------------------------------------------

_SENSITIVE_SUBSTRINGS: frozenset[str] = frozenset(
    {"key", "secret", "token", "password", "credential", "auth", "apikey"}
)


def _is_sensitive(field_name: str) -> bool:
    lower = field_name.lower()
    return any(sub in lower for sub in _SENSITIVE_SUBSTRINGS)


def sanitize_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively redact any field whose name contains a sensitive substring.

    Sensitive values are replaced with ``"***REDACTED***"``.
    """
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if _is_sensitive(k):
            out[k] = "***REDACTED***"
        elif isinstance(v, dict):
            out[k] = sanitize_settings(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# DiagnosticReport
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticReport:
    """
    In-memory collection of all diagnostic artefacts.

    Attributes
    ----------
    health_report:
        JSON-serialisable dict from HealthChecker.
    logs:
        List of JSONL log line strings (up to 50).
    tasks:
        List of task dicts (up to 10).
    settings_sanitized:
        Settings dict with sensitive fields redacted.
    system_info:
        Platform / Python / timestamp dict.
    errors:
        List of stack-trace strings (up to 5).
    transport_audit:
        List of transport audit record dicts (up to 100).
    """

    health_report: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    settings_sanitized: dict[str, Any] = field(default_factory=dict)
    system_info: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    transport_audit: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default injectable implementations
# ---------------------------------------------------------------------------


def _default_get_health() -> HealthReport:
    return HealthChecker().run_all()


def _default_get_system_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def _default_write_zip(
    path: str, files: dict[str, bytes]
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)


# ---------------------------------------------------------------------------
# DiagnosticReporter
# ---------------------------------------------------------------------------


class DiagnosticReporter:
    """
    Collects system state and exports a diagnostic ZIP archive.

    Parameters
    ----------
    _get_health_fn:
        ``() -> HealthReport``.
    _get_logs_fn:
        ``(n: int) -> list[str]``.  Last *n* JSONL log lines.
    _get_tasks_fn:
        ``(n: int) -> list[dict]``.  Last *n* task records.
    _get_settings_fn:
        ``() -> dict``.  Raw settings as a plain dict.
    _get_system_info_fn:
        ``() -> dict``.  System information.
    _get_errors_fn:
        ``() -> list[str]``.  Last 5 error stack traces.
    _get_transport_fn:
        ``(n: int) -> list[dict]``.  Last *n* transport audit rows.
    _write_zip_fn:
        ``(path: str, files: dict[str, bytes]) -> None``.
    """

    def __init__(
        self,
        *,
        _get_health_fn: Callable[[], HealthReport] | None = None,
        _get_logs_fn: Callable[[int], list[str]] | None = None,
        _get_tasks_fn: Callable[[int], list[dict[str, Any]]] | None = None,
        _get_settings_fn: Callable[[], dict[str, Any]] | None = None,
        _get_system_info_fn: Callable[[], dict[str, Any]] | None = None,
        _get_errors_fn: Callable[[], list[str]] | None = None,
        _get_transport_fn: (
            Callable[[int], list[dict[str, Any]]] | None
        ) = None,
        _write_zip_fn: (
            Callable[[str, dict[str, bytes]], None] | None
        ) = None,
    ) -> None:
        self._get_health = _get_health_fn or _default_get_health
        self._get_logs: Callable[[int], list[str]] = _get_logs_fn or (
            lambda _n: []
        )
        self._get_tasks: Callable[[int], list[dict[str, Any]]] = (
            _get_tasks_fn or (lambda _n: [])
        )
        self._get_settings: Callable[[], dict[str, Any]] = (
            _get_settings_fn or (lambda: {})
        )
        self._get_system_info = _get_system_info_fn or _default_get_system_info
        self._get_errors: Callable[[], list[str]] = _get_errors_fn or (
            lambda: []
        )
        self._get_transport: Callable[[int], list[dict[str, Any]]] = (
            _get_transport_fn or (lambda _n: [])
        )
        self._write_zip = _write_zip_fn or _default_write_zip

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> DiagnosticReport:
        """Collect all diagnostic artefacts into a DiagnosticReport."""
        health = self._get_health()
        health_dict: dict[str, Any] = {
            "overall": health.overall,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "message": c.message,
                    "fix_hint": c.fix_hint,
                }
                for c in health.checks
            ],
        }

        raw_settings = self._get_settings()
        sanitized = sanitize_settings(raw_settings)

        return DiagnosticReport(
            health_report=health_dict,
            logs=self._get_logs(50),
            tasks=self._get_tasks(10),
            settings_sanitized=sanitized,
            system_info=self._get_system_info(),
            errors=self._get_errors()[:5],
            transport_audit=self._get_transport(100),
        )

    def export_zip(self, output_path: str) -> str:
        """
        Generate a DiagnosticReport and write it as a ZIP archive.

        Parameters
        ----------
        output_path:
            Destination file path for the ZIP archive.

        Returns
        -------
        The resolved *output_path* string.
        """
        report = self.generate()
        files = self._build_zip_files(report)
        self._write_zip(output_path, files)
        _log.info("diagnostic_zip_exported", path=output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_zip_files(report: DiagnosticReport) -> dict[str, bytes]:
        """Serialise all report artefacts to bytes for ZIP storage."""

        def _json(obj: Any) -> bytes:
            return json.dumps(obj, indent=2, default=str).encode()

        files: dict[str, bytes] = {
            "health_report.json": _json(report.health_report),
            "last_50_logs.jsonl": "\n".join(report.logs).encode(),
            "last_10_tasks.json": _json(report.tasks),
            "settings_sanitized.json": _json(report.settings_sanitized),
            "system_info.json": _json(report.system_info),
            "last_5_errors_stacktrace.txt": (
                "\n\n" + "=" * 60 + "\n\n"
            ).join(report.errors).encode(),
            "transport_audit_last_100.json": _json(report.transport_audit),
        }
        return files


# ---------------------------------------------------------------------------
# CrashHandler
# ---------------------------------------------------------------------------


class CrashHandler:
    """
    Intercepts unhandled Python exceptions and records a crash report.

    Parameters
    ----------
    _write_crash_fn:
        ``(trace: str) -> None``.  Persist the formatted stack trace.
    _generate_diagnostic_fn:
        ``(output_path: str) -> str | None``.  Optional: export diagnostic ZIP.
    _print_fn:
        ``(text: str) -> None``.  Output for the "Tanı dosyası" message.
    diagnostic_zip_path:
        Destination path for the crash diagnostic ZIP.
    """

    def __init__(
        self,
        *,
        _write_crash_fn: Callable[[str], None] | None = None,
        _generate_diagnostic_fn: (
            Callable[[str], str | None] | None
        ) = None,
        _print_fn: Callable[[str], None] | None = None,
        diagnostic_zip_path: str = "nexus_diagnostic.zip",
    ) -> None:
        self._write_crash = _write_crash_fn or (lambda _trace: None)
        self._generate_diagnostic = _generate_diagnostic_fn or (
            lambda _path: None
        )
        self._print = _print_fn or (lambda t: print(t, file=sys.stderr))
        self._zip_path = diagnostic_zip_path
        self._original_excepthook = sys.excepthook

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Override sys.excepthook to route all unhandled exceptions here."""
        self._installed_hook = self.handle_crash
        sys.excepthook = self._installed_hook
        _log.debug("crash_handler_installed")

    def uninstall(self) -> None:
        """Restore the original sys.excepthook."""
        sys.excepthook = self._original_excepthook
        _log.debug("crash_handler_uninstalled")

    def handle_crash(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: Any,
    ) -> None:
        """
        Handle an unhandled exception.

        Steps
        -----
        1. Format the full stack trace.
        2. Call _write_crash_fn to persist it.
        3. Call _generate_diagnostic_fn to export a ZIP archive.
        4. Print the diagnostic file path to stderr.
        """
        try:
            trace_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            trace = "".join(trace_lines)

            _log.error(
                "unhandled_exception",
                exc_type=exc_type.__name__,
                message=str(exc_value),
            )

            self._write_crash(trace)
            self._generate_diagnostic(self._zip_path)
            self._print(f"Tanı dosyası: {self._zip_path}")
        except Exception:
            # CrashHandler itself must never raise
            pass


# ---------------------------------------------------------------------------
# Convenience: _build_zip_bytes (for tests that need the bytes directly)
# ---------------------------------------------------------------------------


def build_zip_bytes(files: dict[str, bytes]) -> bytes:
    """
    Pack *files* into a ZIP archive and return the raw bytes.

    Useful for tests that want to inspect archive contents without touching
    the filesystem.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()
