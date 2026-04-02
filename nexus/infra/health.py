"""
nexus/infra/health.py
System health checker for Nexus Agent.

Each check is implemented as a module-level probe function so that tests
can patch individual probes without mocking entire modules.

Checks
------
CHECK_PYTHON_VERSION      — Python >= 3.11
CHECK_WINDOWS_VERSION     — Windows 10 / 11
CHECK_DISK_SPACE          — >= 2 GB free on working directory
CHECK_RAM                 — >= 4 GB physical RAM
CHECK_DPI_AWARENESS       — DPI-aware process (SHCORE accessible)
CHECK_DB_ACCESSIBLE       — SQLite DB path is readable/writable
CHECK_TESSERACT_BINARY    — tesseract binary on PATH
CHECK_DXCAM               — dxcam importable
CHECK_WRITE_PERMISSION    — write permission on log/work directory
CHECK_CREDENTIAL_MANAGER  — Windows Credential Manager accessible

Status levels
-------------
ok   — requirement met
warn — partially met or degraded (agent can start but may misbehave)
fail — hard requirement missing (agent cannot function)
"""
from __future__ import annotations

import platform
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Literal

CheckStatus = Literal["ok", "warn", "fail"]
Overall = Literal["ok", "warn", "fail"]

# ---------------------------------------------------------------------------
# Check name constants
# ---------------------------------------------------------------------------

CHECK_PYTHON_VERSION = "CHECK_PYTHON_VERSION"
CHECK_WINDOWS_VERSION = "CHECK_WINDOWS_VERSION"
CHECK_DISK_SPACE = "CHECK_DISK_SPACE"
CHECK_RAM = "CHECK_RAM"
CHECK_DPI_AWARENESS = "CHECK_DPI_AWARENESS"
CHECK_DB_ACCESSIBLE = "CHECK_DB_ACCESSIBLE"
CHECK_TESSERACT_BINARY = "CHECK_TESSERACT_BINARY"
CHECK_DXCAM = "CHECK_DXCAM"
CHECK_WRITE_PERMISSION = "CHECK_WRITE_PERMISSION"
CHECK_CREDENTIAL_MANAGER = "CHECK_CREDENTIAL_MANAGER"

_ALL_CHECKS = [
    CHECK_PYTHON_VERSION,
    CHECK_WINDOWS_VERSION,
    CHECK_DISK_SPACE,
    CHECK_RAM,
    CHECK_DPI_AWARENESS,
    CHECK_DB_ACCESSIBLE,
    CHECK_TESSERACT_BINARY,
    CHECK_DXCAM,
    CHECK_WRITE_PERMISSION,
    CHECK_CREDENTIAL_MANAGER,
]

# Thresholds
_MIN_DISK_BYTES: int = 2 * 1024 ** 3   # 2 GB
_MIN_RAM_BYTES: int = 4 * 1024 ** 3    # 4 GB
_MIN_PYTHON: tuple[int, int] = (3, 11)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    fix_hint: str = ""


@dataclass
class HealthReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def overall(self) -> Overall:
        statuses = {r.status for r in self.checks}
        if "fail" in statuses:
            return "fail"
        if "warn" in statuses:
            return "warn"
        return "ok"

    @property
    def exit_code(self) -> int:
        """0 = ok, 1 = warn, 2 = fail."""
        return {"ok": 0, "warn": 1, "fail": 2}[self.overall]

    def by_name(self, name: str) -> CheckResult | None:
        return next((r for r in self.checks if r.name == name), None)


# ---------------------------------------------------------------------------
# Module-level probe functions (patch these in tests)
# ---------------------------------------------------------------------------


def _probe_python_version() -> tuple[int, int]:
    vi = sys.version_info
    return (vi.major, vi.minor)


def _probe_platform_system() -> str:
    return platform.system()


def _probe_windows_release() -> str:
    return platform.release()


def _probe_windows_version_str() -> str:
    return platform.version()


def _probe_disk_free_bytes(path: str) -> int:
    return shutil.disk_usage(path).free


def _probe_ram_bytes() -> int:
    """Return total physical RAM in bytes via Windows kernel32."""
    if sys.platform != "win32":
        return 0
    import ctypes  # noqa: PLC0415

    class _MEMSTATEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = _MEMSTATEX()
    stat.dwLength = ctypes.sizeof(stat)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullTotalPhys)


def _probe_dpi_awareness() -> int | None:
    """Return DPI awareness level (0=unaware, 1=sys, 2=per-monitor) or None."""
    if sys.platform != "win32":
        return None
    import ctypes  # noqa: PLC0415

    try:
        awareness = ctypes.c_int(-1)
        ret = ctypes.windll.shcore.GetProcessDpiAwareness(
            0, ctypes.byref(awareness)
        )
        return int(awareness.value) if ret == 0 else None
    except (OSError, AttributeError):
        return None


def _probe_tesseract_path() -> str | None:
    return shutil.which("tesseract")


def _probe_dxcam_importable() -> bool:
    try:
        import dxcam  # type: ignore[import-untyped]  # noqa: F401, PLC0415
        return True
    except Exception:
        return False


def _probe_write_permission(path: str) -> bool:
    try:
        with tempfile.NamedTemporaryFile(dir=path, delete=True):
            pass
        return True
    except Exception:
        return False


def _probe_db_accessible(db_path: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1")
        conn.close()
        return True
    except Exception:
        return False


def _probe_credential_manager() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import win32cred  # noqa: PLC0415
        win32cred.CredEnumerate(None, 0)
        return True
    except Exception:
        # win32cred accessible but no creds stored is still OK
        try:
            import win32cred  # type: ignore[import-untyped]  # noqa: F401, PLC0415
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# HealthChecker
# ---------------------------------------------------------------------------


class HealthChecker:
    """
    Runs all system health checks and returns a HealthReport.

    Parameters
    ----------
    db_path:
        SQLite database path to probe.  Defaults to ``"nexus.db"``.
    write_dir:
        Directory to probe for write permission.  Defaults to ``"."``.
    """

    def __init__(
        self,
        db_path: str = "nexus.db",
        write_dir: str = ".",
    ) -> None:
        self._db_path = db_path
        self._write_dir = write_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> HealthReport:
        """Execute all checks and return a consolidated HealthReport."""
        report = HealthReport()
        for name in _ALL_CHECKS:
            report.checks.append(self._run_one(name))
        return report

    def run_one(self, name: str) -> CheckResult:
        """Run a single named check."""
        return self._run_one(name)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _run_one(self, name: str) -> CheckResult:
        _dispatch = {
            CHECK_PYTHON_VERSION: self._check_python_version,
            CHECK_WINDOWS_VERSION: self._check_windows_version,
            CHECK_DISK_SPACE: self._check_disk_space,
            CHECK_RAM: self._check_ram,
            CHECK_DPI_AWARENESS: self._check_dpi_awareness,
            CHECK_DB_ACCESSIBLE: self._check_db_accessible,
            CHECK_TESSERACT_BINARY: self._check_tesseract_binary,
            CHECK_DXCAM: self._check_dxcam,
            CHECK_WRITE_PERMISSION: self._check_write_permission,
            CHECK_CREDENTIAL_MANAGER: self._check_credential_manager,
        }
        fn = _dispatch.get(name)
        if fn is None:
            return CheckResult(
                name=name,
                status="fail",
                message=f"Unknown check: {name!r}",
                fix_hint="",
            )
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            return CheckResult(
                name=name,
                status="fail",
                message=f"Check raised an unexpected error: {exc}",
                fix_hint="Check nexus logs for details.",
            )

    # ------------------------------------------------------------------
    # Individual check methods
    # ------------------------------------------------------------------

    def _check_python_version(self) -> CheckResult:
        ver = _probe_python_version()
        major, minor = ver[0], ver[1]
        ver_str = f"{major}.{minor}"
        if (major, minor) >= _MIN_PYTHON:
            return CheckResult(
                name=CHECK_PYTHON_VERSION,
                status="ok",
                message=f"Python {ver_str} — OK (>= 3.11 required).",
            )
        return CheckResult(
            name=CHECK_PYTHON_VERSION,
            status="fail",
            message=f"Python {ver_str} is too old (>= 3.11 required).",
            fix_hint="Install Python 3.11 or later from https://python.org.",
        )

    def _check_windows_version(self) -> CheckResult:
        os_system = _probe_platform_system()
        if os_system != "Windows":
            return CheckResult(
                name=CHECK_WINDOWS_VERSION,
                status="fail",
                message=f"Unsupported OS: {os_system!r} (Windows required).",
                fix_hint="Nexus Agent requires Windows 10 or Windows 11.",
            )
        release = _probe_windows_release()
        ver_str = _probe_windows_version_str()
        # Build number is the 3rd component of the dotted version string
        try:
            build = int(ver_str.split(".")[2])
        except (IndexError, ValueError):
            build = 0
        # Windows 10: build >= 10240; Windows 11: build >= 22000
        if build >= 10240:
            return CheckResult(
                name=CHECK_WINDOWS_VERSION,
                status="ok",
                message=(
                    f"Windows {release} (build {build}) — OK."
                ),
            )
        return CheckResult(
            name=CHECK_WINDOWS_VERSION,
            status="warn",
            message=(
                f"Windows version appears to be older than Windows 10 "
                f"(build {build})."
            ),
            fix_hint="Upgrade to Windows 10 version 1903 or later.",
        )

    def _check_disk_space(self) -> CheckResult:
        free = _probe_disk_free_bytes(self._write_dir)
        free_gb = free / 1024 ** 3
        if free >= _MIN_DISK_BYTES:
            return CheckResult(
                name=CHECK_DISK_SPACE,
                status="ok",
                message=f"{free_gb:.1f} GB free — OK (>= 2 GB required).",
            )
        return CheckResult(
            name=CHECK_DISK_SPACE,
            status="fail",
            message=f"Only {free_gb:.2f} GB free (>= 2 GB required).",
            fix_hint=(
                "Free up disk space or move the Nexus working directory "
                "to a volume with at least 2 GB available."
            ),
        )

    def _check_ram(self) -> CheckResult:
        ram = _probe_ram_bytes()
        ram_gb = ram / 1024 ** 3
        if sys.platform != "win32":
            return CheckResult(
                name=CHECK_RAM,
                status="warn",
                message="RAM check skipped on non-Windows platform.",
                fix_hint="Run on Windows to verify RAM availability.",
            )
        if ram >= _MIN_RAM_BYTES:
            return CheckResult(
                name=CHECK_RAM,
                status="ok",
                message=f"{ram_gb:.1f} GB RAM — OK (>= 4 GB required).",
            )
        return CheckResult(
            name=CHECK_RAM,
            status="fail",
            message=f"Only {ram_gb:.1f} GB RAM detected (>= 4 GB required).",
            fix_hint=(
                "Install more physical RAM or close "
                "memory-intensive applications."
            ),
        )

    def _check_dpi_awareness(self) -> CheckResult:
        awareness = _probe_dpi_awareness()
        if awareness is None:
            return CheckResult(
                name=CHECK_DPI_AWARENESS,
                status="warn",
                message="Could not query DPI awareness (SHCORE unavailable).",
                fix_hint=(
                    "Ensure the application is running on Windows 8.1 or later "
                    "and SHCORE.dll is accessible."
                ),
            )
        labels = {0: "Unaware", 1: "System Aware", 2: "Per-Monitor Aware"}
        label = labels.get(awareness, f"Unknown ({awareness})")
        if awareness >= 1:
            return CheckResult(
                name=CHECK_DPI_AWARENESS,
                status="ok",
                message=f"DPI awareness: {label} — OK.",
            )
        return CheckResult(
            name=CHECK_DPI_AWARENESS,
            status="warn",
            message=f"DPI awareness: {label} — UI coordinates may be incorrect.",
            fix_hint=(
                "Add a DPI-aware manifest or call "
                "SetProcessDpiAwarenessContext("
                "DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) at startup."
            ),
        )

    def _check_db_accessible(self) -> CheckResult:
        ok = _probe_db_accessible(self._db_path)
        if ok:
            return CheckResult(
                name=CHECK_DB_ACCESSIBLE,
                status="ok",
                message=f"Database '{self._db_path}' is accessible.",
            )
        return CheckResult(
            name=CHECK_DB_ACCESSIBLE,
            status="fail",
            message=f"Database '{self._db_path}' is not accessible.",
            fix_hint=(
                f"Ensure the path '{self._db_path}' exists and the process "
                "has read/write permission.  Run `nexus db init` to create it."
            ),
        )

    def _check_tesseract_binary(self) -> CheckResult:
        path = _probe_tesseract_path()
        if path:
            return CheckResult(
                name=CHECK_TESSERACT_BINARY,
                status="ok",
                message=f"tesseract binary found at: {path}",
            )
        return CheckResult(
            name=CHECK_TESSERACT_BINARY,
            status="fail",
            message="tesseract binary not found on PATH.",
            fix_hint=(
                "Install Tesseract OCR from "
                "https://github.com/UB-Mannheim/tesseract/wiki and add its "
                "installation directory to the system PATH."
            ),
        )

    def _check_dxcam(self) -> CheckResult:
        ok = _probe_dxcam_importable()
        if ok:
            return CheckResult(
                name=CHECK_DXCAM,
                status="ok",
                message="dxcam is importable — screen capture ready.",
            )
        return CheckResult(
            name=CHECK_DXCAM,
            status="fail",
            message="dxcam cannot be imported.",
            fix_hint=(
                "Run `pip install dxcam` and ensure DirectX 11 is available. "
                "A display adapter supporting DXGI is required."
            ),
        )

    def _check_write_permission(self) -> CheckResult:
        ok = _probe_write_permission(self._write_dir)
        if ok:
            return CheckResult(
                name=CHECK_WRITE_PERMISSION,
                status="ok",
                message=f"Write permission confirmed for '{self._write_dir}'.",
            )
        return CheckResult(
            name=CHECK_WRITE_PERMISSION,
            status="fail",
            message=f"No write permission in directory '{self._write_dir}'.",
            fix_hint=(
                f"Grant write access to '{self._write_dir}' for the current "
                "user, or change the working / log directory in settings."
            ),
        )

    def _check_credential_manager(self) -> CheckResult:
        ok = _probe_credential_manager()
        if ok:
            return CheckResult(
                name=CHECK_CREDENTIAL_MANAGER,
                status="ok",
                message="Windows Credential Manager is accessible.",
            )
        return CheckResult(
            name=CHECK_CREDENTIAL_MANAGER,
            status="warn",
            message="Windows Credential Manager is not accessible.",
            fix_hint=(
                "Ensure pywin32 is installed (`pip install pywin32`) and the "
                "Credential Manager service (VaultSvc) is running."
            ),
        )
