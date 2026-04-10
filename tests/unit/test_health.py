"""
Unit tests for nexus/infra/health.py — HealthChecker and HealthReport.

Strategy
--------
Every probe function lives at module level in nexus.infra.health and is
named ``_probe_*``.  Tests use monkeypatch to substitute controlled return
values, keeping the tests hermetic (no real disk / RAM / registry calls).
"""
from __future__ import annotations

import pytest

import nexus.infra.health as health_mod
from nexus.infra.health import (
    CHECK_CREDENTIAL_MANAGER,
    CHECK_DB_ACCESSIBLE,
    CHECK_DISK_SPACE,
    CHECK_DPI_AWARENESS,
    CHECK_DXCAM,
    CHECK_PYTHON_VERSION,
    CHECK_RAM,
    CHECK_TESSERACT_BINARY,
    CHECK_WINDOWS_VERSION,
    CHECK_WRITE_PERMISSION,
    CheckResult,
    HealthChecker,
    HealthReport,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _checker(**kwargs) -> HealthChecker:  # type: ignore[type-arg]
    return HealthChecker(db_path=kwargs.get("db_path", ":memory:"), write_dir=".")


# ---------------------------------------------------------------------------
# HealthReport
# ---------------------------------------------------------------------------


class TestHealthReport:
    def test_overall_ok_when_all_ok(self) -> None:
        r = HealthReport(
            checks=[
                CheckResult(CHECK_PYTHON_VERSION, "ok", "fine"),
                CheckResult(CHECK_DISK_SPACE, "ok", "fine"),
            ]
        )
        assert r.overall == "ok"

    def test_overall_warn_when_any_warn(self) -> None:
        r = HealthReport(
            checks=[
                CheckResult(CHECK_PYTHON_VERSION, "ok", "fine"),
                CheckResult(CHECK_DISK_SPACE, "warn", "low"),
            ]
        )
        assert r.overall == "warn"

    def test_overall_fail_when_any_fail(self) -> None:
        r = HealthReport(
            checks=[
                CheckResult(CHECK_PYTHON_VERSION, "ok", "fine"),
                CheckResult(CHECK_DISK_SPACE, "warn", "low"),
                CheckResult(CHECK_RAM, "fail", "no ram"),
            ]
        )
        assert r.overall == "fail"

    def test_fail_overrides_warn(self) -> None:
        r = HealthReport(
            checks=[
                CheckResult(CHECK_PYTHON_VERSION, "warn", "old"),
                CheckResult(CHECK_DISK_SPACE, "fail", "empty"),
            ]
        )
        assert r.overall == "fail"

    def test_exit_code_ok(self) -> None:
        r = HealthReport(
            checks=[CheckResult(CHECK_PYTHON_VERSION, "ok", "ok")]
        )
        assert r.exit_code == 0

    def test_exit_code_warn(self) -> None:
        r = HealthReport(
            checks=[CheckResult(CHECK_PYTHON_VERSION, "warn", "meh")]
        )
        assert r.exit_code == 1

    def test_exit_code_fail(self) -> None:
        r = HealthReport(
            checks=[CheckResult(CHECK_PYTHON_VERSION, "fail", "bad")]
        )
        assert r.exit_code == 2

    def test_by_name_found(self) -> None:
        r = HealthReport(
            checks=[CheckResult(CHECK_PYTHON_VERSION, "ok", "msg")]
        )
        assert r.by_name(CHECK_PYTHON_VERSION) is not None

    def test_by_name_not_found(self) -> None:
        r = HealthReport(checks=[])
        assert r.by_name(CHECK_PYTHON_VERSION) is None

    def test_empty_report_is_ok(self) -> None:
        assert HealthReport().overall == "ok"
        assert HealthReport().exit_code == 0


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_fields(self) -> None:
        r = CheckResult(
            name=CHECK_PYTHON_VERSION,
            status="ok",
            message="Python 3.12",
            fix_hint="",
        )
        assert r.name == CHECK_PYTHON_VERSION
        assert r.status == "ok"

    def test_fix_hint_default_empty(self) -> None:
        r = CheckResult(CHECK_PYTHON_VERSION, "ok", "fine")
        assert r.fix_hint == ""


# ---------------------------------------------------------------------------
# CHECK_PYTHON_VERSION
# ---------------------------------------------------------------------------


class TestCheckPythonVersion:
    def test_ok_on_3_12(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_python_version", lambda: (3, 12))
        r = _checker().run_one(CHECK_PYTHON_VERSION)
        assert r.status == "ok"
        assert "3.12" in r.message

    def test_ok_on_3_11(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_python_version", lambda: (3, 11))
        r = _checker().run_one(CHECK_PYTHON_VERSION)
        assert r.status == "ok"

    def test_fail_on_3_10(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_python_version", lambda: (3, 10))
        r = _checker().run_one(CHECK_PYTHON_VERSION)
        assert r.status == "fail"
        assert r.fix_hint != ""
        assert "3.10" in r.message

    def test_fail_on_2_7(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_python_version", lambda: (2, 7))
        r = _checker().run_one(CHECK_PYTHON_VERSION)
        assert r.status == "fail"


# ---------------------------------------------------------------------------
# CHECK_WINDOWS_VERSION
# ---------------------------------------------------------------------------


class TestCheckWindowsVersion:
    def test_ok_on_windows_11(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_platform_system", lambda: "Windows")
        monkeypatch.setattr(health_mod, "_probe_windows_release", lambda: "11")
        monkeypatch.setattr(
            health_mod, "_probe_windows_version_str", lambda: "10.0.22621"
        )
        r = _checker().run_one(CHECK_WINDOWS_VERSION)
        assert r.status == "ok"
        assert "22621" in r.message

    def test_ok_on_windows_10(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_platform_system", lambda: "Windows")
        monkeypatch.setattr(health_mod, "_probe_windows_release", lambda: "10")
        monkeypatch.setattr(
            health_mod, "_probe_windows_version_str", lambda: "10.0.19045"
        )
        r = _checker().run_one(CHECK_WINDOWS_VERSION)
        assert r.status == "ok"

    def test_fail_on_non_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_platform_system", lambda: "Linux")
        r = _checker().run_one(CHECK_WINDOWS_VERSION)
        assert r.status == "fail"
        assert r.fix_hint != ""

    def test_warn_on_old_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_platform_system", lambda: "Windows")
        monkeypatch.setattr(health_mod, "_probe_windows_release", lambda: "7")
        monkeypatch.setattr(
            health_mod, "_probe_windows_version_str", lambda: "6.1.7601"
        )
        r = _checker().run_one(CHECK_WINDOWS_VERSION)
        assert r.status == "warn"
        assert r.fix_hint != ""


# ---------------------------------------------------------------------------
# CHECK_DISK_SPACE
# ---------------------------------------------------------------------------


class TestCheckDiskSpace:
    _GB = 1024 ** 3

    def test_ok_when_enough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            health_mod, "_probe_disk_free_bytes", lambda p: 10 * self._GB
        )
        r = _checker().run_one(CHECK_DISK_SPACE)
        assert r.status == "ok"
        assert "10.0 GB" in r.message

    def test_fail_when_too_little(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            health_mod, "_probe_disk_free_bytes", lambda p: 1 * self._GB
        )
        r = _checker().run_one(CHECK_DISK_SPACE)
        assert r.status == "fail"
        assert r.fix_hint != ""

    def test_fail_on_empty_disk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_disk_free_bytes", lambda p: 0)
        r = _checker().run_one(CHECK_DISK_SPACE)
        assert r.status == "fail"

    def test_ok_at_exact_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            health_mod, "_probe_disk_free_bytes", lambda p: 2 * self._GB
        )
        r = _checker().run_one(CHECK_DISK_SPACE)
        assert r.status == "ok"


# ---------------------------------------------------------------------------
# CHECK_RAM
# ---------------------------------------------------------------------------


class TestCheckRam:
    _GB = 1024 ** 3

    def test_ok_on_16gb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_ram_bytes", lambda: 16 * self._GB)
        monkeypatch.setattr(health_mod.sys, "platform", "win32")
        r = _checker().run_one(CHECK_RAM)
        assert r.status == "ok"
        assert "16.0 GB" in r.message

    def test_fail_on_2gb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_ram_bytes", lambda: 2 * self._GB)
        monkeypatch.setattr(health_mod.sys, "platform", "win32")
        r = _checker().run_one(CHECK_RAM)
        assert r.status == "fail"
        assert r.fix_hint != ""

    def test_ok_at_exact_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_ram_bytes", lambda: 4 * self._GB)
        monkeypatch.setattr(health_mod.sys, "platform", "win32")
        r = _checker().run_one(CHECK_RAM)
        assert r.status == "ok"

    def test_warn_on_non_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod.sys, "platform", "linux")
        r = _checker().run_one(CHECK_RAM)
        assert r.status == "warn"


# ---------------------------------------------------------------------------
# CHECK_DPI_AWARENESS
# ---------------------------------------------------------------------------


class TestCheckDpiAwareness:
    def test_ok_on_per_monitor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_dpi_awareness", lambda: 2)
        r = _checker().run_one(CHECK_DPI_AWARENESS)
        assert r.status == "ok"
        assert "Per-Monitor Aware" in r.message

    def test_ok_on_system_aware(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_dpi_awareness", lambda: 1)
        r = _checker().run_one(CHECK_DPI_AWARENESS)
        assert r.status == "ok"

    def test_warn_on_unaware(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_dpi_awareness", lambda: 0)
        r = _checker().run_one(CHECK_DPI_AWARENESS)
        assert r.status == "warn"
        assert r.fix_hint != ""

    def test_warn_when_shcore_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_dpi_awareness", lambda: None)
        r = _checker().run_one(CHECK_DPI_AWARENESS)
        assert r.status == "warn"
        assert r.fix_hint != ""


# ---------------------------------------------------------------------------
# CHECK_DB_ACCESSIBLE
# ---------------------------------------------------------------------------


class TestCheckDbAccessible:
    def test_ok_when_accessible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_db_accessible", lambda p: True)
        r = _checker().run_one(CHECK_DB_ACCESSIBLE)
        assert r.status == "ok"
        assert ":memory:" in r.message

    def test_fail_when_not_accessible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_db_accessible", lambda p: False)
        r = _checker().run_one(CHECK_DB_ACCESSIBLE)
        assert r.status == "fail"
        assert r.fix_hint != ""

    def test_in_memory_db_accessible(self) -> None:
        # Real probe: :memory: is always accessible
        r = _checker(db_path=":memory:").run_one(CHECK_DB_ACCESSIBLE)
        assert r.status == "ok"


# ---------------------------------------------------------------------------
# CHECK_TESSERACT_BINARY
# ---------------------------------------------------------------------------


class TestCheckTesseractBinary:
    def test_ok_when_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            health_mod, "_probe_tesseract_path", lambda: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
        r = _checker().run_one(CHECK_TESSERACT_BINARY)
        assert r.status == "ok"
        assert "tesseract" in r.message.lower()
        assert r"C:\Program Files\Tesseract-OCR\tesseract.exe" in r.message

    def test_fail_when_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_tesseract_path", lambda: None)
        r = _checker().run_one(CHECK_TESSERACT_BINARY)
        assert r.status == "fail"
        assert r.fix_hint != ""
        assert "PATH" in r.fix_hint

    def test_fix_hint_contains_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_tesseract_path", lambda: None)
        r = _checker().run_one(CHECK_TESSERACT_BINARY)
        assert "https://" in r.fix_hint or "http://" in r.fix_hint


# ---------------------------------------------------------------------------
# CHECK_DXCAM
# ---------------------------------------------------------------------------


class TestCheckDxcam:
    def test_ok_when_importable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_dxcam_importable", lambda: True)
        r = _checker().run_one(CHECK_DXCAM)
        assert r.status == "ok"

    def test_fail_when_not_importable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_dxcam_importable", lambda: False)
        r = _checker().run_one(CHECK_DXCAM)
        assert r.status == "fail"
        assert r.fix_hint != ""
        assert "pip install dxcam" in r.fix_hint


# ---------------------------------------------------------------------------
# CHECK_WRITE_PERMISSION
# ---------------------------------------------------------------------------


class TestCheckWritePermission:
    def test_ok_when_writable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_write_permission", lambda p: True)
        r = _checker().run_one(CHECK_WRITE_PERMISSION)
        assert r.status == "ok"

    def test_fail_when_not_writable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_write_permission", lambda p: False)
        r = _checker().run_one(CHECK_WRITE_PERMISSION)
        assert r.status == "fail"
        assert r.fix_hint != ""

    def test_real_write_to_tmpdir(self, tmp_path) -> None:  # type: ignore[type-arg]
        checker = HealthChecker(write_dir=str(tmp_path))
        r = checker.run_one(CHECK_WRITE_PERMISSION)
        assert r.status == "ok"


# ---------------------------------------------------------------------------
# CHECK_CREDENTIAL_MANAGER
# ---------------------------------------------------------------------------


class TestCheckCredentialManager:
    def test_ok_when_accessible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_credential_manager", lambda: True)
        r = _checker().run_one(CHECK_CREDENTIAL_MANAGER)
        assert r.status == "ok"
        assert "Credential Manager" in r.message

    def test_warn_when_not_accessible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_credential_manager", lambda: False)
        r = _checker().run_one(CHECK_CREDENTIAL_MANAGER)
        assert r.status == "warn"
        assert r.fix_hint != ""
        assert "pywin32" in r.fix_hint


# ---------------------------------------------------------------------------
# run_all — completeness
# ---------------------------------------------------------------------------


class TestRunAll:
    _EXPECTED_CHECKS = {
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
    }

    def _patch_all_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _GB = 1024 ** 3
        monkeypatch.setattr(health_mod, "_probe_python_version", lambda: (3, 12))
        monkeypatch.setattr(
            health_mod, "_probe_platform_system", lambda: "Windows"
        )
        monkeypatch.setattr(health_mod, "_probe_windows_release", lambda: "11")
        monkeypatch.setattr(
            health_mod, "_probe_windows_version_str", lambda: "10.0.22621"
        )
        monkeypatch.setattr(
            health_mod, "_probe_disk_free_bytes", lambda p: 50 * _GB
        )
        monkeypatch.setattr(health_mod, "_probe_ram_bytes", lambda: 16 * _GB)
        monkeypatch.setattr(health_mod.sys, "platform", "win32")
        monkeypatch.setattr(health_mod, "_probe_dpi_awareness", lambda: 2)
        monkeypatch.setattr(health_mod, "_probe_db_accessible", lambda p: True)
        monkeypatch.setattr(
            health_mod,
            "_probe_tesseract_path",
            lambda: r"C:\tesseract\tesseract.exe",
        )
        monkeypatch.setattr(health_mod, "_probe_dxcam_importable", lambda: True)
        monkeypatch.setattr(
            health_mod, "_probe_write_permission", lambda p: True
        )
        monkeypatch.setattr(
            health_mod, "_probe_credential_manager", lambda: True
        )

    def test_all_checks_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_all_ok(monkeypatch)
        report = _checker().run_all()
        names = {r.name for r in report.checks}
        assert names == self._EXPECTED_CHECKS

    def test_all_ok_when_all_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_all_ok(monkeypatch)
        report = _checker().run_all()
        assert report.overall == "ok"
        assert report.exit_code == 0

    def test_exit_code_2_when_any_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_all_ok(monkeypatch)
        monkeypatch.setattr(health_mod, "_probe_dxcam_importable", lambda: False)
        report = _checker().run_all()
        assert report.exit_code == 2

    def test_exit_code_1_when_only_warn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_all_ok(monkeypatch)
        monkeypatch.setattr(health_mod, "_probe_credential_manager", lambda: False)
        report = _checker().run_all()
        assert report.exit_code == 1

    def test_every_check_has_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_all_ok(monkeypatch)
        report = _checker().run_all()
        for r in report.checks:
            assert r.message, f"{r.name} has empty message"

    def test_failed_checks_have_fix_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_all_ok(monkeypatch)
        # Force several failures
        monkeypatch.setattr(health_mod, "_probe_dxcam_importable", lambda: False)
        monkeypatch.setattr(health_mod, "_probe_tesseract_path", lambda: None)
        monkeypatch.setattr(health_mod, "_probe_write_permission", lambda p: False)
        report = _checker().run_all()
        for r in report.checks:
            if r.status == "fail":
                assert r.fix_hint, f"{r.name} has no fix_hint despite failing"

    def test_unknown_check_name_returns_fail(self) -> None:
        c = _checker()
        r = c.run_one("BOGUS_CHECK")
        assert r.status == "fail"
        assert "Unknown" in r.message


# ---------------------------------------------------------------------------
# Fix hints — all non-ok checks must have hints
# ---------------------------------------------------------------------------


class TestFixHints:
    """Verify every check provides a meaningful fix_hint when status != ok."""

    def _force_fail_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(health_mod, "_probe_python_version", lambda: (3, 9))
        monkeypatch.setattr(
            health_mod, "_probe_platform_system", lambda: "Linux"
        )
        monkeypatch.setattr(
            health_mod, "_probe_disk_free_bytes", lambda p: 0
        )
        monkeypatch.setattr(health_mod, "_probe_ram_bytes", lambda: 0)
        monkeypatch.setattr(health_mod.sys, "platform", "win32")
        monkeypatch.setattr(health_mod, "_probe_dpi_awareness", lambda: 0)
        monkeypatch.setattr(health_mod, "_probe_db_accessible", lambda p: False)
        monkeypatch.setattr(health_mod, "_probe_tesseract_path", lambda: None)
        monkeypatch.setattr(health_mod, "_probe_dxcam_importable", lambda: False)
        monkeypatch.setattr(health_mod, "_probe_write_permission", lambda p: False)
        monkeypatch.setattr(health_mod, "_probe_credential_manager", lambda: False)

    def test_all_non_ok_have_fix_hints(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._force_fail_all(monkeypatch)
        report = _checker().run_all()
        for r in report.checks:
            if r.status != "ok":
                assert r.fix_hint, (
                    f"{r.name} (status={r.status}) must have a fix_hint"
                )
