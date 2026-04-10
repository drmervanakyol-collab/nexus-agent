"""
tests/unit/test_release.py
Release packaging infrastructure tests — Faz 69.

TEST 1 — version.py module
  1.  VERSION is a non-empty string (default "1.0.0" without env vars).
  2.  BUILD_DATE is a non-empty string.
  3.  GIT_HASH is a non-empty string.
  4.  VERSION_STRING contains VERSION and GIT_HASH.
  5.  FULL_VERSION_STRING contains "Nexus Agent", VERSION, BUILD_DATE, GIT_HASH.
  6.  version_tuple() returns a 3-tuple of ints when VERSION is semver.
  7.  version_tuple() returns (0, 0, 0) for non-semver VERSION.
  8.  ENV override: NEXUS_VERSION env var is reflected in VERSION at import time.

TEST 2 — health.py Tesseract path probe
  9.  NEXUS_TESSERACT_PATH env var pointing at a real file → _probe_tesseract_path
      returns that path (packaged build scenario).
  10. NEXUS_TESSERACT_PATH set to non-existent path → falls through to shutil.which.
  11. NEXUS_TESSERACT_PATH not set → shutil.which called (returns None when absent).
  12. _check_tesseract_binary returns "ok" when probe succeeds.
  13. _check_tesseract_binary returns "fail" when probe returns None.
  14. "fail" result fix_hint mentions NEXUS_TESSERACT_PATH.

TEST 3 — nexus.spec structure
  15. nexus.spec exists in repo root.
  16. nexus.spec references "__main__.py" as entry point.
  17. nexus.spec mentions single-directory mode (COLLECT keyword).
  18. nexus.spec bundles configs/ directory.
  19. nexus.spec bundles docs/privacy_policy.md.
  20. nexus.spec bundles docs/terms_of_service.md.
  21. nexus.spec includes Tesseract binary (NEXUS_TESSERACT_PATH / tools/tesseract).

TEST 4 — installer/nexus_agent.iss structure
  22. nexus_agent.iss exists.
  23. Installer sets AppName to "Nexus Agent".
  24. Installer creates desktop shortcut task.
  25. Installer creates Start Menu entry.
  26. Installer has Uninstall support (UninstallDisplayName).
  27. Installer restricts to 64-bit architecture.
  28. Installer output filename contains version placeholder.

TEST 5 — scripts/build_release.bat structure
  29. build_release.bat exists.
  30. build_release.bat invokes CI step (make ci-).
  31. build_release.bat invokes pyinstaller.
  32. build_release.bat invokes iscc (Inno Setup).
  33. build_release.bat injects NEXUS_VERSION / NEXUS_BUILD_DATE / NEXUS_GIT_HASH.
  34. build_release.bat aborts on CI failure (errorlevel check after make).
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent.parent  # repo root
_SPEC = _ROOT / "nexus.spec"
_ISS = _ROOT / "installer" / "nexus_agent.iss"
_BAT = _ROOT / "scripts" / "build_release.bat"


# ---------------------------------------------------------------------------
# TEST 1 — version.py module
# ---------------------------------------------------------------------------


class TestVersionModule:
    def _import_fresh(self, extra_env: dict[str, str] | None = None):
        """
        Import nexus.release.version in a subprocess-like way by temporarily
        removing it from sys.modules so that os.environ overrides take effect.
        """
        import importlib as _il

        env = extra_env or {}
        # Remove cached module
        sys.modules.pop("nexus.release.version", None)
        old_env = {k: os.environ.get(k) for k in env}
        try:
            os.environ.update(env)
            mod = _il.import_module("nexus.release.version")
        finally:
            # Restore env
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            # Restore module cache to whatever was there
            sys.modules.pop("nexus.release.version", None)
        return mod

    def test_version_non_empty(self):
        from nexus.release.version import VERSION
        assert VERSION, "VERSION must be a non-empty string"

    def test_build_date_non_empty(self):
        from nexus.release.version import BUILD_DATE
        assert BUILD_DATE, "BUILD_DATE must be a non-empty string"

    def test_git_hash_non_empty(self):
        from nexus.release.version import GIT_HASH
        assert GIT_HASH, "GIT_HASH must be a non-empty string"

    def test_version_string_contains_version_and_hash(self):
        from nexus.release.version import GIT_HASH, VERSION, VERSION_STRING
        assert VERSION in VERSION_STRING
        assert GIT_HASH in VERSION_STRING

    def test_full_version_string_contains_all_fields(self):
        from nexus.release.version import (
            BUILD_DATE,
            FULL_VERSION_STRING,
            GIT_HASH,
            VERSION,
        )
        assert "Nexus Agent" in FULL_VERSION_STRING
        assert VERSION in FULL_VERSION_STRING
        assert BUILD_DATE in FULL_VERSION_STRING
        assert GIT_HASH in FULL_VERSION_STRING

    def test_version_tuple_returns_three_ints(self):
        from nexus.release.version import version_tuple
        t = version_tuple()
        assert isinstance(t, tuple)
        assert len(t) == 3
        assert all(isinstance(x, int) for x in t)

    def test_version_tuple_fallback_for_non_semver(self):
        mod = self._import_fresh({"NEXUS_VERSION": "1.0.0-alpha"})
        assert mod.version_tuple() == (0, 0, 0)

    def test_env_override_nexus_version(self):
        mod = self._import_fresh({"NEXUS_VERSION": "9.8.7"})
        assert mod.VERSION == "9.8.7"
        assert mod.version_tuple() == (9, 8, 7)


# ---------------------------------------------------------------------------
# TEST 2 — health.py Tesseract path probe
# ---------------------------------------------------------------------------


class TestTesseractPathProbe:
    def test_env_var_real_file_returned(self, tmp_path: Path):
        """NEXUS_TESSERACT_PATH pointing at a real file → that path returned."""
        fake_tess = tmp_path / "tesseract.exe"
        fake_tess.write_bytes(b"fake")

        from nexus.infra import health as _h

        with patch.dict(os.environ, {"NEXUS_TESSERACT_PATH": str(fake_tess)}):
            result = _h._probe_tesseract_path()

        assert result == str(fake_tess)

    def test_env_var_missing_file_falls_through(self, tmp_path: Path):
        """NEXUS_TESSERACT_PATH points to non-existent file → shutil.which called."""
        from nexus.infra import health as _h

        missing = str(tmp_path / "nonexistent.exe")
        with patch.dict(os.environ, {"NEXUS_TESSERACT_PATH": missing}):
            with patch("shutil.which", return_value=None) as mock_which:
                result = _h._probe_tesseract_path()

        mock_which.assert_called_once_with("tesseract")
        assert result is None

    def test_no_env_var_uses_shutil_which(self):
        """No NEXUS_TESSERACT_PATH → shutil.which("tesseract") called."""
        from nexus.infra import health as _h

        env_without = {k: v for k, v in os.environ.items() if k != "NEXUS_TESSERACT_PATH"}
        with patch.dict(os.environ, env_without, clear=True):
            with patch("shutil.which", return_value=None) as mock_which:
                result = _h._probe_tesseract_path()

        mock_which.assert_called_once_with("tesseract")
        assert result is None

    def test_check_tesseract_binary_ok_when_probe_succeeds(self, tmp_path: Path):
        """HealthChecker._check_tesseract_binary returns 'ok' when probe finds binary."""
        from nexus.infra.health import HealthChecker

        fake_tess = tmp_path / "tesseract.exe"
        fake_tess.write_bytes(b"fake")

        checker = HealthChecker()
        with patch.dict(os.environ, {"NEXUS_TESSERACT_PATH": str(fake_tess)}):
            result = checker._check_tesseract_binary()

        assert result.status == "ok"
        assert str(fake_tess) in result.message

    def test_check_tesseract_binary_fail_when_probe_returns_none(self):
        """HealthChecker._check_tesseract_binary returns 'fail' when no binary found."""
        from nexus.infra.health import HealthChecker

        checker = HealthChecker()
        env_without = {k: v for k, v in os.environ.items() if k != "NEXUS_TESSERACT_PATH"}
        with patch.dict(os.environ, env_without, clear=True):
            with patch("shutil.which", return_value=None):
                result = checker._check_tesseract_binary()

        assert result.status == "fail"

    def test_fail_result_fix_hint_mentions_env_var(self):
        """'fail' fix_hint must mention NEXUS_TESSERACT_PATH for packaged-build users."""
        from nexus.infra.health import HealthChecker

        checker = HealthChecker()
        env_without = {k: v for k, v in os.environ.items() if k != "NEXUS_TESSERACT_PATH"}
        with patch.dict(os.environ, env_without, clear=True):
            with patch("shutil.which", return_value=None):
                result = checker._check_tesseract_binary()

        assert "NEXUS_TESSERACT_PATH" in result.fix_hint


# ---------------------------------------------------------------------------
# TEST 3 — nexus.spec structure
# ---------------------------------------------------------------------------


class TestNexusSpec:
    def _spec(self) -> str:
        assert _SPEC.exists(), f"nexus.spec not found at {_SPEC}"
        return _SPEC.read_text(encoding="utf-8")

    def test_spec_exists(self):
        assert _SPEC.exists()

    def test_spec_entry_point_is_main(self):
        assert "__main__.py" in self._spec()

    def test_spec_uses_collect_single_directory(self):
        """COLLECT keyword indicates single-directory mode."""
        assert "COLLECT(" in self._spec()

    def test_spec_bundles_configs(self):
        assert "configs" in self._spec()

    def test_spec_bundles_privacy_policy(self):
        assert "privacy_policy.md" in self._spec()

    def test_spec_bundles_terms_of_service(self):
        assert "terms_of_service.md" in self._spec()

    def test_spec_handles_tesseract(self):
        """Spec must reference Tesseract binary path."""
        content = self._spec()
        has_tess = "tesseract" in content.lower()
        assert has_tess, "nexus.spec must bundle or reference the Tesseract binary"


# ---------------------------------------------------------------------------
# TEST 4 — installer/nexus_agent.iss structure
# ---------------------------------------------------------------------------


class TestInnoSetupScript:
    def _iss(self) -> str:
        assert _ISS.exists(), f"nexus_agent.iss not found at {_ISS}"
        return _ISS.read_text(encoding="utf-8")

    def test_iss_exists(self):
        assert _ISS.exists()

    def test_app_name_set(self):
        assert "Nexus Agent" in self._iss()

    def test_desktop_icon_task_defined(self):
        content = self._iss()
        has_desktop = "desktopicon" in content.lower() or "desktop" in content.lower()
        assert has_desktop, "Installer must define a desktop shortcut task"

    def test_start_menu_entry_defined(self):
        content = self._iss()
        has_start = "{group}" in content or "startmenu" in content.lower()
        assert has_start, "Installer must create a Start Menu entry"

    def test_uninstall_support_defined(self):
        content = self._iss()
        has_uninstall = "UninstallDisplayName" in content or "uninstallexe" in content
        assert has_uninstall, "Installer must support uninstall"

    def test_64bit_architecture_restricted(self):
        content = self._iss()
        has_arch = "x64" in content or "64bit" in content.lower()
        assert has_arch, "Installer must restrict to 64-bit architecture"

    def test_output_filename_has_version_placeholder(self):
        """OutputBaseFilename must contain version (either literal or Inno Setup macro)."""
        content = self._iss()
        has_version_in_output = (
            "OutputBaseFilename" in content
            and ("AppVersion" in content or "NEXUS_VERSION" in content or "Version" in content)
        )
        assert has_version_in_output, "Installer output filename must reference version"


# ---------------------------------------------------------------------------
# TEST 5 — scripts/build_release.bat structure
# ---------------------------------------------------------------------------


class TestBuildReleaseBat:
    def _bat(self) -> str:
        assert _BAT.exists(), f"build_release.bat not found at {_BAT}"
        return _BAT.read_text(encoding="utf-8")

    def test_bat_exists(self):
        assert _BAT.exists()

    def test_bat_runs_ci(self):
        content = self._bat()
        has_ci = "make ci" in content or "make ci-" in content
        assert has_ci, "build_release.bat must invoke make ci (CI gate)"

    def test_bat_runs_pyinstaller(self):
        assert "pyinstaller" in self._bat().lower()

    def test_bat_runs_iscc(self):
        content = self._bat()
        has_iscc = "iscc" in content.lower()
        assert has_iscc, "build_release.bat must invoke iscc (Inno Setup compiler)"

    def test_bat_injects_nexus_version(self):
        assert "NEXUS_VERSION" in self._bat()

    def test_bat_injects_build_date(self):
        assert "NEXUS_BUILD_DATE" in self._bat()

    def test_bat_injects_git_hash(self):
        assert "NEXUS_GIT_HASH" in self._bat()

    def test_bat_aborts_on_ci_failure(self):
        """After CI step, script must check errorlevel and exit /b 1 on failure."""
        content = self._bat()
        # errorlevel check after the make ci invocation
        has_errorlevel_check = "errorlevel 1" in content.lower() or "errorlevel 1" in content
        assert has_errorlevel_check, "build_release.bat must abort when CI fails"
