"""
Unit tests for nexus/release/signing.py

Strategy
--------
All subprocess.run calls and filesystem access are mocked so tests run
without Windows SDK or actual binaries present.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import nexus.release.signing as signing_mod
from nexus.release.signing import (
    TIMESTAMP_URL,
    _build_sign_command,
    _build_verify_command,
    _find_signtool,
    sign_binary,
    sign_release_binaries,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_binary(tmp_path: Path) -> Path:
    """A real (non-empty) file masquerading as a signed binary."""
    exe = tmp_path / "nexus-agent.exe"
    exe.write_bytes(b"MZ\x00\x00")  # minimal PE magic
    return exe


@pytest.fixture()
def fake_installer(tmp_path: Path) -> Path:
    exe = tmp_path / "nexus-agent-v1.0.0-setup.exe"
    exe.write_bytes(b"MZ\x00\x00")
    return exe


@pytest.fixture()
def fake_pfx(tmp_path: Path) -> Path:
    pfx = tmp_path / "dev.pfx"
    pfx.write_bytes(b"FAKEPFX")
    return pfx


def _ok_result() -> MagicMock:
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = 0
    r.stdout = "Successfully signed"
    r.stderr = ""
    return r


def _fail_result() -> MagicMock:
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = 1
    r.stdout = ""
    r.stderr = "SignTool Error: No certificates"
    return r


# ---------------------------------------------------------------------------
# _find_signtool
# ---------------------------------------------------------------------------


class TestFindSigntool:
    def test_env_var_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = tmp_path / "signtool.exe"
        fake.touch()
        monkeypatch.setenv("NEXUS_SIGNTOOL", str(fake))
        assert _find_signtool() == str(fake)

    def test_which_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEXUS_SIGNTOOL", raising=False)
        with patch("shutil.which", return_value=r"C:\fake\signtool.exe"):
            result = _find_signtool()
        assert result == r"C:\fake\signtool.exe"

    def test_raises_when_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEXUS_SIGNTOOL", raising=False)
        with (
            patch("shutil.which", return_value=None),
            patch.object(Path, "is_dir", return_value=False),
            patch.object(Path, "is_file", return_value=False),
            pytest.raises(FileNotFoundError, match="signtool.exe not found"),
        ):
            _find_signtool()


# ---------------------------------------------------------------------------
# _build_sign_command
# ---------------------------------------------------------------------------


class TestBuildSignCommand:
    def test_auto_mode_when_no_env(
        self, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEXUS_SIGN_PFX", raising=False)
        monkeypatch.delenv("NEXUS_SIGN_CERT", raising=False)
        cmd = _build_sign_command("signtool.exe", fake_binary)
        assert "/a" in cmd
        assert TIMESTAMP_URL in cmd

    def test_pfx_mode(
        self, fake_binary: Path, fake_pfx: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEXUS_SIGN_PFX", str(fake_pfx))
        monkeypatch.setenv("NEXUS_SIGN_PFX_PASS", "s3cr3t")
        monkeypatch.delenv("NEXUS_SIGN_CERT", raising=False)
        cmd = _build_sign_command("signtool.exe", fake_binary)
        assert "/f" in cmd
        assert str(fake_pfx) in cmd
        assert "/p" in cmd
        assert "s3cr3t" in cmd

    def test_cert_thumbprint_mode(
        self, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thumbprint = "a" * 40
        monkeypatch.setenv("NEXUS_SIGN_CERT", thumbprint)
        monkeypatch.delenv("NEXUS_SIGN_PFX", raising=False)
        cmd = _build_sign_command("signtool.exe", fake_binary)
        assert "/sha1" in cmd
        assert thumbprint in cmd

    def test_cert_subject_mode(
        self, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEXUS_SIGN_CERT", "Nexus Agent OV")
        monkeypatch.delenv("NEXUS_SIGN_PFX", raising=False)
        cmd = _build_sign_command("signtool.exe", fake_binary)
        assert "/n" in cmd
        assert "Nexus Agent OV" in cmd

    def test_sha256_and_timestamp_always_present(
        self, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEXUS_SIGN_PFX", raising=False)
        monkeypatch.delenv("NEXUS_SIGN_CERT", raising=False)
        cmd = _build_sign_command("signtool.exe", fake_binary)
        assert "/fd" in cmd
        assert "SHA256" in cmd
        assert "/tr" in cmd
        assert TIMESTAMP_URL in cmd
        assert "/td" in cmd
        assert str(fake_binary) in cmd


# ---------------------------------------------------------------------------
# _build_verify_command
# ---------------------------------------------------------------------------


class TestBuildVerifyCommand:
    def test_structure(self, fake_binary: Path) -> None:
        cmd = _build_verify_command("signtool.exe", fake_binary)
        assert cmd[0] == "signtool.exe"
        assert "verify" in cmd
        assert "/pa" in cmd
        assert str(fake_binary) in cmd


# ---------------------------------------------------------------------------
# sign_binary
# ---------------------------------------------------------------------------


class TestSignBinary:
    def test_returns_false_for_missing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEXUS_SIGNTOOL", str(tmp_path / "signtool.exe"))
        result = sign_binary(tmp_path / "ghost.exe")
        assert result is False

    def test_returns_false_when_signtool_not_found(
        self, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEXUS_SIGNTOOL", raising=False)
        with (
            patch("shutil.which", return_value=None),
            patch.object(Path, "is_dir", return_value=False),
            patch.object(Path, "is_file", side_effect=lambda self=None: self == fake_binary if self else False),
            patch.object(signing_mod, "_find_signtool", side_effect=FileNotFoundError("not found")),
        ):
            # patch _find_signtool directly to keep test simple
            result = sign_binary(fake_binary)
        assert result is False

    def test_success_path(
        self, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEXUS_SIGN_PFX", raising=False)
        monkeypatch.delenv("NEXUS_SIGN_CERT", raising=False)
        with (
            patch.object(signing_mod, "_find_signtool", return_value="signtool.exe"),
            patch("subprocess.run", return_value=_ok_result()) as mock_run,
        ):
            result = sign_binary(fake_binary)
        assert result is True
        assert mock_run.call_count == 2  # sign + verify

    def test_sign_failure_returns_false(
        self, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEXUS_SIGN_PFX", raising=False)
        monkeypatch.delenv("NEXUS_SIGN_CERT", raising=False)
        with (
            patch.object(signing_mod, "_find_signtool", return_value="signtool.exe"),
            patch("subprocess.run", return_value=_fail_result()),
        ):
            result = sign_binary(fake_binary)
        assert result is False

    def test_verify_failure_returns_false(
        self, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEXUS_SIGN_PFX", raising=False)
        monkeypatch.delenv("NEXUS_SIGN_CERT", raising=False)
        with (
            patch.object(signing_mod, "_find_signtool", return_value="signtool.exe"),
            patch("subprocess.run", side_effect=[_ok_result(), _fail_result()]),
        ):
            result = sign_binary(fake_binary)
        assert result is False

    def test_string_path_accepted(
        self, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEXUS_SIGN_PFX", raising=False)
        monkeypatch.delenv("NEXUS_SIGN_CERT", raising=False)
        with (
            patch.object(signing_mod, "_find_signtool", return_value="signtool.exe"),
            patch("subprocess.run", return_value=_ok_result()),
        ):
            result = sign_binary(str(fake_binary))
        assert result is True


# ---------------------------------------------------------------------------
# sign_release_binaries
# ---------------------------------------------------------------------------


class TestSignReleaseBinaries:
    def test_signs_both_binaries(
        self,
        tmp_path: Path,
        fake_binary: Path,
        fake_installer: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # fake_binary and fake_installer are already in tmp_path
        monkeypatch.delenv("NEXUS_SIGN_PFX", raising=False)
        monkeypatch.delenv("NEXUS_SIGN_CERT", raising=False)
        with (
            patch.object(signing_mod, "_find_signtool", return_value="signtool.exe"),
            patch("subprocess.run", return_value=_ok_result()),
        ):
            results = sign_release_binaries(tmp_path)

        assert results["nexus-agent.exe"] is True
        assert results["nexus-agent-v1.0.0-setup.exe"] is True

    def test_empty_dist_returns_empty(self, tmp_path: Path) -> None:
        results = sign_release_binaries(tmp_path)
        assert results == {}

    def test_partial_success(
        self, tmp_path: Path, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only agent exe present, no installer
        monkeypatch.delenv("NEXUS_SIGN_PFX", raising=False)
        monkeypatch.delenv("NEXUS_SIGN_CERT", raising=False)
        with (
            patch.object(signing_mod, "_find_signtool", return_value="signtool.exe"),
            patch("subprocess.run", side_effect=[_ok_result(), _fail_result()]),
        ):
            results = sign_release_binaries(tmp_path)

        # sign OK, verify FAIL → False
        assert results["nexus-agent.exe"] is False

    def test_installer_only(
        self, tmp_path: Path, fake_installer: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NEXUS_SIGN_PFX", raising=False)
        monkeypatch.delenv("NEXUS_SIGN_CERT", raising=False)
        with (
            patch.object(signing_mod, "_find_signtool", return_value="signtool.exe"),
            patch("subprocess.run", return_value=_ok_result()),
        ):
            results = sign_release_binaries(tmp_path)

        assert "nexus-agent-v1.0.0-setup.exe" in results
        assert results["nexus-agent-v1.0.0-setup.exe"] is True
