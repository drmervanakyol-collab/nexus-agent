"""
nexus/release/signing.py
Code signing support for Nexus Agent release binaries.

Development: self-signed certificate (makecert / New-SelfSignedCertificate)
Production : EV/OV certificate installed in Windows Certificate Store

Environment variables
---------------------
NEXUS_SIGN_CERT      — certificate thumbprint or subject name (optional;
                       if omitted signtool picks the first valid cert)
NEXUS_SIGN_PFX       — path to a PFX file (used when no store cert is set)
NEXUS_SIGN_PFX_PASS  — password for PFX file
NEXUS_SIGNTOOL       — full path to signtool.exe (optional; discovered via
                       default install paths when not set)
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMESTAMP_URL = "http://timestamp.digicert.com"

_SIGNTOOL_SEARCH_PATHS: list[str] = [
    r"C:\Program Files (x86)\Windows Kits\10\bin\x64\signtool.exe",
    r"C:\Program Files (x86)\Windows Kits\10\bin\x86\signtool.exe",
    r"C:\Program Files\Windows Kits\10\bin\x64\signtool.exe",
    r"C:\Program Files\Windows Kits\10\bin\x86\signtool.exe",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_signtool() -> str:
    """Return path to signtool.exe or raise FileNotFoundError."""
    # 1. Explicit env var
    env_path = os.environ.get("NEXUS_SIGNTOOL", "")
    if env_path and Path(env_path).is_file():
        return env_path

    # 2. PATH
    which = shutil.which("signtool")
    if which:
        return which

    # 3. Well-known SDK locations (newest Windows Kit first)
    sdk_root = r"C:\Program Files (x86)\Windows Kits\10\bin"
    if Path(sdk_root).is_dir():
        # Prefer highest SDK version
        for version_dir in sorted(Path(sdk_root).iterdir(), reverse=True):
            candidate = version_dir / "x64" / "signtool.exe"
            if candidate.is_file():
                return str(candidate)

    for fallback in _SIGNTOOL_SEARCH_PATHS:
        if Path(fallback).is_file():
            return fallback

    raise FileNotFoundError(
        "signtool.exe not found. Install Windows SDK or set NEXUS_SIGNTOOL."
    )


def _build_sign_command(signtool: str, path: Path) -> list[str]:
    """Build the signtool sign command for *path*."""
    cmd: list[str] = [
        signtool,
        "sign",
        "/fd", "SHA256",
        "/tr", TIMESTAMP_URL,
        "/td", "SHA256",
    ]

    pfx = os.environ.get("NEXUS_SIGN_PFX", "")
    cert = os.environ.get("NEXUS_SIGN_CERT", "")

    if pfx and Path(pfx).is_file():
        cmd += ["/f", pfx]
        pfx_pass = os.environ.get("NEXUS_SIGN_PFX_PASS", "")
        if pfx_pass:
            cmd += ["/p", pfx_pass]
    elif cert:
        # Certificate by thumbprint or subject
        if len(cert) == 40 and all(c in "0123456789abcdefABCDEF" for c in cert):
            cmd += ["/sha1", cert]
        else:
            cmd += ["/n", cert]
    else:
        # Let signtool pick the first available cert (dev self-signed scenario)
        cmd += ["/a"]

    cmd.append(str(path))
    return cmd


def _build_verify_command(signtool: str, path: Path) -> list[str]:
    """Build the signtool verify command for *path*."""
    return [
        signtool,
        "verify",
        "/pa",      # verify using default policy (includes self-signed in dev)
        "/v",
        str(path),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sign_binary(path: str | Path) -> bool:
    """
    Sign *path* with signtool.exe and verify the resulting signature.

    Parameters
    ----------
    path:
        Absolute or relative path to the binary (.exe) to sign.

    Returns
    -------
    bool
        True if signing and verification both succeed, False otherwise.
    """
    target = Path(path).resolve()

    if not target.is_file():
        logger.error("sign_binary: file not found: %s", target)
        return False

    try:
        signtool = _find_signtool()
    except FileNotFoundError as exc:
        logger.error("sign_binary: %s", exc)
        return False

    # --- Sign ---
    sign_cmd = _build_sign_command(signtool, target)
    logger.info("Signing: %s", target.name)
    logger.debug("Command: %s", " ".join(sign_cmd))

    sign_result = subprocess.run(
        sign_cmd,
        capture_output=True,
        text=True,
    )

    if sign_result.returncode != 0:
        logger.error(
            "Signing failed for %s:\n%s",
            target.name,
            sign_result.stdout + sign_result.stderr,
        )
        return False

    logger.info("Signed OK: %s", target.name)

    # --- Verify ---
    verify_cmd = _build_verify_command(signtool, target)
    logger.debug("Verifying: %s", " ".join(verify_cmd))

    verify_result = subprocess.run(
        verify_cmd,
        capture_output=True,
        text=True,
    )

    if verify_result.returncode != 0:
        logger.error(
            "Verification failed for %s:\n%s",
            target.name,
            verify_result.stdout + verify_result.stderr,
        )
        return False

    logger.info("Verified OK: %s", target.name)
    return True


def sign_release_binaries(dist_dir: str | Path) -> dict[str, bool]:
    """
    Sign the standard set of Nexus Agent release binaries under *dist_dir*.

    Targets
    -------
    - nexus-agent.exe   (main executable)
    - installer.exe     (setup executable, name pattern: *-setup.exe)

    Returns
    -------
    dict mapping filename → success flag for every attempted binary.
    """
    dist = Path(dist_dir).resolve()
    results: dict[str, bool] = {}

    candidates: list[Path] = []

    agent_exe = dist / "nexus-agent.exe"
    if agent_exe.is_file():
        candidates.append(agent_exe)
    else:
        logger.warning("sign_release_binaries: nexus-agent.exe not found in %s", dist)

    # Installer: dist/nexus-agent-vX.Y.Z-setup.exe
    for installer in dist.glob("*-setup.exe"):
        candidates.append(installer)

    if not candidates:
        logger.error("sign_release_binaries: no binaries found in %s", dist)

    for binary in candidates:
        results[binary.name] = sign_binary(binary)

    return results
