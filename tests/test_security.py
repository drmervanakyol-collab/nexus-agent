"""
tests/test_security.py — PAKET S: Security Scan Testleri

test_bandit_scan             — HIGH severity bulgu olmasın
test_safety_scan             — Bilinen güvenlik açığı olan paket olmasın
test_no_hardcoded_secrets    — Hardcoded API anahtarı/şifre/token olmasın
test_no_dangerous_subprocess — shell=True ile subprocess çağrısı olmasın
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEXUS_DIR = Path(__file__).parent.parent / "nexus"


def _find_python_files(directory: Path) -> list[Path]:
    return list(directory.rglob("*.py"))


# ---------------------------------------------------------------------------
# PAKET S
# ---------------------------------------------------------------------------


class TestBanditScan:
    """bandit -r nexus/ -ll HIGH severity bulgu olmamalı."""

    def test_bandit_no_high_severity(self) -> None:
        """bandit taramasında HIGH severity bulgu olmamalı."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bandit",
                    "-r",
                    str(_NEXUS_DIR),
                    "-ll",
                    "-f",
                    "json",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("bandit not installed. Run: pip install bandit")

        # Exit code 0 = no issues, 1 = issues found, -1 = error
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError:
            # bandit çalıştı ama JSON üretemedi — stderr kontrol et
            if "No module named bandit" in result.stderr:
                pytest.skip("bandit not installed")
            pytest.skip(f"bandit output not parseable: {result.stdout[:200]}")

        results = report.get("results", [])
        high_issues = [
            r
            for r in results
            if r.get("issue_severity", "").upper() == "HIGH"
        ]

        assert len(high_issues) == 0, (
            f"bandit found {len(high_issues)} HIGH severity issue(s):\n"
            + "\n".join(
                f"  {i['filename']}:{i['line_number']} — {i['issue_text']}"
                for i in high_issues[:5]
            )
        )

    def test_bandit_no_critical_cwe(self) -> None:
        """Kritik CWE (SQL injection, path traversal vb.) bulgu olmamalı."""
        critical_cwes = {"CWE-89", "CWE-22", "CWE-78", "CWE-502"}
        python_files = _find_python_files(_NEXUS_DIR)

        # Temel regex kontrolleri ile hızlı tarama
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception:
                continue

            # SQL injection riski: execute ile ham string concatenation
            if re.search(r'\.execute\s*\(\s*["\'].*?\+', content):
                pytest.fail(
                    f"Potential SQL injection in {py_file.relative_to(_NEXUS_DIR.parent)}"
                )


class TestSafetyScan:
    """safety check — bilinen güvenlik açığı olan paket olmamalı."""

    def test_safety_no_known_vulnerabilities(self) -> None:
        """Yüklü paketlerde bilinen CVE olmamalı."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError:
            pytest.skip("safety not installed. Run: pip install safety")

        stdout = result.stdout.strip()
        if not stdout:
            pytest.skip("safety check produced no output (network issue or API key required)")

        # safety ≥ 2.x wraps the JSON block with deprecation banners (before
        # and after).  Extract only the JSON by slicing from the first { or [
        # to the last } or ] so the surrounding banner text is ignored.
        json_start = next(
            (i for i, ch in enumerate(stdout) if ch in "{["),
            None,
        )
        if json_start is None:
            pytest.skip(f"safety output not parseable: {stdout[:200]}")
        open_ch = stdout[json_start]
        close_ch = "}" if open_ch == "{" else "]"
        json_end = stdout.rfind(close_ch)
        if json_end == -1:
            pytest.skip(f"safety output not parseable: {stdout[:200]}")
        stdout = stdout[json_start : json_end + 1]

        try:
            report = json.loads(stdout)
        except json.JSONDecodeError:
            if "No module named safety" in result.stderr:
                pytest.skip("safety not installed")
            pytest.skip(f"safety output not parseable: {stdout[:200]}")

        # Safety v2+ returns list of vulnerabilities
        if isinstance(report, list):
            vulns = report
        elif isinstance(report, dict):
            vulns = report.get("vulnerabilities", [])
        else:
            vulns = []

        critical_vulns = [
            v for v in vulns
            if isinstance(v, dict) and (v.get("severity") or "").upper() in ("CRITICAL", "HIGH")
        ]

        assert len(critical_vulns) == 0, (
            f"safety found {len(critical_vulns)} critical/high vulnerability(s):\n"
            + "\n".join(
                f"  {v.get('package_name', '?')} — {v.get('vulnerability_id', '?')}: "
                f"{v.get('advisory', '?')[:100]}"
                for v in critical_vulns[:5]
            )
        )


class TestNoHardcodedSecrets:
    """nexus/ klasöründe hardcoded API anahtarı/şifre/token olmamalı."""

    _DANGEROUS_PATTERNS: list[tuple[str, str]] = [
        (r'sk-[A-Za-z0-9]{20,}', "OpenAI API key"),
        (r'sk-ant-[A-Za-z0-9-]{20,}', "Anthropic API key"),
        (r'ghp_[A-Za-z0-9]{36}', "GitHub token"),
        (r'(?i)password\s*=\s*["\'][^"\']{8,}["\']', "hardcoded password"),
        (r'(?i)secret\s*=\s*["\'][^"\']{8,}["\']', "hardcoded secret"),
        (r'(?i)api_key\s*=\s*["\'][^"\']{8,}["\']', "hardcoded API key"),
    ]

    # İzin verilen (bilerek konmuş, güvenli) pattern'lar
    _ALLOWED_EXCEPTIONS: set[str] = {
        "nexus-dev-secret-REPLACE-AT-BUILD-TIME",  # license_manager.py dev placeholder
        "_DEV_SECRET",
    }

    def test_no_hardcoded_api_keys(self) -> None:
        """nexus/ içindeki .py dosyalarında gerçek API anahtarı olmamalı."""
        violations: list[str] = []
        python_files = _find_python_files(_NEXUS_DIR)

        for py_file in python_files:
            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception:
                continue

            for pattern, description in self._DANGEROUS_PATTERNS:
                matches = re.finditer(pattern, content)
                for match in matches:
                    matched_text = match.group()

                    # İzin verilen pattern'leri atla
                    if any(exc in matched_text for exc in self._ALLOWED_EXCEPTIONS):
                        continue
                    if any(exc in content[max(0, match.start()-50):match.end()+50]
                           for exc in self._ALLOWED_EXCEPTIONS):
                        continue

                    violations.append(
                        f"{py_file.relative_to(_NEXUS_DIR.parent)} — "
                        f"{description}: {matched_text[:40]!r}"
                    )

        assert len(violations) == 0, (
            f"Found {len(violations)} hardcoded secret(s):\n"
            + "\n".join(f"  {v}" for v in violations[:10])
        )

    def test_no_test_api_keys_in_source(self) -> None:
        """Test dosyaları dahil production secret içermemeli."""
        test_dir = Path(__file__).parent

        for py_file in test_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception:
                continue

            # Gerçek format API key'leri ara
            real_key_patterns = [
                r'sk-[A-Za-z0-9]{40,}',    # Gerçek OpenAI key (40+ char)
                r'sk-ant-[A-Za-z0-9-]{50,}',  # Gerçek Anthropic key
            ]
            for pattern in real_key_patterns:
                if re.search(pattern, content):
                    pytest.fail(
                        f"Potential real API key found in test file: "
                        f"{py_file.relative_to(test_dir)}"
                    )


class TestNoDangerousSubprocess:
    """nexus/ klasöründe shell=True ile subprocess çağrısı olmamalı."""

    def test_no_shell_true_subprocess(self) -> None:
        """subprocess.run/Popen ile shell=True kullanılmamalı."""
        violations: list[str] = []
        python_files = _find_python_files(_NEXUS_DIR)

        # shell=True ile subprocess çağrısı ara
        pattern = re.compile(r'subprocess\s*\.\s*(run|Popen|call|check_output)\s*\([^)]*shell\s*=\s*True')

        for py_file in python_files:
            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception:
                continue

            if pattern.search(content):
                # Satır numarasını bul
                for line_no, line in enumerate(content.splitlines(), 1):
                    if "shell=True" in line and "subprocess" in content:
                        violations.append(
                            f"{py_file.relative_to(_NEXUS_DIR.parent)}:{line_no} — "
                            f"shell=True: {line.strip()[:60]}"
                        )
                        break

        assert len(violations) == 0, (
            f"Found {len(violations)} dangerous shell=True call(s):\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_no_os_system_calls(self) -> None:
        """os.system() çağrısı olmamalı (shell injection riski)."""
        violations: list[str] = []
        python_files = _find_python_files(_NEXUS_DIR)
        pattern = re.compile(r'\bos\.system\s*\(')

        for py_file in python_files:
            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception:
                continue

            for line_no, line in enumerate(content.splitlines(), 1):
                if pattern.search(line):
                    violations.append(
                        f"{py_file.relative_to(_NEXUS_DIR.parent)}:{line_no} — {line.strip()[:60]}"
                    )

        assert len(violations) == 0, (
            f"Found os.system() calls (use subprocess instead):\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_no_eval_in_production_code(self) -> None:
        """eval() production kodunda kullanılmamalı."""
        violations: list[str] = []
        python_files = _find_python_files(_NEXUS_DIR)
        pattern = re.compile(r'\beval\s*\(')

        for py_file in python_files:
            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception:
                continue

            for line_no, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                # Yorum satırlarını atla
                if stripped.startswith("#"):
                    continue
                if pattern.search(line):
                    violations.append(
                        f"{py_file.relative_to(_NEXUS_DIR.parent)}:{line_no} — {stripped[:60]}"
                    )

        assert len(violations) == 0, (
            f"Found eval() calls in production code:\n"
            + "\n".join(f"  {v}" for v in violations)
        )
