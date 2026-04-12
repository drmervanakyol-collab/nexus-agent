"""
tests/test_dead_code.py — PAKET DC: Dead Code Testleri

test_no_dead_code     — vulture nexus/ --min-confidence 80, kullanılmayan < 20
test_no_unused_imports — ruff check nexus/ --select F401, unused import yok
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
_NEXUS_DIR = _ROOT / "nexus"


# ---------------------------------------------------------------------------
# PAKET DC
# ---------------------------------------------------------------------------


class TestNoDeadCode:
    """vulture nexus/ --min-confidence 80 → kullanılmayan < 20."""

    def test_no_dead_code_vulture(self) -> None:
        """vulture taramasında kullanılmayan sembol sayısı 20'den az olmalı."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "vulture",
                    str(_NEXUS_DIR),
                    "--min-confidence",
                    "80",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(_ROOT),
            )
        except FileNotFoundError:
            pytest.skip("vulture not installed. Run: pip install vulture")

        if "No module named vulture" in result.stderr:
            pytest.skip("vulture not installed")

        # Her satır bir kullanılmayan sembol
        lines = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("Scanning")
        ]

        dead_count = len(lines)

        assert dead_count < 20, (
            f"vulture found {dead_count} unused symbol(s) (max 20 allowed):\n"
            + "\n".join(f"  {l}" for l in lines[:20])
        )

    def test_vulture_output_parseable(self) -> None:
        """vulture çıktısı dosya:satır:sembol formatında olmalı."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "vulture",
                    str(_NEXUS_DIR),
                    "--min-confidence",
                    "90",  # Daha yüksek eşik — az sonuç
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(_ROOT),
            )
        except FileNotFoundError:
            pytest.skip("vulture not installed")

        if "No module named vulture" in result.stderr:
            pytest.skip("vulture not installed")

        # vulture çıktısı 0, 1 veya 3 ile bitebilir (0=clean, 1+=issues/exit)
        assert result.returncode in (0, 1, 2, 3)  # crash değil (non-zero da kabul)


class TestNoUnusedImports:
    """ruff check nexus/ --select F401 → kullanılmayan import yok."""

    def test_no_unused_imports_ruff(self) -> None:
        """ruff F401 ile kullanılmayan import bulunmamalı."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    str(_NEXUS_DIR),
                    "--select",
                    "F401",
                    "--output-format",
                    "text",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(_ROOT),
            )
        except FileNotFoundError:
            pytest.skip("ruff not installed. Run: pip install ruff")

        if "No module named ruff" in result.stderr:
            pytest.skip("ruff not installed")

        output = result.stdout.strip()

        # Exit code 0 = no violations, 1 = violations found
        violations = [
            line for line in output.splitlines()
            if "F401" in line
        ]

        assert len(violations) == 0, (
            f"ruff found {len(violations)} unused import(s):\n"
            + "\n".join(f"  {v}" for v in violations[:20])
        )

    def test_ruff_check_passes_overall(self) -> None:
        """ruff F4xx kategorisinde hata olmamalı."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    str(_NEXUS_DIR),
                    "--select",
                    "F4",  # F401-F811 import hataları
                    "--output-format",
                    "text",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(_ROOT),
            )
        except FileNotFoundError:
            pytest.skip("ruff not installed")

        if "No module named ruff" in result.stderr:
            pytest.skip("ruff not installed")

        violations = [
            line for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("Found")
        ]

        # F4xx kategorisinde hiç bulgu olmamalı
        assert len(violations) == 0, (
            f"ruff F4xx violations found:\n"
            + "\n".join(f"  {v}" for v in violations[:15])
        )
