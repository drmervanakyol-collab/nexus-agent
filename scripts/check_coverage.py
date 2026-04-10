#!/usr/bin/env python3
"""
scripts/check_coverage.py
Per-module coverage threshold checker for Nexus Agent.

Reads the coverage.json produced by pytest-cov and checks that each
nexus sub-package meets its required coverage percentage.

Exit codes:
  0 — all thresholds met
  1 — one or more thresholds missed (or coverage.json not found)

Usage:
  # After: pytest tests/unit/ --cov=nexus --cov-report=json
  python scripts/check_coverage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-module coverage thresholds (%)
# ---------------------------------------------------------------------------
THRESHOLDS: dict[str, int] = {
    "nexus/core/": 95,
    "nexus/source/": 90,
    "nexus/capture/": 90,
    "nexus/perception/": 85,
    "nexus/action/": 90,
    "nexus/verification/": 90,
    "nexus/cloud/": 85,
}

COVERAGE_JSON = Path("coverage.json")


def load_coverage() -> dict:
    if not COVERAGE_JSON.exists():
        print(f"ERROR: {COVERAGE_JSON} not found. Run pytest with --cov-report=json first.")
        sys.exit(1)
    with COVERAGE_JSON.open() as f:
        return json.load(f)


def aggregate(files: dict) -> dict[str, tuple[int, int]]:
    """Sum (covered_lines, num_statements) per threshold prefix."""
    stats: dict[str, tuple[int, int]] = {}
    for filepath, fdata in files.items():
        norm = filepath.replace("\\", "/")
        for prefix in THRESHOLDS:
            if prefix in norm:
                covered = fdata["summary"]["covered_lines"]
                total = fdata["summary"]["num_statements"]
                prev_c, prev_t = stats.get(prefix, (0, 0))
                stats[prefix] = (prev_c + covered, prev_t + total)
                break
    return stats


def main() -> int:
    data = load_coverage()
    files = data.get("files", {})
    stats = aggregate(files)

    col_w = 30
    print()
    print("Per-module coverage report")
    print("=" * 60)
    print(f"{'Module':<{col_w}} {'Coverage':>10} {'Target':>8} {'Status':>8}")
    print("-" * 60)

    failed = False
    skipped = []

    for prefix, threshold in THRESHOLDS.items():
        if prefix not in stats:
            skipped.append(prefix)
            print(f"{prefix:<{col_w}} {'N/A':>10} {threshold:>7}% {'SKIP':>8}")
            continue

        covered, total = stats[prefix]
        pct = (covered / total * 100) if total > 0 else 0.0
        status = "PASS" if pct >= threshold else "FAIL"
        if status == "FAIL":
            failed = True
        flag = " <--" if status == "FAIL" else ""
        print(f"{prefix:<{col_w}} {pct:>9.1f}% {threshold:>7}% {status:>8}{flag}")

    print("-" * 60)

    if skipped:
        print(f"NOTE: {len(skipped)} module(s) had no coverage data (not yet tested).")

    if failed:
        print("\nFAIL: one or more coverage thresholds not met.")
        return 1

    print("\nPASS: all coverage thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
