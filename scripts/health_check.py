#!/usr/bin/env python
"""
scripts/health_check.py
Nexus Agent — interactive health check with coloured terminal output.

Exit codes
----------
0  all checks passed (ok)
1  at least one warning, no failures
2  at least one failure

Usage
-----
    python scripts/health_check.py [--db-path nexus.db] [--write-dir .]
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure the repo root is on sys.path when run directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nexus.infra.health import (  # noqa: E402
    CheckResult,
    HealthChecker,
    HealthReport,
)

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RED = "\033[31m"
_ANSI_CYAN = "\033[36m"
_ANSI_WHITE = "\033[37m"


def _enable_ansi() -> None:
    """Enable ANSI escape processing on Windows 10+."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


def _colour(text: str, colour: str) -> str:
    return f"{colour}{text}{_ANSI_RESET}"


def _status_colour(status: str) -> str:
    return {
        "ok": _ANSI_GREEN,
        "warn": _ANSI_YELLOW,
        "fail": _ANSI_RED,
    }.get(status, _ANSI_WHITE)


def _safe_char(unicode_char: str, ascii_fallback: str) -> str:
    """Return *unicode_char* if the console can encode it, else *ascii_fallback*."""
    try:
        unicode_char.encode(sys.stdout.encoding or "ascii")
        return unicode_char
    except (UnicodeEncodeError, LookupError):
        return ascii_fallback


def _status_badge(status: str) -> str:
    symbols = {
        "ok": _safe_char("✔", "v"),
        "warn": _safe_char("⚠", "!"),
        "fail": _safe_char("✘", "x"),
    }
    sym = symbols.get(status, "?")
    return _colour(f"[{sym}]", _status_colour(status))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_check(result: CheckResult) -> None:
    badge = _status_badge(result.status)
    name = _colour(result.name, _ANSI_CYAN)
    print(f"  {badge}  {name}")
    print(f"       {result.message}")
    if result.fix_hint and result.status != "ok":
        hint_lines = result.fix_hint.splitlines()
        arrow = _safe_char("↳", "->")
        for i, line in enumerate(hint_lines):
            prefix = f"  {arrow}  " if i == 0 else "      "
            print(_colour(f"{prefix}{line}", _ANSI_YELLOW))
    print()


def _render_report(report: HealthReport) -> None:
    _h = _safe_char("═", "=") * 60
    _d = _safe_char("─", "-") * 60

    print()
    print(_colour(_h, _ANSI_BOLD))
    print(_colour("  NEXUS AGENT — SYSTEM HEALTH CHECK", _ANSI_BOLD))
    print(_colour(_h, _ANSI_BOLD))
    print()

    for result in report.checks:
        _render_check(result)

    # Summary line
    counts = {"ok": 0, "warn": 0, "fail": 0}
    for r in report.checks:
        counts[r.status] += 1

    overall_colour = _status_colour(report.overall)
    print(_colour(_d, _ANSI_BOLD))
    print(
        f"  Overall: {_colour(report.overall.upper(), overall_colour)}  "
        f"({_colour(str(counts['ok']), _ANSI_GREEN)} ok  "
        f"{_colour(str(counts['warn']), _ANSI_YELLOW)} warn  "
        f"{_colour(str(counts['fail']), _ANSI_RED)} fail)"
    )
    print(_colour(_d, _ANSI_BOLD))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nexus Agent system health check",
    )
    parser.add_argument(
        "--db-path",
        default="nexus.db",
        help="SQLite database path to probe (default: nexus.db)",
    )
    parser.add_argument(
        "--write-dir",
        default=".",
        help="Directory to probe for write permission (default: .)",
    )
    args = parser.parse_args()

    _enable_ansi()

    checker = HealthChecker(db_path=args.db_path, write_dir=args.write_dir)
    report = checker.run_all()
    _render_report(report)

    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
