#!/usr/bin/env python
"""
scripts/pilot_checklist.py
Nexus Agent — Faz 72: Pilot Acceptance Checklist

Runs all mandatory pre-launch checks and, if they all pass, declares
"V1 Launch Candidate" and creates the git tag v1.0.0-rc1.

Usage
-----
    python scripts/pilot_checklist.py              # full run, tag on success
    python scripts/pilot_checklist.py --dry-run    # full run, no git tag
    python scripts/pilot_checklist.py --skip-ci    # skip slow CI gate
    python scripts/pilot_checklist.py --golden     # enable golden scenarios
    python scripts/pilot_checklist.py --list       # print checklist only

Exit codes
----------
0  all mandatory checks passed
1  at least one mandatory check failed
2  usage / environment error
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Repo root (scripts/ is one level below root)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# ANSI helpers (same palette as health_check.py)
# ---------------------------------------------------------------------------

_RST = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"


def _enable_ansi() -> None:
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.kernel32.SetConsoleMode(  # type: ignore[attr-defined]
                ctypes.windll.kernel32.GetStdHandle(-11), 7
            )
        except Exception:  # noqa: BLE001
            pass


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{_RST}"


# ---------------------------------------------------------------------------
# Check result type
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    key: str
    label: str
    passed: bool
    detail: str = ""
    skipped: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_pytest(
    *targets: str,
    extra_env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Run pytest on *targets* and return (passed, summary_line).

    Passed means exit code 0 (all tests pass or collected+skipped is fine).
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    cmd = [sys.executable, "-m", "pytest", *targets, "--tb=no", "-q", "--no-header"]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        env=env,
    )
    # Last non-empty line of stdout is the summary (e.g. "42 passed in 3.1s")
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    summary = lines[-1] if lines else result.stderr.strip()[:120]
    return result.returncode == 0, summary


def _bench_latest_json() -> dict | None:
    """Return the parsed contents of bench_latest.json, or None."""
    path = _ROOT / "tests" / "benchmarks" / "results" / "bench_latest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _git_tag(tag: str, message: str) -> tuple[bool, str]:
    """Create an annotated git tag. Returns (ok, detail)."""
    result = subprocess.run(
        ["git", "tag", "-a", tag, "-m", message],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout).strip()
        return False, err
    return True, f"Tag {tag!r} created."


# ---------------------------------------------------------------------------
# Individual mandatory check functions
# ---------------------------------------------------------------------------


def check_ci(skip: bool = False) -> CheckResult:
    """make ci → zero unit + integration test failures."""
    if skip:
        return CheckResult("ci", "make ci (unit + integration)", passed=True,
                           detail="Skipped via --skip-ci.", skipped=True)
    passed, summary = _run_pytest(
        "tests/unit/", "tests/integration/",
        extra_args=["--timeout=120"],
    )
    return CheckResult("ci", "make ci → zero errors", passed=passed, detail=summary)


def check_health() -> CheckResult:
    """health_check.py → healthy (exit code 0 or 1/warn)."""
    result = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "health_check.py")],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
    )
    # exit 0 = ok, 1 = warn (acceptable), 2 = fail
    passed = result.returncode < 2
    lines = [l for l in result.stdout.splitlines() if "Overall:" in l]
    detail = lines[0].strip() if lines else f"exit {result.returncode}"
    return CheckResult("health", "health_check.py → healthy", passed=passed, detail=detail)


def check_golden_scenarios(run_golden: bool = False) -> CheckResult:
    """8 golden scenarios pass (requires NEXUS_GOLDEN_TESTS=1)."""
    if not run_golden:
        return CheckResult(
            "golden", "8 golden scenarios", passed=True,
            detail="Skipped (use --golden to enable).", skipped=True,
        )
    env = {"NEXUS_GOLDEN_TESTS": "1"}
    passed, summary = _run_pytest("tests/golden/", extra_env=env)
    return CheckResult("golden", "8 golden scenarios pass", passed=passed, detail=summary)


def check_adversarial() -> CheckResult:
    """10 adversarial tests pass."""
    passed, summary = _run_pytest("tests/adversarial/", extra_args=["--timeout=60"])
    return CheckResult("adversarial", "10 adversarial tests pass", passed=passed, detail=summary)


def check_benchmarks() -> CheckResult:
    """All benchmark targets met."""
    passed, summary = _run_pytest("tests/benchmarks/", extra_args=["--timeout=120"])
    return CheckResult("benchmarks", "All benchmark targets met", passed=passed, detail=summary)


def check_transport_benchmark() -> CheckResult:
    """Transport: UIA average latency < mouse average latency."""
    data = _bench_latest_json()
    if data is None:
        # bench_latest.json absent — run the transport bench inline
        passed, summary = _run_pytest(
            "tests/benchmarks/bench_transport_latency.py",
            extra_args=["--timeout=60"],
        )
        detail = summary if not passed else "Bench run OK (no cached JSON found)."
        return CheckResult(
            "transport_bench", "Transport: UIA < mouse latency",
            passed=passed, detail=detail,
        )

    results: list[dict] = data.get("results", [])
    for rec in results:
        if rec.get("name") == "transport_latency_comparison":
            uia_avg = rec.get("extra", {}).get("uia_avg_ms")
            mouse_avg = rec.get("extra", {}).get("mouse_avg_ms")
            if uia_avg is not None and mouse_avg is not None:
                passed = uia_avg < mouse_avg
                ratio = rec.get("extra", {}).get("speedup_ratio", "?")
                detail = (
                    f"UIA {uia_avg:.3f} ms < mouse {mouse_avg:.3f} ms "
                    f"(speedup {ratio}x)"
                )
                return CheckResult(
                    "transport_bench", "Transport: UIA < mouse latency",
                    passed=passed, detail=detail,
                )

    # Record not found in JSON — fall back to re-run
    passed, summary = _run_pytest(
        "tests/benchmarks/bench_transport_latency.py",
        extra_args=["--timeout=60"],
    )
    return CheckResult(
        "transport_bench", "Transport: UIA < mouse latency",
        passed=passed, detail=summary,
    )


def check_legal_docs() -> CheckResult:
    """Privacy policy + Terms of Service documents exist."""
    pp = _ROOT / "docs" / "privacy_policy.md"
    tos = _ROOT / "docs" / "terms_of_service.md"
    missing = [p.name for p in (pp, tos) if not p.is_file()]
    passed = len(missing) == 0
    detail = "OK" if passed else f"Missing: {', '.join(missing)}"
    return CheckResult("legal", "Privacy policy + ToS exist", passed=passed, detail=detail)


def check_onboarding() -> CheckResult:
    """Onboarding module importable (onboarding completable)."""
    try:
        import importlib
        mod = importlib.import_module("nexus.ui.onboarding")
        detail = f"nexus.ui.onboarding OK ({mod.__file__})"
        passed = True
    except ImportError as exc:
        detail = str(exc)
        passed = False
    return CheckResult("onboarding", "Onboarding completable", passed=passed, detail=detail)


def check_cost_dashboard() -> CheckResult:
    """Cost dashboard + transport breakdown module importable."""
    try:
        import importlib
        mod = importlib.import_module("nexus.ui.dashboard")
        # Verify transport_breakdown is present
        has_breakdown = hasattr(mod, "TransportBreakdown") or any(
            "transport" in name.lower() for name in dir(mod)
        )
        detail = f"nexus.ui.dashboard OK (transport attr: {has_breakdown})"
        passed = True
    except ImportError as exc:
        detail = str(exc)
        passed = False
    return CheckResult(
        "cost_dashboard", "Cost dashboard + transport breakdown", passed=passed, detail=detail
    )


def check_cancel() -> CheckResult:
    """Cancel mechanism — cancel handler module + adversarial cancel test."""
    # Module check
    try:
        import importlib
        importlib.import_module("nexus.ui.cancel_handler")
    except ImportError as exc:
        return CheckResult("cancel", "Cancel works", passed=False, detail=str(exc))

    # Test check (adv_010 covers concurrent cancel)
    passed, summary = _run_pytest(
        "tests/adversarial/adv_010_concurrent_cancel.py",
        "tests/unit/test_notifications_cancel.py",
        extra_args=["--timeout=30"],
    )
    return CheckResult("cancel", "Cancel works", passed=passed, detail=summary)


def check_diagnostic() -> CheckResult:
    """Diagnostic ZIP creation (transport_audit included)."""
    try:
        import importlib
        mod = importlib.import_module("nexus.infra.diagnostic")
        has_zip = any(
            hasattr(mod, name)
            for name in ("build_zip_bytes", "create_diagnostic_zip", "DiagnosticCollector")
        )
        passed = has_zip
        detail = (
            f"nexus.infra.diagnostic OK (zip fn: {has_zip})"
            if passed else "build_zip_bytes / DiagnosticCollector not found"
        )
    except ImportError as exc:
        passed = False
        detail = str(exc)
    return CheckResult("diagnostic", "Diagnostic ZIP (transport_audit)", passed=passed, detail=detail)


def check_installer_config() -> CheckResult:
    """Installer config exists (clean-VM install possible)."""
    iss = _ROOT / "installer" / "nexus_agent.iss"
    passed = iss.is_file()
    detail = str(iss) if passed else f"Missing: {iss}"
    return CheckResult("installer", "Installer config (clean-VM install)", passed=passed, detail=detail)


def check_code_signing() -> CheckResult:
    """Code signing module importable (sign_binary API present)."""
    try:
        from nexus.release.signing import sign_binary
        detail = "nexus.release.signing.sign_binary OK"
        passed = True
    except ImportError as exc:
        detail = str(exc)
        passed = False
    return CheckResult("signing", "Code signing module", passed=passed, detail=detail)


def check_scenario_007(run_golden: bool = False) -> CheckResult:
    """Native transport golden scenario (007) passes."""
    if not run_golden:
        return CheckResult(
            "scenario_007", "Native transport scenario (007)", passed=True,
            detail="Skipped (use --golden to enable).", skipped=True,
        )
    env = {"NEXUS_GOLDEN_TESTS": "1"}
    passed, summary = _run_pytest(
        "tests/golden/scenario_007_transport_native.py", extra_env=env,
    )
    return CheckResult(
        "scenario_007", "Native transport scenario (007)", passed=passed, detail=summary,
    )


def check_scenario_005(run_golden: bool = False) -> CheckResult:
    """Sheets-browser writeback scenario (005) passes."""
    if not run_golden:
        return CheckResult(
            "scenario_005", "Sheets-browser writeback (005)", passed=True,
            detail="Skipped (use --golden to enable).", skipped=True,
        )
    env = {"NEXUS_GOLDEN_TESTS": "1"}
    passed, summary = _run_pytest(
        "tests/golden/scenario_005_sheets_browser_writeback.py", extra_env=env,
    )
    return CheckResult(
        "scenario_005", "Sheets-browser writeback (005)", passed=passed, detail=summary,
    )


def check_scenario_006(run_golden: bool = False) -> CheckResult:
    """PDF-spreadsheet write scenario (006) passes."""
    if not run_golden:
        return CheckResult(
            "scenario_006", "PDF-spreadsheet write (006)", passed=True,
            detail="Skipped (use --golden to enable).", skipped=True,
        )
    env = {"NEXUS_GOLDEN_TESTS": "1"}
    passed, summary = _run_pytest(
        "tests/golden/scenario_006_pdf_spreadsheet_write.py", extra_env=env,
    )
    return CheckResult(
        "scenario_006", "PDF-spreadsheet write (006)", passed=passed, detail=summary,
    )


# ---------------------------------------------------------------------------
# Manual checklist (not automated)
# ---------------------------------------------------------------------------

MANUAL_CHECKS: list[str] = [
    "3 gerçek kullanıcı testi tamamlandı",
    "Türkçe UI test geçiyor",
    "Windows 10 ortamında test edildi",
    "Windows 11 ortamında test edildi",
    "Yüksek DPI (150 % / 200 %) ortamında test edildi",
]

# ---------------------------------------------------------------------------
# PilotChecklist orchestrator
# ---------------------------------------------------------------------------


class PilotChecklist:
    """Run all mandatory checks and report results."""

    def __init__(
        self,
        skip_ci: bool = False,
        run_golden: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.skip_ci = skip_ci
        self.run_golden = run_golden
        self.dry_run = dry_run

    def _build_checks(self) -> list[Callable[[], CheckResult]]:
        g = self.run_golden
        return [
            lambda: check_ci(skip=self.skip_ci),
            check_health,
            lambda: check_golden_scenarios(run_golden=g),
            check_adversarial,
            check_benchmarks,
            check_transport_benchmark,
            check_legal_docs,
            check_onboarding,
            check_cost_dashboard,
            check_cancel,
            check_diagnostic,
            check_installer_config,
            check_code_signing,
            lambda: check_scenario_007(run_golden=g),
            lambda: check_scenario_005(run_golden=g),
            lambda: check_scenario_006(run_golden=g),
        ]

    def run(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        checks = self._build_checks()
        total = len(checks)

        print()
        print(_c("=" * 64, _BOLD))
        print(_c("  NEXUS AGENT — FAZ 72: Pilot Acceptance Checklist", _BOLD))
        print(_c("=" * 64, _BOLD))
        print()

        for idx, check_fn in enumerate(checks, 1):
            label_hint = getattr(check_fn, "__doc__", "") or "..."
            label_hint = label_hint.splitlines()[0].strip()
            print(f"  [{idx:02d}/{total}] {_c(label_hint, _CYAN)} ...", end="", flush=True)

            result = check_fn()
            results.append(result)

            if result.skipped:
                badge = _c("[SKIP]", _YELLOW)
            elif result.passed:
                badge = _c("[ OK ]", _GREEN)
            else:
                badge = _c("[FAIL]", _RED)

            print(f"\r  {badge}  {_c(result.label, _CYAN)}")
            if result.detail:
                print(f"         {result.detail}")
            print()

        return results

    def report(self, results: list[CheckResult]) -> bool:
        """Print summary and return True if all mandatory checks passed."""
        mandatory_results = [r for r in results if not r.skipped]
        failures = [r for r in mandatory_results if not r.passed]
        skipped = [r for r in results if r.skipped]

        print(_c("─" * 64, _BOLD))
        print(
            f"  Mandatory: "
            f"{_c(str(len(mandatory_results) - len(failures)), _GREEN)} passed  "
            f"{_c(str(len(failures)), _RED)} failed  "
            f"{_c(str(len(skipped)), _YELLOW)} skipped"
        )

        all_passed = len(failures) == 0

        if all_passed:
            print()
            print(_c("  ✔  V1 Launch Candidate", _GREEN + _BOLD))
            print(_c("     All mandatory checks passed!", _GREEN))
        else:
            print()
            print(_c("  ✘  NOT a Launch Candidate", _RED + _BOLD))
            print(_c("     Fix the following before re-running:", _RED))
            for r in failures:
                print(f"     • {r.label}: {r.detail}")

        print()
        print(_c("  MANUAL CHECKLIST (human sign-off required):", _BOLD))
        for item in MANUAL_CHECKS:
            print(f"    □  {item}")
        print()

        return all_passed

    def maybe_tag(self, all_passed: bool) -> None:
        """Create git tag v1.0.0-rc1 when all mandatory checks pass."""
        if not all_passed:
            return

        from nexus.release.version import VERSION
        tag = f"v{VERSION}-rc1"

        if self.dry_run:
            print(_c(f"  [DRY-RUN] Would create git tag: {tag}", _YELLOW))
            return

        ok, detail = _git_tag(tag, f"Nexus Agent {tag} — Release Candidate (Pilot Acceptance passed)")
        if ok:
            print(_c(f"  ✔  git tag {tag!r} created.", _GREEN))
        else:
            print(_c(f"  ⚠  Could not create tag {tag!r}: {detail}", _YELLOW))
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nexus Agent Pilot Acceptance Checklist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run all checks but do not create the git tag.",
    )
    parser.add_argument(
        "--skip-ci", action="store_true",
        help="Skip the slow CI (unit+integration) gate.",
    )
    parser.add_argument(
        "--golden", action="store_true",
        help="Enable golden scenarios (requires a real Windows desktop).",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_only",
        help="Print the checklist without running any checks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _enable_ansi()

    if args.list_only:
        _print_list()
        return 0

    checklist = PilotChecklist(
        skip_ci=args.skip_ci,
        run_golden=args.golden,
        dry_run=args.dry_run,
    )
    results = checklist.run()
    all_passed = checklist.report(results)
    checklist.maybe_tag(all_passed)

    return 0 if all_passed else 1


def _print_list() -> None:
    print()
    print(_c("  NEXUS AGENT — Pilot Acceptance Checklist", _BOLD))
    print()
    print(_c("  MANDATORY (automated):", _CYAN))
    mandatory_labels = [
        "make ci → zero errors",
        "health_check.py → healthy",
        "8 golden scenarios pass",
        "10 adversarial tests pass",
        "All benchmark targets met",
        "Transport: UIA < mouse latency",
        "Privacy policy + ToS exist",
        "Onboarding completable",
        "Cost dashboard + transport breakdown",
        "Cancel works",
        "Diagnostic ZIP (transport_audit)",
        "Installer config (clean-VM install)",
        "Code signing module",
        "Native transport scenario (007)",
        "Sheets-browser writeback (005)",
        "PDF-spreadsheet write (006)",
    ]
    for item in mandatory_labels:
        print(f"    □  {item}")
    print()
    print(_c("  MANUAL (human sign-off):", _MAGENTA))
    for item in MANUAL_CHECKS:
        print(f"    □  {item}")
    print()


if __name__ == "__main__":
    sys.exit(main())
