"""
tests/benchmarks/bench_report.py
Benchmark Report Generator — FAZ 64

Collects results from all bench_*.py modules, writes:
  - tests/benchmarks/results/bench_<timestamp>.json
  - tests/benchmarks/results/bench_latest.json  (symlink / copy)
  - tests/benchmarks/results/bench_<timestamp>.md

Regression detection: if NEXUS_BENCH_BASELINE env var points to a prior JSON
report, each metric is compared and a WARNING is emitted when the new value is
worse than the baseline by more than _REGRESSION_THRESHOLD_PCT.

Run standalone:
    python -m pytest tests/benchmarks/ && python tests/benchmarks/bench_report.py

Or integrated via the pytest session-finish hook below (auto-runs after
`pytest tests/benchmarks/`).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when run standalone
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.benchmarks.conftest import RESULTS_DIR, BenchmarkRecord, get_all_results

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REGRESSION_THRESHOLD_PCT: float = 10.0   # warn if metric degrades > 10%
_TIMESTAMP_FMT: str = "%Y%m%d_%H%M%S"


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _is_regression(record: BenchmarkRecord, baseline_value: float) -> bool:
    """
    Return True if *record* represents a regression vs *baseline_value*.

    higher_is_better=True  → regression if new < baseline * (1 - threshold)
    higher_is_better=False → regression if new > baseline * (1 + threshold)
    """
    new = record.measured_value
    threshold = _REGRESSION_THRESHOLD_PCT / 100.0
    if record.higher_is_better:
        return new < baseline_value * (1.0 - threshold)
    else:
        return new > baseline_value * (1.0 + threshold)


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _build_report(records: list[BenchmarkRecord]) -> dict[str, Any]:
    ts = datetime.now(UTC).isoformat()
    return {
        "generated_at": ts,
        "all_passed": all(r.passed for r in records),
        "results": [r.to_dict() for r in records],
    }


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

_STATUS = {True: "PASS", False: "FAIL"}
_ARROW = {True: "↑ higher is better", False: "↓ lower is better"}


def _build_markdown(
    report: dict[str, Any],
    baseline: dict[str, Any] | None,
    regressions: list[str],
) -> str:
    lines: list[str] = []
    ts = report["generated_at"]
    overall = "ALL PASS" if report["all_passed"] else "SOME FAILURES"

    lines.append("# Nexus Agent — Performance Benchmark Report")
    lines.append(f"\nGenerated: `{ts}`  |  Status: **{overall}**\n")

    if regressions:
        lines.append("## Regressions Detected\n")
        for r in regressions:
            lines.append(f"- {r}")
        lines.append("")

    lines.append("## Results\n")
    lines.append(
        "| Benchmark | Measured | Target | Unit | Direction | Status |"
    )
    lines.append(
        "|-----------|----------|--------|------|-----------|--------|"
    )

    for res in report["results"]:
        name = res["name"]
        measured = res["measured_value"]
        target = res["target_value"]
        unit = res["unit"]
        direction = _ARROW[res["higher_is_better"]]
        status = _STATUS[res["passed"]]
        status_md = f"**{status}**" if not res["passed"] else status

        # Baseline delta
        delta = ""
        if baseline:
            for br in baseline.get("results", []):
                if br["name"] == name:
                    prev = br["measured_value"]
                    if prev:
                        diff_pct = (measured - prev) / abs(prev) * 100
                        sign = "+" if diff_pct >= 0 else ""
                        delta = f" ({sign}{diff_pct:.1f}% vs baseline)"
                    break

        lines.append(
            f"| {name} | {measured:.4g}{delta} | {target:.4g} | {unit} "
            f"| {direction} | {status_md} |"
        )

    lines.append("")

    # Extra details
    lines.append("## Detail\n")
    for res in report["results"]:
        lines.append(f"### {res['name']}")
        lines.append(f"- **Target**: {res['target_label']}")
        lines.append(f"- **Samples**: {res['samples_count']}")
        for k, v in res.get("extra", {}).items():
            if k != "top_leaks":
                lines.append(f"- **{k}**: {v}")
        top_leaks = res.get("extra", {}).get("top_leaks")
        if top_leaks:
            lines.append("- **top_leaks**:")
            for leak in top_leaks:
                lines.append(f"  - {leak['location']} → {leak['size_kb']} KB")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main report writer
# ---------------------------------------------------------------------------


def write_report() -> Path:
    """
    Write JSON + Markdown reports from collected benchmark results.

    Returns the path to the JSON report.
    """
    records = get_all_results()
    if not records:
        print("[bench_report] No benchmark results collected — skipping report.")
        return RESULTS_DIR / "bench_empty.json"

    report = _build_report(records)

    # Load baseline if configured
    baseline_path = os.environ.get("NEXUS_BENCH_BASELINE")
    baseline: dict[str, Any] | None = None
    if baseline_path and Path(baseline_path).exists():
        with open(baseline_path) as fh:
            baseline = json.load(fh)
        print(f"[bench_report] Comparing against baseline: {baseline_path}")

    # Detect regressions
    regressions: list[str] = []
    if baseline:
        baseline_by_name = {r["name"]: r for r in baseline.get("results", [])}
        for rec in records:
            br = baseline_by_name.get(rec.name)
            if br is not None and _is_regression(rec, br["measured_value"]):
                direction = "↓" if rec.higher_is_better else "↑"
                regressions.append(
                    f"{rec.name}: {rec.measured_value:.4g} {rec.unit} "
                    f"{direction} vs baseline {br['measured_value']:.4g} "
                    f"(>{_REGRESSION_THRESHOLD_PCT}% degradation)"
                )

    if regressions:
        print("\n[bench_report] WARNING — REGRESSIONS DETECTED:")
        for r in regressions:
            print(f"  !! {r}")
        print()

    # Write files
    ts = datetime.now(UTC).strftime(_TIMESTAMP_FMT)
    json_path = RESULTS_DIR / f"bench_{ts}.json"
    md_path = RESULTS_DIR / f"bench_{ts}.md"
    latest_path = RESULTS_DIR / "bench_latest.json"

    json_text = json.dumps(report, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    shutil.copy(json_path, latest_path)

    md_text = _build_markdown(report, baseline, regressions)
    md_path.write_text(md_text, encoding="utf-8")

    print(f"[bench_report] JSON   : {json_path}")
    print(f"[bench_report] MD     : {md_path}")
    print(f"[bench_report] Latest : {latest_path}")

    # Summary
    passed = sum(1 for r in records if r.passed)
    total = len(records)
    print(f"[bench_report] {passed}/{total} benchmarks passed")
    if regressions:
        print(f"[bench_report] {len(regressions)} regression(s) detected — review required")

    return json_path


# ---------------------------------------------------------------------------
# pytest session-finish hook (auto-invoked after bench suite)
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:  # noqa: ANN401
    """Write report at the end of every pytest session that ran benchmarks."""
    if get_all_results():
        write_report()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    write_report()
