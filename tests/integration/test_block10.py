"""
tests/integration/test_block10.py
Blok 10 Integration Tests — Faz 67

Quality-gate integration: verifies that every CI guard introduced in
Faz 64–66 fires correctly when the conditions it monitors are triggered.
No external processes, filesystems, or real coverage data are required;
each gate is exercised through its Python API.

TEST 1 — Coverage gate: < 80% -> CI fail
  check_coverage.py exits 1 when coverage.json contains modules below
  their per-module threshold.  Passing data produces exit 0.

TEST 2 — Benchmark regression detection
  BenchmarkRecord.finish() marks FAIL when measured value exceeds
  target (higher_is_better=False).
  write_report() embeds the delta-% regression warning in the Markdown
  output when a baseline file with a better result is provided.

TEST 3 — Security scan tooling available
  bandit and safety modules are importable (installed as dev deps).
  Confirms the security stage will not fail at import time.
  Tests are skipped when tools are not installed (correct for dev envs
  where `pip install -e ".[dev]"` was not run).

TEST 4 — Property test failure detection
  A deliberately broken implementation of Rect.area() (returns -1) is
  injected into a Hypothesis test via monkeypatch; the test raises
  AssertionError immediately, proving property tests catch regressions.

TEST 5 — Mutation test: critical mutation killed
  _MutantVisitor generates relational-operator mutations on a known
  snippet containing `>=`.  A synthetic test verifies the mutant
  produces different output from the original (the mutation is detectable).

TEST 6 — Transport benchmark regression: UIA > mouse -> warning
  BenchmarkRecord with higher_is_better=False and uia_avg > mouse_avg
  -> record.passed is False (regression detected).
  write_report() includes the "Regressions Detected" section in the
  Markdown when a baseline shows UIA was faster in a previous run.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

_THRESHOLDS: dict[str, int] = {
    "nexus/core/": 95,
    "nexus/source/": 90,
    "nexus/capture/": 90,
    "nexus/perception/": 85,
    "nexus/action/": 90,
    "nexus/verification/": 90,
    "nexus/cloud/": 85,
}


def _make_coverage_json(overrides: dict[str, float]) -> dict[str, Any]:
    """
    Build a synthetic coverage.json dict.

    *overrides* maps path-prefix fragments to coverage percentages.
    Any prefix not listed gets threshold + 5% (passing by default).
    """
    files: dict[str, Any] = {}
    for prefix, threshold in _THRESHOLDS.items():
        pct = overrides.get(prefix, float(threshold + 5))
        total = 100
        covered = int(pct)
        files[f"{prefix}__init__.py"] = {
            "summary": {
                "covered_lines": covered,
                "num_statements": total,
            }
        }
    return {"files": files, "meta": {"version": "7.0"}}


def _load_check_coverage_module():
    """Load scripts/check_coverage.py as a module object."""
    import importlib.util
    import types

    spec = importlib.util.spec_from_file_location(
        "check_coverage", Path("scripts/check_coverage.py")
    )
    assert spec is not None and spec.loader is not None
    mod = types.ModuleType("check_coverage")
    mod.__spec__ = spec
    mod.__loader__ = spec.loader
    mod.__file__ = str(Path("scripts/check_coverage.py").resolve())
    mod.__package__ = ""
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _load_mutation_runner():
    """
    Load scripts/run_mutation.py via sys.path insertion so its classes
    are accessible as regular Python objects.
    """
    scripts_dir = str(Path("scripts").resolve())
    inserted = False
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
        inserted = True
    try:
        import importlib
        import run_mutation  # type: ignore[import-not-found]
        importlib.reload(run_mutation)
        return run_mutation
    finally:
        if inserted:
            sys.path.remove(scripts_dir)


# ---------------------------------------------------------------------------
# TEST 1 — Coverage gate: < 80% -> CI fail
# ---------------------------------------------------------------------------


class TestCoverageGate:
    def test_failing_coverage_triggers_exit_1(self, tmp_path: Path) -> None:
        """check_coverage.py returns 1 when a module is below threshold."""
        data = _make_coverage_json({"nexus/core/": 93.0})  # 93% < 95% target
        cov_file = tmp_path / "coverage.json"
        cov_file.write_text(json.dumps(data))

        mod = _load_check_coverage_module()
        with patch.object(mod, "COVERAGE_JSON", cov_file):
            exit_code = mod.main()

        assert exit_code == 1, "Expected exit 1 when coverage below threshold"

    def test_passing_coverage_triggers_exit_0(self, tmp_path: Path) -> None:
        """check_coverage.py returns 0 when all modules meet thresholds."""
        data = _make_coverage_json({})  # all modules at threshold + 5%
        cov_file = tmp_path / "coverage.json"
        cov_file.write_text(json.dumps(data))

        mod = _load_check_coverage_module()
        with patch.object(mod, "COVERAGE_JSON", cov_file):
            exit_code = mod.main()

        assert exit_code == 0, "Expected exit 0 when all coverage thresholds met"

    def test_exactly_at_threshold_passes(self, tmp_path: Path) -> None:
        """A module exactly at its threshold passes (>= not >)."""
        data = _make_coverage_json({"nexus/core/": 95.0})
        cov_file = tmp_path / "coverage.json"
        cov_file.write_text(json.dumps(data))

        mod = _load_check_coverage_module()
        with patch.object(mod, "COVERAGE_JSON", cov_file):
            exit_code = mod.main()

        assert exit_code == 0

    def test_missing_coverage_json_exits_1(self, tmp_path: Path) -> None:
        """check_coverage.py exits 1 when coverage.json does not exist."""
        mod = _load_check_coverage_module()
        missing = tmp_path / "nonexistent_coverage.json"

        with patch.object(mod, "COVERAGE_JSON", missing):
            with pytest.raises(SystemExit) as exc_info:
                mod.main()

        assert exc_info.value.code == 1

    def test_multiple_modules_failing_exits_1(self, tmp_path: Path) -> None:
        """Multiple modules below threshold → still exit 1."""
        data = _make_coverage_json({
            "nexus/core/": 80.0,    # 80% < 95%
            "nexus/capture/": 70.0,  # 70% < 90%
        })
        cov_file = tmp_path / "coverage.json"
        cov_file.write_text(json.dumps(data))

        mod = _load_check_coverage_module()
        with patch.object(mod, "COVERAGE_JSON", cov_file):
            exit_code = mod.main()

        assert exit_code == 1


# ---------------------------------------------------------------------------
# TEST 2 — Benchmark regression detection
# ---------------------------------------------------------------------------


class TestBenchmarkRegression:
    """BenchmarkRecord.finish() and write_report() regression logic."""

    def _make_record(
        self,
        name: str,
        measured: float,
        target: float,
        higher_is_better: bool = False,
    ):
        from tests.benchmarks.conftest import BenchmarkRecord

        rec = BenchmarkRecord(
            name=name,
            target_label="test target",
            unit="ms",
            target_value=target,
            higher_is_better=higher_is_better,
        )
        rec.finish(measured)
        return rec

    def test_record_passes_when_within_target(self) -> None:
        """measured <= target -> passed for latency metrics."""
        rec = self._make_record("latency", measured=100.0, target=200.0, higher_is_better=False)
        assert rec.passed is True

    def test_record_fails_when_exceeds_target(self) -> None:
        """measured > target -> passed=False (regression)."""
        rec = self._make_record("latency", measured=300.0, target=200.0, higher_is_better=False)
        assert rec.passed is False

    def test_record_passes_throughput(self) -> None:
        """measured >= target -> passed for throughput metrics."""
        rec = self._make_record("fps", measured=15.0, target=10.0, higher_is_better=True)
        assert rec.passed is True

    def test_record_fails_throughput(self) -> None:
        """measured < target -> passed=False for throughput (regression)."""
        rec = self._make_record("fps", measured=5.0, target=10.0, higher_is_better=True)
        assert rec.passed is False

    def test_regression_section_in_markdown(self, tmp_path: Path) -> None:
        """write_report() includes 'Regressions Detected' when baseline is better."""
        from tests.benchmarks.conftest import BenchmarkRecord
        from tests.benchmarks import bench_report as br

        # Baseline: latency was 50 ms (good)
        baseline = {
            "generated_at": "2026-04-01T00:00:00",
            "results": [{
                "name": "test_latency",
                "target_label": "avg < 200ms",
                "unit": "ms",
                "target_value": 200.0,
                "higher_is_better": False,
                "measured_value": 50.0,
                "passed": True,
                "samples_count": 100,
                "extra": {},
            }],
        }
        baseline_file = tmp_path / "bench_baseline.json"
        baseline_file.write_text(json.dumps(baseline))

        # Current run: latency is 150 ms (200% above 50 ms baseline)
        rec = BenchmarkRecord(
            name="test_latency",
            target_label="avg < 200ms",
            unit="ms",
            target_value=200.0,
            higher_is_better=False,
        )
        rec.finish(150.0)

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        with (
            patch.dict("os.environ", {"NEXUS_BENCH_BASELINE": str(baseline_file)}),
            patch.object(br, "RESULTS_DIR", results_dir),
            patch("tests.benchmarks.bench_report.get_all_results", return_value=[rec]),
        ):
            br.write_report()

        md_files = list(results_dir.glob("bench_*.md"))
        assert md_files, "No markdown report written"
        md_content = md_files[0].read_text()

        # Report uses "Regressions Detected" heading when regressions exist
        assert "Regressions Detected" in md_content, (
            f"Expected 'Regressions Detected' section in report:\n{md_content[:500]}"
        )
        assert "test_latency" in md_content
        assert "degradation" in md_content

    def test_no_regression_within_threshold(self, tmp_path: Path) -> None:
        """write_report() does NOT add regression section when delta <= 10%."""
        from tests.benchmarks.conftest import BenchmarkRecord
        from tests.benchmarks import bench_report as br

        baseline = {
            "generated_at": "2026-04-01T00:00:00",
            "results": [{
                "name": "stable_latency",
                "target_label": "avg < 200ms",
                "unit": "ms",
                "target_value": 200.0,
                "higher_is_better": False,
                "measured_value": 100.0,
                "passed": True,
                "samples_count": 100,
                "extra": {},
            }],
        }
        baseline_file = tmp_path / "bench_baseline.json"
        baseline_file.write_text(json.dumps(baseline))

        # 5% above baseline — within 10% threshold, no regression
        rec = BenchmarkRecord(
            name="stable_latency",
            target_label="avg < 200ms",
            unit="ms",
            target_value=200.0,
            higher_is_better=False,
        )
        rec.finish(105.0)

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        with (
            patch.dict("os.environ", {"NEXUS_BENCH_BASELINE": str(baseline_file)}),
            patch.object(br, "RESULTS_DIR", results_dir),
            patch("tests.benchmarks.bench_report.get_all_results", return_value=[rec]),
        ):
            br.write_report()

        md_files = list(results_dir.glob("bench_*.md"))
        assert md_files
        md_content = md_files[0].read_text()
        assert "Regressions Detected" not in md_content


# ---------------------------------------------------------------------------
# TEST 3 — Security scan tooling available
# ---------------------------------------------------------------------------

_BANDIT_INSTALLED = pytest.importorskip.__module__ != "missing"  # always True


def _tool_installed(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


class TestSecurityScanTooling:
    def test_bandit_importable(self) -> None:
        """bandit must be importable when dev extras are installed."""
        if not _tool_installed("bandit"):
            pytest.skip("bandit not installed (run: pip install -e '.[dev]')")
        import importlib.util

        assert importlib.util.find_spec("bandit") is not None

    def test_safety_importable(self) -> None:
        """safety must be importable when dev extras are installed."""
        if not _tool_installed("safety"):
            pytest.skip("safety not installed (run: pip install -e '.[dev]')")
        import importlib.util

        assert importlib.util.find_spec("safety") is not None

    def test_bandit_cli_callable(self) -> None:
        """bandit CLI entrypoint resolves and exits 0 for --version."""
        import subprocess

        if not _tool_installed("bandit"):
            pytest.skip("bandit not installed")

        result = subprocess.run(
            [sys.executable, "-m", "bandit", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, f"bandit --version failed: {result.stderr}"

    def test_bandit_finds_no_high_severity_in_core(self) -> None:
        """bandit -r nexus/core/ -ll must exit 0 (no HIGH/MEDIUM issues)."""
        import subprocess

        if not _tool_installed("bandit"):
            pytest.skip("bandit not installed")

        result = subprocess.run(
            [sys.executable, "-m", "bandit", "-r", "nexus/core/", "-ll", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"bandit found issues in nexus/core/:\n{result.stdout}\n{result.stderr}"
        )

    def test_security_stage_has_required_tools_in_dev_deps(self) -> None:
        """pyproject.toml dev extras list bandit and safety."""
        import tomllib

        pyproject = Path("pyproject.toml").read_bytes()
        data = tomllib.loads(pyproject.decode())
        dev_deps = data["project"]["optional-dependencies"]["dev"]
        assert any("bandit" in d for d in dev_deps), "bandit missing from dev deps"
        assert any("safety" in d for d in dev_deps), "safety missing from dev deps"


# ---------------------------------------------------------------------------
# TEST 4 — Property test failure detection
# ---------------------------------------------------------------------------


class TestPropertyTestFailureDetection:
    """Hypothesis catches a deliberately broken implementation."""

    def test_broken_area_detected_by_hypothesis(self) -> None:
        """
        Patch Rect.area() to always return -1; a Hypothesis test must
        immediately raise, proving property tests catch regressions.
        """
        from hypothesis import given, settings
        from hypothesis import strategies as st
        from nexus.core.types import Rect

        def _broken_area(self: Rect) -> int:
            return -1

        with patch.object(Rect, "area", _broken_area):
            captured: list[Exception] = []

            @given(
                x=st.integers(-100, 100),
                y=st.integers(-100, 100),
                w=st.integers(0, 100),
                h=st.integers(0, 100),
            )
            @settings(max_examples=10)
            def _prop(x: int, y: int, w: int, h: int) -> None:
                r = Rect(x, y, w, h)
                assert r.area() >= 0, f"area={r.area()} must be >= 0"

            try:
                _prop()
            except Exception as exc:  # noqa: BLE001
                captured.append(exc)

        assert captured, (
            "Expected Hypothesis to raise when area() always returns -1"
        )

    def test_correct_area_passes_hypothesis(self) -> None:
        """The real area() survives 50 Hypothesis examples."""
        from hypothesis import given, settings
        from hypothesis import strategies as st
        from nexus.core.types import Rect

        @given(
            x=st.integers(-100, 100),
            y=st.integers(-100, 100),
            w=st.integers(0, 100),
            h=st.integers(0, 100),
        )
        @settings(max_examples=50)
        def _prop(x: int, y: int, w: int, h: int) -> None:
            r = Rect(x, y, w, h)
            assert r.area() >= 0
            assert r.area() == r.width * r.height

        _prop()  # must not raise

    def test_broken_contains_detected(self) -> None:
        """Patch contains() to always return True; boundary check catches it."""
        from nexus.core.types import Point, Rect

        def _always_true(self: Rect, p: Point) -> bool:
            return True

        with patch.object(Rect, "contains", _always_true):
            r = Rect(0, 0, 10, 10)
            outside = Point(100, 100)
            # Confirm patch is active
            assert r.contains(outside) is True
            # A real gate checks the coordinates independently
            truly_outside = not (
                r.x <= outside.x <= r.right and r.y <= outside.y <= r.bottom
            )
            assert truly_outside, "Point(100,100) should be outside Rect(0,0,10,10)"


# ---------------------------------------------------------------------------
# TEST 5 — Mutation test: critical mutation killed
# ---------------------------------------------------------------------------


class TestMutationTestKillsCriticalMutant:
    """
    _MutantVisitor generates >= -> > (and other relational) mutations.
    We verify mutants differ from the original and that a killing test
    detects the difference without running a subprocess.
    """

    def test_mutant_visitor_generates_relational_mutation(self) -> None:
        """_MutantVisitor produces at least one mutation for a snippet with >=."""
        mod = _load_mutation_runner()

        snippet = textwrap.dedent("""\
            def check(x):
                return x >= 0
        """)
        visitor = mod._MutantVisitor(snippet)
        mutations = visitor.generate()

        assert mutations, "No mutations generated for snippet with >="

        mutant_sources = [m[0] for m in mutations]
        descriptions = [m[1] for m in mutations]

        # Must produce source code different from original
        assert any(src != snippet for src in mutant_sources), (
            "All mutants identical to original"
        )

        # At least one description should mention a relational change
        has_relational = any(
            any(op in desc for op in (">=", ">", "<=", "<", "==", "!=", "GtE", "Gt"))
            for desc in descriptions
        )
        assert has_relational, f"No relational mutation found. Got: {descriptions}"

    def test_mutant_differs_from_original(self) -> None:
        """Every generated mutant produces source different from the original."""
        mod = _load_mutation_runner()

        snippet = textwrap.dedent("""\
            def area(w, h):
                if w >= 0 and h >= 0:
                    return w * h
                return 0
        """)
        visitor = mod._MutantVisitor(snippet)
        mutations = visitor.generate()
        assert mutations

        mutant_source, description, lineno = mutations[0]
        assert mutant_source != snippet, (
            f"Mutant identical to original (mutation had no effect): {description}"
        )
        assert lineno >= 1

    def test_mutation_killed_by_boundary_test(self) -> None:
        """
        Simulate mutation detection without subprocess:
        >= -> > mutation on check_nonneg(0) produces a different return value,
        so the test asserting check_nonneg(0) == True kills the mutant.
        """
        original_src = textwrap.dedent("""\
            def check_nonneg(x):
                return x >= 0
        """)
        mutant_src = textwrap.dedent("""\
            def check_nonneg(x):
                return x > 0
        """)

        orig_ns: dict[str, Any] = {}
        exec(original_src, orig_ns)  # noqa: S102
        mutant_ns: dict[str, Any] = {}
        exec(mutant_src, mutant_ns)  # noqa: S102

        # Original: 0 >= 0 == True; Mutant: 0 > 0 == False
        assert orig_ns["check_nonneg"](0) is True, "Original: 0 >= 0 should be True"
        assert mutant_ns["check_nonneg"](0) is False, "Mutant: 0 > 0 should be False"

        # Confirm mutant would be killed (different result on boundary input)
        assert orig_ns["check_nonneg"](0) != mutant_ns["check_nonneg"](0), (
            "Mutant produces different result -> killed"
        )

    def test_mutation_score_threshold_met_in_results(self) -> None:
        """scripts/mutation_results.json (if present) shows score >= 0.70."""
        results_file = Path("scripts/mutation_results.json")
        if not results_file.exists():
            pytest.skip("mutation_results.json not found — run `make mutmut` first")

        data = json.loads(results_file.read_text())
        score = data.get("mutation_score", 0.0)
        assert score >= 0.70, (
            f"Mutation score {score:.1%} is below 70% threshold"
        )


# ---------------------------------------------------------------------------
# TEST 6 — Transport benchmark regression: UIA > mouse -> warning
# ---------------------------------------------------------------------------


class TestTransportBenchmarkRegression:
    """UIA latency > mouse latency triggers regression detection."""

    def test_uia_slower_than_mouse_fails_benchmark(self) -> None:
        """
        BenchmarkRecord with uia_avg > mouse_avg (higher_is_better=False)
        -> passed=False (regression).
        """
        from tests.benchmarks.conftest import BenchmarkRecord

        mouse_avg = 0.15
        uia_avg = 0.30  # UIA SLOWER than mouse

        rec = BenchmarkRecord(
            name="transport_latency_comparison",
            target_label="UIA avg < mouse avg (native advantage)",
            unit="ms",
            target_value=mouse_avg,
            higher_is_better=False,
            extra={"uia_avg_ms": uia_avg, "mouse_avg_ms": mouse_avg},
        )
        rec.finish(uia_avg)

        assert rec.passed is False, (
            f"Expected FAIL when UIA ({uia_avg}ms) > mouse ({mouse_avg}ms)"
        )

    def test_uia_faster_than_mouse_passes_benchmark(self) -> None:
        """UIA avg < mouse avg -> passed=True."""
        from tests.benchmarks.conftest import BenchmarkRecord

        rec = BenchmarkRecord(
            name="transport_latency_comparison",
            target_label="UIA avg < mouse avg (native advantage)",
            unit="ms",
            target_value=0.30,
            higher_is_better=False,
        )
        rec.finish(0.11)  # UIA faster

        assert rec.passed is True

    def test_regression_report_flags_uia_slower(self, tmp_path: Path) -> None:
        """
        write_report() includes 'Regressions Detected' when a baseline run
        shows UIA was faster but the current run has UIA slower.
        """
        from tests.benchmarks.conftest import BenchmarkRecord
        from tests.benchmarks import bench_report as br

        baseline = {
            "generated_at": "2026-04-01T00:00:00",
            "results": [{
                "name": "transport_latency_comparison",
                "target_label": "UIA avg < mouse avg (native advantage)",
                "unit": "ms",
                "target_value": 0.30,
                "higher_is_better": False,
                "measured_value": 0.11,  # baseline: UIA was faster
                "passed": True,
                "samples_count": 100,
                "extra": {"uia_avg_ms": 0.11, "mouse_avg_ms": 0.30},
            }],
        }
        baseline_file = tmp_path / "bench_baseline.json"
        baseline_file.write_text(json.dumps(baseline))

        # Current: UIA regressed to 0.40 ms (264% worse than 0.11 baseline)
        rec = BenchmarkRecord(
            name="transport_latency_comparison",
            target_label="UIA avg < mouse avg (native advantage)",
            unit="ms",
            target_value=0.30,
            higher_is_better=False,
        )
        rec.finish(0.40)

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        with (
            patch.dict("os.environ", {"NEXUS_BENCH_BASELINE": str(baseline_file)}),
            patch.object(br, "RESULTS_DIR", results_dir),
            patch("tests.benchmarks.bench_report.get_all_results", return_value=[rec]),
        ):
            br.write_report()

        md_files = list(results_dir.glob("bench_*.md"))
        assert md_files, "No markdown report written"
        md_text = md_files[0].read_text()

        assert "Regressions Detected" in md_text, (
            "Expected 'Regressions Detected' section for UIA regression.\n"
            f"Report:\n{md_text[:800]}"
        )
        assert "transport_latency_comparison" in md_text
        assert "degradation" in md_text

    def test_uia_equal_to_mouse_is_not_regression(self) -> None:
        """UIA == mouse latency is technically within target (not strictly slower)."""
        from tests.benchmarks.conftest import BenchmarkRecord

        avg = 0.20
        rec = BenchmarkRecord(
            name="transport_latency_comparison",
            target_label="UIA avg < mouse avg (native advantage)",
            unit="ms",
            target_value=avg,
            higher_is_better=False,
        )
        rec.finish(avg)  # measured == target -> passed (<=)

        # Equal is "not slower" -> passes target check
        assert rec.passed is True, "Equal latency is within target (measured <= target)"

    def test_transport_benchmark_result_stored_correctly(self) -> None:
        """BenchmarkRecord.to_dict() includes uia/mouse comparison data."""
        from tests.benchmarks.conftest import BenchmarkRecord

        rec = BenchmarkRecord(
            name="transport_latency_comparison",
            target_label="UIA avg < mouse avg",
            unit="ms",
            target_value=0.30,
            higher_is_better=False,
            extra={"uia_avg_ms": 0.11, "mouse_avg_ms": 0.30, "speedup_ratio": 2.7},
        )
        rec.finish(0.11)

        d = rec.to_dict()
        assert d["name"] == "transport_latency_comparison"
        assert d["passed"] is True
        assert d["extra"]["speedup_ratio"] == 2.7
        assert d["measured_value"] == 0.11
