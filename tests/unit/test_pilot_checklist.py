"""
Unit tests for scripts/pilot_checklist.py

Strategy
--------
- All subprocess.run calls are mocked so tests never spawn real pytest/git.
- File-system checks use tmp_path + monkeypatch.
- PilotChecklist.run() is tested with individual check functions replaced by
  controlled stubs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import pilot_checklist as pc
from pilot_checklist import (
    MANUAL_CHECKS,
    CheckResult,
    PilotChecklist,
    _bench_latest_json,
    _git_tag,
    _run_pytest,
    check_adversarial,
    check_benchmarks,
    check_cancel,
    check_ci,
    check_code_signing,
    check_cost_dashboard,
    check_diagnostic,
    check_health,
    check_installer_config,
    check_legal_docs,
    check_onboarding,
    check_scenario_005,
    check_scenario_006,
    check_scenario_007,
    check_transport_benchmark,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_proc(stdout: str = "5 passed in 0.5s", returncode: int = 0) -> MagicMock:
    p = MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = ""
    return p


def _fail_proc(stderr: str = "1 failed", returncode: int = 1) -> MagicMock:
    p = MagicMock()
    p.returncode = returncode
    p.stdout = ""
    p.stderr = stderr
    return p


# ---------------------------------------------------------------------------
# _run_pytest
# ---------------------------------------------------------------------------


class TestRunPytest:
    def test_returns_true_on_exit_0(self) -> None:
        with patch("subprocess.run", return_value=_ok_proc()) as mock:
            passed, summary = _run_pytest("tests/unit/")
        assert passed is True
        assert "passed" in summary

    def test_returns_false_on_exit_1(self) -> None:
        with patch("subprocess.run", return_value=_fail_proc(returncode=1)):
            passed, _ = _run_pytest("tests/unit/")
        assert passed is False

    def test_extra_env_forwarded(self) -> None:
        with patch("subprocess.run", return_value=_ok_proc()) as mock:
            _run_pytest("tests/golden/", extra_env={"NEXUS_GOLDEN_TESTS": "1"})
        env = mock.call_args.kwargs["env"]
        assert env["NEXUS_GOLDEN_TESTS"] == "1"

    def test_extra_args_appended(self) -> None:
        with patch("subprocess.run", return_value=_ok_proc()) as mock:
            _run_pytest("tests/unit/", extra_args=["--timeout=30"])
        cmd = mock.call_args.args[0]
        assert "--timeout=30" in cmd


# ---------------------------------------------------------------------------
# _bench_latest_json
# ---------------------------------------------------------------------------


class TestBenchLatestJson:
    def test_returns_none_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        assert _bench_latest_json() is None

    def test_parses_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        data = {"results": [{"name": "transport_latency_comparison", "extra": {"uia_avg_ms": 1.0}}]}
        bench_json = tmp_path / "bench_latest.json"
        bench_json.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(pc, "_ROOT", tmp_path)
        # Recreate expected path structure
        results_dir = tmp_path / "tests" / "benchmarks" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "bench_latest.json").write_text(json.dumps(data), encoding="utf-8")
        result = _bench_latest_json()
        assert result is not None
        assert result["results"][0]["name"] == "transport_latency_comparison"

    def test_returns_none_on_bad_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        results_dir = tmp_path / "tests" / "benchmarks" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "bench_latest.json").write_text("NOT JSON", encoding="utf-8")
        monkeypatch.setattr(pc, "_ROOT", tmp_path)
        assert _bench_latest_json() is None


# ---------------------------------------------------------------------------
# _git_tag
# ---------------------------------------------------------------------------


class TestGitTag:
    def test_success(self) -> None:
        with patch("subprocess.run", return_value=_ok_proc(stdout="")) as mock:
            ok, detail = _git_tag("v1.0.0-rc1", "Test tag")
        assert ok is True
        assert "v1.0.0-rc1" in detail

    def test_failure(self) -> None:
        with patch("subprocess.run", return_value=_fail_proc(stderr="tag already exists")):
            ok, detail = _git_tag("v1.0.0-rc1", "Test tag")
        assert ok is False
        assert "already exists" in detail


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


class TestCheckCi:
    def test_skip_flag(self) -> None:
        result = check_ci(skip=True)
        assert result.passed is True
        assert result.skipped is True

    def test_passes_on_exit_0(self) -> None:
        with patch("subprocess.run", return_value=_ok_proc("10 passed in 2s")):
            result = check_ci(skip=False)
        assert result.passed is True

    def test_fails_on_exit_1(self) -> None:
        with patch("subprocess.run", return_value=_fail_proc(returncode=1)):
            result = check_ci(skip=False)
        assert result.passed is False


class TestCheckHealth:
    def test_exit_0_passes(self) -> None:
        with patch("subprocess.run", return_value=_ok_proc("Overall: OK", 0)):
            result = check_health()
        assert result.passed is True

    def test_exit_1_warn_passes(self) -> None:
        # warn (exit 1) is acceptable
        with patch("subprocess.run", return_value=_ok_proc("Overall: WARN", 1)):
            result = check_health()
        assert result.passed is True

    def test_exit_2_fail(self) -> None:
        with patch("subprocess.run", return_value=_fail_proc(returncode=2)):
            result = check_health()
        assert result.passed is False


class TestCheckGoldenScenarios:
    def test_skip_when_not_enabled(self) -> None:
        result = pc.check_golden_scenarios(run_golden=False)
        assert result.skipped is True
        assert result.passed is True

    def test_passes_when_pytest_ok(self) -> None:
        with patch("subprocess.run", return_value=_ok_proc("8 passed in 5s")):
            result = pc.check_golden_scenarios(run_golden=True)
        assert result.passed is True


class TestCheckAdversarial:
    def test_passes(self) -> None:
        with patch("subprocess.run", return_value=_ok_proc("10 passed in 3s")):
            result = check_adversarial()
        assert result.passed is True
        assert result.key == "adversarial"

    def test_fails(self) -> None:
        with patch("subprocess.run", return_value=_fail_proc(returncode=1)):
            result = check_adversarial()
        assert result.passed is False


class TestCheckBenchmarks:
    def test_passes(self) -> None:
        with patch("subprocess.run", return_value=_ok_proc("8 passed in 10s")):
            result = check_benchmarks()
        assert result.passed is True


class TestCheckTransportBenchmark:
    def test_uses_json_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = {
            "results": [{
                "name": "transport_latency_comparison",
                "extra": {
                    "uia_avg_ms": 0.5,
                    "mouse_avg_ms": 2.0,
                    "speedup_ratio": 4.0,
                },
            }]
        }
        monkeypatch.setattr(pc, "_bench_latest_json", lambda: data)
        result = check_transport_benchmark()
        assert result.passed is True
        assert "0.500" in result.detail

    def test_fails_when_uia_slower(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = {
            "results": [{
                "name": "transport_latency_comparison",
                "extra": {
                    "uia_avg_ms": 3.0,
                    "mouse_avg_ms": 1.0,
                    "speedup_ratio": 0.33,
                },
            }]
        }
        monkeypatch.setattr(pc, "_bench_latest_json", lambda: data)
        result = check_transport_benchmark()
        assert result.passed is False

    def test_falls_back_to_pytest_when_no_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pc, "_bench_latest_json", lambda: None)
        with patch("subprocess.run", return_value=_ok_proc("1 passed in 0.5s")):
            result = check_transport_benchmark()
        assert result.passed is True

    def test_falls_back_to_pytest_when_record_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pc, "_bench_latest_json", lambda: {"results": []})
        with patch("subprocess.run", return_value=_ok_proc("1 passed in 0.5s")):
            result = check_transport_benchmark()
        assert result.passed is True


class TestCheckLegalDocs:
    def test_both_present(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "privacy_policy.md").write_text("pp", encoding="utf-8")
        (docs / "terms_of_service.md").write_text("tos", encoding="utf-8")
        monkeypatch.setattr(pc, "_ROOT", tmp_path)
        result = check_legal_docs()
        assert result.passed is True

    def test_missing_one(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "privacy_policy.md").write_text("pp", encoding="utf-8")
        monkeypatch.setattr(pc, "_ROOT", tmp_path)
        result = check_legal_docs()
        assert result.passed is False
        assert "terms_of_service.md" in result.detail

    def test_both_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pc, "_ROOT", tmp_path)
        result = check_legal_docs()
        assert result.passed is False


class TestCheckOnboarding:
    def test_importable(self) -> None:
        result = check_onboarding()
        # nexus.ui.onboarding exists in the project
        assert result.passed is True

    def test_not_importable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib
        original = importlib.import_module

        def _fail(name, *a, **kw):
            if name == "nexus.ui.onboarding":
                raise ImportError("fake error")
            return original(name, *a, **kw)

        monkeypatch.setattr("importlib.import_module", _fail)
        result = check_onboarding()
        assert result.passed is False


class TestCheckCostDashboard:
    def test_importable(self) -> None:
        result = check_cost_dashboard()
        assert result.passed is True


class TestCheckCancel:
    def test_module_and_test_pass(self) -> None:
        with patch("subprocess.run", return_value=_ok_proc("2 passed in 0.5s")):
            result = check_cancel()
        assert result.passed is True

    def test_module_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib
        original = importlib.import_module

        def _fail(name, *a, **kw):
            if name == "nexus.ui.cancel_handler":
                raise ImportError("fake")
            return original(name, *a, **kw)

        monkeypatch.setattr("importlib.import_module", _fail)
        result = check_cancel()
        assert result.passed is False


class TestCheckDiagnostic:
    def test_importable(self) -> None:
        result = check_diagnostic()
        assert result.passed is True


class TestCheckInstallerConfig:
    def test_present(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        installer = tmp_path / "installer"
        installer.mkdir()
        (installer / "nexus_agent.iss").write_text("[Setup]", encoding="utf-8")
        monkeypatch.setattr(pc, "_ROOT", tmp_path)
        result = check_installer_config()
        assert result.passed is True

    def test_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pc, "_ROOT", tmp_path)
        result = check_installer_config()
        assert result.passed is False


class TestCheckCodeSigning:
    def test_importable(self) -> None:
        result = check_code_signing()
        assert result.passed is True


class TestCheckGoldenSpecific:
    def test_007_skipped_without_flag(self) -> None:
        result = check_scenario_007(run_golden=False)
        assert result.skipped is True

    def test_005_skipped_without_flag(self) -> None:
        result = check_scenario_005(run_golden=False)
        assert result.skipped is True

    def test_006_skipped_without_flag(self) -> None:
        result = check_scenario_006(run_golden=False)
        assert result.skipped is True

    def test_007_runs_pytest_when_enabled(self) -> None:
        with patch("subprocess.run", return_value=_ok_proc("1 passed in 0.5s")):
            result = check_scenario_007(run_golden=True)
        assert result.passed is True

    def test_006_fails_when_pytest_fails(self) -> None:
        with patch("subprocess.run", return_value=_fail_proc(returncode=1)):
            result = check_scenario_006(run_golden=True)
        assert result.passed is False


# ---------------------------------------------------------------------------
# PilotChecklist orchestration
# ---------------------------------------------------------------------------


class TestPilotChecklist:
    def _stub_check(self, key: str, passed: bool) -> CheckResult:
        return CheckResult(key=key, label=key, passed=passed, detail="stub")

    def test_run_returns_all_results(self) -> None:
        cl = PilotChecklist(skip_ci=True, run_golden=False, dry_run=True)
        # Replace all check functions with fast stubs
        stub_fn = lambda: CheckResult("stub", "stub", passed=True, detail="ok")  # noqa: E731
        with patch.object(cl, "_build_checks", return_value=[stub_fn] * 5):
            results = cl.run()
        assert len(results) == 5

    def test_report_true_when_all_pass(self, capsys: pytest.CaptureFixture) -> None:
        cl = PilotChecklist(dry_run=True)
        results = [CheckResult(f"k{i}", f"label{i}", passed=True) for i in range(3)]
        all_passed = cl.report(results)
        assert all_passed is True
        captured = capsys.readouterr()
        assert "V1 Launch Candidate" in captured.out

    def test_report_false_when_any_fail(self, capsys: pytest.CaptureFixture) -> None:
        cl = PilotChecklist(dry_run=True)
        results = [
            CheckResult("a", "check a", passed=True),
            CheckResult("b", "check b", passed=False, detail="error detail"),
        ]
        all_passed = cl.report(results)
        assert all_passed is False
        captured = capsys.readouterr()
        assert "NOT a Launch Candidate" in captured.out
        assert "check b" in captured.out

    def test_skipped_not_counted_as_failure(self, capsys: pytest.CaptureFixture) -> None:
        cl = PilotChecklist(dry_run=True)
        results = [
            CheckResult("a", "check a", passed=True),
            CheckResult("b", "check b", passed=True, skipped=True),
        ]
        all_passed = cl.report(results)
        assert all_passed is True

    def test_manual_checklist_always_printed(self, capsys: pytest.CaptureFixture) -> None:
        cl = PilotChecklist(dry_run=True)
        cl.report([CheckResult("x", "x", passed=True)])
        captured = capsys.readouterr()
        for item in MANUAL_CHECKS:
            assert item in captured.out

    def test_dry_run_no_tag(self, capsys: pytest.CaptureFixture) -> None:
        cl = PilotChecklist(dry_run=True)
        with patch("subprocess.run") as mock:
            cl.maybe_tag(all_passed=True)
        mock.assert_not_called()
        captured = capsys.readouterr()
        assert "DRY-RUN" in captured.out

    def test_tag_created_on_success(self) -> None:
        cl = PilotChecklist(dry_run=False)
        with patch("subprocess.run", return_value=_ok_proc("")) as mock:
            cl.maybe_tag(all_passed=True)
        mock.assert_called_once()
        cmd = mock.call_args.args[0]
        assert "tag" in cmd
        assert "-a" in cmd

    def test_no_tag_when_failed(self) -> None:
        cl = PilotChecklist(dry_run=False)
        with patch("subprocess.run") as mock:
            cl.maybe_tag(all_passed=False)
        mock.assert_not_called()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCli:
    def test_list_only_exits_0(self, capsys: pytest.CaptureFixture) -> None:
        exit_code = pc.main(["--list"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "MANDATORY" in captured.out
        assert "MANUAL" in captured.out

    def test_dry_run_flag_parsed(self) -> None:
        args = pc._parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_skip_ci_flag_parsed(self) -> None:
        args = pc._parse_args(["--skip-ci"])
        assert args.skip_ci is True

    def test_golden_flag_parsed(self) -> None:
        args = pc._parse_args(["--golden"])
        assert args.golden is True
