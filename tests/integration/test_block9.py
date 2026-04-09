"""
tests/integration/test_block9.py
Blok 9 Integration Tests — Faz 61

UI layer pipelines: onboarding, dashboard, diagnostic, crash handler,
cancel handler, and privacy transparency.  No real terminal I/O, OS
signals, filesystem, or network calls are exercised.

TEST 1 — Onboarding + browser setup
  OnboardingFlow.run() completes all 8 steps with injected stubs.
  Browser setup step calls _launch_browser_fn.
  Both privacy + terms consents saved to store.

TEST 2 — Cost dashboard transport breakdown
  CostTracker records native (uia) and fallback (mouse) transport actions.
  TaskDashboard.show_cost_dashboard() renders the correct percentages.

TEST 3 — Diagnostic ZIP + transport_audit
  DiagnosticReporter.export_zip() includes transport_audit_last_100.json.
  Confirm transport rows are present and API key is absent.

TEST 4 — Crash handler
  CrashHandler.install() + handle_crash():
    _write_crash_fn called with full traceback text.
    _generate_diagnostic_fn called with zip path.
    Output contains "Tanı dosyası".

TEST 5 — Cancel flow
  CancelHandler.install(executor) + two SIGINT calls:
    First  → executor.cancel() called, cancel_requested True.
    Second → _force_quit_fn called.

TEST 6 — Privacy: native transport message
  ScreenshotMasker.mask(..., transport="uia"):
    Log event == "screenshot_not_sent".
  PrivacyTransparencyScreen.show_cloud_call_info(..., transport="uia"):
    Output contains "veri gönderilmedi".
    Output does NOT contain "ekran görüntüsü gönderildi".
"""
from __future__ import annotations

import signal
from datetime import date, datetime
from typing import Any

import numpy as np
import pytest

from nexus.core.screenshot_masker import ScreenshotMasker
from nexus.core.task_executor import TaskResult, TransportStats
from nexus.infra.cost_tracker import (
    CostTracker,
    DashboardData,
    DayTotal,
    TaskCost,
    TransportBreakdown,
)
from nexus.infra.diagnostic import DiagnosticReporter, sanitize_settings
from nexus.infra.health import CheckResult, HealthReport
from nexus.ui.cancel_handler import CancelHandler
from nexus.ui.dashboard import SuspendedTask, TaskDashboard, TaskHistoryEntry
from nexus.ui.notifications import NotificationManager
from nexus.ui.onboarding import OnboardingFlow
from nexus.ui.privacy import PrivacyTransparencyScreen

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_API_KEY = "sk-ant-secret-integration-test"


def _ok_health() -> HealthReport:
    return HealthReport(
        checks=[
            CheckResult(
                name="CHECK_PYTHON_VERSION",
                status="ok",
                message="Python 3.14 — OK",
            )
        ]
    )


# ---------------------------------------------------------------------------
# TEST 1 — Onboarding + browser setup
# ---------------------------------------------------------------------------


class TestOnboardingBrowserSetup:
    """
    Full onboarding run with browser setup accepted.
    Consent store receives both 'privacy' and 'terms'.
    _launch_browser_fn is called exactly once.
    """

    def test_full_run_with_browser_launch(self):
        store: dict[str, bool] = {}
        launch_calls: list[bool] = []
        prompts = [
            "",          # welcome: Enter
            "e",         # privacy: accept
            "e",         # terms: accept
            "1",         # provider: anthropic
            "sk-ant-key",# api key
            "e",         # browser setup: yes
        ]

        flow = OnboardingFlow(
            _print_fn=lambda _: None,
            _prompt_fn=lambda _: prompts.pop(0) if prompts else "",
            _has_consent_fn=lambda scope: store.get(scope, False),
            _save_consent_fn=lambda scope: store.update({scope: True}),
            _health_check_fn=lambda: _ok_health(),
            _validate_key_fn=lambda p, k: (True, "ok"),
            _launch_browser_fn=lambda: launch_calls.append(True) or True,  # type: ignore[func-returns-value]
            _test_api_fn=lambda p, k: (True, 0.001),
        )

        result = flow.run()

        assert result is True
        assert store.get("privacy") is True
        assert store.get("terms") is True
        assert len(launch_calls) == 1, "Browser launch must be called exactly once"

    def test_returning_user_skips_all_steps(self):
        """Both consents stored → run() returns True, no prompts issued."""
        store = {"privacy": True, "terms": True}
        prompts_issued: list[str] = []

        flow = OnboardingFlow(
            _print_fn=lambda _: None,
            _prompt_fn=lambda label: prompts_issued.append(label) or "",  # type: ignore[func-returns-value]
            _has_consent_fn=lambda scope: store.get(scope, False),
            _save_consent_fn=lambda scope: None,
        )

        assert flow.run() is True
        assert prompts_issued == []


# ---------------------------------------------------------------------------
# TEST 2 — Cost dashboard transport breakdown
# ---------------------------------------------------------------------------


class TestCostDashboardTransportBreakdown:
    """
    TaskDashboard.show_cost_dashboard renders correct native/fallback %.
    """

    def test_breakdown_percentages_correct(self):
        # 70 native, 30 fallback → 70 % / 30 %
        data = DashboardData(
            today_total_usd=0.50,
            last_7_days=[
                DayTotal(date=date(2026, 4, d), cost_usd=0.07 * d, call_count=d)
                for d in range(3, 10)
            ],
            last_10_tasks=[],
            top_5_expensive=[
                TaskCost(
                    task_id=f"task-{i:03d}",
                    cost_usd=float(i + 1) * 0.1,
                    call_count=i + 1,
                    first_call_at=datetime(2026, 4, 9),
                )
                for i in range(3)
            ],
            transport_breakdown=TransportBreakdown(
                cloud_calls=50,
                cloud_cost_usd=0.50,
                native_calls=70,
                fallback_calls=30,
                native_ratio=0.70,
            ),
        )
        printed: list[str] = []
        dash = TaskDashboard(
            _print_fn=lambda t: printed.append(t),
            _get_dashboard_data_fn=lambda: data,
        )
        dash.show_cost_dashboard()
        text = "\n".join(printed)

        assert "70.0%" in text, "Native percentage missing"
        assert "30.0%" in text, "Fallback percentage missing"

    def test_zero_transport_shows_zero_percent(self):
        data = DashboardData(
            today_total_usd=0.0,
            last_7_days=[],
            last_10_tasks=[],
            top_5_expensive=[],
            transport_breakdown=TransportBreakdown(
                cloud_calls=0,
                cloud_cost_usd=0.0,
                native_calls=0,
                fallback_calls=0,
                native_ratio=0.0,
            ),
        )
        printed: list[str] = []
        dash = TaskDashboard(
            _print_fn=lambda t: printed.append(t),
            _get_dashboard_data_fn=lambda: data,
        )
        dash.show_cost_dashboard()
        assert any("0.0%" in line for line in printed)


# ---------------------------------------------------------------------------
# TEST 3 — Diagnostic ZIP + transport_audit
# ---------------------------------------------------------------------------


class TestDiagnosticZipTransportAudit:
    def test_transport_audit_in_zip(self):
        """ZIP contains transport_audit_last_100.json with injected rows."""
        import json  # noqa: PLC0415

        transport_rows = [
            {"id": i, "method": "uia", "success": True}
            for i in range(15)
        ]
        written: dict[str, Any] = {}

        reporter = DiagnosticReporter(
            _get_health_fn=lambda: _ok_health(),
            _get_transport_fn=lambda n: transport_rows[:n],
            _get_settings_fn=lambda: {"fps": 15, "api_key": _FAKE_API_KEY},
            _get_system_info_fn=lambda: {"platform": "TestOS"},
            _write_zip_fn=lambda path, files: written.update({"files": files}),
        )
        reporter.export_zip("diag.zip")

        files: dict[str, bytes] = written["files"]
        assert "transport_audit_last_100.json" in files

        rows = json.loads(files["transport_audit_last_100.json"])
        assert len(rows) == 15
        assert rows[0]["method"] == "uia"

    def test_api_key_not_in_zip(self):
        """Sensitive fields must be redacted from all ZIP contents."""
        written: dict[str, Any] = {}

        reporter = DiagnosticReporter(
            _get_health_fn=lambda: _ok_health(),
            _get_settings_fn=lambda: {
                "api_key": _FAKE_API_KEY,
                "nested": {"token": _FAKE_API_KEY},
                "fps": 15,
            },
            _get_system_info_fn=lambda: {"platform": "TestOS"},
            _write_zip_fn=lambda path, files: written.update({"files": files}),
        )
        reporter.export_zip("diag.zip")

        for filename, content_bytes in written["files"].items():
            content = content_bytes.decode(errors="replace")
            assert _FAKE_API_KEY not in content, (
                f"API key leaked in {filename!r}"
            )


# ---------------------------------------------------------------------------
# TEST 4 — Crash handler
# ---------------------------------------------------------------------------


class TestCrashHandler:
    def test_crash_handler_full_pipeline(self):
        """
        handle_crash():
          _write_crash_fn receives the traceback text.
          _generate_diagnostic_fn called with the zip path.
          Output contains "Tanı dosyası".
        """
        import sys  # noqa: PLC0415

        traces: list[str] = []
        diag_paths: list[str] = []
        printed: list[str] = []

        from nexus.infra.diagnostic import CrashHandler  # noqa: PLC0415

        handler = CrashHandler(
            _write_crash_fn=lambda t: traces.append(t),
            _generate_diagnostic_fn=lambda p: diag_paths.append(p) or p,  # type: ignore[func-returns-value]
            _print_fn=lambda t: printed.append(t),
            diagnostic_zip_path="crash_test.zip",
        )

        try:
            raise RuntimeError("integration crash test")
        except RuntimeError:
            handler.handle_crash(*sys.exc_info())

        assert len(traces) == 1
        assert "RuntimeError" in traces[0]
        assert "integration crash test" in traces[0]
        assert diag_paths == ["crash_test.zip"]
        assert any("Tanı dosyası" in line for line in printed)


# ---------------------------------------------------------------------------
# TEST 5 — Cancel flow
# ---------------------------------------------------------------------------


class TestCancelFlow:
    def test_graceful_then_force(self):
        """
        First SIGINT → executor.cancel() called.
        Second SIGINT → _force_quit_fn called.
        """
        cancel_calls: list[None] = []
        force_calls: list[None] = []
        printed: list[str] = []

        class _FakeExecutor:
            def cancel(self) -> None:
                cancel_calls.append(None)

        handler = CancelHandler(
            _force_quit_fn=lambda: force_calls.append(None),
            _print_fn=lambda t: printed.append(t),
            _signal_fn=lambda sig, h: None,
        )
        handler.install(_FakeExecutor())

        # First Ctrl+C
        handler._handle_sigint(signal.SIGINT, None)
        assert handler.cancel_requested is True
        assert len(cancel_calls) == 1
        assert len(force_calls) == 0
        assert any("İptal" in line for line in printed)

        # Second Ctrl+C
        handler._handle_sigint(signal.SIGINT, None)
        assert len(force_calls) == 1
        assert any("Zorla" in line for line in printed)

    def test_cancel_flag_fallback(self):
        """Executor without cancel() gets .cancelled = True."""

        class _SimpleExec:
            cancelled: bool = False

        executor = _SimpleExec()
        handler = CancelHandler(
            _print_fn=lambda _: None,
            _signal_fn=lambda sig, h: None,
        )
        handler.install(executor)
        handler._handle_sigint(signal.SIGINT, None)
        assert executor.cancelled is True


# ---------------------------------------------------------------------------
# TEST 6 — Privacy: native transport message
# ---------------------------------------------------------------------------


class TestPrivacyNativeTransport:
    def test_uia_masker_log_is_not_sent(self):
        """
        ScreenshotMasker.mask(transport="uia") emits "screenshot_not_sent"
        with screenshot_sent=False.
        """
        logged: list[dict] = []

        import nexus.core.screenshot_masker as _mod  # noqa: PLC0415

        original_log = _mod._log

        class _CapLog:
            def info(self, event: str, **kw: Any) -> None:
                logged.append({"event": event, **kw})

            def warning(self, *a: Any, **kw: Any) -> None:
                pass

        _mod._log = _CapLog()  # type: ignore[assignment]
        try:
            masker = ScreenshotMasker()
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            masker.mask(img, [], transport="uia")
        finally:
            _mod._log = original_log

        assert any(e["event"] == "screenshot_not_sent" for e in logged)
        assert any(e.get("screenshot_sent") is False for e in logged)

    def test_privacy_screen_native_message(self):
        """
        PrivacyTransparencyScreen.show_cloud_call_info(transport="uia"):
          output contains "veri gönderilmedi".
          output does NOT contain "ekran görüntüsü gönderildi".
        """
        printed: list[str] = []
        screen = PrivacyTransparencyScreen(
            _print_fn=lambda t: printed.append(t),
        )
        screen.show_cloud_call_info(
            masked_screenshot_path="",
            sensitive_count=2,
            provider="anthropic",
            transport="uia",
        )
        text = "\n".join(printed)

        assert "veri gönderilmedi" in text
        assert "ekran görüntüsü gönderildi" not in text

    def test_privacy_screen_visual_message(self):
        """
        PrivacyTransparencyScreen.show_cloud_call_info(transport="visual"):
          output contains "ekran görüntüsü gönderildi".
          output does NOT contain "veri gönderilmedi".
        """
        printed: list[str] = []
        screen = PrivacyTransparencyScreen(
            _print_fn=lambda t: printed.append(t),
        )
        screen.show_cloud_call_info(
            masked_screenshot_path="/tmp/masked.png",
            sensitive_count=1,
            provider="openai",
            transport="visual",
        )
        text = "\n".join(printed)

        assert "ekran görüntüsü gönderildi" in text
        assert "veri gönderilmedi" not in text
