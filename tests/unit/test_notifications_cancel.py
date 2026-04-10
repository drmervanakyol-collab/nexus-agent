"""
tests/unit/test_notifications_cancel.py
Unit tests for nexus/ui/notifications.py and nexus/ui/cancel_handler.py
— Faz 60.

TEST PLAN
---------
NotificationManager.notify:
  1.  info level → "[i]" prefix.
  2.  warn level → "[!]" prefix.
  3.  error level → "[✗]" prefix.
  4.  success level → "[✓]" prefix.
  5.  title and message both appear in output.
  6.  Unknown level defaults to "[i]".

NotificationManager.task_complete — success:
  7.  "[✓]" shown.
  8.  summary in output.
  9.  step count in output.
  10. duration in output.
  11. cost in output.
  12. native transport count and percentage shown.
  13. fallback transport count and percentage shown.

NotificationManager.task_complete — failure:
  14. "[✗]" shown.
  15. error message in output.
  16. transport stats still shown.

NotificationManager.task_complete — zero transport actions:
  17. "aksiyon yok" (or fallback text) shown instead of ratio.

NotificationManager.budget_warning:
  18. current and limit values in output.
  19. percentage in output.
  20. "[!]" prefix shown.

NotificationManager.hitl_needed:
  21. question text in output.
  22. "[!]" prefix shown.

CancelHandler — first Ctrl+C (graceful):
  23. _cancel_fn called.
  24. cancel_requested becomes True.
  25. "[!] İptal" message printed.
  26. Second SIGINT calls _force_quit_fn.

CancelHandler — executor integration:
  27. executor.cancel() called when no _cancel_fn provided.
  28. executor.cancelled set when executor has no cancel() method.

CancelHandler — install / uninstall:
  29. install() registers with _signal_fn.
  30. uninstall() restores original handler.
"""
from __future__ import annotations

import signal
from typing import Any

import pytest

from nexus.core.task_executor import TaskResult, TransportStats
from nexus.ui.cancel_handler import CancelHandler
from nexus.ui.notifications import NotificationManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(
    *,
    success: bool = True,
    summary: str = "Task done",
    error: str | None = None,
    steps: int = 5,
    cost: float = 0.12,
    duration_ms: float = 3000.0,
    native: int = 8,
    fallback: int = 2,
) -> TaskResult:
    return TaskResult(
        task_id="task-001",
        success=success,
        steps_completed=steps,
        total_cost_usd=cost,
        duration_ms=duration_ms,
        status="completed" if success else "failed",
        summary=summary,
        error=error,
        transport_stats=TransportStats(native_count=native, fallback_count=fallback),
    )


def _nm(printed: list[str] | None = None) -> tuple[NotificationManager, list[str]]:
    out: list[str] = printed if printed is not None else []
    nm = NotificationManager(_print_fn=lambda t: out.append(t))
    return nm, out


def _lines(out: list[str]) -> str:
    return "\n".join(out)


# ---------------------------------------------------------------------------
# NotificationManager.notify
# ---------------------------------------------------------------------------


class TestNotify:
    @pytest.mark.parametrize(
        "level, prefix",
        [("info", "[i]"), ("warn", "[!]"), ("error", "[✗]"), ("success", "[✓]")],
    )
    def test_level_prefix(self, level: str, prefix: str):
        nm, out = _nm()
        nm.notify("TITLE", "msg", level=level)
        assert any(prefix in line for line in out)

    def test_title_and_message_in_output(self):
        nm, out = _nm()
        nm.notify("MyTitle", "My message")
        text = _lines(out)
        assert "MyTitle" in text
        assert "My message" in text

    def test_unknown_level_defaults_to_info(self):
        nm, out = _nm()
        nm.notify("T", "M", level="critical")
        assert any("[i]" in line for line in out)


# ---------------------------------------------------------------------------
# NotificationManager.task_complete — success
# ---------------------------------------------------------------------------


class TestTaskCompleteSuccess:
    def test_success_icon(self):
        nm, out = _nm()
        nm.task_complete(_result(success=True))
        assert any("[✓]" in line for line in out)

    def test_summary_in_output(self):
        nm, out = _nm()
        nm.task_complete(_result(summary="Opened Notepad"))
        assert any("Opened Notepad" in line for line in out)

    def test_step_count(self):
        nm, out = _nm()
        nm.task_complete(_result(steps=7))
        text = _lines(out)
        assert "7" in text

    def test_duration_shown(self):
        nm, out = _nm()
        nm.task_complete(_result(duration_ms=5000.0))
        text = _lines(out)
        assert "5.0" in text

    def test_cost_shown(self):
        nm, out = _nm()
        nm.task_complete(_result(cost=0.0512))
        text = _lines(out)
        assert "0.0512" in text

    def test_native_transport_stats(self):
        nm, out = _nm()
        nm.task_complete(_result(native=8, fallback=2))
        text = _lines(out)
        assert "native 8" in text
        assert "80%" in text

    def test_fallback_transport_stats(self):
        nm, out = _nm()
        nm.task_complete(_result(native=3, fallback=7))
        text = _lines(out)
        assert "fallback 7" in text
        assert "70%" in text


# ---------------------------------------------------------------------------
# NotificationManager.task_complete — failure
# ---------------------------------------------------------------------------


class TestTaskCompleteFailure:
    def test_failure_icon(self):
        nm, out = _nm()
        nm.task_complete(_result(success=False, error="Timeout"))
        assert any("[✗]" in line for line in out)

    def test_error_in_output(self):
        nm, out = _nm()
        nm.task_complete(_result(success=False, error="Element not found"))
        assert any("Element not found" in line for line in out)

    def test_transport_stats_shown_on_failure(self):
        nm, out = _nm()
        nm.task_complete(_result(success=False, native=5, fallback=5))
        text = _lines(out)
        assert "native 5" in text


# ---------------------------------------------------------------------------
# NotificationManager.task_complete — zero transport actions
# ---------------------------------------------------------------------------


class TestZeroTransport:
    def test_zero_transport_fallback_message(self):
        nm, out = _nm()
        nm.task_complete(_result(native=0, fallback=0))
        text = _lines(out)
        assert "aksiyon yok" in text


# ---------------------------------------------------------------------------
# NotificationManager.budget_warning
# ---------------------------------------------------------------------------


class TestBudgetWarning:
    def test_values_in_output(self):
        nm, out = _nm()
        nm.budget_warning(0.80, 1.00)
        text = _lines(out)
        assert "0.8000" in text
        assert "1.0000" in text

    def test_percentage_in_output(self):
        nm, out = _nm()
        nm.budget_warning(0.80, 1.00)
        text = _lines(out)
        assert "80.0%" in text

    def test_warn_prefix(self):
        nm, out = _nm()
        nm.budget_warning(0.5, 1.0)
        assert any("[!]" in line for line in out)


# ---------------------------------------------------------------------------
# NotificationManager.hitl_needed
# ---------------------------------------------------------------------------


class TestHitlNeeded:
    def test_question_in_output(self):
        nm, out = _nm()
        nm.hitl_needed("Should I delete the file?")
        assert any("Should I delete the file?" in line for line in out)

    def test_warn_prefix(self):
        nm, out = _nm()
        nm.hitl_needed("Confirm action")
        assert any("[!]" in line for line in out)


# ---------------------------------------------------------------------------
# CancelHandler — first Ctrl+C
# ---------------------------------------------------------------------------


class TestCancelFirstSigint:
    def _make_handler(
        self,
        *,
        cancel_calls: list[None] | None = None,
        force_calls: list[None] | None = None,
        printed: list[str] | None = None,
        signal_log: list[tuple] | None = None,
    ) -> CancelHandler:
        _cancel = cancel_calls if cancel_calls is not None else []
        _force = force_calls if force_calls is not None else []
        _print_out = printed if printed is not None else []
        _slog = signal_log if signal_log is not None else []

        return CancelHandler(
            _cancel_fn=lambda: _cancel.append(None),
            _force_quit_fn=lambda: _force.append(None),
            _print_fn=lambda t: _print_out.append(t),
            _signal_fn=lambda sig, h: _slog.append((sig, h)),
        )

    def test_cancel_fn_called_on_first_sigint(self):
        cancels: list[None] = []
        h = self._make_handler(cancel_calls=cancels)
        h.install(object())
        h._handle_sigint(signal.SIGINT, None)
        assert len(cancels) == 1

    def test_cancel_requested_set(self):
        h = self._make_handler()
        h.install(object())
        assert h.cancel_requested is False
        h._handle_sigint(signal.SIGINT, None)
        assert h.cancel_requested is True

    def test_graceful_message_printed(self):
        printed: list[str] = []
        h = self._make_handler(printed=printed)
        h.install(object())
        h._handle_sigint(signal.SIGINT, None)
        text = "\n".join(printed)
        assert "İptal" in text

    def test_second_sigint_calls_force_quit(self):
        forces: list[None] = []
        h = self._make_handler(force_calls=forces)
        h.install(object())
        h._handle_sigint(signal.SIGINT, None)   # first
        h._handle_sigint(signal.SIGINT, None)   # second
        assert len(forces) == 1

    def test_second_sigint_prints_force_message(self):
        printed: list[str] = []
        h = self._make_handler(printed=printed)
        h.install(object())
        h._handle_sigint(signal.SIGINT, None)
        h._handle_sigint(signal.SIGINT, None)
        text = "\n".join(printed)
        assert "Zorla" in text


# ---------------------------------------------------------------------------
# CancelHandler — executor integration
# ---------------------------------------------------------------------------


class TestCancelExecutorIntegration:
    def test_executor_cancel_method_called(self):
        cancel_calls: list[None] = []

        class _FakeExecutor:
            def cancel(self) -> None:
                cancel_calls.append(None)

        h = CancelHandler(
            _print_fn=lambda _: None,
            _signal_fn=lambda sig, h: None,
        )
        h.install(_FakeExecutor())
        h._handle_sigint(signal.SIGINT, None)
        assert len(cancel_calls) == 1

    def test_executor_cancelled_flag_set_when_no_cancel_method(self):
        class _SimpleExecutor:
            cancelled: bool = False

        executor = _SimpleExecutor()
        h = CancelHandler(
            _print_fn=lambda _: None,
            _signal_fn=lambda sig, h: None,
        )
        h.install(executor)
        h._handle_sigint(signal.SIGINT, None)
        assert executor.cancelled is True


# ---------------------------------------------------------------------------
# CancelHandler — install / uninstall
# ---------------------------------------------------------------------------


class TestCancelInstallUninstall:
    def test_install_registers_signal(self):
        registered: list[tuple] = []
        h = CancelHandler(
            _print_fn=lambda _: None,
            _signal_fn=lambda sig, handler: registered.append((sig, handler)),
        )
        h.install(object())
        assert len(registered) == 1
        assert registered[0][0] == signal.SIGINT

    def test_uninstall_restores_original(self):
        handlers: list[Any] = []

        def _fake_signal(sig: int, handler: Any) -> Any:
            handlers.append(handler)
            return signal.SIG_DFL  # simulate "original" return value

        h = CancelHandler(
            _print_fn=lambda _: None,
            _signal_fn=_fake_signal,
        )
        h.install(object())
        h.uninstall()
        # Second call to _signal_fn is uninstall restoring original
        assert len(handlers) == 2
