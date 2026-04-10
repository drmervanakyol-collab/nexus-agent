"""
tests/unit/test_dashboard.py
Unit tests for nexus/ui/dashboard.py — Faz 57.

TEST PLAN
---------
show_status:
  1. Goal, step count, elapsed time present in output.
  2. Native transport shown when native_ratio >= 0.5.
  3. Visual fallback shown when native_ratio < 0.5.
  4. Task cost and daily cost shown.
  5. Last action truncated at 36 chars.
  6. No action history → last action shows "—".
  7. Keybindings [C] and [P] present.

show_cost_dashboard:
  8. Today total rendered.
  9. 7-day ASCII chart rows present (one per day).
  10. Top-5 tasks rendered in order.
  11. Transport breakdown: native % and fallback % correct.
  12. Zero transport data shows 0.0%.

show_task_history:
  13. Each entry row contains short id, status, cost, step count, goal.
  14. Empty history shows "(görev yok)".
  15. Goal truncated at 24 chars in table.

show_suspended_tasks:
  16. Suspended task rows show id + reason.
  17. Empty suspended list shows "(askıya alınan görev yok)".

show_notification:
  18. "info" level → "[i]" prefix.
  19. "warn" level → "[!]" prefix.
  20. "error" level → "[✗]" prefix.
  21. "success" level → "[✓]" prefix.
  22. Unknown level defaults to "[i]" prefix.

_format_elapsed (unit):
  23. 0 s → "00:00:00".
  24. 83 s → "00:01:23".
  25. 3661 s → "01:01:01".

_ascii_bar (unit):
  26. Full bar when value == max_value.
  27. Empty bar when value == 0.
  28. Half bar when value == max_value / 2.
  29. max_value == 0 → all empty.
"""
from __future__ import annotations

from datetime import date

import pytest

from nexus.cloud.prompt_builder import ActionRecord
from nexus.core.task_executor import TaskContext, TransportStats
from nexus.infra.cost_tracker import (
    DashboardData,
    DayTotal,
    TaskCost,
    TransportBreakdown,
)
from nexus.ui.dashboard import (
    SuspendedTask,
    TaskDashboard,
    TaskHistoryEntry,
    _ascii_bar,
    _format_elapsed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = 1_000_000.0  # fixed monotonic baseline


def _action(type_: str = "click", target: str = "OK button") -> ActionRecord:
    return ActionRecord(
        action_type=type_,
        target_description=target,
        outcome="success",
        timestamp="2026-04-09T10:00:00Z",
    )


def _task_ctx(
    *,
    goal: str = "Open Notepad",
    action_count: int = 3,
    total_cost_usd: float = 0.12,
    action_history: list[ActionRecord] | None = None,
    native_count: int = 8,
    fallback_count: int = 2,
    started_at: float = _NOW - 83.0,  # 1 min 23 s ago
) -> TaskContext:
    return TaskContext(
        task_id="task-001",
        goal=goal,
        started_at=started_at,
        action_count=action_count,
        total_cost_usd=total_cost_usd,
        status="running",
        action_history=action_history or [],
        transport_stats=TransportStats(
            native_count=native_count, fallback_count=fallback_count
        ),
    )


def _dashboard_data(
    *,
    today: float = 0.45,
    native_calls: int = 80,
    fallback_calls: int = 20,
    top_5: list[TaskCost] | None = None,
    last_7: list[DayTotal] | None = None,
) -> DashboardData:
    total_t = native_calls + fallback_calls
    ratio = native_calls / total_t if total_t else 0.0
    return DashboardData(
        today_total_usd=today,
        last_7_days=last_7 or [
            DayTotal(date=date(2026, 4, i), cost_usd=0.1 * i, call_count=i)
            for i in range(3, 10)
        ],
        last_10_tasks=[],
        top_5_expensive=top_5 or [
            TaskCost(
                task_id=f"task-{i:03d}",
                cost_usd=0.5 - i * 0.05,
                call_count=10 - i,
                first_call_at=__import__("datetime").datetime(2026, 4, 9),
            )
            for i in range(5)
        ],
        transport_breakdown=TransportBreakdown(
            cloud_calls=100,
            cloud_cost_usd=1.23,
            native_calls=native_calls,
            fallback_calls=fallback_calls,
            native_ratio=ratio,
        ),
    )


def _make_dashboard(
    *,
    dashboard_data: DashboardData | None = None,
    history: list[TaskHistoryEntry] | None = None,
    suspended: list[SuspendedTask] | None = None,
    printed: list[str] | None = None,
    now: float = _NOW,
) -> tuple[TaskDashboard, list[str]]:
    output: list[str] = printed if printed is not None else []
    _data = dashboard_data or _dashboard_data()
    _history = history or []
    _suspended = suspended or []

    dash = TaskDashboard(
        _print_fn=lambda t: output.append(t),
        _get_dashboard_data_fn=lambda: _data,
        _get_task_history_fn=lambda limit: _history[:limit],
        _get_suspended_fn=lambda: _suspended,
        _now_fn=lambda: now,
    )
    return dash, output


def _lines(output: list[str]) -> str:
    return "\n".join(output)


# ---------------------------------------------------------------------------
# show_status
# ---------------------------------------------------------------------------


class TestShowStatus:
    def test_goal_and_step_in_output(self):
        dash, out = _make_dashboard()
        dash.show_status(_task_ctx(goal="Open Notepad", action_count=5))
        text = _lines(out)
        assert "Open Notepad" in text
        assert "5/?" in text

    def test_native_transport_label(self):
        dash, out = _make_dashboard()
        # native_count=8, fallback_count=2 → ratio=0.8
        dash.show_status(_task_ctx(native_count=8, fallback_count=2))
        text = _lines(out)
        assert "native" in text.lower()

    def test_visual_fallback_label(self):
        dash, out = _make_dashboard()
        # native=1, fallback=9 → ratio=0.1 < 0.5
        dash.show_status(_task_ctx(native_count=1, fallback_count=9))
        text = _lines(out)
        assert "fallback" in text.lower()

    def test_cost_displayed(self):
        dash, out = _make_dashboard(dashboard_data=_dashboard_data(today=0.45))
        dash.show_status(_task_ctx(total_cost_usd=0.12))
        text = _lines(out)
        assert "0.1200" in text
        assert "0.4500" in text

    def test_last_action_truncated(self):
        long_desc = "A" * 40  # > 36 chars
        action = _action(target=long_desc)
        dash, out = _make_dashboard()
        dash.show_status(_task_ctx(action_history=[action]))
        text = _lines(out)
        assert "..." in text

    def test_no_history_shows_dash(self):
        dash, out = _make_dashboard()
        dash.show_status(_task_ctx(action_history=[]))
        text = _lines(out)
        assert "—" in text

    def test_elapsed_time_format(self):
        dash, out = _make_dashboard(now=_NOW)
        # started 83 s ago
        dash.show_status(_task_ctx(started_at=_NOW - 83.0))
        text = _lines(out)
        assert "00:01:23" in text

    def test_keybindings_present(self):
        dash, out = _make_dashboard()
        dash.show_status(_task_ctx())
        text = _lines(out)
        assert "[C]" in text
        assert "[P]" in text


# ---------------------------------------------------------------------------
# show_cost_dashboard
# ---------------------------------------------------------------------------


class TestShowCostDashboard:
    def test_today_total_rendered(self):
        dash, out = _make_dashboard(dashboard_data=_dashboard_data(today=1.2345))
        dash.show_cost_dashboard()
        assert any("1.2345" in line for line in out)

    def test_7_day_chart_rows(self):
        days = [
            DayTotal(date=date(2026, 4, i), cost_usd=0.1 * i, call_count=1)
            for i in range(3, 10)
        ]
        dash, out = _make_dashboard(
            dashboard_data=_dashboard_data(last_7=days)
        )
        dash.show_cost_dashboard()
        text = _lines(out)
        # All 7 dates should appear
        for i in range(3, 10):
            assert f"2026-04-0{i}" in text

    def test_top5_tasks_rendered(self):
        import datetime as dt

        top5 = [
            TaskCost(
                task_id=f"task-{i:03d}",
                cost_usd=float(i + 1) * 0.1,
                call_count=i,
                first_call_at=dt.datetime(2026, 4, 9),
            )
            for i in range(5)
        ]
        dash, out = _make_dashboard(
            dashboard_data=_dashboard_data(top_5=top5)
        )
        dash.show_cost_dashboard()
        text = _lines(out)
        assert "task-000" in text
        assert "task-004" in text

    def test_transport_breakdown_percentages(self):
        # 80 native, 20 fallback → 80%, 20%
        dash, out = _make_dashboard(
            dashboard_data=_dashboard_data(native_calls=80, fallback_calls=20)
        )
        dash.show_cost_dashboard()
        text = _lines(out)
        assert "80.0%" in text
        assert "20.0%" in text

    def test_zero_transport_shows_zero(self):
        dash, out = _make_dashboard(
            dashboard_data=_dashboard_data(native_calls=0, fallback_calls=0)
        )
        dash.show_cost_dashboard()
        text = _lines(out)
        assert "0.0%" in text


# ---------------------------------------------------------------------------
# show_task_history
# ---------------------------------------------------------------------------


class TestShowTaskHistory:
    def test_entry_row_fields(self):
        history = [
            TaskHistoryEntry(
                task_id="abc12345-def",
                goal="Open Notepad",
                status="completed",
                total_cost_usd=0.05,
                step_count=7,
                started_at="2026-04-09T10:00:00Z",
            )
        ]
        dash, out = _make_dashboard(history=history)
        dash.show_task_history()
        text = _lines(out)
        assert "abc12345" in text
        assert "completed" in text
        assert "0.0500" in text
        assert "7" in text

    def test_empty_history(self):
        dash, out = _make_dashboard(history=[])
        dash.show_task_history()
        text = _lines(out)
        assert "görev yok" in text

    def test_goal_truncated_at_24(self):
        long_goal = "A" * 30
        history = [
            TaskHistoryEntry(
                task_id="x" * 12,
                goal=long_goal,
                status="running",
                total_cost_usd=0.0,
                step_count=1,
                started_at="",
            )
        ]
        dash, out = _make_dashboard(history=history)
        dash.show_task_history()
        text = _lines(out)
        assert "..." in text


# ---------------------------------------------------------------------------
# show_suspended_tasks
# ---------------------------------------------------------------------------


class TestShowSuspendedTasks:
    def test_suspended_task_shown(self):
        tasks = [
            SuspendedTask(
                task_id="susp-abc-123",
                reason="HITL confirmation required",
                suspended_at="2026-04-09T11:00:00Z",
            )
        ]
        dash, out = _make_dashboard(suspended=tasks)
        dash.show_suspended_tasks()
        text = _lines(out)
        assert "susp-abc" in text
        assert "HITL" in text

    def test_empty_suspended_list(self):
        dash, out = _make_dashboard(suspended=[])
        dash.show_suspended_tasks()
        text = _lines(out)
        assert "askıya alınan görev yok" in text


# ---------------------------------------------------------------------------
# show_notification
# ---------------------------------------------------------------------------


class TestShowNotification:
    @pytest.mark.parametrize(
        "level, expected_prefix",
        [
            ("info",    "[i]"),
            ("warn",    "[!]"),
            ("error",   "[✗]"),
            ("success", "[✓]"),
        ],
    )
    def test_level_prefix(self, level: str, expected_prefix: str):
        output: list[str] = []
        dash = TaskDashboard(_print_fn=lambda t: output.append(t))
        dash.show_notification("Test message", level=level)
        assert any(expected_prefix in line for line in output)

    def test_unknown_level_defaults_to_info(self):
        output: list[str] = []
        dash = TaskDashboard(_print_fn=lambda t: output.append(t))
        dash.show_notification("hello", level="critical")
        assert any("[i]" in line for line in output)

    def test_message_in_output(self):
        output: list[str] = []
        dash = TaskDashboard(_print_fn=lambda t: output.append(t))
        dash.show_notification("Disk space low!")
        assert any("Disk space low!" in line for line in output)


# ---------------------------------------------------------------------------
# _format_elapsed (unit)
# ---------------------------------------------------------------------------


class TestFormatElapsed:
    def test_zero(self):
        assert _format_elapsed(0) == "00:00:00"

    def test_83_seconds(self):
        assert _format_elapsed(83) == "00:01:23"

    def test_3661_seconds(self):
        assert _format_elapsed(3661) == "01:01:01"


# ---------------------------------------------------------------------------
# _ascii_bar (unit)
# ---------------------------------------------------------------------------


class TestAsciiBar:
    def test_full_bar(self):
        bar = _ascii_bar(10.0, 10.0, width=10)
        assert bar == "█" * 10

    def test_empty_bar_when_zero_value(self):
        bar = _ascii_bar(0.0, 10.0, width=10)
        assert bar == "░" * 10

    def test_half_bar(self):
        bar = _ascii_bar(5.0, 10.0, width=10)
        assert bar == "█" * 5 + "░" * 5

    def test_zero_max_all_empty(self):
        bar = _ascii_bar(1.0, 0.0, width=8)
        assert bar == "░" * 8
