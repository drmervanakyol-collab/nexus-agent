"""
nexus/ui/dashboard.py
Task Dashboard for Nexus Agent — real-time status, cost, and history views.

TaskDashboard
-------------
Renders four distinct views to a pluggable output callable (_print_fn).
All data is provided via injectable callables so the class is testable
without a running agent, live database, or real terminal.

show_status(task_context)
  Live single-task status panel (goal, step, transport, cost, last action,
  elapsed time, keybindings).

show_cost_dashboard()
  Today's total, a 7-day ASCII bar-chart, top-5-expensive tasks, and a
  transport breakdown (native UIA/DOM vs visual fallback percentages).

show_task_history(limit=20)
  Tabular list of the most recent tasks with id, goal, cost, and step count.

show_suspended_tasks()
  List of tasks currently in suspended state.

show_notification(message, level)
  Single-line notification prefixed with a level indicator.
  Levels: "info", "warn", "error", "success".

Injectable callables
--------------------
_print_fn            : (text: str) -> None
_get_dashboard_data_fn : () -> DashboardData
_get_task_history_fn : (limit: int) -> list[TaskHistoryEntry]
_get_suspended_fn    : () -> list[SuspendedTask]
_now_fn              : () -> float   (monotonic seconds — for elapsed time)
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nexus.core.task_executor import TaskContext
from nexus.infra.cost_tracker import DashboardData, TransportBreakdown
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


@dataclass
class TaskHistoryEntry:
    """One row in the task history view."""

    task_id: str
    goal: str
    status: str
    total_cost_usd: float
    step_count: int
    started_at: str


@dataclass
class SuspendedTask:
    """Mirror of SuspendManager.SuspendedTask for display purposes."""

    task_id: str
    reason: str
    suspended_at: str


# ---------------------------------------------------------------------------
# Notification levels
# ---------------------------------------------------------------------------

_LEVEL_PREFIX: dict[str, str] = {
    "info":    "[i]",
    "warn":    "[!]",
    "error":   "[✗]",
    "success": "[✓]",
}

# ---------------------------------------------------------------------------
# ASCII bar-chart constants
# ---------------------------------------------------------------------------

_BAR_WIDTH = 20   # max bar length in characters
_BAR_CHAR = "█"
_EMPTY_CHAR = "░"

# ---------------------------------------------------------------------------
# Border characters
# ---------------------------------------------------------------------------

_BORDER = "═" * 43
_THIN   = "─" * 43


# ---------------------------------------------------------------------------
# TaskDashboard
# ---------------------------------------------------------------------------


class TaskDashboard:
    """
    Terminal-oriented dashboard for Nexus Agent.

    Parameters
    ----------
    _print_fn:
        Output callable.  Default: ``print``.
    _get_dashboard_data_fn:
        ``() -> DashboardData``.  Provides cost + transport data.
    _get_task_history_fn:
        ``(limit: int) -> list[TaskHistoryEntry]``.
    _get_suspended_fn:
        ``() -> list[SuspendedTask]``.
    _now_fn:
        ``() -> float``.  Current monotonic time.  Default: ``time.monotonic``.
    """

    def __init__(
        self,
        *,
        _print_fn: Callable[[str], None] | None = None,
        _get_dashboard_data_fn: Callable[[], DashboardData] | None = None,
        _get_task_history_fn: (
            Callable[[int], list[TaskHistoryEntry]] | None
        ) = None,
        _get_suspended_fn: Callable[[], list[SuspendedTask]] | None = None,
        _now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._print = _print_fn or print
        self._get_dashboard_data = _get_dashboard_data_fn or (
            lambda: DashboardData(
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
        )
        self._get_task_history: Callable[[int], list[TaskHistoryEntry]] = (
            _get_task_history_fn or (lambda _limit: [])
        )
        self._get_suspended: Callable[[], list[SuspendedTask]] = (
            _get_suspended_fn or (lambda: [])
        )
        self._now = _now_fn or time.monotonic

    # ------------------------------------------------------------------
    # show_status
    # ------------------------------------------------------------------

    def show_status(self, task_context: TaskContext) -> None:
        """
        Render the live single-task status panel.

        Parameters
        ----------
        task_context:
            The currently executing TaskContext.
        """
        elapsed_s = self._now() - task_context.started_at
        elapsed_str = _format_elapsed(elapsed_s)

        # Last action description
        last_action = "—"
        if task_context.action_history:
            last = task_context.action_history[-1]
            last_action = f"{last.action_type}: {last.target_description}"
            if len(last_action) > 36:
                last_action = last_action[:33] + "..."

        # Transport label
        ts = task_context.transport_stats
        if ts.total == 0:
            transport_label = "—"
        elif ts.native_ratio >= 0.5:
            transport_label = f"native (UIA/DOM) ✓  {ts.native_count}/{ts.total}"
        else:
            transport_label = f"visual fallback  ⚠  {ts.fallback_count}/{ts.total}"

        # Daily cost (via dashboard data if available)
        daily_cost = 0.0
        try:
            data = self._get_dashboard_data()
            daily_cost = data.today_total_usd
        except Exception:
            pass

        p = self._print
        p(f"\n{_BORDER}")
        p("  NEXUS AGENT — Çalışıyor")
        p(_BORDER)
        p(f"  Görev    : {_truncate(task_context.goal, 36)}")
        p(f"  Adım     : {task_context.action_count}/?")
        p(f"  Transport: {transport_label}")
        p(
            f"  Maliyet  : ${task_context.total_cost_usd:.4f}"
            f" / günlük ${daily_cost:.4f}"
        )
        p(f"  Son aksiyon: {last_action}")
        p(f"  Süre     : {elapsed_str}")
        p(_BORDER)
        p("  [C] İptal  [P] Duraklat")
        p(_BORDER)

    # ------------------------------------------------------------------
    # show_cost_dashboard
    # ------------------------------------------------------------------

    def show_cost_dashboard(self) -> None:
        """
        Render the cost + transport breakdown dashboard.
        """
        data = self._get_dashboard_data()
        bd = data.transport_breakdown
        p = self._print

        p(f"\n{_BORDER}")
        p("  MALİYET PANOSU")
        p(_BORDER)

        # Today
        p(f"  Bugünkü toplam : ${data.today_total_usd:.4f}")
        p("")

        # 7-day ASCII chart
        p("  Son 7 gün:")
        if data.last_7_days:
            max_cost = max((d.cost_usd for d in data.last_7_days), default=0.0)
            for day_total in data.last_7_days:
                bar = _ascii_bar(day_total.cost_usd, max_cost, _BAR_WIDTH)
                p(
                    f"  {day_total.date}  {bar}"
                    f"  ${day_total.cost_usd:.4f}"
                )
        else:
            p("  (veri yok)")
        p("")

        # Top-5 expensive tasks
        p("  En pahalı 5 görev:")
        if data.top_5_expensive:
            for i, task in enumerate(data.top_5_expensive, 1):
                short_id = task.task_id[:8]
                p(
                    f"  {i}. [{short_id}]"
                    f"  ${task.cost_usd:.4f}"
                    f"  ({task.call_count} çağrı)"
                )
        else:
            p("  (veri yok)")
        p("")

        # Transport breakdown
        p("  Transport dağılımı:")
        total_t = bd.native_calls + bd.fallback_calls
        if total_t > 0:
            native_pct = bd.native_ratio * 100
            fallback_pct = (1.0 - bd.native_ratio) * 100
            p(f"    Native (UIA/DOM) : {native_pct:5.1f}%  ({bd.native_calls} aksiyon)")
            p(f"    Visual fallback  : {fallback_pct:5.1f}%  ({bd.fallback_calls} aksiyon)")
        else:
            p("    Native (UIA/DOM) :   0.0%")
            p("    Visual fallback  :   0.0%")
        p(f"    Bulut çağrısı    : {bd.cloud_calls}  (${bd.cloud_cost_usd:.4f})")
        p(_BORDER)

    # ------------------------------------------------------------------
    # show_task_history
    # ------------------------------------------------------------------

    def show_task_history(self, limit: int = 20) -> None:
        """
        Render the most recent *limit* tasks as a table.
        """
        history = self._get_task_history(limit)
        p = self._print

        p(f"\n{_BORDER}")
        p(f"  GÖREV GEÇMİŞİ (son {limit})")
        p(_BORDER)
        if not history:
            p("  (görev yok)")
        else:
            header = f"  {'ID':8}  {'Durum':10}  {'Maliyet':8}  {'Adım':4}  Görev"
            p(header)
            p(f"  {_THIN}")
            for entry in history:
                short_id = entry.task_id[:8]
                goal_trunc = _truncate(entry.goal, 24)
                p(
                    f"  {short_id:<8}  {entry.status:<10}"
                    f"  ${entry.total_cost_usd:6.4f}  {entry.step_count:4}"
                    f"  {goal_trunc}"
                )
        p(_BORDER)

    # ------------------------------------------------------------------
    # show_suspended_tasks
    # ------------------------------------------------------------------

    def show_suspended_tasks(self) -> None:
        """
        Render all currently suspended tasks.
        """
        tasks = self._get_suspended()
        p = self._print

        p(f"\n{_BORDER}")
        p("  ASKIYA ALINAN GÖREVLER")
        p(_BORDER)
        if not tasks:
            p("  (askıya alınan görev yok)")
        else:
            for task in tasks:
                short_id = task.task_id[:8]
                p(f"  [{short_id}]  {task.suspended_at}")
                p(f"    Neden: {task.reason}")
        p(_BORDER)

    # ------------------------------------------------------------------
    # show_notification
    # ------------------------------------------------------------------

    def show_notification(self, message: str, level: str = "info") -> None:
        """
        Display a single-line notification.

        Parameters
        ----------
        message:
            The notification text.
        level:
            One of ``"info"``, ``"warn"``, ``"error"``, ``"success"``.
            Unknown levels default to ``"info"``.
        """
        prefix = _LEVEL_PREFIX.get(level, _LEVEL_PREFIX["info"])
        self._print(f"  {prefix} {message}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _format_elapsed(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS``."""
    s = int(seconds)
    hh, rem = divmod(s, 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _truncate(text: str, max_len: int) -> str:
    """Return *text* truncated to *max_len* chars with an ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _ascii_bar(value: float, max_value: float, width: int) -> str:
    """
    Return a fixed-width ASCII bar representing *value* relative to *max_value*.

    Example with width=10: ``"████░░░░░░"``
    """
    if max_value <= 0:
        filled = 0
    else:
        filled = round(value / max_value * width)
    filled = max(0, min(filled, width))
    return _BAR_CHAR * filled + _EMPTY_CHAR * (width - filled)
