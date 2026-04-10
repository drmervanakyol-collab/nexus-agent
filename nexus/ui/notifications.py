"""
nexus/ui/notifications.py
Notification manager for Nexus Agent.

NotificationManager
--------------------
Renders structured task and budget notifications to a pluggable output
callable.  All output is injectable for testability.

notify(title, message, level)
  Generic one-liner notification with a level prefix.
  Levels: "info" [i], "warn" [!], "error" [✗], "success" [✓].

task_complete(task_result)
  Rich completion panel:
    Success → ✓ title + goal summary + steps + duration + cost
    Failure → ✗ title + error message
    Both    → transport stats (native ratio, native/fallback counts).

budget_warning(current, limit)
  Budget threshold alert with spend-to-limit ratio.

hitl_needed(question)
  Human-in-the-loop attention notice.

Injectable callables
--------------------
_print_fn : (text: str) -> None
"""
from __future__ import annotations

from collections.abc import Callable

from nexus.core.task_executor import TaskResult
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

_LEVEL_PREFIX: dict[str, str] = {
    "info":    "[i]",
    "warn":    "[!]",
    "error":   "[✗]",
    "success": "[✓]",
}

_BORDER = "─" * 48


class NotificationManager:
    """
    Structured notification renderer.

    Parameters
    ----------
    _print_fn:
        Output callable.  Default: ``print``.
    """

    def __init__(
        self,
        *,
        _print_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._print = _print_fn or print

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify(
        self,
        title: str,
        message: str,
        level: str = "info",
    ) -> None:
        """
        Display a generic notification line.

        Parameters
        ----------
        title:
            Short label shown before the message.
        message:
            Notification body.
        level:
            One of ``"info"``, ``"warn"``, ``"error"``, ``"success"``.
        """
        prefix = _LEVEL_PREFIX.get(level, _LEVEL_PREFIX["info"])
        self._print(f"  {prefix} [{title}] {message}")

    def task_complete(self, task_result: TaskResult) -> None:
        """
        Render a full task completion panel.

        Shows success/failure indicator, goal summary, steps, duration,
        cost, and transport statistics (native vs fallback ratio).

        Parameters
        ----------
        task_result:
            The TaskResult returned by TaskExecutor.execute().
        """
        ts = task_result.transport_stats
        p = self._print

        p(f"\n  {_BORDER}")
        if task_result.success:
            p("  [✓] GÖREV TAMAMLANDI")
            p(f"  {_BORDER}")
            p(f"  Özet    : {_truncate(task_result.summary, 44)}")
            p(f"  Adım    : {task_result.steps_completed}")
            p(f"  Süre    : {task_result.duration_ms / 1000:.1f} s")
        else:
            p("  [✗] GÖREV BAŞARISIZ")
            p(f"  {_BORDER}")
            p(f"  Hata    : {_truncate(task_result.error or 'bilinmiyor', 44)}")

        p(f"  Maliyet : ${task_result.total_cost_usd:.4f}")
        p(f"  {_BORDER}")

        # Transport stats
        if ts.total > 0:
            native_pct = ts.native_ratio * 100
            fallback_pct = (1.0 - ts.native_ratio) * 100
            p(f"  Transport: native {ts.native_count} ({native_pct:.0f}%)"
              f"  |  fallback {ts.fallback_count} ({fallback_pct:.0f}%)")
        else:
            p("  Transport: — (aksiyon yok)")

        p(f"  {_BORDER}")

        _log.info(
            "task_complete_notification",
            task_id=task_result.task_id,
            success=task_result.success,
            steps=task_result.steps_completed,
            cost_usd=task_result.total_cost_usd,
            native_ratio=ts.native_ratio,
        )

    def budget_warning(self, current: float, limit: float) -> None:
        """
        Display a budget threshold alert.

        Parameters
        ----------
        current:
            Spend accrued so far (USD).
        limit:
            Configured budget cap (USD).
        """
        pct = current / limit * 100 if limit > 0 else 100.0

        self._print(f"\n  {_BORDER}")
        self._print("  [!] BÜTÇE UYARISI")
        self._print(f"  {_BORDER}")
        self._print(f"  Harcama : ${current:.4f} / ${limit:.4f}  ({pct:.1f}%)")
        bar = _budget_bar(current, limit)
        self._print(f"  [{bar}]")
        self._print(f"  {_BORDER}")

        _log.warning(
            "budget_warning_shown",
            current_usd=current,
            limit_usd=limit,
            pct=pct,
        )

    def hitl_needed(self, question: str) -> None:
        """
        Display a human-in-the-loop attention notice.

        Parameters
        ----------
        question:
            The question or situation requiring human attention.
        """
        self._print(f"\n  {_BORDER}")
        self._print("  [!] İNSAN ONAY GEREKLİ")
        self._print(f"  {_BORDER}")
        self._print(f"  Soru: {_truncate(question, 44)}")
        self._print(f"  {_BORDER}")

        _log.info("hitl_notification_shown", question=question)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _budget_bar(current: float, limit: float, width: int = 40) -> str:
    """ASCII spend bar: filled portion = current / limit."""
    ratio = 1.0 if limit <= 0 else min(current / limit, 1.0)
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)
