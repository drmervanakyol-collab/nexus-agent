"""
nexus/infra/cost_tracker.py
In-process cost tracker for Nexus Agent.

Responsibilities
----------------
- Calculate per-call cost from BudgetSettings pricing tables.
- Accumulate task-level and daily totals.
- Fire budget alerts at 50 % / 80 % / 100 % thresholds.
- Provide a rich dashboard snapshot (today, last-7-days, last-10-tasks,
  top-5-expensive, transport breakdown).

Design notes
------------
- All monetary arithmetic uses plain float; precision is ±0.000001 USD
  (sub-cent), well within spec requirements.
- A pluggable ``clock`` callable makes the tracker fully testable without
  patching ``datetime``.
- Thread-safe via a single reentrant lock.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

from nexus.core.settings import BudgetSettings, NexusSettings
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AlertLevel = Literal["none", "info", "warn", "block"]

_THRESHOLD_INFO: float = 0.50
_THRESHOLD_WARN: float = 0.80
_THRESHOLD_BLOCK: float = 1.00

_NATIVE_TRANSPORTS: frozenset[str] = frozenset({"uia", "dom", "file"})
_FALLBACK_TRANSPORTS: frozenset[str] = frozenset({"mouse", "keyboard"})


# ---------------------------------------------------------------------------
# Internal entry types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CostEntry:
    task_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    recorded_at: datetime


@dataclass(frozen=True)
class _TransportEntry:
    task_id: str
    method: str
    recorded_at: datetime


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertResult:
    """Returned by ``record()`` after every cloud call."""

    level: AlertLevel
    task_pct: float  # task_cost / max_cost_per_task
    daily_pct: float  # daily_cost / max_cost_per_day
    task_cost_usd: float
    daily_cost_usd: float
    message: str


@dataclass
class DayTotal:
    date: date
    cost_usd: float
    call_count: int


@dataclass
class TaskCost:
    task_id: str
    cost_usd: float
    call_count: int
    first_call_at: datetime


@dataclass
class TransportBreakdown:
    cloud_calls: int
    cloud_cost_usd: float
    native_calls: int
    fallback_calls: int
    native_ratio: float  # native / (native + fallback), 0 if none


@dataclass
class CostSummary:
    total_cost_usd: float
    total_calls: int
    unique_tasks: int
    most_expensive_model: str | None
    cheapest_model: str | None
    avg_cost_per_call_usd: float
    avg_tokens_per_call: float


@dataclass
class DashboardData:
    today_total_usd: float
    last_7_days: list[DayTotal]
    last_10_tasks: list[TaskCost]
    top_5_expensive: list[TaskCost]
    transport_breakdown: TransportBreakdown


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """
    Tracks LLM cloud costs and transport usage for a running agent session.

    Parameters
    ----------
    settings:
        NexusSettings instance; uses ``.budget`` sub-settings.
    clock:
        Callable returning the current UTC datetime.  Defaults to
        ``datetime.now(timezone.utc)``.  Override in tests for
        deterministic time-based queries.
    on_alert:
        Optional callback invoked whenever the alert level is ``"warn"``
        or ``"block"``.  Signature: ``(alert: AlertResult) -> None``.
    """

    def __init__(
        self,
        settings: NexusSettings,
        *,
        clock: Callable[[], datetime] | None = None,
        on_alert: Callable[[AlertResult], None] | None = None,
    ) -> None:
        self._budget: BudgetSettings = settings.budget
        self._clock: Callable[[], datetime] = clock or (
            lambda: datetime.now(UTC)
        )
        self._on_alert = on_alert
        self._lock = threading.Lock()
        self._entries: list[_CostEntry] = []
        self._transport_entries: list[_TransportEntry] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        task_id: str,
        model: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> AlertResult:
        """
        Record one LLM call and return the resulting budget alert state.

        Cost is calculated from BudgetSettings pricing tables.
        Raises ``KeyError`` for unknown models.
        """
        pricing = self._budget.pricing_for(model)
        cost = (
            input_tokens / 1000.0 * pricing.input_per_1k
            + output_tokens / 1000.0 * pricing.output_per_1k
        )
        now = self._clock()
        entry = _CostEntry(
            task_id=task_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            recorded_at=now,
        )
        with self._lock:
            self._entries.append(entry)
            task_cost = sum(
                e.cost_usd for e in self._entries if e.task_id == task_id
            )
            today = now.date()
            daily_cost = sum(
                e.cost_usd
                for e in self._entries
                if e.recorded_at.date() == today
            )

        alert = self._evaluate_alert(task_id, task_cost, daily_cost)
        self._fire_alert(alert)
        return alert

    def record_transport(self, task_id: str, method: str) -> None:
        """Track one transport action (no monetary cost)."""
        with self._lock:
            self._transport_entries.append(
                _TransportEntry(
                    task_id=task_id,
                    method=method,
                    recorded_at=self._clock(),
                )
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_task_cost(self, task_id: str) -> float:
        """Return total USD spent for *task_id*."""
        with self._lock:
            return sum(e.cost_usd for e in self._entries if e.task_id == task_id)

    def get_daily_cost(self, day: date | None = None) -> float:
        """Return total USD spent on *day* (defaults to today)."""
        target = day or self._clock().date()
        with self._lock:
            return sum(
                e.cost_usd
                for e in self._entries
                if e.recorded_at.date() == target
            )

    def get_summary(self) -> CostSummary:
        """Return aggregate statistics across all recorded entries."""
        with self._lock:
            entries = list(self._entries)

        if not entries:
            return CostSummary(
                total_cost_usd=0.0,
                total_calls=0,
                unique_tasks=0,
                most_expensive_model=None,
                cheapest_model=None,
                avg_cost_per_call_usd=0.0,
                avg_tokens_per_call=0.0,
            )

        total_cost = sum(e.cost_usd for e in entries)
        total_calls = len(entries)

        # Per-model aggregates
        model_cost: dict[str, float] = defaultdict(float)
        for e in entries:
            model_cost[e.model] += e.cost_usd
        most_expensive = max(model_cost, key=model_cost.__getitem__)
        cheapest = min(model_cost, key=model_cost.__getitem__)

        total_tokens = sum(e.input_tokens + e.output_tokens for e in entries)
        avg_tokens = total_tokens / total_calls

        return CostSummary(
            total_cost_usd=total_cost,
            total_calls=total_calls,
            unique_tasks=len({e.task_id for e in entries}),
            most_expensive_model=most_expensive,
            cheapest_model=cheapest,
            avg_cost_per_call_usd=total_cost / total_calls,
            avg_tokens_per_call=avg_tokens,
        )

    def get_dashboard_data(self) -> DashboardData:
        """
        Build a rich dashboard snapshot.

        Includes today's total, last-7-days breakdown, last-10-tasks,
        top-5-expensive tasks, and transport method breakdown.
        """
        now = self._clock()
        today = now.date()

        with self._lock:
            entries = list(self._entries)
            transport_entries = list(self._transport_entries)

        # --- today's total ---
        today_total = sum(
            e.cost_usd for e in entries if e.recorded_at.date() == today
        )

        # --- last 7 days ---
        day_buckets: dict[date, list[_CostEntry]] = defaultdict(list)
        for e in entries:
            day_buckets[e.recorded_at.date()].append(e)

        last_7: list[DayTotal] = []
        from datetime import timedelta

        for delta in range(6, -1, -1):
            d = today - timedelta(days=delta)
            bucket = day_buckets.get(d, [])
            last_7.append(
                DayTotal(
                    date=d,
                    cost_usd=sum(x.cost_usd for x in bucket),
                    call_count=len(bucket),
                )
            )

        # --- per-task aggregates ---
        task_buckets: dict[str, list[_CostEntry]] = defaultdict(list)
        for e in entries:
            task_buckets[e.task_id].append(e)

        task_costs: list[TaskCost] = [
            TaskCost(
                task_id=tid,
                cost_usd=sum(x.cost_usd for x in bucket),
                call_count=len(bucket),
                first_call_at=min(x.recorded_at for x in bucket),
            )
            for tid, bucket in task_buckets.items()
        ]

        # last 10 tasks (by first call time, most recent first)
        last_10 = sorted(task_costs, key=lambda t: t.first_call_at, reverse=True)[:10]

        # top 5 most expensive
        top_5 = sorted(task_costs, key=lambda t: t.cost_usd, reverse=True)[:5]

        # --- transport breakdown ---
        cloud_calls = len(entries)
        cloud_cost = sum(e.cost_usd for e in entries)
        native = sum(1 for t in transport_entries if t.method in _NATIVE_TRANSPORTS)
        fallback = sum(
            1 for t in transport_entries if t.method in _FALLBACK_TRANSPORTS
        )
        total_transport = native + fallback
        native_ratio = native / total_transport if total_transport > 0 else 0.0

        breakdown = TransportBreakdown(
            cloud_calls=cloud_calls,
            cloud_cost_usd=cloud_cost,
            native_calls=native,
            fallback_calls=fallback,
            native_ratio=native_ratio,
        )

        return DashboardData(
            today_total_usd=today_total,
            last_7_days=last_7,
            last_10_tasks=last_10,
            top_5_expensive=top_5,
            transport_breakdown=breakdown,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded data (useful between tests)."""
        with self._lock:
            self._entries.clear()
            self._transport_entries.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_alert(
        self,
        task_id: str,
        task_cost: float,
        daily_cost: float,
    ) -> AlertResult:
        task_cap = self._budget.max_cost_per_task_usd
        daily_cap = self._budget.max_cost_per_day_usd

        task_pct = task_cost / task_cap if task_cap > 0 else float("inf")
        daily_pct = daily_cost / daily_cap if daily_cap > 0 else float("inf")
        max_pct = max(task_pct, daily_pct)

        if max_pct >= _THRESHOLD_BLOCK:
            level: AlertLevel = "block"
            msg = (
                f"Budget cap reached — "
                f"task ${task_cost:.4f}/{task_cap:.4f} "
                f"({task_pct:.0%}), "
                f"daily ${daily_cost:.4f}/{daily_cap:.4f} "
                f"({daily_pct:.0%})."
            )
        elif max_pct >= _THRESHOLD_WARN:
            level = "warn"
            msg = (
                f"Budget warning ({max_pct:.0%} of cap used) — "
                f"task ${task_cost:.4f}, daily ${daily_cost:.4f}."
            )
        elif max_pct >= _THRESHOLD_INFO:
            level = "info"
            msg = (
                f"Budget notice ({max_pct:.0%} of cap used) — "
                f"task ${task_cost:.4f}, daily ${daily_cost:.4f}."
            )
        else:
            level = "none"
            msg = ""

        return AlertResult(
            level=level,
            task_pct=task_pct,
            daily_pct=daily_pct,
            task_cost_usd=task_cost,
            daily_cost_usd=daily_cost,
            message=msg,
        )

    def _fire_alert(self, alert: AlertResult) -> None:
        if alert.level == "block":
            _log.warning(
                "budget_block",
                task_cost_usd=alert.task_cost_usd,
                daily_cost_usd=alert.daily_cost_usd,
                task_pct=alert.task_pct,
                daily_pct=alert.daily_pct,
            )
        elif alert.level == "warn":
            _log.warning(
                "budget_warn",
                task_cost_usd=alert.task_cost_usd,
                daily_cost_usd=alert.daily_cost_usd,
                task_pct=alert.task_pct,
                daily_pct=alert.daily_pct,
            )
        elif alert.level == "info":
            _log.info(
                "budget_info",
                task_cost_usd=alert.task_cost_usd,
                daily_cost_usd=alert.daily_cost_usd,
            )

        if alert.level in ("warn", "block") and self._on_alert is not None:
            self._on_alert(alert)
