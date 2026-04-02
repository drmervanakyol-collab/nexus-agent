"""
Unit tests for nexus/infra/cost_tracker.py.

Coverage targets
----------------
- Cost calculation accuracy (cent-level precision for every model)
- Alert thresholds: 50 % / 80 % / 100 %
- on_alert callback fired at correct levels
- get_task_cost / get_daily_cost correctness
- get_summary correctness
- get_dashboard_data — today, 7-day, last-10-tasks, top-5, transport breakdown
- Transport breakdown (native vs fallback vs cloud)
- Thread safety
- Unknown model raises KeyError
- reset() clears all state
"""
from __future__ import annotations

import threading
from datetime import date, datetime, timedelta, timezone

import pytest

from nexus.core.settings import BudgetSettings, NexusSettings
from nexus.infra.cost_tracker import (
    AlertResult,
    CostSummary,
    CostTracker,
    DashboardData,
    DayTotal,
    TaskCost,
    TransportBreakdown,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Deterministic UTC times
_D0 = datetime(2026, 4, 3, 12, 0, 0, tzinfo=timezone.utc)  # "today"
_D1 = _D0 - timedelta(days=1)
_D2 = _D0 - timedelta(days=2)
_D6 = _D0 - timedelta(days=6)
_D7 = _D0 - timedelta(days=7)  # outside 7-day window


def _settings(
    max_task: float = 1.0,
    max_daily: float = 10.0,
    warn_at: float = 0.8,
) -> NexusSettings:
    return NexusSettings(
        budget=BudgetSettings(
            max_cost_per_task_usd=max_task,
            max_cost_per_day_usd=max_daily,
            warn_at_percent=warn_at,
        )
    )


def _tracker(
    max_task: float = 1.0,
    max_daily: float = 10.0,
    clock: datetime | None = None,
    on_alert=None,
) -> CostTracker:
    fixed = clock or _D0
    return CostTracker(
        _settings(max_task, max_daily),
        clock=lambda: fixed,
        on_alert=on_alert,
    )


# ---------------------------------------------------------------------------
# Cost calculation accuracy
# ---------------------------------------------------------------------------


class TestCostCalculation:
    """Verify per-model cost to ±0.000001 USD (sub-cent)."""

    def test_gpt4o_cost(self) -> None:
        t = _tracker()
        alert = t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=500)
        # 1000/1000*0.005 + 500/1000*0.015 = 0.005 + 0.0075 = 0.0125
        assert alert.task_cost_usd == pytest.approx(0.0125, abs=1e-6)

    def test_gpt4o_mini_cost(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o-mini", input_tokens=2000, output_tokens=1000)
        # 2000/1000*0.00015 + 1000/1000*0.0006 = 0.0003 + 0.0006 = 0.0009
        assert t.get_task_cost("t1") == pytest.approx(0.0009, abs=1e-6)

    def test_claude_sonnet_cost(self) -> None:
        t = _tracker()
        t.record(
            "t1",
            "claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=1000,
        )
        # 1000/1000*0.003 + 1000/1000*0.015 = 0.003 + 0.015 = 0.018
        assert t.get_task_cost("t1") == pytest.approx(0.018, abs=1e-6)

    def test_claude_haiku_cost(self) -> None:
        t = _tracker()
        t.record(
            "t1", "claude-3-haiku-20240307", input_tokens=4000, output_tokens=2000
        )
        # 4000/1000*0.00025 + 2000/1000*0.00125 = 0.001 + 0.0025 = 0.0035
        assert t.get_task_cost("t1") == pytest.approx(0.0035, abs=1e-6)

    def test_zero_tokens(self) -> None:
        t = _tracker()
        alert = t.record("t1", "gpt-4o", input_tokens=0, output_tokens=0)
        assert alert.task_cost_usd == pytest.approx(0.0, abs=1e-9)

    def test_multiple_calls_cumulative(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=500)
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=500)
        # 0.0125 * 2 = 0.025
        assert t.get_task_cost("t1") == pytest.approx(0.025, abs=1e-6)

    def test_different_tasks_isolated(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)
        t.record("t2", "gpt-4o-mini", input_tokens=1000, output_tokens=0)
        assert t.get_task_cost("t1") == pytest.approx(0.005, abs=1e-6)
        assert t.get_task_cost("t2") == pytest.approx(0.00015, abs=1e-6)

    def test_unknown_model_raises(self) -> None:
        t = _tracker()
        with pytest.raises(KeyError):
            t.record("t1", "gpt-999-ultra", input_tokens=100, output_tokens=100)

    def test_large_token_count(self) -> None:
        t = _tracker()
        # 1M input + 1M output with gpt-4o-mini
        t.record("t1", "gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
        # 1000*0.00015 + 1000*0.0006 = 0.15 + 0.6 = 0.75
        assert t.get_task_cost("t1") == pytest.approx(0.75, abs=1e-4)


# ---------------------------------------------------------------------------
# get_task_cost
# ---------------------------------------------------------------------------


class TestGetTaskCost:
    def test_zero_for_unknown_task(self) -> None:
        t = _tracker()
        assert t.get_task_cost("nonexistent") == pytest.approx(0.0)

    def test_single_entry(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)
        assert t.get_task_cost("t1") == pytest.approx(0.005, abs=1e-6)

    def test_multi_model_sum(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)
        t.record("t1", "gpt-4o-mini", input_tokens=1000, output_tokens=0)
        expected = 0.005 + 0.00015
        assert t.get_task_cost("t1") == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# get_daily_cost
# ---------------------------------------------------------------------------


class TestGetDailyCost:
    def test_zero_for_empty(self) -> None:
        t = _tracker()
        assert t.get_daily_cost() == pytest.approx(0.0)

    def test_today_only(self) -> None:
        calls = [_D0, _D1]
        idx = [0]

        def multi_clock() -> datetime:
            val = calls[idx[0] % len(calls)]
            idx[0] += 1
            return val

        s = _settings()
        t = CostTracker(s, clock=multi_clock)
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)  # D0
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)  # D1
        assert t.get_daily_cost(_D0.date()) == pytest.approx(0.005, abs=1e-6)
        assert t.get_daily_cost(_D1.date()) == pytest.approx(0.005, abs=1e-6)

    def test_default_day_is_today(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)
        assert t.get_daily_cost() == pytest.approx(0.005, abs=1e-6)


# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------


class TestAlertThresholds:
    def test_no_alert_below_50pct(self) -> None:
        # cap=1.0, cost=0.49 → 49 % → none
        t = _tracker(max_task=1.0)
        alert = t.record("t1", "gpt-4o", input_tokens=49_000, output_tokens=0)
        # 49000/1000*0.005 = 0.245 → 24.5 %
        assert alert.level == "none"

    def test_info_at_50pct(self) -> None:
        # cap=0.1, need ≥ 0.05 to hit 50 %
        t = _tracker(max_task=0.1)
        # 10000 input → 10*0.005 = 0.05 → exactly 50 %
        alert = t.record("t1", "gpt-4o", input_tokens=10_000, output_tokens=0)
        assert alert.level == "info"

    def test_warn_at_80pct(self) -> None:
        # cap=0.1, need > 0.08 to stay clearly above 80 %
        # 17000 input → 17*0.005 = 0.085 → 85 % (avoids float boundary)
        t = _tracker(max_task=0.1)
        alert = t.record("t1", "gpt-4o", input_tokens=17_000, output_tokens=0)
        assert alert.level == "warn"

    def test_block_at_100pct(self) -> None:
        t = _tracker(max_task=0.1)
        # 20000 input → 20*0.005 = 0.10 → exactly 100 %
        alert = t.record("t1", "gpt-4o", input_tokens=20_000, output_tokens=0)
        assert alert.level == "block"

    def test_block_above_100pct(self) -> None:
        t = _tracker(max_task=0.01)
        # small cap, easy to exceed
        alert = t.record("t1", "gpt-4o", input_tokens=10_000, output_tokens=0)
        assert alert.level == "block"

    def test_daily_cap_triggers_alert(self) -> None:
        # task cap huge, daily cap tiny
        t = _tracker(max_task=100.0, max_daily=0.01)
        # 2000 input → 2*0.005 = 0.01 → 100 % daily
        alert = t.record("t1", "gpt-4o", input_tokens=2_000, output_tokens=0)
        assert alert.level == "block"

    def test_task_pct_and_daily_pct_reported(self) -> None:
        t = _tracker(max_task=1.0, max_daily=10.0)
        alert = t.record("t1", "gpt-4o", input_tokens=1_000, output_tokens=0)
        assert alert.task_pct == pytest.approx(0.005, abs=1e-5)
        assert alert.daily_pct == pytest.approx(0.0005, abs=1e-6)

    def test_warn_message_not_empty(self) -> None:
        t = _tracker(max_task=0.1)
        alert = t.record("t1", "gpt-4o", input_tokens=16_000, output_tokens=0)
        assert len(alert.message) > 0

    def test_none_message_empty(self) -> None:
        t = _tracker(max_task=1.0)
        alert = t.record("t1", "gpt-4o", input_tokens=100, output_tokens=0)
        assert alert.message == ""


# ---------------------------------------------------------------------------
# on_alert callback
# ---------------------------------------------------------------------------


class TestOnAlertCallback:
    def test_callback_fired_on_warn(self) -> None:
        received: list[AlertResult] = []
        t = _tracker(max_task=0.1, on_alert=received.append)
        # 17000 input → 85 % of cap → warn
        t.record("t1", "gpt-4o", input_tokens=17_000, output_tokens=0)
        assert len(received) == 1
        assert received[0].level == "warn"

    def test_callback_fired_on_block(self) -> None:
        received: list[AlertResult] = []
        t = _tracker(max_task=0.01, on_alert=received.append)
        t.record("t1", "gpt-4o", input_tokens=10_000, output_tokens=0)
        assert len(received) == 1
        assert received[0].level == "block"

    def test_callback_not_fired_on_none(self) -> None:
        received: list[AlertResult] = []
        t = _tracker(max_task=1.0, on_alert=received.append)
        t.record("t1", "gpt-4o", input_tokens=100, output_tokens=0)
        assert received == []

    def test_callback_not_fired_on_info(self) -> None:
        received: list[AlertResult] = []
        t = _tracker(max_task=0.1, on_alert=received.append)
        # 50 % → info, no callback
        t.record("t1", "gpt-4o", input_tokens=10_000, output_tokens=0)
        assert received == []

    def test_no_callback_when_none(self) -> None:
        t = _tracker(max_task=0.01, on_alert=None)
        # Must not raise when no callback is set
        t.record("t1", "gpt-4o", input_tokens=10_000, output_tokens=0)


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_empty_summary(self) -> None:
        s = _tracker().get_summary()
        assert isinstance(s, CostSummary)
        assert s.total_cost_usd == pytest.approx(0.0)
        assert s.total_calls == 0
        assert s.unique_tasks == 0
        assert s.most_expensive_model is None

    def test_single_entry(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)
        s = t.get_summary()
        assert s.total_calls == 1
        assert s.unique_tasks == 1
        assert s.most_expensive_model == "gpt-4o"

    def test_unique_tasks_counted(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o", input_tokens=100, output_tokens=0)
        t.record("t1", "gpt-4o", input_tokens=100, output_tokens=0)
        t.record("t2", "gpt-4o", input_tokens=100, output_tokens=0)
        assert t.get_summary().unique_tasks == 2

    def test_avg_cost_per_call(self) -> None:
        t = _tracker()
        # Each call: 1000 input → 0.005
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)
        s = t.get_summary()
        assert s.avg_cost_per_call_usd == pytest.approx(0.005, abs=1e-6)

    def test_avg_tokens_per_call(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o", input_tokens=600, output_tokens=400)
        t.record("t1", "gpt-4o", input_tokens=200, output_tokens=800)
        # (1000 + 1000) / 2 = 1000
        assert t.get_summary().avg_tokens_per_call == pytest.approx(1000.0)

    def test_most_expensive_model(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o-mini", input_tokens=1000, output_tokens=0)
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)
        s = t.get_summary()
        assert s.most_expensive_model == "gpt-4o"
        assert s.cheapest_model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# get_dashboard_data — structure
# ---------------------------------------------------------------------------


class TestGetDashboardData:
    def test_returns_dashboard_type(self) -> None:
        assert isinstance(_tracker().get_dashboard_data(), DashboardData)

    def test_today_total(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)
        d = t.get_dashboard_data()
        assert d.today_total_usd == pytest.approx(0.005, abs=1e-6)

    def test_last_7_days_length(self) -> None:
        d = _tracker().get_dashboard_data()
        assert len(d.last_7_days) == 7

    def test_last_7_days_ordered_oldest_first(self) -> None:
        d = _tracker().get_dashboard_data()
        dates = [entry.date for entry in d.last_7_days]
        assert dates == sorted(dates)

    def test_last_7_days_today_is_last(self) -> None:
        t = _tracker()
        d = t.get_dashboard_data()
        assert d.last_7_days[-1].date == _D0.date()

    def test_last_7_days_cost_bucketed(self) -> None:
        clocks = [_D0, _D1, _D6]
        idx = [0]

        def seq_clock() -> datetime:
            v = clocks[idx[0] % len(clocks)]
            idx[0] += 1
            return v

        t = CostTracker(_settings(), clock=seq_clock)
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)  # D0
        t.record("t2", "gpt-4o", input_tokens=1000, output_tokens=0)  # D1
        t.record("t3", "gpt-4o", input_tokens=1000, output_tokens=0)  # D6

        d = t.get_dashboard_data()
        day_map = {entry.date: entry.cost_usd for entry in d.last_7_days}
        assert day_map[_D0.date()] == pytest.approx(0.005, abs=1e-6)
        assert day_map[_D1.date()] == pytest.approx(0.005, abs=1e-6)
        assert day_map[_D6.date()] == pytest.approx(0.005, abs=1e-6)

    def test_7_days_outside_window_excluded(self) -> None:
        # record() at D7, then fix clock to D0 for everything else
        entry_clocks = [_D7, _D0]
        idx = [0]

        def seq_clock() -> datetime:
            # First 2 calls come from record(); remainder from get_dashboard_data()
            v = entry_clocks[min(idx[0], len(entry_clocks) - 1)]
            idx[0] += 1
            return v

        t = CostTracker(_settings(), clock=seq_clock)
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)  # _D7
        t.record("t2", "gpt-4o", input_tokens=1000, output_tokens=0)  # _D0

        d = t.get_dashboard_data()  # clock now returns _D0 (clamped)
        # Last-7-day window relative to _D0: _D0-6 … _D0
        # _D7 = _D0 - 7 days → outside the window
        dates = {entry.date for entry in d.last_7_days}
        assert _D7.date() not in dates

    def test_last_10_tasks_at_most_10(self) -> None:
        t = _tracker()
        for i in range(15):
            t.record(f"task-{i}", "gpt-4o-mini", input_tokens=100, output_tokens=0)
        d = t.get_dashboard_data()
        assert len(d.last_10_tasks) <= 10

    def test_last_10_tasks_most_recent_first(self) -> None:
        clocks = [_D2, _D1, _D0]
        idx = [0]

        def seq_clock() -> datetime:
            v = clocks[idx[0] % len(clocks)]
            idx[0] += 1
            return v

        t = CostTracker(_settings(), clock=seq_clock)
        t.record("old", "gpt-4o-mini", input_tokens=100, output_tokens=0)  # D2
        t.record("mid", "gpt-4o-mini", input_tokens=100, output_tokens=0)  # D1
        t.record("new", "gpt-4o-mini", input_tokens=100, output_tokens=0)  # D0

        d = t.get_dashboard_data()
        order = [tc.task_id for tc in d.last_10_tasks]
        assert order == ["new", "mid", "old"]

    def test_top_5_expensive_at_most_5(self) -> None:
        t = _tracker()
        for i in range(8):
            # Vary cost: 100*(i+1) input tokens
            t.record(
                f"t{i}", "gpt-4o", input_tokens=100 * (i + 1), output_tokens=0
            )
        d = t.get_dashboard_data()
        assert len(d.top_5_expensive) <= 5

    def test_top_5_expensive_descending(self) -> None:
        t = _tracker()
        for i, cost_tokens in enumerate([100, 500, 200, 800, 300, 50]):
            t.record(f"t{i}", "gpt-4o", input_tokens=cost_tokens, output_tokens=0)
        d = t.get_dashboard_data()
        costs = [tc.cost_usd for tc in d.top_5_expensive]
        assert costs == sorted(costs, reverse=True)

    def test_top_5_expensive_most_costly_first(self) -> None:
        t = _tracker()
        t.record("cheap", "gpt-4o-mini", input_tokens=100, output_tokens=0)
        t.record("pricey", "gpt-4o", input_tokens=10_000, output_tokens=0)
        d = t.get_dashboard_data()
        assert d.top_5_expensive[0].task_id == "pricey"


# ---------------------------------------------------------------------------
# Transport breakdown
# ---------------------------------------------------------------------------


class TestTransportBreakdown:
    def test_empty_breakdown(self) -> None:
        d = _tracker().get_dashboard_data()
        b = d.transport_breakdown
        assert b.cloud_calls == 0
        assert b.native_calls == 0
        assert b.fallback_calls == 0
        assert b.native_ratio == pytest.approx(0.0)

    def test_cloud_calls_counted(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o", input_tokens=100, output_tokens=0)
        t.record("t1", "gpt-4o", input_tokens=100, output_tokens=0)
        b = t.get_dashboard_data().transport_breakdown
        assert b.cloud_calls == 2

    def test_native_calls_counted(self) -> None:
        t = _tracker()
        for m in ("uia", "dom", "file"):
            t.record_transport("t1", m)
        b = t.get_dashboard_data().transport_breakdown
        assert b.native_calls == 3
        assert b.fallback_calls == 0

    def test_fallback_calls_counted(self) -> None:
        t = _tracker()
        t.record_transport("t1", "mouse")
        t.record_transport("t1", "keyboard")
        b = t.get_dashboard_data().transport_breakdown
        assert b.fallback_calls == 2
        assert b.native_calls == 0

    def test_native_ratio_all_native(self) -> None:
        t = _tracker()
        for _ in range(4):
            t.record_transport("t1", "uia")
        assert t.get_dashboard_data().transport_breakdown.native_ratio == pytest.approx(
            1.0
        )

    def test_native_ratio_mixed(self) -> None:
        t = _tracker()
        t.record_transport("t1", "uia")
        t.record_transport("t1", "uia")
        t.record_transport("t1", "mouse")
        t.record_transport("t1", "keyboard")
        # 2 native / 4 total = 0.5
        assert t.get_dashboard_data().transport_breakdown.native_ratio == pytest.approx(
            0.5
        )

    def test_cloud_cost_in_breakdown(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)
        b = t.get_dashboard_data().transport_breakdown
        assert b.cloud_cost_usd == pytest.approx(0.005, abs=1e-6)

    def test_native_ratio_zero_when_no_transport(self) -> None:
        t = _tracker()
        assert t.get_dashboard_data().transport_breakdown.native_ratio == pytest.approx(
            0.0
        )


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_entries(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)
        t.reset()
        assert t.get_task_cost("t1") == pytest.approx(0.0)
        assert t.get_summary().total_calls == 0

    def test_reset_clears_transport(self) -> None:
        t = _tracker()
        t.record_transport("t1", "uia")
        t.reset()
        b = t.get_dashboard_data().transport_breakdown
        assert b.native_calls == 0

    def test_record_after_reset(self) -> None:
        t = _tracker()
        t.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)
        t.reset()
        t.record("t2", "gpt-4o", input_tokens=500, output_tokens=0)
        assert t.get_task_cost("t1") == pytest.approx(0.0)
        assert t.get_task_cost("t2") == pytest.approx(0.0025, abs=1e-6)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record(self) -> None:
        t = _tracker()
        n = 100

        def worker() -> None:
            for _ in range(n):
                t.record("shared", "gpt-4o-mini", input_tokens=10, output_tokens=0)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert t.get_summary().total_calls == 4 * n

    def test_concurrent_transport(self) -> None:
        t = _tracker()

        def worker() -> None:
            for _ in range(50):
                t.record_transport("t1", "uia")

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        b = t.get_dashboard_data().transport_breakdown
        assert b.native_calls == 200
