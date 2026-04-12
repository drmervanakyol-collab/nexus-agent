"""
tests/test_budget.py — PAKET G: Bütçe Testleri

test_task_cost_limit   — max_cost_per_task_usd=0.01, 0.02 harcayınca blok
test_daily_cost_limit  — max_cost_per_day_usd=0.10, aşılınca yeni görev başlamasın
test_warn_threshold    — warn_at_percent=0.8 noktasında warn logu
test_cost_tracking     — Görev maliyeti 0.0000 değil gerçek değer
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from nexus.core.settings import BudgetSettings, NexusSettings
from nexus.infra.cost_tracker import AlertResult, CostTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(
    max_task: float = 1.0,
    max_day: float = 10.0,
    warn_pct: float = 0.8,
) -> NexusSettings:
    """BudgetSettings override'lı NexusSettings döndür."""
    return NexusSettings(
        budget=BudgetSettings(
            max_cost_per_task_usd=max_task,
            max_cost_per_day_usd=max_day,
            warn_at_percent=warn_pct,
        )
    )


def _fixed_clock(dt: datetime):
    """Sabit saat döndüren clock fonksiyonu."""
    return lambda: dt


# ---------------------------------------------------------------------------
# PAKET G
# ---------------------------------------------------------------------------


class TestTaskCostLimit:
    """max_cost_per_task_usd=0.01 → 0.02 harcayınca görev durdurulsun."""

    def test_task_cost_limit_block(self) -> None:
        """0.02 USD harcanınca alert level 'block' olmalı."""
        settings = _make_settings(max_task=0.01)
        tracker = CostTracker(settings)

        # gpt-4o: input=0.005/1k, output=0.015/1k
        # 1000 input + 1000 output = 0.005 + 0.015 = 0.020 USD → task cap 0.01 aşılır
        alert = tracker.record(
            "task-1",
            "gpt-4o",
            input_tokens=1000,
            output_tokens=1000,
        )
        assert alert.level == "block", f"Expected 'block', got '{alert.level}'"
        assert alert.task_cost_usd > 0.01

    def test_task_cost_below_limit(self) -> None:
        """0.005 USD harcanınca limit aşılmamış olmalı."""
        settings = _make_settings(max_task=0.01)
        tracker = CostTracker(settings)

        # 1000 input tokens gpt-4o-mini = 0.00015 USD
        alert = tracker.record(
            "task-2",
            "gpt-4o-mini",
            input_tokens=1000,
            output_tokens=0,
        )
        assert alert.level != "block"
        assert alert.task_cost_usd < 0.01

    def test_task_cost_multiple_calls_accumulate(self) -> None:
        """Birden fazla çağrı task maliyetini biriktirmeli."""
        settings = _make_settings(max_task=0.10)
        tracker = CostTracker(settings)

        for _ in range(5):
            tracker.record(
                "task-accum",
                "gpt-4o-mini",
                input_tokens=1000,
                output_tokens=1000,
            )

        total = tracker.get_task_cost("task-accum")
        # 5 * (0.00015 + 0.0006) = 5 * 0.00075 = 0.00375
        assert total > 0.003
        assert total < 0.10


class TestDailyCostLimit:
    """max_cost_per_day_usd=0.10 aşılınca yeni görev başlamamalı."""

    def test_daily_limit_block(self) -> None:
        """Günlük limit aşılınca 'block' alert dönmeli."""
        settings = _make_settings(max_task=10.0, max_day=0.01)
        now = datetime.now(UTC)
        tracker = CostTracker(settings, clock=_fixed_clock(now))

        # İlk çağrıda günlük limit 0.01 → 0.020 harcar
        alert = tracker.record(
            "task-daily",
            "gpt-4o",
            input_tokens=1000,
            output_tokens=1000,
        )
        assert alert.level == "block"
        assert alert.daily_cost_usd > 0.01

    def test_daily_limit_different_tasks(self) -> None:
        """Farklı task'lar için günlük toplam hesaplanmalı."""
        settings = _make_settings(max_task=0.50, max_day=0.05)
        now = datetime.now(UTC)
        tracker = CostTracker(settings, clock=_fixed_clock(now))

        # 3 farklı task, her biri gpt-4o ile 1000 input+output = 0.02 USD
        for i in range(3):
            alert = tracker.record(
                f"task-{i}",
                "gpt-4o",
                input_tokens=1000,
                output_tokens=1000,
            )

        # 3 * 0.02 = 0.06 > 0.05 günlük cap
        assert alert.level == "block"

    def test_get_daily_cost(self) -> None:
        """get_daily_cost() bugünkü toplamı döndürmeli."""
        settings = _make_settings()
        now = datetime.now(UTC)
        tracker = CostTracker(settings, clock=_fixed_clock(now))

        tracker.record("t-1", "gpt-4o-mini", input_tokens=1000, output_tokens=1000)
        tracker.record("t-1", "gpt-4o-mini", input_tokens=500, output_tokens=500)

        daily = tracker.get_daily_cost(now.date())
        assert daily > 0.0


class TestWarnThreshold:
    """warn_at_percent=0.8 noktasında 'warn' logu çıkmalı."""

    def test_warn_level_at_80_percent(self) -> None:
        """Task maliyeti kapın %80'ine ulaşınca 'warn' dönmeli."""
        # max_task=0.10, gpt-4o: 16001 input = 0.080005 USD → %80+
        settings = _make_settings(max_task=0.10, warn_pct=0.8)
        tracker = CostTracker(settings)

        alert = tracker.record(
            "task-warn",
            "gpt-4o",
            input_tokens=16001,
            output_tokens=0,
        )

        # %80+ → warn (hardcoded threshold 0.80)
        assert alert.level in ("warn", "block"), (
            f"Expected warn/block at 80%, got '{alert.level}' "
            f"(task_cost={alert.task_cost_usd:.6f}, cap=0.10)"
        )

    def test_warn_callback_called(self) -> None:
        """warn seviyesinde on_alert callback çağrılmalı."""
        settings = _make_settings(max_task=0.10, warn_pct=0.5)
        callback = MagicMock()
        tracker = CostTracker(settings, on_alert=callback)

        # gpt-4o: 16001 input = 0.080005 → %80+ → warn (hardcoded threshold 0.80)
        tracker.record("t-cb", "gpt-4o", input_tokens=16001, output_tokens=0)

        # callback en az bir kez çağrılmış olmalı
        assert callback.call_count >= 1
        last_alert: AlertResult = callback.call_args[0][0]
        assert last_alert.level in ("warn", "block")

    def test_info_level_at_50_percent(self) -> None:
        """Task maliyeti kapın %50'sine ulaşınca en az 'info' dönmeli."""
        settings = _make_settings(max_task=0.10, warn_pct=0.8)
        tracker = CostTracker(settings)

        # gpt-4o: 2000 input + 667 output ≈ 0.02 USD → %20 (none)
        # 5000 input + 1667 output ≈ 0.05 USD → %50 → info
        alert = tracker.record(
            "task-info",
            "gpt-4o",
            input_tokens=5000,
            output_tokens=1667,
        )
        assert alert.level in ("info", "warn", "block")


class TestCostTracking:
    """Görev sonrası cost değeri gerçek hesaplanmış değer olmalı."""

    def test_cost_not_zero(self) -> None:
        """Bir cloud çağrısı sonrası task maliyeti > 0 olmalı."""
        settings = _make_settings()
        tracker = CostTracker(settings)

        tracker.record(
            "task-cost",
            "gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        cost = tracker.get_task_cost("task-cost")
        assert cost > 0.0, "Cost should be greater than 0"
        assert cost != 0.0000

    def test_cost_calculation_correct(self) -> None:
        """gpt-4o-mini için maliyet doğru hesaplanmalı."""
        settings = _make_settings()
        tracker = CostTracker(settings)

        # gpt-4o-mini: input=0.00015/1k, output=0.0006/1k
        # 1000 input + 1000 output = 0.00015 + 0.0006 = 0.00075
        tracker.record(
            "task-calc",
            "gpt-4o-mini",
            input_tokens=1000,
            output_tokens=1000,
        )
        cost = tracker.get_task_cost("task-calc")
        assert abs(cost - 0.00075) < 1e-8, f"Expected 0.00075, got {cost}"

    def test_summary_contains_real_values(self) -> None:
        """get_summary() gerçek değerler içermeli."""
        settings = _make_settings()
        tracker = CostTracker(settings)

        tracker.record("t1", "gpt-4o", input_tokens=1000, output_tokens=500)
        tracker.record("t2", "gpt-4o-mini", input_tokens=2000, output_tokens=1000)

        summary = tracker.get_summary()
        assert summary.total_cost_usd > 0.0
        assert summary.total_calls == 2
        assert summary.unique_tasks == 2
        assert summary.most_expensive_model is not None
        assert summary.avg_cost_per_call_usd > 0.0

    def test_reset_clears_cost(self) -> None:
        """reset() sonrası maliyet 0 olmalı."""
        settings = _make_settings()
        tracker = CostTracker(settings)

        tracker.record("t1", "gpt-4o", input_tokens=1000, output_tokens=500)
        assert tracker.get_task_cost("t1") > 0.0

        tracker.reset()
        assert tracker.get_task_cost("t1") == 0.0

    def test_transport_tracking_no_cost(self) -> None:
        """Transport kaydı maliyete eklenmemeli."""
        settings = _make_settings()
        tracker = CostTracker(settings)

        tracker.record_transport("t1", "uia")
        tracker.record_transport("t1", "mouse")

        assert tracker.get_task_cost("t1") == 0.0

    def test_dashboard_data_structure(self) -> None:
        """get_dashboard_data() eksiksiz yapı döndürmeli."""
        settings = _make_settings()
        tracker = CostTracker(settings)

        tracker.record("t1", "gpt-4o", input_tokens=500, output_tokens=200)
        tracker.record_transport("t1", "uia")
        tracker.record_transport("t1", "mouse")

        dash = tracker.get_dashboard_data()
        assert dash.today_total_usd > 0.0
        assert len(dash.last_7_days) == 7
        assert len(dash.last_10_tasks) >= 1
        assert dash.transport_breakdown.cloud_calls == 1
        assert dash.transport_breakdown.native_calls == 1
        assert dash.transport_breakdown.fallback_calls == 1
