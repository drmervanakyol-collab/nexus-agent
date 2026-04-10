"""
tests/adversarial/adv_004_cost_cap_mid_task.py
Adversarial Test 004 — Budget cap exceeded mid-task → policy blocks

Scenario:
  The per-task budget cap is set to $0.05.  The simulated task_cost_usd
  already exceeds this cap when the next action is evaluated by PolicyEngine.

Success criteria:
  - PolicyEngine.check_action() returns verdict == "block".
  - rule == RULE_TASK_BUDGET.
  - No exception raised.

All I/O is injected — no real OS calls.
"""
from __future__ import annotations

import pytest

from nexus.core.policy import (
    RULE_DAILY_BUDGET,
    RULE_TASK_BUDGET,
    ActionContext,
    PolicyEngine,
)
from nexus.core.settings import BudgetSettings, NexusSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings_with_task_budget(task_cap_usd: float) -> NexusSettings:
    """Build NexusSettings with a tight per-task budget cap."""
    return NexusSettings(
        budget=BudgetSettings(
            max_cost_per_task_usd=task_cap_usd,
            max_cost_per_day_usd=100.0,
        )
    )


def _ctx(task_cost: float, daily_cost: float = 0.0) -> ActionContext:
    return ActionContext(
        action_type="click",
        transport="mouse",
        task_cost_usd=task_cost,
        daily_cost_usd=daily_cost,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.adversarial
class TestCostCapMidTask:
    """ADV-004: Exceeding per-task budget cap blocks via PolicyEngine."""

    def test_task_budget_exceeded_blocks(self):
        """
        task_cost_usd >= task_budget → verdict 'block', RULE_TASK_BUDGET.
        """
        settings = _settings_with_task_budget(0.05)
        engine = PolicyEngine(settings)

        result = engine.check_action(_ctx(task_cost=0.07))

        assert result.verdict == "block", (
            f"Expected verdict 'block'; got {result.verdict!r}"
        )
        assert result.rule == RULE_TASK_BUDGET, (
            f"Expected RULE_TASK_BUDGET; got {result.rule!r}"
        )

    def test_under_budget_is_allowed(self):
        """task_cost_usd < task_budget → verdict 'allow'."""
        settings = _settings_with_task_budget(0.05)
        engine = PolicyEngine(settings)

        result = engine.check_action(_ctx(task_cost=0.03))

        assert result.verdict == "allow", (
            f"Expected verdict 'allow'; got {result.verdict!r}"
        )

    def test_daily_budget_exceeded_blocks(self):
        """
        daily_cost_usd >= daily_budget → verdict 'block', RULE_DAILY_BUDGET.
        Task budget is generous so RULE_DAILY_BUDGET fires.
        """
        settings = NexusSettings(
            budget=BudgetSettings(
                max_cost_per_task_usd=100.0,
                max_cost_per_day_usd=1.00,
            )
        )
        engine = PolicyEngine(settings)

        result = engine.check_action(
            ActionContext(
                action_type="click",
                transport="mouse",
                task_cost_usd=0.10,
                daily_cost_usd=1.50,  # exceeds daily cap
            )
        )

        assert result.verdict == "block"
        assert result.rule == RULE_DAILY_BUDGET, (
            f"Expected RULE_DAILY_BUDGET; got {result.rule!r}"
        )

    def test_exact_cap_boundary_is_blocked(self):
        """
        task_cost_usd == task_budget_usd triggers block
        (policy uses strict <; at-cap is not-less-than → block).
        """
        settings = _settings_with_task_budget(0.05)
        engine = PolicyEngine(settings)

        result = engine.check_action(_ctx(task_cost=0.05))
        assert result.verdict == "block"
