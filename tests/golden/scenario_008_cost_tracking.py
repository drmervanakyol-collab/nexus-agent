"""
tests/golden/scenario_008_cost_tracking.py
Golden Scenario 008 — CostTracker records non-zero cost for a cloud call

Goal: simulate a task that makes a real (or injected) cloud LLM call and
verify that CostTracker.get_task_cost() returns a value > 0 after the call.

Setup:
  Build a CostTracker with real BudgetSettings (default pricing).
Execute:
  Record one LLM call (model=claude-3-5-sonnet-20241022, 500 in + 200 out).
Assert:
  get_task_cost(task_id) > 0
  alert.level in {"ok", "warn"}   — not "block" at default thresholds.
Teardown:
  (no-op — CostTracker is in-memory)
Report:
  Recorded cost, alert level.
"""
from __future__ import annotations

import pytest

_TASK_ID = "golden-cost-008"
_MODEL = "claude-3-5-sonnet-20241022"
_INPUT_TOKENS = 500
_OUTPUT_TOKENS = 200


@pytest.mark.golden
class TestCostTracking:
    """Scenario 008 — CostTracker records non-zero cost after a cloud call."""

    def test_cost_recorded_after_cloud_call(self, scenario_report):
        """
        Record an LLM call and assert the accumulated cost is > 0.
        Uses real BudgetSettings pricing tables — no mocking.
        """
        # ------------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------------
        from nexus.core.settings import BudgetSettings, NexusSettings
        from nexus.infra.cost_tracker import CostTracker

        settings = NexusSettings()
        tracker = CostTracker(settings)

        # ------------------------------------------------------------------
        # Execute — record one cloud call
        # ------------------------------------------------------------------
        alert = tracker.record(
            _TASK_ID,
            _MODEL,
            input_tokens=_INPUT_TOKENS,
            output_tokens=_OUTPUT_TOKENS,
        )

        # Also record a transport action so transport coverage shows in report
        tracker.record_transport(_TASK_ID, "uia")

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        cost = tracker.get_task_cost(_TASK_ID)

        scenario_report.cost_usd = cost
        scenario_report.steps = 1
        scenario_report.native_count = 1  # one UIA transport action

        scenario_report.assert_ok(
            cost > 0,
            f"get_task_cost must be > 0 after cloud call; got {cost}",
        )

        # Expected cost: 500/1000 * 0.003 + 200/1000 * 0.015 = 0.0015 + 0.003 = 0.0045
        expected_cost = (
            _INPUT_TOKENS / 1000.0 * settings.budget.claude_sonnet_input_per_1k
            + _OUTPUT_TOKENS / 1000.0 * settings.budget.claude_sonnet_output_per_1k
        )
        scenario_report.assert_ok(
            abs(cost - expected_cost) < 1e-9,
            f"Cost must match pricing formula: expected {expected_cost}, got {cost}",
        )

        scenario_report.assert_ok(
            alert.level in {"ok", "warn"},
            f"Alert level must be 'ok' or 'warn' (not blocked); got {alert.level!r}",
        )

        # ------------------------------------------------------------------
        # Report
        # ------------------------------------------------------------------
        print(f"\n  Alert level: {alert.level} | Recorded cost: ${cost:.6f}")
