"""
tests/property/test_cost.py
Cost tracker property tests — Faz 65

Invariants tested
-----------------
- record() cost is always >= 0 for any valid token count
- get_task_cost() never decreases as more calls are recorded
- get_daily_cost() never decreases within a day
- AlertResult.task_cost_usd and .daily_cost_usd are always >= 0
- Budget cap: when task_cost >= max_cost_per_task → alert.level == "block"
- AlertResult.task_pct and .daily_pct are in [0.0, ∞) and consistent with costs
- CostTracker.get_task_cost() == sum of individually recorded costs
"""
from __future__ import annotations

from datetime import UTC, datetime

from hypothesis import assume, given
from hypothesis import strategies as st

from nexus.core.settings import BudgetSettings, NexusSettings
from nexus.infra.cost_tracker import CostTracker

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_TOKENS = st.integers(min_value=0, max_value=100_000)
_TASK_ID = st.text(min_size=1, max_size=32, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")))

_KNOWN_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-5-sonnet-20241022",
    "claude-3-haiku-20240307",
]

_MODEL = st.sampled_from(_KNOWN_MODELS)

_FIXED_NOW = datetime(2026, 4, 10, 12, 0, 0, tzinfo=UTC)


def _make_tracker(max_task_usd: float = 100.0, max_day_usd: float = 1000.0) -> CostTracker:
    settings = NexusSettings(
        budget=BudgetSettings(
            max_cost_per_task_usd=max_task_usd,
            max_cost_per_day_usd=max_day_usd,
        )
    )
    return CostTracker(settings, clock=lambda: _FIXED_NOW)


# ---------------------------------------------------------------------------
# Property: single record cost is always >= 0
# ---------------------------------------------------------------------------


@given(_MODEL, _TOKENS, _TOKENS)
def test_single_record_cost_nonnegative(model: str, in_tok: int, out_tok: int) -> None:
    tracker = _make_tracker()
    alert = tracker.record("t1", model, input_tokens=in_tok, output_tokens=out_tok)
    assert alert.task_cost_usd >= 0.0
    assert alert.daily_cost_usd >= 0.0


@given(_MODEL, _TOKENS, _TOKENS)
def test_get_task_cost_nonnegative(model: str, in_tok: int, out_tok: int) -> None:
    tracker = _make_tracker()
    tracker.record("t1", model, input_tokens=in_tok, output_tokens=out_tok)
    assert tracker.get_task_cost("t1") >= 0.0


# ---------------------------------------------------------------------------
# Property: costs are monotonically non-decreasing
# ---------------------------------------------------------------------------


@given(
    _MODEL,
    st.lists(_TOKENS, min_size=2, max_size=5),
    st.lists(_TOKENS, min_size=2, max_size=5),
)
def test_task_cost_monotonically_nondecreasing(
    model: str, in_tokens: list[int], out_tokens: list[int]
) -> None:
    assume(len(in_tokens) == len(out_tokens))
    tracker = _make_tracker()
    prev_cost = 0.0
    for in_t, out_t in zip(in_tokens, out_tokens, strict=False):
        tracker.record("t1", model, input_tokens=in_t, output_tokens=out_t)
        cur_cost = tracker.get_task_cost("t1")
        assert cur_cost >= prev_cost - 1e-12  # small float tolerance
        prev_cost = cur_cost


@given(
    _MODEL,
    st.lists(st.tuples(_TOKENS, _TOKENS), min_size=1, max_size=5),
)
def test_daily_cost_monotonically_nondecreasing(
    model: str, calls: list[tuple[int, int]]
) -> None:
    tracker = _make_tracker()
    prev = 0.0
    for in_t, out_t in calls:
        tracker.record("t1", model, input_tokens=in_t, output_tokens=out_t)
        cur = tracker.get_daily_cost()
        assert cur >= prev - 1e-12
        prev = cur


# ---------------------------------------------------------------------------
# Property: task_cost matches cumulative sum
# ---------------------------------------------------------------------------


@given(
    _MODEL,
    st.lists(st.tuples(_TOKENS, _TOKENS), min_size=1, max_size=8),
)
def test_task_cost_equals_cumulative_sum(
    model: str, calls: list[tuple[int, int]]
) -> None:
    tracker = _make_tracker()
    pricing = NexusSettings().budget.pricing_for(model)
    expected = 0.0
    for in_t, out_t in calls:
        expected += in_t / 1000.0 * pricing.input_per_1k + out_t / 1000.0 * pricing.output_per_1k
        tracker.record("t1", model, input_tokens=in_t, output_tokens=out_t)

    assert abs(tracker.get_task_cost("t1") - expected) < 1e-9


# ---------------------------------------------------------------------------
# Property: budget cap triggers block verdict
# ---------------------------------------------------------------------------


@given(
    _MODEL,
    st.integers(min_value=1_000, max_value=100_000),
    st.integers(min_value=1_000, max_value=100_000),
)
def test_budget_cap_triggers_block(model: str, in_tok: int, out_tok: int) -> None:
    """When task cost exceeds cap, alert level must be 'block'."""
    # Set a tiny cap so any non-zero call exceeds it
    tracker = _make_tracker(max_task_usd=0.000001)
    alert = tracker.record("t1", model, input_tokens=in_tok, output_tokens=out_tok)

    pricing = NexusSettings().budget.pricing_for(model)
    cost = in_tok / 1000.0 * pricing.input_per_1k + out_tok / 1000.0 * pricing.output_per_1k

    if cost > 0.000001:
        assert alert.level == "block", (
            f"cost={cost} > cap=0.000001, expected 'block', got '{alert.level}'"
        )


# ---------------------------------------------------------------------------
# Property: AlertResult percentage fields are consistent with costs
# ---------------------------------------------------------------------------


@given(_MODEL, _TOKENS, _TOKENS)
def test_alert_pct_consistent_with_costs(model: str, in_tok: int, out_tok: int) -> None:
    max_task = 10.0
    max_day = 100.0
    tracker = _make_tracker(max_task_usd=max_task, max_day_usd=max_day)
    alert = tracker.record("t1", model, input_tokens=in_tok, output_tokens=out_tok)

    expected_task_pct = alert.task_cost_usd / max_task
    expected_day_pct = alert.daily_cost_usd / max_day

    assert abs(alert.task_pct - expected_task_pct) < 1e-9
    assert abs(alert.daily_pct - expected_day_pct) < 1e-9


# ---------------------------------------------------------------------------
# Property: separate tasks don't bleed into each other
# ---------------------------------------------------------------------------


@given(
    _MODEL,
    _TOKENS,
    _TOKENS,
)
def test_separate_tasks_isolated(model: str, in_tok: int, out_tok: int) -> None:
    tracker = _make_tracker()
    tracker.record("task-A", model, input_tokens=in_tok, output_tokens=out_tok)
    tracker.record("task-B", model, input_tokens=in_tok, output_tokens=out_tok)

    cost_a = tracker.get_task_cost("task-A")
    cost_b = tracker.get_task_cost("task-B")
    # Both should be equal (same call) and not double-counted
    assert abs(cost_a - cost_b) < 1e-12
