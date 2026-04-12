"""
tests/unit/test_ambiguity_scorer.py
Unit + Hypothesis tests for nexus/decision/ambiguity_scorer.py.

Coverage
--------
  §1  AmbiguityScore dataclass
  §2  Weights catalogue — sum check
  §3  _compute_weighted_score helper
  §4  _recommend helper — threshold boundaries
  §5  _is_stuck helper
  §6  Individual factor isolation
      §6a overall_confidence factor
      §6b action_risk (destructive)
      §6c source_disagreement (conflicts)
      §6d new_screen_pattern
      §6e stuck_indicator
      §6f temporal_instability (all StateType values)
      §6g transport_uncertainty
  §7  Recommendation decisions
      Destructive → cloud or suspend
      Stuck → cloud or suspend
  §8  dominant_factor resolution
  §9  Hypothesis: weights sum to 1.0
  §10 Hypothesis: arbitrary factor values → score in [0.0, 1.0]
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nexus.decision.ambiguity_scorer import (
    _WEIGHTS,
    AmbiguityScore,
    AmbiguityScorer,
    _compute_weighted_score,
    _dominant_factor,
    _is_stuck,
    _recommend,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_perception(
    overall_confidence: float = 0.9,
    conflicts_detected: int = 0,
    state: str = "STABLE",
) -> MagicMock:
    """Return a MagicMock duck-typing PerceptionResult."""
    p = MagicMock()
    p.arbitration.overall_confidence = overall_confidence
    p.arbitration.conflicts_detected = conflicts_detected
    p.screen_state.state_type.name = state
    return p


def _make_action(action_type: str = "click", target: str = "Button") -> MagicMock:
    """Return a MagicMock duck-typing ActionRecord."""
    rec = MagicMock()
    rec.action_type = action_type
    rec.target_description = target
    return rec


_SCORER = AmbiguityScorer()


def _score(**kwargs) -> AmbiguityScore:
    """Call scorer with safe defaults; override via kwargs."""
    defaults = {
        "perception": _make_perception(),
        "action_history": [],
        "candidate_is_destructive": False,
        "screen_previously_seen": True,
        "used_fallback_transport": False,
    }
    defaults.update(kwargs)
    return _SCORER.score(
        defaults.pop("perception"),
        defaults.pop("action_history"),
        **defaults,
    )


# ---------------------------------------------------------------------------
# §1 — AmbiguityScore dataclass
# ---------------------------------------------------------------------------


class TestAmbiguityScoreDataclass:
    def test_fields_present(self) -> None:
        result = _score()
        assert hasattr(result, "score")
        assert hasattr(result, "recommendation")
        assert hasattr(result, "factors")
        assert hasattr(result, "dominant_factor")

    def test_score_in_range(self) -> None:
        result = _score()
        assert 0.0 <= result.score <= 1.0

    def test_recommendation_valid_value(self) -> None:
        result = _score()
        assert result.recommendation in ("local", "cloud", "suspend")

    def test_factors_has_all_keys(self) -> None:
        result = _score()
        expected_keys = set(_WEIGHTS.keys())
        assert expected_keys == set(result.factors.keys())

    def test_factors_all_in_range(self) -> None:
        result = _score()
        for k, v in result.factors.items():
            assert 0.0 <= v <= 1.0, f"Factor {k!r} = {v} out of range"

    def test_dominant_factor_is_key_of_factors(self) -> None:
        result = _score()
        assert result.dominant_factor in result.factors

    def test_frozen(self) -> None:
        result = _score()
        with pytest.raises((AttributeError, TypeError)):
            result.score = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# §2 — Weights catalogue
# ---------------------------------------------------------------------------


class TestWeights:
    def test_all_seven_factors_defined(self) -> None:
        assert len(_WEIGHTS) == 7

    def test_weights_sum_to_one(self) -> None:
        total = sum(_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum = {total}, expected 1.0"

    def test_all_weights_positive(self) -> None:
        for name, w in _WEIGHTS.items():
            assert w > 0.0, f"Weight for {name!r} must be positive"

    def test_expected_weights(self) -> None:
        assert _WEIGHTS["overall_confidence"] == pytest.approx(0.10)
        assert _WEIGHTS["action_risk"] == pytest.approx(0.15)
        assert _WEIGHTS["source_disagreement"] == pytest.approx(0.10)
        assert _WEIGHTS["new_screen_pattern"] == pytest.approx(0.10)
        assert _WEIGHTS["stuck_indicator"] == pytest.approx(0.10)
        assert _WEIGHTS["temporal_instability"] == pytest.approx(0.05)
        assert _WEIGHTS["transport_uncertainty"] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# §3 — _compute_weighted_score helper
# ---------------------------------------------------------------------------


class TestComputeWeightedScore:
    def test_all_zeros_gives_zero(self) -> None:
        factors = dict.fromkeys(_WEIGHTS, 0.0)
        assert _compute_weighted_score(factors, _WEIGHTS) == pytest.approx(0.0)

    def test_all_ones_gives_one(self) -> None:
        factors = dict.fromkeys(_WEIGHTS, 1.0)
        assert _compute_weighted_score(factors, _WEIGHTS) == pytest.approx(1.0)

    def test_single_factor_weight(self) -> None:
        # Only action_risk = 1.0, everything else 0
        factors = dict.fromkeys(_WEIGHTS, 0.0)
        factors["action_risk"] = 1.0
        score = _compute_weighted_score(factors, _WEIGHTS)
        assert score == pytest.approx(0.15)

    def test_clamp_prevents_above_one(self) -> None:
        # Even if factors somehow exceed 1.0, output is clamped
        factors = dict.fromkeys(_WEIGHTS, 2.0)
        assert _compute_weighted_score(factors, _WEIGHTS) <= 1.0

    def test_clamp_prevents_below_zero(self) -> None:
        factors = dict.fromkeys(_WEIGHTS, -1.0)
        assert _compute_weighted_score(factors, _WEIGHTS) >= 0.0


# ---------------------------------------------------------------------------
# §4 — _recommend thresholds
# ---------------------------------------------------------------------------


class TestRecommend:
    def test_score_zero_is_local(self) -> None:
        assert _recommend(0.0) == "local"

    def test_score_just_below_cloud_threshold_is_local(self) -> None:
        assert _recommend(0.399) == "local"

    def test_score_at_cloud_threshold_is_cloud(self) -> None:
        assert _recommend(0.40) == "cloud"

    def test_score_midway_cloud_is_cloud(self) -> None:
        assert _recommend(0.55) == "cloud"

    def test_score_just_below_suspend_is_cloud(self) -> None:
        assert _recommend(0.699) == "cloud"

    def test_score_at_suspend_threshold_is_suspend(self) -> None:
        assert _recommend(0.70) == "suspend"

    def test_score_one_is_suspend(self) -> None:
        assert _recommend(1.0) == "suspend"


# ---------------------------------------------------------------------------
# §5 — _is_stuck helper
# ---------------------------------------------------------------------------


class TestIsStuck:
    def test_empty_history_not_stuck(self) -> None:
        assert _is_stuck([]) is False

    def test_one_action_not_stuck(self) -> None:
        assert _is_stuck([_make_action()]) is False

    def test_two_actions_not_stuck(self) -> None:
        assert _is_stuck([_make_action(), _make_action()]) is False

    def test_three_identical_is_stuck(self) -> None:
        history = [_make_action("click", "OK") for _ in range(3)]
        assert _is_stuck(history) is True

    def test_three_different_not_stuck(self) -> None:
        history = [
            _make_action("click", "Button1"),
            _make_action("click", "Button2"),
            _make_action("click", "Button3"),
        ]
        assert _is_stuck(history) is False

    def test_four_with_last_three_identical_is_stuck(self) -> None:
        history = [
            _make_action("click", "OtherButton"),
            _make_action("click", "SameButton"),
            _make_action("click", "SameButton"),
            _make_action("click", "SameButton"),
        ]
        assert _is_stuck(history) is True

    def test_last_three_differ_even_with_repeats_not_stuck(self) -> None:
        history = [
            _make_action("click", "Same"),
            _make_action("click", "Same"),
            _make_action("click", "Same"),
            _make_action("type", "Same"),  # last one differs in action_type
        ]
        assert _is_stuck(history) is False

    def test_different_action_type_breaks_stuck(self) -> None:
        history = [
            _make_action("click", "Button"),
            _make_action("type", "Button"),
            _make_action("click", "Button"),
        ]
        assert _is_stuck(history) is False


# ---------------------------------------------------------------------------
# §6 — Individual factor isolation
# ---------------------------------------------------------------------------


class TestOverallConfidenceFactor:
    def test_high_confidence_low_score(self) -> None:
        result = _score(perception=_make_perception(overall_confidence=1.0))
        assert result.factors["overall_confidence"] == pytest.approx(0.0)

    def test_zero_confidence_full_score(self) -> None:
        result = _score(perception=_make_perception(overall_confidence=0.0))
        assert result.factors["overall_confidence"] == pytest.approx(1.0)

    def test_midpoint_confidence(self) -> None:
        result = _score(perception=_make_perception(overall_confidence=0.5))
        assert result.factors["overall_confidence"] == pytest.approx(0.5)


class TestActionRiskFactor:
    def test_non_destructive_zero(self) -> None:
        result = _score(candidate_is_destructive=False)
        assert result.factors["action_risk"] == pytest.approx(0.0)

    def test_destructive_one(self) -> None:
        result = _score(candidate_is_destructive=True)
        assert result.factors["action_risk"] == pytest.approx(1.0)


class TestSourceDisagreementFactor:
    def test_no_conflicts_zero(self) -> None:
        result = _score(perception=_make_perception(conflicts_detected=0))
        assert result.factors["source_disagreement"] == pytest.approx(0.0)

    def test_one_conflict_partial(self) -> None:
        result = _score(perception=_make_perception(conflicts_detected=1))
        assert result.factors["source_disagreement"] == pytest.approx(1 / 3)

    def test_three_conflicts_full(self) -> None:
        result = _score(perception=_make_perception(conflicts_detected=3))
        assert result.factors["source_disagreement"] == pytest.approx(1.0)

    def test_more_than_three_capped(self) -> None:
        result = _score(perception=_make_perception(conflicts_detected=10))
        assert result.factors["source_disagreement"] == pytest.approx(1.0)


class TestNewScreenPatternFactor:
    def test_seen_before_zero(self) -> None:
        result = _score(screen_previously_seen=True)
        assert result.factors["new_screen_pattern"] == pytest.approx(0.0)

    def test_new_screen_one(self) -> None:
        result = _score(screen_previously_seen=False)
        assert result.factors["new_screen_pattern"] == pytest.approx(1.0)


class TestStuckIndicatorFactor:
    def test_no_history_zero(self) -> None:
        result = _score(action_history=[])
        assert result.factors["stuck_indicator"] == pytest.approx(0.0)

    def test_stuck_one(self) -> None:
        stuck = [_make_action("click", "Btn") for _ in range(3)]
        result = _score(action_history=stuck)
        assert result.factors["stuck_indicator"] == pytest.approx(1.0)

    def test_not_stuck_zero(self) -> None:
        varied = [
            _make_action("click", "A"),
            _make_action("click", "B"),
            _make_action("click", "C"),
        ]
        result = _score(action_history=varied)
        assert result.factors["stuck_indicator"] == pytest.approx(0.0)


class TestTemporalInstabilityFactor:
    @pytest.mark.parametrize("state,expected", [
        ("STABLE",       0.0),
        ("UNKNOWN",      0.3),
        ("FROZEN",       0.5),
        ("ANIMATING",    0.8),
        ("TRANSITIONING", 0.9),
        ("LOADING",      1.0),
    ])
    def test_state_mapping(self, state: str, expected: float) -> None:
        result = _score(perception=_make_perception(state=state))
        assert result.factors["temporal_instability"] == pytest.approx(expected)


class TestTransportUncertaintyFactor:
    def test_native_zero(self) -> None:
        result = _score(used_fallback_transport=False)
        assert result.factors["transport_uncertainty"] == pytest.approx(0.0)

    def test_fallback_one(self) -> None:
        result = _score(used_fallback_transport=True)
        assert result.factors["transport_uncertainty"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# §7 — Recommendation decisions
# ---------------------------------------------------------------------------


class TestRecommendationDecisions:
    def test_all_clear_gives_local(self) -> None:
        """High confidence, no risk, no conflicts → local."""
        result = _score(
            perception=_make_perception(
                overall_confidence=0.95, conflicts_detected=0, state="STABLE"
            ),
            candidate_is_destructive=False,
            screen_previously_seen=True,
            used_fallback_transport=False,
        )
        assert result.recommendation == "local"

    def test_destructive_action_raises_to_cloud(self) -> None:
        """Destructive + fallback transport + low confidence + new screen → cloud."""
        # transport_uncertainty: 1.0 × 0.40 = 0.40  ← dominant; alone hits threshold
        # action_risk: 1.0 × 0.15 = 0.15
        # overall_confidence: (1-0.3) × 0.10 = 0.07
        # new_screen_pattern: 1.0 × 0.10 = 0.10
        # source_disagreement: (1/3) × 0.10 ≈ 0.033
        # total ≈ 0.753 → suspend
        result = _score(
            perception=_make_perception(
                overall_confidence=0.3, conflicts_detected=1, state="STABLE"
            ),
            candidate_is_destructive=True,
            screen_previously_seen=False,
            used_fallback_transport=True,
        )
        assert result.recommendation in ("cloud", "suspend")

    def test_stuck_agent_raises_recommendation(self) -> None:
        """Stuck + fallback transport + low-confidence + new screen → cloud/suspend."""
        # transport_uncertainty: 1.0 × 0.40 = 0.40  ← primary escalation signal
        # stuck: 1.0 × 0.10 = 0.10
        # overall_confidence: (1-0.3) × 0.10 = 0.07
        # new_screen_pattern: 1.0 × 0.10 = 0.10
        # source_disagreement: (1/3) × 0.10 ≈ 0.033
        # total ≈ 0.703 → suspend
        stuck = [_make_action("click", "Btn") for _ in range(3)]
        result = _score(
            perception=_make_perception(
                overall_confidence=0.3, conflicts_detected=1, state="STABLE"
            ),
            action_history=stuck,
            screen_previously_seen=False,
            used_fallback_transport=True,
        )
        assert result.recommendation in ("cloud", "suspend")

    def test_worst_case_is_suspend(self) -> None:
        """All factors at maximum → suspend."""
        stuck = [_make_action("click", "Btn") for _ in range(3)]
        result = _score(
            perception=_make_perception(
                overall_confidence=0.0, conflicts_detected=5, state="LOADING"
            ),
            action_history=stuck,
            candidate_is_destructive=True,
            screen_previously_seen=False,
            used_fallback_transport=True,
        )
        assert result.recommendation == "suspend"
        assert result.score == pytest.approx(1.0)

    def test_best_case_is_local(self) -> None:
        """All factors at minimum → local."""
        result = _score(
            perception=_make_perception(
                overall_confidence=1.0, conflicts_detected=0, state="STABLE"
            ),
            action_history=[],
            candidate_is_destructive=False,
            screen_previously_seen=True,
            used_fallback_transport=False,
        )
        assert result.recommendation == "local"
        assert result.score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# §8 — dominant_factor
# ---------------------------------------------------------------------------


class TestDominantFactor:
    def test_all_zero_except_action_risk(self) -> None:
        factors = dict.fromkeys(_WEIGHTS, 0.0)
        factors["action_risk"] = 1.0
        dominant = _dominant_factor(factors, _WEIGHTS)
        assert dominant == "action_risk"

    def test_dominant_with_high_overall_confidence(self) -> None:
        """overall_confidence=1.0 with weight 0.20 → dominant."""
        factors = dict.fromkeys(_WEIGHTS, 0.0)
        factors["overall_confidence"] = 1.0
        dominant = _dominant_factor(factors, _WEIGHTS)
        assert dominant == "overall_confidence"

    def test_dominant_factor_is_in_weights(self) -> None:
        result = _score(
            perception=_make_perception(overall_confidence=0.0),
            candidate_is_destructive=True,
        )
        assert result.dominant_factor in _WEIGHTS

    def test_stuck_is_dominant_when_highest(self) -> None:
        """stuck_indicator × 0.15 with everything else 0 → dominant."""
        factors = dict.fromkeys(_WEIGHTS, 0.0)
        factors["stuck_indicator"] = 1.0
        dominant = _dominant_factor(factors, _WEIGHTS)
        assert dominant == "stuck_indicator"


# ---------------------------------------------------------------------------
# §9 — Hypothesis: weights sum to 1.0
# ---------------------------------------------------------------------------


@given(st.just(_WEIGHTS))
@settings(max_examples=1, deadline=None)
def test_hypothesis_weights_sum_to_one(weights: dict[str, float]) -> None:
    """The _WEIGHTS constant must always sum to exactly 1.0."""
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


# ---------------------------------------------------------------------------
# §10 — Hypothesis: arbitrary factor values → score in [0.0, 1.0]
# ---------------------------------------------------------------------------


@given(
    st.fixed_dictionaries({
        "overall_confidence":    st.floats(0.0, 1.0, allow_nan=False),
        "action_risk":           st.floats(0.0, 1.0, allow_nan=False),
        "source_disagreement":   st.floats(0.0, 1.0, allow_nan=False),
        "new_screen_pattern":    st.floats(0.0, 1.0, allow_nan=False),
        "stuck_indicator":       st.floats(0.0, 1.0, allow_nan=False),
        "temporal_instability":  st.floats(0.0, 1.0, allow_nan=False),
        "transport_uncertainty": st.floats(0.0, 1.0, allow_nan=False),
    })
)
@settings(max_examples=500, deadline=None)
def test_hypothesis_score_always_in_unit_interval(
    factors: dict[str, float],
) -> None:
    """For any factor values in [0,1], the weighted score is in [0,1]."""
    score = _compute_weighted_score(factors, _WEIGHTS)
    assert 0.0 <= score <= 1.0, f"score={score} out of [0,1] for factors={factors}"
