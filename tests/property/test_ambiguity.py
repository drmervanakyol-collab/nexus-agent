"""
tests/property/test_ambiguity.py
Ambiguity scorer property tests — Faz 65

Invariants tested
-----------------
- AmbiguityScore.score is always in [0.0, 1.0]
- _WEIGHTS values are all in (0, 1] and sum to 1.0 (±1e-9)
- _compute_weighted_score returns a value in [0.0, 1.0] for any factor map
- recommendation is always one of {"local", "cloud", "suspend"}
- recommendation is consistent with score thresholds:
    score < 0.40  → "local"
    0.40 ≤ score < 0.70 → "cloud"
    score ≥ 0.70  → "suspend"
- AmbiguityScorer.score() never raises for valid PerceptionResult inputs
"""
from __future__ import annotations

import datetime

from hypothesis import given
from hypothesis import strategies as st

from nexus.decision.ambiguity_scorer import (
    _WEIGHTS,
    AmbiguityScorer,
    _compute_weighted_score,
)

# ---------------------------------------------------------------------------
# Helpers: build minimal PerceptionResult for scorer
# ---------------------------------------------------------------------------


def _make_perception(
    overall_confidence: float = 1.0,
    conflicts_detected: int = 0,
    state_type_name: str = "STABLE",
) -> object:
    from nexus.perception.arbitration.arbitrator import ArbitrationResult
    from nexus.perception.orchestrator import PerceptionResult
    from nexus.perception.spatial_graph import SpatialGraph
    from nexus.perception.temporal.temporal_expert import ScreenState, StateType
    from nexus.source.resolver import SourceResult

    state = ScreenState(
        state_type=StateType[state_type_name],
        confidence=1.0,
        blocks_perception=False,
        reason="test",
        retry_after_ms=0,
    )
    arb = ArbitrationResult(
        resolved_elements=(),
        resolved_labels=(),
        conflicts_detected=conflicts_detected,
        conflicts_resolved=0,
        temporal_blocked=False,
        overall_confidence=overall_confidence,
    )
    source = SourceResult(source_type="uia", data=[], confidence=1.0, latency_ms=0.0)
    return PerceptionResult(
        spatial_graph=SpatialGraph([], [], {}),
        screen_state=state,
        arbitration=arb,
        source_result=source,
        perception_ms=0.0,
        frame_sequence=1,
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
    )


# ---------------------------------------------------------------------------
# Weights: static invariants
# ---------------------------------------------------------------------------


def test_weights_sum_to_one() -> None:
    total = sum(_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


def test_weights_each_in_unit_interval() -> None:
    for name, w in _WEIGHTS.items():
        assert 0 < w <= 1.0, f"Weight '{name}' = {w} out of (0, 1]"


def test_weights_cover_all_expected_factors() -> None:
    expected = {
        "overall_confidence",
        "action_risk",
        "source_disagreement",
        "new_screen_pattern",
        "stuck_indicator",
        "temporal_instability",
        "transport_uncertainty",
    }
    assert set(_WEIGHTS.keys()) == expected


# ---------------------------------------------------------------------------
# _compute_weighted_score: output always in [0, 1]
# ---------------------------------------------------------------------------

_factor_value = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@given(
    st.fixed_dictionaries(
        dict.fromkeys(_WEIGHTS, _factor_value)
    )
)
def test_weighted_score_in_unit_interval(factors: dict) -> None:
    score = _compute_weighted_score(factors, _WEIGHTS)
    assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"


@given(
    st.fixed_dictionaries(
        {k: st.just(1.0) for k in _WEIGHTS}
    )
)
def test_weighted_score_max_factors_gives_one(factors: dict) -> None:
    score = _compute_weighted_score(factors, _WEIGHTS)
    assert abs(score - 1.0) < 1e-9


@given(
    st.fixed_dictionaries(
        {k: st.just(0.0) for k in _WEIGHTS}
    )
)
def test_weighted_score_zero_factors_gives_zero(factors: dict) -> None:
    score = _compute_weighted_score(factors, _WEIGHTS)
    assert score == 0.0


# ---------------------------------------------------------------------------
# AmbiguityScorer.score(): output invariants
# ---------------------------------------------------------------------------

_confidence = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_conflicts = st.integers(min_value=0, max_value=10)
_state_name = st.sampled_from(
    ["STABLE", "LOADING", "TRANSITIONING", "ANIMATING", "FROZEN", "UNKNOWN"]
)


@given(_confidence, _conflicts, _state_name)
def test_scorer_score_in_unit_interval(
    confidence: float, conflicts: int, state: str
) -> None:
    scorer = AmbiguityScorer()
    perception = _make_perception(
        overall_confidence=confidence,
        conflicts_detected=conflicts,
        state_type_name=state,
    )
    result = scorer.score(perception, action_history=[])
    assert 0.0 <= result.score <= 1.0, f"Score {result.score} out of [0, 1]"


@given(_confidence, _conflicts, _state_name)
def test_scorer_recommendation_is_valid(
    confidence: float, conflicts: int, state: str
) -> None:
    scorer = AmbiguityScorer()
    perception = _make_perception(
        overall_confidence=confidence,
        conflicts_detected=conflicts,
        state_type_name=state,
    )
    result = scorer.score(perception, action_history=[])
    assert result.recommendation in {"local", "cloud", "suspend"}


@given(_confidence, _conflicts, _state_name)
def test_scorer_recommendation_consistent_with_score(
    confidence: float, conflicts: int, state: str
) -> None:
    """recommendation must match the documented threshold bands."""
    scorer = AmbiguityScorer()
    perception = _make_perception(
        overall_confidence=confidence,
        conflicts_detected=conflicts,
        state_type_name=state,
    )
    result = scorer.score(perception, action_history=[])
    s = result.score

    if s < 0.40:
        assert result.recommendation == "local", (
            f"score={s} → expected 'local', got '{result.recommendation}'"
        )
    elif s < 0.70:
        assert result.recommendation == "cloud", (
            f"score={s} → expected 'cloud', got '{result.recommendation}'"
        )
    else:
        assert result.recommendation == "suspend", (
            f"score={s} → expected 'suspend', got '{result.recommendation}'"
        )


def test_scorer_all_uncertain_recommends_suspend() -> None:
    """Maximum uncertainty (all factors=1.0) must give 'suspend'."""
    scorer = AmbiguityScorer()
    perception = _make_perception(
        overall_confidence=0.0,   # max uncertainty factor
        conflicts_detected=10,    # saturates source_disagreement
        state_type_name="LOADING",
    )
    result = scorer.score(
        perception,
        action_history=[],
        candidate_is_destructive=True,
        screen_previously_seen=False,
        used_fallback_transport=True,
    )
    assert result.recommendation == "suspend"
    assert result.score >= 0.70


def test_scorer_fully_confident_recommends_local() -> None:
    """Full confidence, no risk → 'local'."""
    scorer = AmbiguityScorer()
    perception = _make_perception(
        overall_confidence=1.0,
        conflicts_detected=0,
        state_type_name="STABLE",
    )
    result = scorer.score(
        perception,
        action_history=[],
        candidate_is_destructive=False,
        screen_previously_seen=True,
        used_fallback_transport=False,
    )
    assert result.recommendation == "local"
    assert result.score < 0.40
