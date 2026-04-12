"""
nexus/decision/ambiguity_scorer.py
Ambiguity Scorer — estimates how uncertain the agent is about the next action.

AmbiguityScore
--------------
  score           float  0.0 = certain, 1.0 = maximally uncertain
  recommendation  str    "local" | "cloud" | "suspend"
  factors         dict   raw per-factor value in [0.0, 1.0] before weighting
  dominant_factor str    name of the factor with the highest weighted contribution

Recommendation thresholds
--------------------------
  score < 0.40  → "local"   (handle locally, no cloud needed)
  0.40 ≤ score < 0.70 → "cloud"   (consult the LLM planner)
  score ≥ 0.70  → "suspend" (pause and ask for human help)

Factor catalogue (7 factors, weights sum to 1.00)
--------------------------------------------------
  overall_confidence   (0.10)
      1 − perception.arbitration.overall_confidence.
      Low perception confidence = high ambiguity.

  action_risk          (0.15)
      1.0 if the candidate action targets a destructive element; 0.0 otherwise.

  source_disagreement  (0.10)
      min(1.0, conflicts_detected / 3).
      Proxy for UIA vs visual disagreement: arbitration conflict count.

  new_screen_pattern   (0.10)
      1.0 when the current screen fingerprint was not seen before; 0.0 otherwise.

  stuck_indicator      (0.10)
      1.0 when the last three (action_type, target_description) tuples are all
      identical (agent is looping); 0.0 otherwise.

  temporal_instability (0.05)
      Continuous score based on StateType:
        LOADING → 1.0, TRANSITIONING → 0.9, ANIMATING → 0.8,
        FROZEN → 0.5, UNKNOWN → 0.3, STABLE → 0.0.

  transport_uncertainty (0.40)
      1.0 when the most recent transport used a fallback (mouse/keyboard)
      instead of the native (UIA/DOM) channel; 0.0 otherwise.
      High weight so a single fallback step immediately triggers cloud planning.

Public API
----------
  AmbiguityScore             frozen output dataclass
  AmbiguityScorer.score()    single method; returns AmbiguityScore
  _WEIGHTS                   exposed for tests (Hypothesis weight-sum check)
  _compute_weighted_score()  exposed for Hypothesis score-range test
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from nexus.infra.logger import get_logger
from nexus.perception.orchestrator import PerceptionResult

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Weights (must sum to exactly 1.00)
# ---------------------------------------------------------------------------

_WEIGHTS: dict[str, float] = {
    "overall_confidence":   0.10,
    "action_risk":          0.15,
    "source_disagreement":  0.10,
    "new_screen_pattern":   0.10,
    "stuck_indicator":      0.10,
    "temporal_instability": 0.05,
    "transport_uncertainty": 0.40,
}

# ---------------------------------------------------------------------------
# Recommendation thresholds
# ---------------------------------------------------------------------------

_THRESHOLD_CLOUD: float = 0.40
_THRESHOLD_SUSPEND: float = 0.70

# ---------------------------------------------------------------------------
# Temporal instability mapping (StateType name → raw factor value)
# ---------------------------------------------------------------------------

_TEMPORAL_FACTOR: dict[str, float] = {
    "LOADING":       1.0,
    "TRANSITIONING": 0.9,
    "ANIMATING":     0.8,
    "FROZEN":        0.5,
    "UNKNOWN":       0.3,
    "STABLE":        0.0,
}

# Number of conflicts that saturate source_disagreement to 1.0
_CONFLICT_SATURATION: int = 3


# ---------------------------------------------------------------------------
# AmbiguityScore
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmbiguityScore:
    """
    Output of :meth:`AmbiguityScorer.score`.

    Attributes
    ----------
    score:
        Weighted sum of all factor values, in [0.0, 1.0].
        0.0 = perfectly certain, 1.0 = maximally uncertain.
    recommendation:
        ``"local"`` — proceed without cloud consultation.
        ``"cloud"`` — consult the LLM planner before acting.
        ``"suspend"`` — pause and request human guidance.
    factors:
        Raw (unweighted) value for each factor in [0.0, 1.0].
        Useful for debugging and logging.
    dominant_factor:
        Name of the factor contributing the most to the final score
        (highest ``factor_value × weight``).
    """

    score: float
    recommendation: str
    factors: dict[str, float]
    dominant_factor: str


# ---------------------------------------------------------------------------
# AmbiguityScorer
# ---------------------------------------------------------------------------


class AmbiguityScorer:
    """
    Stateless scorer; create one instance and reuse it across planning cycles.

    All inputs that cannot be derived from :class:`PerceptionResult` are
    passed explicitly as keyword arguments.
    """

    def score(
        self,
        perception: PerceptionResult,
        action_history: Sequence[Any],
        *,
        candidate_is_destructive: bool = False,
        screen_previously_seen: bool = True,
        used_fallback_transport: bool = False,
    ) -> AmbiguityScore:
        """
        Compute an ambiguity score for the current planning context.

        Parameters
        ----------
        perception:
            Current perception result (provides confidence and conflict data).
        action_history:
            All completed :class:`~nexus.cloud.prompt_builder.ActionRecord`
            objects so far.  Duck-typed: only ``.action_type`` and
            ``.target_description`` attributes are accessed.
        candidate_is_destructive:
            True when the candidate next action targets an element flagged as
            destructive (e.g. a Delete / Sil button).
        screen_previously_seen:
            True when the current screen fingerprint is present in memory.
            Pass False for never-before-seen screens.
        used_fallback_transport:
            True when the most recent transport action used a fallback
            (mouse/keyboard) instead of native UIA/DOM.

        Returns
        -------
        AmbiguityScore
        """
        factors = _compute_factors(
            perception=perception,
            action_history=action_history,
            candidate_is_destructive=candidate_is_destructive,
            screen_previously_seen=screen_previously_seen,
            used_fallback_transport=used_fallback_transport,
        )

        total = _compute_weighted_score(factors, _WEIGHTS)
        dominant = _dominant_factor(factors, _WEIGHTS)
        recommendation = _recommend(total)

        _log.debug(
            "ambiguity_scored",
            score=round(total, 4),
            recommendation=recommendation,
            dominant=dominant,
            factors={k: round(v, 4) for k, v in factors.items()},
        )

        return AmbiguityScore(
            score=total,
            recommendation=recommendation,
            factors=factors,
            dominant_factor=dominant,
        )


# ---------------------------------------------------------------------------
# Factor computation
# ---------------------------------------------------------------------------


def _compute_factors(
    *,
    perception: PerceptionResult,
    action_history: Sequence[Any],
    candidate_is_destructive: bool,
    screen_previously_seen: bool,
    used_fallback_transport: bool,
) -> dict[str, float]:
    """Compute raw factor values in [0.0, 1.0] for each named factor."""
    arb = perception.arbitration
    state_name = perception.screen_state.state_type.name

    overall_conf_factor = 1.0 - max(0.0, min(1.0, arb.overall_confidence))

    action_risk_factor = 1.0 if candidate_is_destructive else 0.0

    source_disagree_factor = min(
        1.0, arb.conflicts_detected / _CONFLICT_SATURATION
    )

    new_pattern_factor = 0.0 if screen_previously_seen else 1.0

    stuck_factor = 1.0 if _is_stuck(action_history) else 0.0

    temporal_factor = _TEMPORAL_FACTOR.get(state_name, 0.3)

    transport_factor = 1.0 if used_fallback_transport else 0.0

    return {
        "overall_confidence":    overall_conf_factor,
        "action_risk":           action_risk_factor,
        "source_disagreement":   source_disagree_factor,
        "new_screen_pattern":    new_pattern_factor,
        "stuck_indicator":       stuck_factor,
        "temporal_instability":  temporal_factor,
        "transport_uncertainty": transport_factor,
    }


# ---------------------------------------------------------------------------
# Helpers (module-level, exposed for testing)
# ---------------------------------------------------------------------------


def _compute_weighted_score(
    factors: dict[str, float],
    weights: dict[str, float],
) -> float:
    """
    Return the weighted sum of *factors* using *weights*.

    Result is clamped to [0.0, 1.0] to guard against floating-point drift.
    """
    total = sum(factors[k] * weights[k] for k in weights if k in factors)
    return max(0.0, min(1.0, total))


def _dominant_factor(
    factors: dict[str, float],
    weights: dict[str, float],
) -> str:
    """Return the name of the factor with the largest weighted contribution."""
    contributions = {k: factors.get(k, 0.0) * weights.get(k, 0.0) for k in weights}
    return max(contributions, key=contributions.__getitem__)


def _recommend(score: float) -> str:
    """Map a score to a recommendation string."""
    if score >= _THRESHOLD_SUSPEND:
        return "suspend"
    if score >= _THRESHOLD_CLOUD:
        return "cloud"
    return "local"


def _is_stuck(history: Sequence[Any]) -> bool:
    """
    Return True when the last three history entries are all identical
    in (action_type, target_description).

    Returns False when fewer than three entries are present.
    """
    if len(history) < 3:
        return False
    last_three = list(history)[-3:]
    fingerprints = {
        (getattr(rec, "action_type", ""), getattr(rec, "target_description", ""))
        for rec in last_three
    }
    return len(fingerprints) == 1
