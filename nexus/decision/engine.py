"""
nexus/decision/engine.py
Decision Engine — routes each planning cycle to local, cloud, or human.

Pipeline (DecisionEngine.decide)
---------------------------------
  0. Hard-stuck check   — 5 actions with same target → hitl immediately.
  1. Anti-loop check    — last 3 identical (type, target) → force cloud.
  2. PolicyEngine.check_action() → block/abort → suspend; warn → proceed.
  3. AmbiguityScorer.score() → recommendation.
     If anti-loop flag is set: override recommendation to "cloud".
  4. recommendation == "local":
       LocalResolver.resolve() → Decision or None.
       If None: fall through to "cloud".
  5. recommendation == "cloud":
       CloudPlanner.plan() → PlannerDecision or CloudError → suspend.
  6. recommendation == "suspend" / hitl path: return hitl Decision.

Data classes
------------
  TargetSpec     — where/how to interact with the target element.
  Decision       — full output of one decide() call.
  DecisionContext — caller-supplied task metadata.

transport_hint mapping (source_result.source_type)
---------------------------------------------------
  "uia"    → "uia"
  "dom"    → "dom"
  "visual" → "mouse"
  "file"   → None

LocalResolver confidence threshold
------------------------------------
  semantic.confidence < _LOCAL_CONFIDENCE_THRESHOLD → return None
  (falls through to cloud path)
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from nexus.cloud.planner import CloudPlanner, PlannerDecision
from nexus.cloud.prompt_builder import ActionRecord
from nexus.core.errors import CloudError
from nexus.core.policy import ActionContext, PolicyEngine
from nexus.core.types import Rect
from nexus.decision.ambiguity_scorer import AmbiguityScorer
from nexus.infra.logger import get_logger
from nexus.perception.orchestrator import PerceptionResult

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOCAL_CONFIDENCE_THRESHOLD: float = 0.50
_ANTI_LOOP_WINDOW: int = 3      # N identical (type, target) → cloud
_HARD_STUCK_WINDOW: int = 5     # N actions same target → hitl

# transport_hint from source_type
_TRANSPORT_MAP: dict[str, str | None] = {
    "uia":    "uia",
    "dom":    "dom",
    "visual": "mouse",
    "file":   None,
}

# ---------------------------------------------------------------------------
# TargetSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSpec:
    """
    Describes *where* and *how* to interact with a target UI element.

    Attributes
    ----------
    element_id:
        Stable identifier from the perception layer, or ``None``.
    coordinates:
        (x, y) screen coordinates (centre of the bounding box), or ``None``.
    description:
        Human-readable description of the element.
    preferred_transport:
        Recommended execution channel: ``"uia"``, ``"dom"``, ``"mouse"``,
        or ``None`` when no preference can be determined.
    """

    element_id: str | None
    coordinates: tuple[int, int] | None
    description: str
    preferred_transport: Literal["uia", "dom", "mouse"] | None


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """
    Output of one :meth:`DecisionEngine.decide` call.

    Attributes
    ----------
    source:
        Which path produced this decision:
        ``"local"`` | ``"cloud"`` | ``"hitl"`` | ``"suspend"``.
    action_type:
        Verb for the next action (e.g. ``"click"``, ``"type"``).
    target:
        Target specification (element id, coordinates, description, transport).
    value:
        Text/key value for type/press_key actions, or ``None``.
    confidence:
        Decision confidence in [0.0, 1.0].
    reasoning:
        Explanation of why this action was chosen.
    cost_incurred:
        LLM cost (USD) incurred during this decide() call.
        Zero for local decisions.
    transport_hint:
        Recommended transport channel, derived from source_result.source_type.
        ``"uia"`` | ``"dom"`` | ``"mouse"`` | ``None``.
    """

    source: Literal["local", "cloud", "hitl", "suspend"]
    action_type: str
    target: TargetSpec
    value: str | None
    confidence: float
    reasoning: str
    cost_incurred: float
    transport_hint: str | None


# ---------------------------------------------------------------------------
# DecisionContext
# ---------------------------------------------------------------------------


@dataclass
class DecisionContext:
    """
    Caller-supplied snapshot of task state used by the decision pipeline.

    Attributes
    ----------
    task_id:
        Identifier for cost-tracking and policy evaluation.
    actions_so_far:
        Number of actions already executed (used by PolicyEngine).
    elapsed_seconds:
        Seconds since the task started (used by PolicyEngine).
    task_cost_usd:
        Accumulated LLM cost for this task (used by PolicyEngine).
    daily_cost_usd:
        Accumulated LLM cost today (used by PolicyEngine).
    candidate_is_destructive:
        True when the candidate action is potentially destructive.
    screen_previously_seen:
        False when the current screen fingerprint is new.
    used_fallback_transport:
        True when the most recent transport used a fallback channel.
    screenshot:
        Optional raw HxWx3 uint8 RGB array forwarded to CloudPlanner.
    sensitive_regions:
        Screen rects to mask in the screenshot before sending to cloud.
    """

    task_id: str
    actions_so_far: int = 0
    elapsed_seconds: float = 0.0
    task_cost_usd: float = 0.0
    daily_cost_usd: float = 0.0
    candidate_is_destructive: bool = False
    screen_previously_seen: bool = True
    used_fallback_transport: bool = False
    screenshot: np.ndarray | None = None
    sensitive_regions: list[Rect] | None = field(default=None)


# ---------------------------------------------------------------------------
# LocalResolver
# ---------------------------------------------------------------------------


class LocalResolver:
    """
    Attempts to resolve the next action using only local perception data.

    Uses :meth:`~nexus.perception.spatial_graph.SpatialGraph.find_best_target`
    to locate the target element.  Returns ``None`` when confidence is
    insufficient (the caller should fall through to cloud).
    """

    def resolve(
        self,
        goal: str,
        perception: PerceptionResult,
    ) -> Decision | None:
        """
        Try to build a :class:`Decision` from local perception.

        Parameters
        ----------
        goal:
            The agent's current objective (used as ``find_best_target`` query).
        perception:
            Current perception result.

        Returns
        -------
        Decision or None
            ``None`` when no suitable target is found locally.
        """
        node = perception.spatial_graph.find_best_target(goal)
        if node is None:
            _log.debug("local_resolver_no_target", goal=goal)
            return None

        if node.semantic.confidence < _LOCAL_CONFIDENCE_THRESHOLD:
            _log.debug(
                "local_resolver_low_confidence",
                confidence=node.semantic.confidence,
                threshold=_LOCAL_CONFIDENCE_THRESHOLD,
            )
            return None

        # Derive coordinates from bounding box centre
        bb = node.element.bounding_box
        cx = bb.x + bb.width // 2
        cy = bb.y + bb.height // 2

        # Derive transport from source_type
        source_type = perception.source_result.source_type
        transport = _transport_from_source(source_type)

        target = TargetSpec(
            element_id=str(node.id),
            coordinates=(cx, cy),
            description=node.semantic.primary_label or node.text or goal,
            preferred_transport=transport,
        )

        _log.debug(
            "local_resolver_resolved",
            element_id=target.element_id,
            confidence=node.semantic.confidence,
        )

        return Decision(
            source="local",
            action_type=_action_from_affordance(node.semantic.affordance.name),
            target=target,
            value=None,
            confidence=node.semantic.confidence,
            reasoning=(
                f"Local resolution via spatial graph. "
                f"Best match: '{target.description}' "
                f"(confidence {node.semantic.confidence:.2f})."
            ),
            cost_incurred=0.0,
            transport_hint=transport,
        )


# ---------------------------------------------------------------------------
# DecisionEngine
# ---------------------------------------------------------------------------


class DecisionEngine:
    """
    Orchestrates the full decision pipeline for one agent step.

    Parameters
    ----------
    policy:
        PolicyEngine for safety checks.
    scorer:
        AmbiguityScorer for routing decisions.
    resolver:
        LocalResolver for low-ambiguity cases.
    planner:
        CloudPlanner for high-ambiguity cases.
    cost_before_fn:
        Callable that returns the current task cost before the plan call.
        Signature: ``(task_id: str) -> float``.
    """

    def __init__(
        self,
        policy: PolicyEngine,
        scorer: AmbiguityScorer,
        resolver: LocalResolver,
        planner: CloudPlanner,
        cost_before_fn: Any = None,
    ) -> None:
        self._policy = policy
        self._scorer = scorer
        self._resolver = resolver
        self._planner = planner
        self._cost_before_fn = cost_before_fn or (lambda _: 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decide(
        self,
        goal: str,
        perception: PerceptionResult,
        action_history: Sequence[ActionRecord],
        context: DecisionContext,
    ) -> Decision:
        """
        Run one decision cycle and return a :class:`Decision`.

        Parameters
        ----------
        goal:
            The agent's current high-level objective.
        perception:
            Current perception result.
        action_history:
            All completed action records for this task.
        context:
            Task-level metadata for policy and cost tracking.

        Returns
        -------
        Decision
        """
        source_type = perception.source_result.source_type
        transport_hint = _transport_from_source(source_type)

        # ------------------------------------------------------------------
        # Step 0 — Hard-stuck detection (5 same target → hitl)
        # ------------------------------------------------------------------
        if _is_hard_stuck(action_history):
            _log.warning("decision_hard_stuck", actions=len(action_history))
            return _make_suspend_decision(
                source="hitl",
                reasoning=(
                    "Agent is hard-stuck: the last "
                    f"{_HARD_STUCK_WINDOW} actions all targeted the same element. "
                    "Human intervention required."
                ),
                transport_hint=transport_hint,
            )

        # ------------------------------------------------------------------
        # Step 1 — Anti-loop detection (3 identical type+target → force cloud)
        # ------------------------------------------------------------------
        force_cloud = _is_anti_loop(action_history)
        if force_cloud:
            _log.warning("decision_anti_loop", actions=len(action_history))

        # ------------------------------------------------------------------
        # Step 2 — Policy check
        # ------------------------------------------------------------------
        policy_ctx = ActionContext(
            action_type="plan",
            transport=str(source_type),
            is_destructive=context.candidate_is_destructive,
            target_rect=None,
            actions_so_far=context.actions_so_far,
            elapsed_seconds=context.elapsed_seconds,
            task_cost_usd=context.task_cost_usd,
            daily_cost_usd=context.daily_cost_usd,
        )
        policy_result = self._policy.check_action(policy_ctx)
        if policy_result.verdict in ("block", "abort"):
            _log.warning(
                "decision_policy_blocked",
                verdict=policy_result.verdict,
                rule=policy_result.rule,
            )
            return _make_suspend_decision(
                source="suspend",
                reasoning=(
                    f"PolicyEngine blocked the action: {policy_result.message}"
                ),
                transport_hint=transport_hint,
            )

        # ------------------------------------------------------------------
        # Step 3 — Ambiguity scoring
        # ------------------------------------------------------------------
        amb = self._scorer.score(
            perception,
            action_history,
            candidate_is_destructive=context.candidate_is_destructive,
            screen_previously_seen=context.screen_previously_seen,
            used_fallback_transport=context.used_fallback_transport,
        )

        recommendation = "cloud" if force_cloud else amb.recommendation

        _log.debug(
            "decision_routing",
            recommendation=recommendation,
            score=round(amb.score, 4),
            dominant=amb.dominant_factor,
        )

        # ------------------------------------------------------------------
        # Step 4 — Local path
        # ------------------------------------------------------------------
        if recommendation == "local":
            local = self._resolver.resolve(goal, perception)
            if local is not None:
                return local
            # LocalResolver failed — escalate to cloud
            _log.debug("decision_local_failed_escalating_cloud")
            recommendation = "cloud"

        # ------------------------------------------------------------------
        # Step 5 — Cloud path
        # ------------------------------------------------------------------
        if recommendation == "cloud":
            cost_before = self._cost_before_fn(context.task_id)
            try:
                plan: PlannerDecision = await self._planner.plan(
                    goal,
                    perception,
                    list(action_history),
                    screenshot=context.screenshot,
                    sensitive_regions=context.sensitive_regions,
                )
            except CloudError as exc:
                _log.error("decision_cloud_failed", error=str(exc))
                return _make_suspend_decision(
                    source="suspend",
                    reasoning=f"CloudPlanner failed: {exc}",
                    transport_hint=transport_hint,
                )

            cost_after = self._cost_before_fn(context.task_id)
            cost_incurred = max(0.0, cost_after - cost_before)

            target = _target_from_plan(plan, transport_hint)
            return Decision(
                source="cloud",
                action_type=plan.action_type,
                target=target,
                value=plan.value,
                confidence=plan.confidence,
                reasoning=plan.reasoning,
                cost_incurred=cost_incurred,
                transport_hint=transport_hint,
            )

        # ------------------------------------------------------------------
        # Step 6 — Suspend / HITL
        # ------------------------------------------------------------------
        return _make_suspend_decision(
            source="hitl",
            reasoning=(
                f"Ambiguity score {amb.score:.3f} exceeds suspend threshold. "
                f"Dominant factor: {amb.dominant_factor}."
            ),
            transport_hint=transport_hint,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _transport_from_source(source_type: str) -> Literal["uia", "dom", "mouse"] | None:
    """Map source_result.source_type to a transport_hint string."""
    result = _TRANSPORT_MAP.get(str(source_type))
    return result  # type: ignore[return-value]


def _action_from_affordance(affordance_name: str) -> str:
    """Derive a default action_type from an affordance name."""
    mapping = {
        "CLICKABLE":   "click",
        "TYPEABLE":    "type",
        "SCROLLABLE":  "scroll",
        "SELECTABLE":  "click",
    }
    return mapping.get(affordance_name, "click")


def _target_from_plan(
    plan: PlannerDecision,
    transport_hint: str | None,
) -> TargetSpec:
    """Build a TargetSpec from a PlannerDecision."""
    return TargetSpec(
        element_id=plan.target_element_id,
        coordinates=None,
        description=plan.target_description,
        preferred_transport=transport_hint,  # type: ignore[arg-type]
    )


def _make_suspend_decision(
    *,
    source: Literal["hitl", "suspend"],
    reasoning: str,
    transport_hint: str | None,
) -> Decision:
    """Build a no-op Decision that requests human or suspends the task."""
    return Decision(
        source=source,
        action_type="none",
        target=TargetSpec(
            element_id=None,
            coordinates=None,
            description="",
            preferred_transport=None,
        ),
        value=None,
        confidence=0.0,
        reasoning=reasoning,
        cost_incurred=0.0,
        transport_hint=transport_hint,
    )


def _is_anti_loop(history: Sequence[Any]) -> bool:
    """
    Return True when the last ``_ANTI_LOOP_WINDOW`` entries are all identical
    in (action_type, target_description).
    """
    if len(history) < _ANTI_LOOP_WINDOW:
        return False
    last = list(history)[-_ANTI_LOOP_WINDOW:]
    fingerprints = {
        (
            getattr(r, "action_type", ""),
            getattr(r, "target_description", ""),
        )
        for r in last
    }
    return len(fingerprints) == 1


def _is_hard_stuck(history: Sequence[Any]) -> bool:
    """
    Return True when the last ``_HARD_STUCK_WINDOW`` entries all target the
    same element (regardless of action_type).
    """
    if len(history) < _HARD_STUCK_WINDOW:
        return False
    last = list(history)[-_HARD_STUCK_WINDOW:]
    targets = {getattr(r, "target_description", "") for r in last}
    return len(targets) == 1
