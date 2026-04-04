"""
tests/unit/test_decision_engine.py
Unit tests for nexus/decision/engine.py.

Coverage
--------
  LocalResolver        — no graph, low confidence, success paths
  DecisionEngine       — local / cloud / hitl / suspend routing
  Anti-loop detection  — 3 identical (type, target) → force cloud
  Hard-stuck detection — 5 same target → hitl
  Policy blocking      — block/abort verdict → suspend Decision
  transport_hint       — correct mapping from source_type
  Cloud fallback       — local miss → cloud
  Cloud failure        — CloudError → suspend
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from nexus.cloud.planner import PlannerDecision
from nexus.core.errors import CloudError
from nexus.core.policy import PolicyResult
from nexus.decision.ambiguity_scorer import AmbiguityScore
from nexus.decision.engine import (
    Decision,
    DecisionContext,
    DecisionEngine,
    LocalResolver,
    TargetSpec,
    _ANTI_LOOP_WINDOW,
    _HARD_STUCK_WINDOW,
    _is_anti_loop,
    _is_hard_stuck,
    _transport_from_source,
)
from nexus.infra.cost_tracker import AlertResult


# ---------------------------------------------------------------------------
# Shared helpers / factories
# ---------------------------------------------------------------------------


def _alert(level: str = "none") -> AlertResult:
    return AlertResult(
        level=level,  # type: ignore[arg-type]
        task_pct=0.0,
        daily_pct=0.0,
        task_cost_usd=0.0,
        daily_cost_usd=0.0,
        message="",
    )


def _make_action_record(
    action_type: str = "click",
    target_description: str = "Submit",
) -> MagicMock:
    rec = MagicMock()
    rec.action_type = action_type
    rec.target_description = target_description
    return rec


def _make_semantic(
    primary_label: str = "Submit",
    confidence: float = 0.90,
    affordance_name: str = "CLICKABLE",
    is_destructive: bool = False,
) -> MagicMock:
    sem = MagicMock()
    sem.primary_label = primary_label
    sem.confidence = confidence
    aff = MagicMock()
    aff.name = affordance_name
    sem.affordance = aff
    sem.is_destructive = is_destructive
    sem.secondary_labels = []
    return sem


def _make_node(
    element_id: str = "el-1",
    x: int = 100,
    y: int = 200,
    width: int = 80,
    height: int = 30,
    primary_label: str = "Submit",
    confidence: float = 0.90,
    affordance_name: str = "CLICKABLE",
) -> MagicMock:
    node = MagicMock()
    node.id = element_id
    node.text = primary_label
    node.semantic = _make_semantic(primary_label, confidence, affordance_name)

    bbox = MagicMock()
    bbox.x = x
    bbox.y = y
    bbox.width = width
    bbox.height = height
    node.element = MagicMock()
    node.element.bounding_box = bbox
    node.element.id = element_id
    return node


def _make_spatial_graph(node: MagicMock | None = None) -> MagicMock:
    graph = MagicMock()
    graph.find_best_target.return_value = node
    return graph


def _make_perception(
    source_type: str = "visual",
    best_node: MagicMock | None = None,
    overall_confidence: float = 0.80,
    conflicts_detected: int = 0,
    state_name: str = "STABLE",
) -> MagicMock:
    perception = MagicMock()

    # source_result
    src = MagicMock()
    src.source_type = source_type
    perception.source_result = src

    # spatial_graph
    perception.spatial_graph = _make_spatial_graph(best_node)

    # screen_state
    ss = MagicMock()
    st = MagicMock()
    st.name = state_name
    ss.state_type = st
    ss.confidence = overall_confidence
    perception.screen_state = ss

    # arbitration
    arb = MagicMock()
    arb.overall_confidence = overall_confidence
    arb.conflicts_detected = conflicts_detected
    perception.arbitration = arb

    return perception


def _make_planner_decision(
    action_type: str = "click",
    target_description: str = "Submit",
    confidence: float = 0.85,
    task_status: str = "in_progress",
    target_element_id: str | None = "el-1",
    value: str | None = None,
    reasoning: str = "LLM chose this.",
    tokens_input: int = 100,
    tokens_output: int = 50,
) -> PlannerDecision:
    return PlannerDecision(
        action_type=action_type,
        target_description=target_description,
        target_element_id=target_element_id,
        value=value,
        reasoning=reasoning,
        confidence=confidence,
        task_status=task_status,
        raw_response='{"action_type":"click"}',
        tokens_used=tokens_input + tokens_output,
        alert=_alert(),
    )


def _make_context(
    task_id: str = "task-1",
    actions_so_far: int = 0,
    elapsed_seconds: float = 0.0,
    task_cost_usd: float = 0.0,
    daily_cost_usd: float = 0.0,
    candidate_is_destructive: bool = False,
    screen_previously_seen: bool = True,
    used_fallback_transport: bool = False,
) -> DecisionContext:
    return DecisionContext(
        task_id=task_id,
        actions_so_far=actions_so_far,
        elapsed_seconds=elapsed_seconds,
        task_cost_usd=task_cost_usd,
        daily_cost_usd=daily_cost_usd,
        candidate_is_destructive=candidate_is_destructive,
        screen_previously_seen=screen_previously_seen,
        used_fallback_transport=used_fallback_transport,
    )


def _make_engine(
    policy_verdict: str = "allow",
    policy_rule: str | None = None,
    ambiguity_score: float = 0.20,
    ambiguity_recommendation: str = "local",
    planner_decision: PlannerDecision | None = None,
    planner_side_effect: Exception | None = None,
    cost_before_value: float = 0.0,
) -> tuple[DecisionEngine, MagicMock, MagicMock, MagicMock, MagicMock]:
    """
    Build a DecisionEngine with mocked dependencies.

    Returns (engine, mock_policy, mock_scorer, mock_resolver, mock_planner).
    """
    policy = MagicMock()
    policy.check_action.return_value = PolicyResult(
        verdict=policy_verdict,  # type: ignore[arg-type]
        rule=policy_rule,
        severity=policy_verdict,
        message=f"Policy: {policy_verdict}",
    )

    amb_score = AmbiguityScore(
        score=ambiguity_score,
        recommendation=ambiguity_recommendation,
        factors={},
        dominant_factor="overall_confidence",
    )
    scorer = MagicMock()
    scorer.score.return_value = amb_score

    resolver = MagicMock(spec=LocalResolver)
    resolver.resolve.return_value = None  # default: fail

    planner = MagicMock()
    if planner_side_effect is not None:
        planner.plan = AsyncMock(side_effect=planner_side_effect)
    elif planner_decision is not None:
        planner.plan = AsyncMock(return_value=planner_decision)
    else:
        planner.plan = AsyncMock(return_value=_make_planner_decision())

    cost_fn = MagicMock(return_value=cost_before_value)

    engine = DecisionEngine(
        policy=policy,
        scorer=scorer,
        resolver=resolver,
        planner=planner,
        cost_before_fn=cost_fn,
    )
    return engine, policy, scorer, resolver, planner


# ===========================================================================
# Section 1 — _transport_from_source
# ===========================================================================


class TestTransportFromSource:
    def test_uia_maps_to_uia(self) -> None:
        assert _transport_from_source("uia") == "uia"

    def test_dom_maps_to_dom(self) -> None:
        assert _transport_from_source("dom") == "dom"

    def test_visual_maps_to_mouse(self) -> None:
        assert _transport_from_source("visual") == "mouse"

    def test_file_maps_to_none(self) -> None:
        assert _transport_from_source("file") is None

    def test_unknown_maps_to_none(self) -> None:
        assert _transport_from_source("something_else") is None


# ===========================================================================
# Section 2 — _is_anti_loop
# ===========================================================================


class TestIsAntiLoop:
    def test_fewer_than_window_returns_false(self) -> None:
        history = [_make_action_record() for _ in range(_ANTI_LOOP_WINDOW - 1)]
        assert _is_anti_loop(history) is False

    def test_all_same_returns_true(self) -> None:
        history = [
            _make_action_record("click", "Submit")
            for _ in range(_ANTI_LOOP_WINDOW)
        ]
        assert _is_anti_loop(history) is True

    def test_mixed_types_returns_false(self) -> None:
        history = [
            _make_action_record("click", "Submit"),
            _make_action_record("type", "Submit"),
            _make_action_record("click", "Submit"),
        ]
        assert _is_anti_loop(history) is False

    def test_mixed_targets_returns_false(self) -> None:
        history = [
            _make_action_record("click", "Submit"),
            _make_action_record("click", "Cancel"),
            _make_action_record("click", "Submit"),
        ]
        assert _is_anti_loop(history) is False

    def test_only_last_window_checked(self) -> None:
        # First entry is different but outside the window
        history = [_make_action_record("scroll", "Page")]
        history += [
            _make_action_record("click", "Submit")
            for _ in range(_ANTI_LOOP_WINDOW)
        ]
        assert _is_anti_loop(history) is True

    def test_empty_history_returns_false(self) -> None:
        assert _is_anti_loop([]) is False


# ===========================================================================
# Section 3 — _is_hard_stuck
# ===========================================================================


class TestIsHardStuck:
    def test_fewer_than_window_returns_false(self) -> None:
        history = [_make_action_record() for _ in range(_HARD_STUCK_WINDOW - 1)]
        assert _is_hard_stuck(history) is False

    def test_all_same_target_returns_true(self) -> None:
        history = [
            _make_action_record(f"action_{i}", "Submit")
            for i in range(_HARD_STUCK_WINDOW)
        ]
        assert _is_hard_stuck(history) is True

    def test_different_targets_returns_false(self) -> None:
        history = [
            _make_action_record("click", f"target_{i}")
            for i in range(_HARD_STUCK_WINDOW)
        ]
        assert _is_hard_stuck(history) is False

    def test_same_type_different_target_returns_false(self) -> None:
        history = [_make_action_record("click", "A")] * 3
        history += [_make_action_record("click", "B")] * (_HARD_STUCK_WINDOW - 3)
        assert _is_hard_stuck(history) is False

    def test_empty_returns_false(self) -> None:
        assert _is_hard_stuck([]) is False


# ===========================================================================
# Section 4 — LocalResolver
# ===========================================================================


class TestLocalResolver:
    def test_no_graph_target_returns_none(self) -> None:
        resolver = LocalResolver()
        perception = _make_perception(best_node=None)
        result = resolver.resolve("click Submit", perception)
        assert result is None

    def test_low_confidence_returns_none(self) -> None:
        resolver = LocalResolver()
        node = _make_node(confidence=0.30)
        perception = _make_perception(best_node=node)
        result = resolver.resolve("click Submit", perception)
        assert result is None

    def test_threshold_boundary_just_below_returns_none(self) -> None:
        resolver = LocalResolver()
        node = _make_node(confidence=0.499)
        perception = _make_perception(best_node=node)
        assert resolver.resolve("Submit", perception) is None

    def test_threshold_boundary_at_threshold_returns_decision(self) -> None:
        resolver = LocalResolver()
        node = _make_node(confidence=0.50)
        perception = _make_perception(best_node=node)
        result = resolver.resolve("Submit", perception)
        assert result is not None
        assert result.source == "local"

    def test_success_returns_local_decision(self) -> None:
        resolver = LocalResolver()
        node = _make_node(
            element_id="btn-1",
            x=100, y=200, width=80, height=30,
            primary_label="Submit",
            confidence=0.90,
        )
        perception = _make_perception(best_node=node, source_type="uia")
        result = resolver.resolve("Submit", perception)
        assert result is not None
        assert result.source == "local"
        assert result.action_type == "click"
        assert result.target.element_id == "btn-1"
        assert result.target.coordinates == (140, 215)  # 100+40, 200+15
        assert result.target.preferred_transport == "uia"
        assert result.confidence == 0.90
        assert result.cost_incurred == 0.0

    def test_coordinates_calculated_from_bbox_centre(self) -> None:
        resolver = LocalResolver()
        node = _make_node(x=50, y=60, width=100, height=40, confidence=0.80)
        perception = _make_perception(best_node=node)
        result = resolver.resolve("something", perception)
        assert result is not None
        assert result.target.coordinates == (100, 80)  # 50+50, 60+20

    def test_transport_hint_visual_gives_mouse(self) -> None:
        resolver = LocalResolver()
        node = _make_node(confidence=0.80)
        perception = _make_perception(best_node=node, source_type="visual")
        result = resolver.resolve("btn", perception)
        assert result is not None
        assert result.transport_hint == "mouse"
        assert result.target.preferred_transport == "mouse"

    def test_transport_hint_dom_gives_dom(self) -> None:
        resolver = LocalResolver()
        node = _make_node(confidence=0.80)
        perception = _make_perception(best_node=node, source_type="dom")
        result = resolver.resolve("btn", perception)
        assert result is not None
        assert result.transport_hint == "dom"

    def test_typeable_affordance_gives_type_action(self) -> None:
        resolver = LocalResolver()
        node = _make_node(confidence=0.80, affordance_name="TYPEABLE")
        perception = _make_perception(best_node=node)
        result = resolver.resolve("input", perception)
        assert result is not None
        assert result.action_type == "type"

    def test_scrollable_affordance_gives_scroll_action(self) -> None:
        resolver = LocalResolver()
        node = _make_node(confidence=0.80, affordance_name="SCROLLABLE")
        perception = _make_perception(best_node=node)
        result = resolver.resolve("list", perception)
        assert result is not None
        assert result.action_type == "scroll"

    def test_unknown_affordance_defaults_to_click(self) -> None:
        resolver = LocalResolver()
        node = _make_node(confidence=0.80, affordance_name="UNKNOWN")
        perception = _make_perception(best_node=node)
        result = resolver.resolve("thing", perception)
        assert result is not None
        assert result.action_type == "click"


# ===========================================================================
# Section 5 — DecisionEngine: local path
# ===========================================================================


class TestDecisionEngineLocalPath:
    @pytest.mark.asyncio
    async def test_local_path_returns_local_decision(self) -> None:
        local_decision = Decision(
            source="local",
            action_type="click",
            target=TargetSpec(
                element_id="btn-1",
                coordinates=(100, 200),
                description="Submit",
                preferred_transport="uia",
            ),
            value=None,
            confidence=0.90,
            reasoning="local",
            cost_incurred=0.0,
            transport_hint="uia",
        )
        engine, policy, scorer, resolver, planner = _make_engine(
            ambiguity_recommendation="local",
        )
        resolver.resolve.return_value = local_decision
        perception = _make_perception(source_type="uia")

        result = await engine.decide("Submit", perception, [], _make_context())

        assert result.source == "local"
        assert result.action_type == "click"
        assert result.cost_incurred == 0.0
        planner.plan.assert_not_called()

    @pytest.mark.asyncio
    async def test_local_fail_escalates_to_cloud(self) -> None:
        engine, _, _, resolver, planner = _make_engine(
            ambiguity_recommendation="local",
            planner_decision=_make_planner_decision(),
        )
        resolver.resolve.return_value = None  # local fails
        perception = _make_perception(source_type="visual")

        result = await engine.decide("Submit", perception, [], _make_context())

        assert result.source == "cloud"
        planner.plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_local_success_does_not_call_planner(self) -> None:
        local_decision = Decision(
            source="local",
            action_type="click",
            target=TargetSpec(None, None, "X", None),
            value=None,
            confidence=0.9,
            reasoning="",
            cost_incurred=0.0,
            transport_hint=None,
        )
        engine, _, _, resolver, planner = _make_engine(
            ambiguity_recommendation="local",
        )
        resolver.resolve.return_value = local_decision
        perception = _make_perception()

        await engine.decide("X", perception, [], _make_context())
        planner.plan.assert_not_called()


# ===========================================================================
# Section 6 — DecisionEngine: cloud path
# ===========================================================================


class TestDecisionEngineCloudPath:
    @pytest.mark.asyncio
    async def test_cloud_path_returns_cloud_decision(self) -> None:
        plan = _make_planner_decision(
            action_type="click",
            target_description="OK",
            confidence=0.85,
        )
        engine, _, _, _, _ = _make_engine(
            ambiguity_recommendation="cloud",
            planner_decision=plan,
        )
        perception = _make_perception(source_type="dom")

        result = await engine.decide("OK", perception, [], _make_context())

        assert result.source == "cloud"
        assert result.action_type == "click"
        assert result.confidence == 0.85
        assert result.transport_hint == "dom"

    @pytest.mark.asyncio
    async def test_cloud_failure_returns_suspend(self) -> None:
        engine, _, _, _, _ = _make_engine(
            ambiguity_recommendation="cloud",
            planner_side_effect=CloudError("LLM unavailable"),
        )
        perception = _make_perception()

        result = await engine.decide("goal", perception, [], _make_context())

        assert result.source == "suspend"
        assert result.action_type == "none"

    @pytest.mark.asyncio
    async def test_cloud_decision_sets_target_spec(self) -> None:
        plan = _make_planner_decision(
            target_description="Login button",
            target_element_id="btn-login",
        )
        engine, _, _, _, _ = _make_engine(
            ambiguity_recommendation="cloud",
            planner_decision=plan,
        )
        perception = _make_perception(source_type="uia")

        result = await engine.decide("Login", perception, [], _make_context())

        assert result.target.element_id == "btn-login"
        assert result.target.description == "Login button"
        assert result.target.preferred_transport == "uia"

    @pytest.mark.asyncio
    async def test_cloud_cost_incurred_captured(self) -> None:
        plan = _make_planner_decision()
        engine, _, _, _, _ = _make_engine(
            ambiguity_recommendation="cloud",
            planner_decision=plan,
            cost_before_value=0.05,
        )
        # First call returns 0.05 (before), second returns 0.08 (after)
        call_count = 0

        def cost_fn(task_id: str) -> float:
            nonlocal call_count
            call_count += 1
            return 0.05 if call_count == 1 else 0.08

        engine._cost_before_fn = cost_fn
        perception = _make_perception()

        result = await engine.decide("goal", perception, [], _make_context())
        assert result.cost_incurred == pytest.approx(0.03, abs=1e-6)

    @pytest.mark.asyncio
    async def test_cloud_resolver_not_called(self) -> None:
        plan = _make_planner_decision()
        engine, _, _, resolver, _ = _make_engine(
            ambiguity_recommendation="cloud",
            planner_decision=plan,
        )
        perception = _make_perception()

        await engine.decide("goal", perception, [], _make_context())
        resolver.resolve.assert_not_called()


# ===========================================================================
# Section 7 — DecisionEngine: suspend / hitl path
# ===========================================================================


class TestDecisionEngineSuspendPath:
    @pytest.mark.asyncio
    async def test_suspend_recommendation_returns_hitl(self) -> None:
        engine, _, _, _, planner = _make_engine(
            ambiguity_recommendation="suspend",
        )
        perception = _make_perception()

        result = await engine.decide("goal", perception, [], _make_context())

        assert result.source == "hitl"
        assert result.action_type == "none"
        planner.plan.assert_not_called()

    @pytest.mark.asyncio
    async def test_policy_block_returns_suspend(self) -> None:
        engine, _, _, _, planner = _make_engine(
            policy_verdict="block",
            policy_rule="RULE_MAX_ACTIONS",
        )
        perception = _make_perception()

        result = await engine.decide("goal", perception, [], _make_context())

        assert result.source == "suspend"
        assert result.action_type == "none"
        planner.plan.assert_not_called()

    @pytest.mark.asyncio
    async def test_policy_abort_returns_suspend(self) -> None:
        engine, _, _, _, planner = _make_engine(
            policy_verdict="abort",
        )
        perception = _make_perception()

        result = await engine.decide("goal", perception, [], _make_context())

        assert result.source == "suspend"
        planner.plan.assert_not_called()

    @pytest.mark.asyncio
    async def test_policy_warn_proceeds_normally(self) -> None:
        local_decision = Decision(
            source="local",
            action_type="click",
            target=TargetSpec(None, None, "X", None),
            value=None,
            confidence=0.9,
            reasoning="",
            cost_incurred=0.0,
            transport_hint=None,
        )
        engine, _, _, resolver, _ = _make_engine(
            policy_verdict="warn",
            ambiguity_recommendation="local",
        )
        resolver.resolve.return_value = local_decision
        perception = _make_perception()

        result = await engine.decide("X", perception, [], _make_context())

        assert result.source == "local"


# ===========================================================================
# Section 8 — Anti-loop: force cloud
# ===========================================================================


class TestAntiLoop:
    @pytest.mark.asyncio
    async def test_anti_loop_forces_cloud(self) -> None:
        """3 identical (type, target) → cloud even if ambiguity says local."""
        # Build history with 3 identical records
        history = [
            _make_action_record("click", "Submit")
            for _ in range(_ANTI_LOOP_WINDOW)
        ]
        plan = _make_planner_decision()
        engine, _, scorer, resolver, planner = _make_engine(
            ambiguity_recommendation="local",  # would normally go local
            planner_decision=plan,
        )
        perception = _make_perception()

        result = await engine.decide("Submit", perception, history, _make_context())

        assert result.source == "cloud"
        planner.plan.assert_called_once()
        # Resolver should not have been called since we forced cloud
        resolver.resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_anti_loop_with_2_identical(self) -> None:
        """2 identical records — not enough to trigger anti-loop."""
        history = [
            _make_action_record("click", "Submit")
            for _ in range(_ANTI_LOOP_WINDOW - 1)
        ]
        local_decision = Decision(
            source="local",
            action_type="click",
            target=TargetSpec(None, None, "Submit", None),
            value=None,
            confidence=0.9,
            reasoning="",
            cost_incurred=0.0,
            transport_hint=None,
        )
        engine, _, _, resolver, _ = _make_engine(
            ambiguity_recommendation="local",
        )
        resolver.resolve.return_value = local_decision
        perception = _make_perception()

        result = await engine.decide("Submit", perception, history, _make_context())
        assert result.source == "local"

    @pytest.mark.asyncio
    async def test_anti_loop_only_checks_last_window(self) -> None:
        """Anti-loop uses only the last _ANTI_LOOP_WINDOW entries."""
        old_recs = [_make_action_record("scroll", "List") for _ in range(10)]
        new_recs = [
            _make_action_record("click", "OK")
            for _ in range(_ANTI_LOOP_WINDOW)
        ]
        history = old_recs + new_recs
        plan = _make_planner_decision()
        engine, _, _, _, planner = _make_engine(
            ambiguity_recommendation="local",
            planner_decision=plan,
        )
        perception = _make_perception()

        result = await engine.decide("OK", perception, history, _make_context())
        assert result.source == "cloud"


# ===========================================================================
# Section 9 — Hard-stuck: force hitl
# ===========================================================================


class TestHardStuck:
    @pytest.mark.asyncio
    async def test_hard_stuck_returns_hitl(self) -> None:
        history = [
            _make_action_record(f"action_{i}", "Submit")
            for i in range(_HARD_STUCK_WINDOW)
        ]
        engine, _, _, _, planner = _make_engine()
        perception = _make_perception()

        result = await engine.decide("goal", perception, history, _make_context())

        assert result.source == "hitl"
        assert result.action_type == "none"
        planner.plan.assert_not_called()

    @pytest.mark.asyncio
    async def test_hard_stuck_short_history_ok(self) -> None:
        """Fewer than _HARD_STUCK_WINDOW same-target entries → no hard-stuck."""
        # Use different action_types to avoid triggering anti-loop (window=3)
        history = [
            _make_action_record(f"action_{i}", "Submit")
            for i in range(_HARD_STUCK_WINDOW - 1)
        ]
        local_decision = Decision(
            source="local",
            action_type="click",
            target=TargetSpec(None, None, "Submit", None),
            value=None,
            confidence=0.9,
            reasoning="",
            cost_incurred=0.0,
            transport_hint=None,
        )
        engine, _, _, resolver, _ = _make_engine(
            ambiguity_recommendation="local",
        )
        resolver.resolve.return_value = local_decision
        perception = _make_perception()

        result = await engine.decide("Submit", perception, history, _make_context())
        assert result.source == "local"

    @pytest.mark.asyncio
    async def test_hard_stuck_takes_priority_over_anti_loop(self) -> None:
        """Hard-stuck check runs before anti-loop; result should be hitl."""
        history = [
            _make_action_record("click", "Submit")
            for _ in range(max(_HARD_STUCK_WINDOW, _ANTI_LOOP_WINDOW))
        ]
        engine, _, _, _, _ = _make_engine()
        perception = _make_perception()

        result = await engine.decide("Submit", perception, history, _make_context())
        assert result.source == "hitl"


# ===========================================================================
# Section 10 — transport_hint integration
# ===========================================================================


class TestTransportHintIntegration:
    @pytest.mark.asyncio
    async def test_uia_source_gives_uia_transport_hint(self) -> None:
        plan = _make_planner_decision()
        engine, _, _, _, _ = _make_engine(
            ambiguity_recommendation="cloud",
            planner_decision=plan,
        )
        perception = _make_perception(source_type="uia")

        result = await engine.decide("goal", perception, [], _make_context())
        assert result.transport_hint == "uia"

    @pytest.mark.asyncio
    async def test_visual_source_gives_mouse_transport_hint(self) -> None:
        plan = _make_planner_decision()
        engine, _, _, _, _ = _make_engine(
            ambiguity_recommendation="cloud",
            planner_decision=plan,
        )
        perception = _make_perception(source_type="visual")

        result = await engine.decide("goal", perception, [], _make_context())
        assert result.transport_hint == "mouse"

    @pytest.mark.asyncio
    async def test_file_source_gives_none_transport_hint(self) -> None:
        plan = _make_planner_decision()
        engine, _, _, _, _ = _make_engine(
            ambiguity_recommendation="cloud",
            planner_decision=plan,
        )
        perception = _make_perception(source_type="file")

        result = await engine.decide("goal", perception, [], _make_context())
        assert result.transport_hint is None

    @pytest.mark.asyncio
    async def test_suspend_sets_transport_hint_from_source(self) -> None:
        engine, _, _, _, _ = _make_engine(
            ambiguity_recommendation="suspend",
        )
        perception = _make_perception(source_type="dom")

        result = await engine.decide("goal", perception, [], _make_context())
        assert result.source == "hitl"
        assert result.transport_hint == "dom"


# ===========================================================================
# Section 11 — TargetSpec and Decision dataclass properties
# ===========================================================================


class TestDataclasses:
    def test_target_spec_is_frozen(self) -> None:
        ts = TargetSpec(
            element_id="e1",
            coordinates=(10, 20),
            description="btn",
            preferred_transport="uia",
        )
        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            ts.description = "changed"  # type: ignore[misc]

    def test_decision_is_frozen(self) -> None:
        d = Decision(
            source="local",
            action_type="click",
            target=TargetSpec(None, None, "", None),
            value=None,
            confidence=0.5,
            reasoning="",
            cost_incurred=0.0,
            transport_hint=None,
        )
        with pytest.raises(Exception):
            d.source = "cloud"  # type: ignore[misc]

    def test_decision_context_is_mutable(self) -> None:
        ctx = _make_context()
        ctx.actions_so_far = 5
        assert ctx.actions_so_far == 5


# ===========================================================================
# Section 12 — AmbiguityScorer is called with context params
# ===========================================================================


class TestScorerIntegration:
    @pytest.mark.asyncio
    async def test_scorer_receives_context_flags(self) -> None:
        local_decision = Decision(
            source="local",
            action_type="click",
            target=TargetSpec(None, None, "X", None),
            value=None,
            confidence=0.9,
            reasoning="",
            cost_incurred=0.0,
            transport_hint=None,
        )
        engine, _, scorer, resolver, _ = _make_engine(
            ambiguity_recommendation="local",
        )
        resolver.resolve.return_value = local_decision
        perception = _make_perception()
        context = _make_context(
            candidate_is_destructive=True,
            screen_previously_seen=False,
            used_fallback_transport=True,
        )

        await engine.decide("X", perception, [], context)

        scorer.score.assert_called_once_with(
            perception,
            [],
            candidate_is_destructive=True,
            screen_previously_seen=False,
            used_fallback_transport=True,
        )

    @pytest.mark.asyncio
    async def test_policy_check_uses_context_values(self) -> None:
        local_decision = Decision(
            source="local",
            action_type="click",
            target=TargetSpec(None, None, "X", None),
            value=None,
            confidence=0.9,
            reasoning="",
            cost_incurred=0.0,
            transport_hint=None,
        )
        engine, policy, _, resolver, _ = _make_engine(
            ambiguity_recommendation="local",
        )
        resolver.resolve.return_value = local_decision
        perception = _make_perception(source_type="uia")
        context = _make_context(
            actions_so_far=7,
            elapsed_seconds=120.0,
            task_cost_usd=0.10,
            daily_cost_usd=0.50,
        )

        await engine.decide("X", perception, [], context)

        call_args = policy.check_action.call_args[0][0]
        assert call_args.actions_so_far == 7
        assert call_args.elapsed_seconds == 120.0
        assert call_args.task_cost_usd == 0.10
        assert call_args.daily_cost_usd == 0.50
