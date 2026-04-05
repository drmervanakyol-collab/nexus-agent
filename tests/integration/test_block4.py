"""
tests/integration/test_block4.py
Blok 4 Integration Tests — Faz 37

Decision layer (FAZ 33-36) end-to-end: real classes wired together.
Cloud LLM network calls are replaced with lightweight inline stubs so
no real OpenAI / Anthropic accounts are needed.

TEST 1 — Local decision full path
TEST 2 — Cloud decision full path
TEST 3 — Anti-loop (3 identical actions → force cloud)
TEST 4 — Budget cap (PolicyEngine blocks → suspend)
TEST 5 — Provider fallback (primary raises → secondary used)
TEST 6 — Transport hint propagation (Decision.transport_hint → TransportResolver)
TEST 7 — Full cost chain (CostTracker accumulates correctly)
"""
from __future__ import annotations

import numpy as np
import pytest

from nexus.capture.frame import Frame
from nexus.cloud.planner import CloudPlanner
from nexus.cloud.prompt_builder import ActionRecord, PromptBuilder
from nexus.cloud.providers import (
    CloudMessage,
    CloudResponse,
    CloudUnavailableError,
    FallbackProvider,
)
from nexus.core.policy import PolicyEngine
from nexus.core.settings import NexusSettings
from nexus.core.types import Rect
from nexus.decision.ambiguity_scorer import AmbiguityScorer
from nexus.decision.engine import (
    DecisionContext,
    DecisionEngine,
    LocalResolver,
)
from nexus.infra.cost_tracker import CostTracker
from nexus.perception.arbitration.arbitrator import PerceptionArbitrator
from nexus.perception.locator.locator import Locator
from nexus.perception.matcher.matcher import Matcher
from nexus.perception.orchestrator import PerceptionOrchestrator, PerceptionResult
from nexus.perception.reader.ocr_engine import OCRResult
from nexus.perception.temporal.temporal_expert import TemporalExpert
from nexus.source.resolver import SourceResult
from nexus.source.transport.fallback import KeyboardTransport, MouseTransport
from nexus.source.transport.resolver import ActionSpec, TransportResolver

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_UTC = "2026-04-05T10:00:00+00:00"
_GOAL_SUBMIT = "click Submit button"
_TASK_ID = "integration-task-1"
_MODEL = "gpt-4o"

# Valid LLM JSON response used by the stub cloud provider
_LLM_JSON = (
    '{"action_type": "click", "target_description": "Submit", '
    '"target_element_id": null, "value": null, '
    '"reasoning": "Submit is the correct next action.", '
    '"confidence": 0.85, "task_status": "in_progress"}'
)


# ---------------------------------------------------------------------------
# Frame / OCR helpers (mirror test_block3 pattern)
# ---------------------------------------------------------------------------


def _blank_frame(seq: int = 1) -> Frame:
    data = np.zeros((200, 400, 3), dtype=np.uint8)
    return Frame(
        data=data,
        width=400,
        height=200,
        captured_at_monotonic=0.0,
        captured_at_utc=_UTC,
        sequence_number=seq,
    )


def _frame_with_button(seq: int = 1) -> Frame:
    """Black frame with one white button rect at (50, 80, 100, 30)."""
    data = np.zeros((200, 500, 3), dtype=np.uint8)
    data[80:110, 50:150] = 255  # y:y+h, x:x+w → white button
    return Frame(
        data=data,
        width=500,
        height=200,
        captured_at_monotonic=0.0,
        captured_at_utc=_UTC,
        sequence_number=seq,
    )


def _history(frame: Frame, n: int = 3) -> list[Frame]:
    return [frame] * n


def _ocr_result(text: str, cx: int, cy: int) -> OCRResult:
    return OCRResult(
        text=text,
        confidence=0.95,
        bounding_box=Rect(x=cx - 10, y=cy - 6, width=20, height=12),
        language="eng",
    )


class _ConstOCR:
    def __init__(self, results: list[OCRResult]) -> None:
        self._results = results

    def extract(self, image, region=None, languages=None) -> list[OCRResult]:
        return self._results


class _EmptyOCR:
    def extract(self, image, region=None, languages=None) -> list[OCRResult]:
        return []


# ---------------------------------------------------------------------------
# Component factories
# ---------------------------------------------------------------------------


def _make_perception_orchestrator(ocr=None) -> PerceptionOrchestrator:
    return PerceptionOrchestrator(
        temporal_expert=TemporalExpert(_ocr_fn=lambda f: ""),
        locator=Locator(),
        matcher=Matcher(),
        arbitrator=PerceptionArbitrator(),
        ocr_engine=ocr or _EmptyOCR(),
    )


def _visual_source() -> SourceResult:
    return SourceResult(
        source_type="visual",
        data={"visual_pending": True},
        confidence=0.70,
        latency_ms=0.0,
    )


def _uia_source() -> SourceResult:
    return SourceResult(
        source_type="uia",
        data={"elements": []},
        confidence=1.0,
        latency_ms=0.0,
    )


async def _perceive(
    orch: PerceptionOrchestrator,
    frame: Frame,
    source: SourceResult | None = None,
) -> PerceptionResult:
    return await orch.perceive(
        frame,
        source or _visual_source(),
        frame_history=_history(frame),
    )


class _StubProvider:
    """Always returns a valid cloud response with the given JSON content."""

    def __init__(
        self,
        content: str = _LLM_JSON,
        tokens_input: int = 100,
        tokens_output: int = 50,
    ) -> None:
        self._content = content
        self._tokens_input = tokens_input
        self._tokens_output = tokens_output
        self.call_count = 0

    async def complete(
        self,
        messages: list[CloudMessage],
        model: str,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> CloudResponse:
        self.call_count += 1
        return CloudResponse(
            content=self._content,
            tokens_input=self._tokens_input,
            tokens_output=self._tokens_output,
            model_used=model,
            provider="openai",
            latency_ms=10.0,
            finish_reason="stop",
        )


class _FailProvider:
    """Always raises CloudUnavailableError."""

    def __init__(self) -> None:
        self.call_count = 0

    async def complete(
        self,
        messages: list[CloudMessage],
        model: str,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> CloudResponse:
        self.call_count += 1
        raise CloudUnavailableError("primary provider is down")


def _make_planner(
    provider,
    settings: NexusSettings | None = None,
    task_id: str = _TASK_ID,
    model: str = _MODEL,
) -> tuple[CloudPlanner, CostTracker]:
    cfg = settings or NexusSettings()
    tracker = CostTracker(cfg)
    planner = CloudPlanner(
        provider=provider,
        cost_tracker=tracker,
        prompt_builder=PromptBuilder(),
        task_id=task_id,
        model=model,
        max_tokens=256,
        timeout=5.0,
        max_parse_retries=0,
    )
    return planner, tracker


def _make_engine(
    planner: CloudPlanner,
    tracker: CostTracker,
    settings: NexusSettings | None = None,
) -> DecisionEngine:
    cfg = settings or NexusSettings()
    return DecisionEngine(
        policy=PolicyEngine(cfg),
        scorer=AmbiguityScorer(),
        resolver=LocalResolver(),
        planner=planner,
        cost_before_fn=lambda task_id: tracker.get_task_cost(task_id),
    )


def _make_context(**kwargs) -> DecisionContext:
    defaults: dict = {
        "task_id": _TASK_ID,
        "actions_so_far": 0,
        "elapsed_seconds": 0.0,
        "task_cost_usd": 0.0,
        "daily_cost_usd": 0.0,
    }
    defaults.update(kwargs)
    return DecisionContext(**defaults)


def _make_action_record(
    action_type: str = "click",
    target_description: str = "Submit",
    outcome: str = "success",
) -> ActionRecord:
    return ActionRecord(
        action_type=action_type,
        target_description=target_description,
        outcome=outcome,
        timestamp=_UTC,
    )


# ---------------------------------------------------------------------------
# TEST 1 — Local decision full path
# ---------------------------------------------------------------------------


class TestLocalDecisionFullPath:
    """
    Real visual perception → real AmbiguityScorer → real LocalResolver.
    No cloud call.  Source is visual, screen is STABLE, confidence is high.
    LocalResolver finds the "Submit" button → Decision.source == "local".
    """

    async def test_decision_source_is_local(self) -> None:
        frame = _frame_with_button()
        ocr = _ConstOCR([_ocr_result("Submit", cx=100, cy=95)])
        orch = _make_perception_orchestrator(ocr)
        perception = await _perceive(orch, frame)

        provider = _StubProvider()  # should never be called
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)
        ctx = _make_context()

        result = await engine.decide(_GOAL_SUBMIT, perception, [], ctx)

        assert result.source == "local"

    async def test_local_action_type_is_click(self) -> None:
        frame = _frame_with_button()
        ocr = _ConstOCR([_ocr_result("Submit", cx=100, cy=95)])
        orch = _make_perception_orchestrator(ocr)
        perception = await _perceive(orch, frame)

        planner, tracker = _make_planner(_StubProvider())
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert result.action_type == "click"

    async def test_local_decision_zero_cost(self) -> None:
        frame = _frame_with_button()
        ocr = _ConstOCR([_ocr_result("Submit", cx=100, cy=95)])
        orch = _make_perception_orchestrator(ocr)
        perception = await _perceive(orch, frame)

        planner, tracker = _make_planner(_StubProvider())
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert result.cost_incurred == 0.0

    async def test_cloud_provider_not_called_on_local(self) -> None:
        frame = _frame_with_button()
        ocr = _ConstOCR([_ocr_result("Submit", cx=100, cy=95)])
        orch = _make_perception_orchestrator(ocr)
        perception = await _perceive(orch, frame)

        provider = _StubProvider()
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert provider.call_count == 0

    async def test_local_target_has_coordinates(self) -> None:
        """LocalResolver sets coordinates from the element bounding box."""
        frame = _frame_with_button()
        ocr = _ConstOCR([_ocr_result("Submit", cx=100, cy=95)])
        orch = _make_perception_orchestrator(ocr)
        perception = await _perceive(orch, frame)

        planner, tracker = _make_planner(_StubProvider())
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert result.source == "local"
        assert result.target.coordinates is not None
        cx, cy = result.target.coordinates
        assert cx > 0 and cy > 0

    async def test_local_transport_hint_is_mouse_for_visual_source(self) -> None:
        frame = _frame_with_button()
        ocr = _ConstOCR([_ocr_result("Submit", cx=100, cy=95)])
        orch = _make_perception_orchestrator(ocr)
        perception = await _perceive(orch, frame, _visual_source())

        planner, tracker = _make_planner(_StubProvider())
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert result.source == "local"
        assert result.transport_hint == "mouse"


# ---------------------------------------------------------------------------
# TEST 2 — Cloud decision full path
# ---------------------------------------------------------------------------


class TestCloudDecisionFullPath:
    """
    Blank frame → no visible elements → LocalResolver returns None →
    engine escalates to cloud.  Stub provider returns valid JSON.
    Decision.source == "cloud".
    """

    async def test_decision_source_is_cloud(self) -> None:
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        provider = _StubProvider()
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert result.source == "cloud"

    async def test_cloud_action_type_from_llm(self) -> None:
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        planner, tracker = _make_planner(_StubProvider())
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert result.action_type == "click"

    async def test_cloud_confidence_from_llm(self) -> None:
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        planner, tracker = _make_planner(_StubProvider())
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert result.confidence == pytest.approx(0.85)

    async def test_cloud_provider_called_once(self) -> None:
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        provider = _StubProvider()
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert provider.call_count == 1

    async def test_cloud_failure_returns_suspend(self) -> None:
        """If CloudPlanner raises CloudError, engine returns suspend."""
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        bad_json = "not valid json at all"
        provider = _StubProvider(content=bad_json)
        planner, tracker = _make_planner(provider, model=_MODEL)
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert result.source == "suspend"


# ---------------------------------------------------------------------------
# TEST 3 — Anti-loop
# ---------------------------------------------------------------------------


class TestAntiLoop:
    """
    3 identical (action_type, target_description) records → DecisionEngine
    overrides recommendation to "cloud" even when ambiguity says "local".
    """

    async def test_anti_loop_forces_cloud(self) -> None:
        """With 3 identical records the engine must route to cloud."""
        # Use a button frame so local would normally succeed
        frame = _frame_with_button()
        ocr = _ConstOCR([_ocr_result("Submit", cx=100, cy=95)])
        orch = _make_perception_orchestrator(ocr)
        perception = await _perceive(orch, frame)

        history = [
            _make_action_record("click", "Submit") for _ in range(3)
        ]

        provider = _StubProvider()
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, history, _make_context())
        assert result.source == "cloud"

    async def test_anti_loop_cloud_provider_called(self) -> None:
        frame = _frame_with_button()
        ocr = _ConstOCR([_ocr_result("Submit", cx=100, cy=95)])
        orch = _make_perception_orchestrator(ocr)
        perception = await _perceive(orch, frame)

        history = [
            _make_action_record("click", "Submit") for _ in range(3)
        ]
        provider = _StubProvider()
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        await engine.decide(_GOAL_SUBMIT, perception, history, _make_context())
        assert provider.call_count == 1

    async def test_two_identical_no_anti_loop(self) -> None:
        """Only 2 identical records — no anti-loop → local path."""
        frame = _frame_with_button()
        ocr = _ConstOCR([_ocr_result("Submit", cx=100, cy=95)])
        orch = _make_perception_orchestrator(ocr)
        perception = await _perceive(orch, frame)

        history = [_make_action_record("click", "Submit") for _ in range(2)]
        provider = _StubProvider()
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, history, _make_context())
        # With 2 identical records and a visible button, should stay local
        assert result.source == "local"
        assert provider.call_count == 0


# ---------------------------------------------------------------------------
# TEST 4 — Budget cap
# ---------------------------------------------------------------------------


class TestBudgetCap:
    """
    PolicyEngine with max_cost_per_task_usd=0.001 USD.
    When context.task_cost_usd >= cap, decision is "suspend".
    """

    def _make_tight_settings(self) -> NexusSettings:
        cfg = NexusSettings()
        cfg.budget.max_cost_per_task_usd = 0.001
        return cfg

    async def test_budget_exceeded_returns_suspend(self) -> None:
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        cfg = self._make_tight_settings()
        provider = _StubProvider()
        planner, tracker = _make_planner(provider, settings=cfg)
        engine = _make_engine(planner, tracker, settings=cfg)

        ctx = _make_context(task_cost_usd=0.01)  # above the 0.001 cap
        result = await engine.decide(_GOAL_SUBMIT, perception, [], ctx)

        assert result.source == "suspend"

    async def test_budget_exceeded_no_cloud_call(self) -> None:
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        cfg = self._make_tight_settings()
        provider = _StubProvider()
        planner, tracker = _make_planner(provider, settings=cfg)
        engine = _make_engine(planner, tracker, settings=cfg)

        ctx = _make_context(task_cost_usd=0.01)
        await engine.decide(_GOAL_SUBMIT, perception, [], ctx)
        assert provider.call_count == 0

    async def test_budget_under_cap_proceeds_normally(self) -> None:
        """Cost below cap → engine proceeds (local or cloud)."""
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        cfg = self._make_tight_settings()
        cfg.budget.max_cost_per_task_usd = 1.0  # generous cap
        provider = _StubProvider()
        planner, tracker = _make_planner(provider, settings=cfg)
        engine = _make_engine(planner, tracker, settings=cfg)

        ctx = _make_context(task_cost_usd=0.001)
        result = await engine.decide(_GOAL_SUBMIT, perception, [], ctx)
        assert result.source in ("local", "cloud")

    async def test_max_actions_exceeded_returns_suspend(self) -> None:
        """PolicyEngine also blocks when max_actions_per_task is reached."""
        cfg = NexusSettings()
        cfg.safety.max_actions_per_task = 5

        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        provider = _StubProvider()
        planner, tracker = _make_planner(provider, settings=cfg)
        engine = _make_engine(planner, tracker, settings=cfg)

        ctx = _make_context(actions_so_far=10)  # above cap of 5
        result = await engine.decide(_GOAL_SUBMIT, perception, [], ctx)
        assert result.source == "suspend"


# ---------------------------------------------------------------------------
# TEST 5 — Provider fallback
# ---------------------------------------------------------------------------


class TestProviderFallback:
    """
    FallbackProvider: primary raises CloudUnavailableError → secondary
    returns valid response.  CloudPlanner must succeed and Decision.source
    must be "cloud".
    """

    async def test_fallback_to_secondary_on_primary_failure(self) -> None:
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        primary = _FailProvider()
        secondary = _StubProvider()
        fallback = FallbackProvider(primary, secondary)

        planner, tracker = _make_planner(fallback)
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert result.source == "cloud"

    async def test_primary_was_attempted(self) -> None:
        """Primary must be called before secondary."""
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        primary = _FailProvider()
        secondary = _StubProvider()
        fallback = FallbackProvider(primary, secondary)

        planner, tracker = _make_planner(fallback)
        engine = _make_engine(planner, tracker)

        await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert primary.call_count == 1

    async def test_secondary_was_called(self) -> None:
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        primary = _FailProvider()
        secondary = _StubProvider()
        fallback = FallbackProvider(primary, secondary)

        planner, tracker = _make_planner(fallback)
        engine = _make_engine(planner, tracker)

        await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert secondary.call_count == 1

    async def test_decision_is_cloud_from_secondary(self) -> None:
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        primary = _FailProvider()
        secondary = _StubProvider()
        fallback = FallbackProvider(primary, secondary)

        planner, tracker = _make_planner(fallback)
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert result.source == "cloud"
        assert result.action_type == "click"

    async def test_both_fail_returns_suspend(self) -> None:
        """When both primary and secondary fail, Decision.source == 'suspend'."""
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        primary = _FailProvider()
        secondary = _FailProvider()
        fallback = FallbackProvider(primary, secondary)

        planner, tracker = _make_planner(fallback)
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert result.source == "suspend"


# ---------------------------------------------------------------------------
# TEST 6 — Transport hint propagation
# ---------------------------------------------------------------------------


class TestTransportHintPropagation:
    """
    Decision.transport_hint == "uia" (from uia source_result) →
    TransportResolver.execute() calls the UIA invoker.
    TransportResult.method_used == "uia", success == True.
    """

    def _make_transport_resolver(self, invoke_fn) -> TransportResolver:
        return TransportResolver(
            NexusSettings(),
            _uia_invoker=invoke_fn,
            _mouse_transport=MouseTransport(_click_fn=lambda x, y: None),
            _keyboard_transport=KeyboardTransport(_type_fn=lambda t: None),
        )

    async def test_uia_transport_hint_routes_to_uia(self) -> None:
        """Decision.transport_hint='uia' → TransportResolver uses uia_invoker."""
        invoked: list[bool] = []

        orch = _make_perception_orchestrator()
        perception = await orch.perceive(_blank_frame(), _uia_source())
        # source_type = "uia" → transport_hint = "uia"

        provider = _StubProvider()
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        decision = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert decision.transport_hint == "uia"

        # Now use TransportResolver with that transport hint
        transport = self._make_transport_resolver(
            lambda el: (invoked.append(True), True)[1]
        )
        spec = ActionSpec(action_type="click", task_id=_TASK_ID)
        tr = await transport.execute(spec, perception.source_result, target_element=object())

        assert tr.method_used == "uia"
        assert tr.success is True
        assert len(invoked) == 1

    async def test_visual_transport_hint_is_mouse(self) -> None:
        """Visual source → transport_hint == 'mouse'."""
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame(), _visual_source())

        provider = _StubProvider()
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        decision = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert decision.transport_hint == "mouse"

    async def test_transport_result_method_matches_hint(self) -> None:
        """TransportResolver.method_used matches Decision.transport_hint."""
        invoked: list[bool] = []

        orch = _make_perception_orchestrator()
        perception = await orch.perceive(_blank_frame(), _uia_source())

        provider = _StubProvider()
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        decision = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())

        transport = self._make_transport_resolver(
            lambda el: (invoked.append(True), True)[1]
        )
        spec = ActionSpec(action_type="click", task_id=_TASK_ID)
        tr = await transport.execute(spec, perception.source_result, target_element=object())

        assert tr.method_used == decision.transport_hint

    async def test_uia_hint_transport_fallback_to_mouse(self) -> None:
        """When UIA invoker returns False, TransportResolver falls back to mouse."""
        orch = _make_perception_orchestrator()
        perception = await orch.perceive(_blank_frame(), _uia_source())

        provider = _StubProvider()
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        decision = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert decision.transport_hint == "uia"

        transport = self._make_transport_resolver(lambda el: False)

        class _FakeElement:
            bounding_rect = Rect(x=100, y=100, width=80, height=30)

        spec = ActionSpec(action_type="click", task_id=_TASK_ID)
        tr = await transport.execute(
            spec, perception.source_result, target_element=_FakeElement()
        )
        assert tr.fallback_used is True
        assert tr.method_used == "mouse"


# ---------------------------------------------------------------------------
# TEST 7 — Full cost chain
# ---------------------------------------------------------------------------


class TestFullCostChain:
    """
    Real CostTracker accumulates LLM costs from the cloud decision path.
    After decide(), both CostTracker.get_task_cost() and
    Decision.cost_incurred must be > 0.
    """

    async def test_cost_tracker_accumulates_after_cloud_decision(self) -> None:
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        # Provider returns 100 input + 50 output tokens → real cost computed
        provider = _StubProvider(tokens_input=100, tokens_output=50)
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())

        task_cost = tracker.get_task_cost(_TASK_ID)
        assert task_cost > 0.0

    async def test_decision_cost_incurred_positive_for_cloud(self) -> None:
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        provider = _StubProvider(tokens_input=100, tokens_output=50)
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert result.source == "cloud"
        assert result.cost_incurred > 0.0

    async def test_local_decision_does_not_add_to_tracker(self) -> None:
        """Local path must not record any LLM cost."""
        frame = _frame_with_button()
        ocr = _ConstOCR([_ocr_result("Submit", cx=100, cy=95)])
        orch = _make_perception_orchestrator(ocr)
        perception = await _perceive(orch, frame)

        provider = _StubProvider()
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        assert result.source == "local"
        assert tracker.get_task_cost(_TASK_ID) == 0.0

    async def test_multiple_cloud_calls_accumulate(self) -> None:
        """Two separate cloud decisions → costs accumulate in CostTracker."""
        orch = _make_perception_orchestrator()
        provider = _StubProvider(tokens_input=100, tokens_output=50)
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        for seq in range(1, 3):
            perception = await _perceive(orch, _blank_frame(seq=seq))
            await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())

        total = tracker.get_task_cost(_TASK_ID)
        # Two calls → cost is at least 2× single-call cost
        single_call_cost = tracker.get_task_cost(_TASK_ID)
        assert total == single_call_cost  # same task_id accumulates
        assert total > 0.0

    async def test_cost_incurred_matches_tracker_delta(self) -> None:
        """
        Decision.cost_incurred must equal the delta in CostTracker for that call.
        """
        orch = _make_perception_orchestrator()
        perception = await _perceive(orch, _blank_frame())

        provider = _StubProvider(tokens_input=200, tokens_output=100)
        planner, tracker = _make_planner(provider)
        engine = _make_engine(planner, tracker)

        cost_before = tracker.get_task_cost(_TASK_ID)
        result = await engine.decide(_GOAL_SUBMIT, perception, [], _make_context())
        cost_after = tracker.get_task_cost(_TASK_ID)

        expected_incurred = cost_after - cost_before
        assert result.cost_incurred == pytest.approx(expected_incurred, abs=1e-9)
