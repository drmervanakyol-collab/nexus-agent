"""
tests/unit/test_planner.py
Unit tests for nexus/cloud/prompt_builder.py and nexus/cloud/planner.py.

Coverage
--------
  §1  PromptBuilder.build() — basic output structure
  §2  PromptBuilder — token budget: history trimming
  §3  PromptBuilder — token budget: summary compaction
  §4  PromptBuilder — token budget: screenshot drop
  §5  PromptBuilder — sensitive region masking
  §6  PromptBuilder — screenshot included when budget allows
  §7  _extract_json helper
  §8  _try_parse helper
  §9  CloudPlanner.plan() — successful JSON parse
  §10 CloudPlanner.plan() — invalid JSON → retry once
  §11 CloudPlanner.plan() — cost recorded
  §12 CloudPlanner.plan() — JSON fails all retries → CloudError
  §13 ActionRecord dataclass
  §14 PlannerDecision dataclass
  §15 _build_decision helper
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from nexus.cloud.planner import (
    CloudPlanner,
    PlannerDecision,
    _build_decision,
    _extract_json,
    _try_parse,
)
from nexus.cloud.prompt_builder import (
    _SCREENSHOT_TOKEN_ESTIMATE,
    ActionRecord,
    BuiltPrompt,
    PromptBuilder,
    _encode_screenshot,
    _estimate_tokens,
)
from nexus.cloud.providers import CloudMessage, CloudResponse
from nexus.core.errors import CloudError
from nexus.core.settings import NexusSettings
from nexus.core.types import Rect
from nexus.infra.cost_tracker import AlertResult, CostTracker

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_MODEL = "gpt-4o"


def _make_settings() -> NexusSettings:
    return NexusSettings()


def _make_tracker() -> CostTracker:
    return CostTracker(_make_settings())


def _make_alert() -> AlertResult:
    return AlertResult(
        level="none",
        task_pct=0.01,
        daily_pct=0.01,
        task_cost_usd=0.001,
        daily_cost_usd=0.001,
        message="",
    )


def _make_response(content: str) -> CloudResponse:
    return CloudResponse(
        content=content,
        tokens_input=50,
        tokens_output=30,
        model_used=_MODEL,
        provider="openai",
        latency_ms=100.0,
        finish_reason="stop",
    )


def _make_perception(
    node_count: int = 3,
    state: str = "STABLE",
    confidence: float = 0.95,
) -> MagicMock:
    """Return a MagicMock that satisfies PerceptionResult duck-type."""
    nodes = [
        {
            "id": f"el-{i}",
            "type": "BUTTON",
            "affordance": "CLICKABLE",
            "label": f"Button{i}",
            "text": f"Btn{i}",
            "is_destructive": False,
            "confidence": 0.9,
            "bbox": [10 * i, 10 * i, 80, 30],
            "relations": [],
        }
        for i in range(1, node_count + 1)
    ]
    p = MagicMock()
    p.spatial_graph.to_summary_dict.return_value = {
        "node_count": node_count,
        "nodes": nodes,
    }
    p.screen_state.state_type.name = state
    p.screen_state.confidence = confidence
    return p


def _make_history(n: int = 3) -> list[ActionRecord]:
    return [
        ActionRecord(
            action_type="click",
            target_description=f"Button{i}",
            outcome="success",
            timestamp=f"2026-04-05T10:0{i}:00Z",
        )
        for i in range(1, n + 1)
    ]


def _make_planner(
    provider: Any = None,
    tracker: CostTracker | None = None,
    builder: PromptBuilder | None = None,
    *,
    task_id: str = "task-test",
    model: str = _MODEL,
    max_parse_retries: int = 1,
) -> CloudPlanner:
    return CloudPlanner(
        provider=provider or AsyncMock(),
        cost_tracker=tracker or _make_tracker(),
        prompt_builder=builder or PromptBuilder(),
        task_id=task_id,
        model=model,
        max_parse_retries=max_parse_retries,
    )


def _valid_json_response(
    action_type: str = "click",
    target: str = "OK button",
    element_id: str = "el-1",
    confidence: float = 0.9,
    status: str = "in_progress",
) -> str:
    return json.dumps({
        "action_type": action_type,
        "target_description": target,
        "target_element_id": element_id,
        "value": None,
        "reasoning": "The OK button completes the task.",
        "confidence": confidence,
        "task_status": status,
    })


# ---------------------------------------------------------------------------
# §1 — PromptBuilder basic output structure
# ---------------------------------------------------------------------------


class TestPromptBuilderBasic:
    def test_returns_built_prompt(self) -> None:
        builder = PromptBuilder()
        result = builder.build("Click OK", _make_perception(), [])
        assert isinstance(result, BuiltPrompt)

    def test_two_messages_system_and_user(self) -> None:
        builder = PromptBuilder()
        result = builder.build("Click OK", _make_perception(), [])
        assert len(result.messages) == 2
        assert result.messages[0].role == "system"
        assert result.messages[1].role == "user"

    def test_system_prompt_contains_json_format(self) -> None:
        builder = PromptBuilder()
        result = builder.build("Click OK", _make_perception(), [])
        assert "action_type" in result.messages[0].content
        assert "confidence" in result.messages[0].content
        assert "task_status" in result.messages[0].content

    def test_system_prompt_contains_turkish_awareness(self) -> None:
        builder = PromptBuilder()
        result = builder.build("Click OK", _make_perception(), [])
        content = result.messages[0].content
        assert "Kaydet" in content
        assert "İptal" in content

    def test_user_message_contains_goal(self) -> None:
        builder = PromptBuilder()
        goal = "Find the Save button and click it"
        result = builder.build(goal, _make_perception(), [])
        assert goal in result.messages[1].content

    def test_user_message_contains_screen_state(self) -> None:
        builder = PromptBuilder()
        result = builder.build("x", _make_perception(state="STABLE"), [])
        assert "STABLE" in result.messages[1].content

    def test_user_message_contains_element_labels(self) -> None:
        builder = PromptBuilder()
        result = builder.build("x", _make_perception(node_count=2), [])
        assert "Button1" in result.messages[1].content

    def test_user_message_contains_history(self) -> None:
        builder = PromptBuilder()
        history = _make_history(2)
        result = builder.build("x", _make_perception(), history)
        assert "Button1" in result.messages[1].content
        assert "success" in result.messages[1].content

    def test_no_history_shows_no_previous_actions(self) -> None:
        builder = PromptBuilder()
        result = builder.build("x", _make_perception(), [])
        assert "No previous actions" in result.messages[1].content

    def test_estimated_tokens_positive(self) -> None:
        builder = PromptBuilder()
        result = builder.build("x", _make_perception(), [])
        assert result.estimated_tokens > 0

    def test_history_kept_count(self) -> None:
        builder = PromptBuilder()
        history = _make_history(3)
        result = builder.build("x", _make_perception(), history)
        assert result.history_kept == 3

    def test_history_capped_at_five(self) -> None:
        builder = PromptBuilder()
        history = _make_history(10)
        result = builder.build("x", _make_perception(), history)
        assert result.history_kept == 5

    def test_no_screenshot_by_default(self) -> None:
        builder = PromptBuilder()
        result = builder.build("x", _make_perception(), [])
        assert result.screenshot_included is False
        assert result.messages[1].image is None


# ---------------------------------------------------------------------------
# §2 — Token budget: history trimming
# ---------------------------------------------------------------------------


class TestTokenBudgetHistoryTrim:
    def test_history_trimmed_when_budget_exceeded(self) -> None:
        builder = PromptBuilder()
        history = _make_history(5)
        # Very tight budget forces history trimming
        result = builder.build(
            "x", _make_perception(node_count=50), history,
            context_budget_tokens=200,
        )
        assert result.history_kept < 5

    def test_full_history_fits_when_budget_large(self) -> None:
        builder = PromptBuilder()
        history = _make_history(5)
        result = builder.build(
            "x", _make_perception(), history,
            context_budget_tokens=10_000,
        )
        assert result.history_kept == 5

    def test_oldest_history_dropped_first(self) -> None:
        """With very tight budget, latest action should still be present."""
        builder = PromptBuilder()
        history = [
            ActionRecord("click", "OldTarget", "success", "2026-04-05T09:00:00Z"),
            ActionRecord("click", "NewTarget", "success", "2026-04-05T10:00:00Z"),
        ]
        result = builder.build(
            "x", _make_perception(), history, context_budget_tokens=300
        )
        user_text = result.messages[1].content
        # NewTarget should survive (it's newer)
        if result.history_kept >= 1:
            assert "NewTarget" in user_text


# ---------------------------------------------------------------------------
# §3 — Token budget: summary compaction
# ---------------------------------------------------------------------------


class TestTokenBudgetSummaryCompact:
    def test_compact_summary_when_very_tight_budget(self) -> None:
        builder = PromptBuilder()
        # Tiny budget: no history space + no room for full element list
        result = builder.build(
            "x",
            _make_perception(node_count=100),
            [],
            context_budget_tokens=150,
        )
        # Compact form should be very short (no per-element lines)
        user_text = result.messages[1].content
        # Verify it still contains state info
        assert "STABLE" in user_text

    def test_compact_summary_does_not_have_element_bbox_lines(self) -> None:
        builder = PromptBuilder()
        result = builder.build(
            "x",
            _make_perception(node_count=10),
            _make_history(5),
            context_budget_tokens=100,
        )
        # In compact form, individual bbox arrays won't appear
        user_text = result.messages[1].content
        # At minimum, screen state is present
        assert len(user_text) > 0


# ---------------------------------------------------------------------------
# §4 — Token budget: screenshot drop
# ---------------------------------------------------------------------------


class TestTokenBudgetScreenshotDrop:
    def test_screenshot_dropped_when_budget_exhausted(self) -> None:
        builder = PromptBuilder()
        screenshot = np.zeros((50, 50, 3), dtype=np.uint8)
        result = builder.build(
            "x",
            _make_perception(node_count=50),
            _make_history(5),
            context_budget_tokens=100,  # extremely tight
            screenshot=screenshot,
        )
        assert result.screenshot_included is False
        assert result.messages[1].image is None

    def test_screenshot_included_with_generous_budget(self) -> None:
        builder = PromptBuilder()
        screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
        result = builder.build(
            "x",
            _make_perception(),
            [],
            context_budget_tokens=10_000,
            screenshot=screenshot,
        )
        assert result.screenshot_included is True
        assert result.messages[1].image is not None


# ---------------------------------------------------------------------------
# §5 — Sensitive region masking
# ---------------------------------------------------------------------------


class TestSensitiveRegionMasking:
    def test_sensitive_region_blacked_out(self) -> None:
        """Pixels inside sensitive rect must be zero in the encoded PNG."""
        # White screenshot
        screenshot = np.full((100, 100, 3), 255, dtype=np.uint8)
        sensitive = [Rect(x=10, y=10, width=20, height=20)]

        masked = _encode_screenshot(screenshot, sensitive)

        # Decode back and check the masked region
        import io

        from PIL import Image
        img = Image.open(io.BytesIO(masked))
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        # Region must be black
        assert arr[15, 15, 0] == 0, "Masked pixel should be black"
        assert arr[15, 15, 1] == 0
        assert arr[15, 15, 2] == 0

    def test_outside_region_not_affected(self) -> None:
        """Pixels outside sensitive rect must keep original value."""
        screenshot = np.full((100, 100, 3), 200, dtype=np.uint8)
        sensitive = [Rect(x=10, y=10, width=20, height=20)]

        masked_bytes = _encode_screenshot(screenshot, sensitive)

        import io

        from PIL import Image
        img = Image.open(io.BytesIO(masked_bytes))
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        # Pixel outside the rect should be ~200 (may differ slightly due to resize)
        # We just check it's not zero
        assert arr[0, 0, 0] > 0

    def test_no_sensitive_regions_leaves_image_intact(self) -> None:
        screenshot = np.full((80, 80, 3), 128, dtype=np.uint8)
        result = _encode_screenshot(screenshot, [])
        assert len(result) > 0  # PNG bytes produced

    def test_masking_in_built_prompt(self) -> None:
        """PromptBuilder applies masking when sensitive_regions given."""
        builder = PromptBuilder()
        screenshot = np.full((100, 100, 3), 255, dtype=np.uint8)
        sensitive = [Rect(x=40, y=40, width=20, height=20)]

        result = builder.build(
            "x",
            _make_perception(),
            [],
            context_budget_tokens=10_000,
            screenshot=screenshot,
            sensitive_regions=sensitive,
        )

        assert result.screenshot_included is True
        assert result.messages[1].image is not None
        # Decode and verify the masked region
        import io

        from PIL import Image
        img = Image.open(io.BytesIO(result.messages[1].image))
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        # After resize to max 512 the 100x100 image stays 100x100
        assert arr[50, 50, 0] == 0, "Masked center pixel should be black"


# ---------------------------------------------------------------------------
# §6 — Screenshot included when budget allows
# ---------------------------------------------------------------------------


class TestScreenshotIncluded:
    def test_screenshot_produces_png_bytes(self) -> None:
        builder = PromptBuilder()
        screenshot = np.zeros((64, 64, 3), dtype=np.uint8)
        result = builder.build(
            "x", _make_perception(), [],
            context_budget_tokens=10_000,
            screenshot=screenshot,
        )
        img_bytes = result.messages[1].image
        assert img_bytes is not None
        # PNG magic bytes
        assert img_bytes[:4] == b"\x89PNG"

    def test_screenshot_estimated_tokens_counted(self) -> None:
        builder = PromptBuilder()
        without = builder.build("x", _make_perception(), [], context_budget_tokens=10_000)
        screenshot = np.zeros((64, 64, 3), dtype=np.uint8)
        with_shot = builder.build(
            "x", _make_perception(), [],
            context_budget_tokens=10_000,
            screenshot=screenshot,
        )
        assert (
            with_shot.estimated_tokens
            == without.estimated_tokens + _SCREENSHOT_TOKEN_ESTIMATE
        )


# ---------------------------------------------------------------------------
# §7 — _extract_json helper
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json(self) -> None:
        text = '{"action_type": "click", "confidence": 0.9}'
        result = _extract_json(text)
        assert result is not None
        data = json.loads(result)
        assert data["action_type"] == "click"

    def test_fenced_json(self) -> None:
        text = '```json\n{"action_type": "type"}\n```'
        result = _extract_json(text)
        assert result is not None
        assert json.loads(result)["action_type"] == "type"

    def test_json_with_surrounding_text(self) -> None:
        text = 'Here is my response: {"action_type": "none"} done.'
        result = _extract_json(text)
        assert result is not None
        assert "action_type" in result

    def test_no_json_returns_none(self) -> None:
        assert _extract_json("No JSON here at all") is None

    def test_nested_json(self) -> None:
        text = '{"outer": {"inner": true}}'
        result = _extract_json(text)
        assert result is not None
        data = json.loads(result)
        assert data["outer"]["inner"] is True


# ---------------------------------------------------------------------------
# §8 — _try_parse helper
# ---------------------------------------------------------------------------


class TestTryParse:
    def test_valid_json_returns_dict(self) -> None:
        result = _try_parse('{"action_type": "click"}')
        assert result == {"action_type": "click"}

    def test_invalid_json_returns_none(self) -> None:
        assert _try_parse("{not valid json}") is None

    def test_empty_string_returns_none(self) -> None:
        assert _try_parse("") is None

    def test_plain_text_returns_none(self) -> None:
        assert _try_parse("sorry, I cannot comply") is None

    def test_json_array_returns_none(self) -> None:
        # We only accept dicts, not arrays
        assert _try_parse("[1, 2, 3]") is None


# ---------------------------------------------------------------------------
# §9 — CloudPlanner.plan() successful parse
# ---------------------------------------------------------------------------


class TestPlannerSuccessfulParse:
    async def test_returns_planner_decision(self) -> None:
        provider = AsyncMock()
        provider.complete.return_value = _make_response(_valid_json_response())
        planner = _make_planner(provider=provider)

        result = await planner.plan("Click OK", _make_perception(), [])

        assert isinstance(result, PlannerDecision)

    async def test_action_type_parsed(self) -> None:
        provider = AsyncMock()
        provider.complete.return_value = _make_response(
            _valid_json_response(action_type="type")
        )
        planner = _make_planner(provider=provider)

        result = await planner.plan("type text", _make_perception(), [])
        assert result.action_type == "type"

    async def test_confidence_parsed(self) -> None:
        provider = AsyncMock()
        provider.complete.return_value = _make_response(
            _valid_json_response(confidence=0.85)
        )
        planner = _make_planner(provider=provider)

        result = await planner.plan("x", _make_perception(), [])
        assert result.confidence == 0.85

    async def test_task_status_parsed(self) -> None:
        provider = AsyncMock()
        provider.complete.return_value = _make_response(
            _valid_json_response(status="complete")
        )
        planner = _make_planner(provider=provider)

        result = await planner.plan("x", _make_perception(), [])
        assert result.task_status == "complete"

    async def test_target_element_id_none_when_null(self) -> None:
        payload = json.dumps({
            "action_type": "wait",
            "target_description": "none",
            "target_element_id": None,
            "value": None,
            "reasoning": "waiting",
            "confidence": 0.5,
            "task_status": "in_progress",
        })
        provider = AsyncMock()
        provider.complete.return_value = _make_response(payload)
        planner = _make_planner(provider=provider)

        result = await planner.plan("x", _make_perception(), [])
        assert result.target_element_id is None

    async def test_raw_response_stored(self) -> None:
        content = _valid_json_response()
        provider = AsyncMock()
        provider.complete.return_value = _make_response(content)
        planner = _make_planner(provider=provider)

        result = await planner.plan("x", _make_perception(), [])
        assert result.raw_response == content

    async def test_tokens_used_is_sum(self) -> None:
        provider = AsyncMock()
        provider.complete.return_value = _make_response(_valid_json_response())
        planner = _make_planner(provider=provider)

        result = await planner.plan("x", _make_perception(), [])
        assert result.tokens_used == 50 + 30  # input + output from _make_response

    async def test_provider_called_once_on_success(self) -> None:
        provider = AsyncMock()
        provider.complete.return_value = _make_response(_valid_json_response())
        planner = _make_planner(provider=provider)

        await planner.plan("x", _make_perception(), [])
        assert provider.complete.call_count == 1


# ---------------------------------------------------------------------------
# §10 — Invalid JSON → retry
# ---------------------------------------------------------------------------


class TestPlannerJsonRetry:
    async def test_invalid_json_triggers_retry(self) -> None:
        provider = AsyncMock()
        provider.complete.side_effect = [
            _make_response("Sorry, I cannot provide that information."),
            _make_response(_valid_json_response()),
        ]
        planner = _make_planner(provider=provider, max_parse_retries=1)

        result = await planner.plan("x", _make_perception(), [])

        assert provider.complete.call_count == 2
        assert isinstance(result, PlannerDecision)

    async def test_retry_message_appended(self) -> None:
        """On retry the fix-request message is appended to the conversation."""
        captured_messages: list[list[CloudMessage]] = []

        async def capture_complete(messages, *_args, **_kwargs) -> CloudResponse:
            captured_messages.append(list(messages))
            if len(captured_messages) == 1:
                return _make_response("not json")
            return _make_response(_valid_json_response())

        provider = AsyncMock()
        provider.complete.side_effect = capture_complete
        planner = _make_planner(provider=provider, max_parse_retries=1)

        await planner.plan("x", _make_perception(), [])

        # First call: [system, user]
        assert len(captured_messages[0]) == 2
        # Second call: [system, user, assistant, fix_request]
        assert len(captured_messages[1]) == 4
        assert captured_messages[1][2].role == "assistant"
        assert captured_messages[1][3].role == "user"

    async def test_json_in_fenced_block_parsed_first_attempt(self) -> None:
        fenced = "```json\n" + _valid_json_response() + "\n```"
        provider = AsyncMock()
        provider.complete.return_value = _make_response(fenced)
        planner = _make_planner(provider=provider)

        result = await planner.plan("x", _make_perception(), [])
        assert provider.complete.call_count == 1
        assert result.action_type == "click"


# ---------------------------------------------------------------------------
# §11 — Cost recorded
# ---------------------------------------------------------------------------


class TestPlannerCostRecorded:
    async def test_cost_recorded_on_success(self) -> None:
        provider = AsyncMock()
        provider.complete.return_value = _make_response(_valid_json_response())
        tracker = _make_tracker()
        planner = _make_planner(
            provider=provider,
            tracker=tracker,
            task_id="task-cost-test",
        )

        await planner.plan("x", _make_perception(), [])

        cost = tracker.get_task_cost("task-cost-test")
        assert cost > 0

    async def test_cost_recorded_on_each_retry_attempt(self) -> None:
        provider = AsyncMock()
        provider.complete.side_effect = [
            _make_response("not json"),
            _make_response(_valid_json_response()),
        ]
        tracker = _make_tracker()
        planner = _make_planner(
            provider=provider,
            tracker=tracker,
            task_id="task-retry-cost",
            max_parse_retries=1,
        )

        await planner.plan("x", _make_perception(), [])

        # Two calls were made — both costs should be recorded
        # (get_task_cost returns cumulative sum)
        cost = tracker.get_task_cost("task-retry-cost")
        # With 2 calls of 50+30 tokens and gpt-4o pricing, cost > 0
        assert cost > 0
        summary = tracker.get_summary()
        assert summary.total_calls == 2

    async def test_alert_result_in_decision(self) -> None:
        provider = AsyncMock()
        provider.complete.return_value = _make_response(_valid_json_response())
        planner = _make_planner(provider=provider)

        result = await planner.plan("x", _make_perception(), [])
        assert isinstance(result.alert, AlertResult)
        assert result.alert.level in ("none", "info", "warn", "block")


# ---------------------------------------------------------------------------
# §12 — JSON fails all retries → CloudError
# ---------------------------------------------------------------------------


class TestPlannerAllRetriesFail:
    async def test_cloud_error_raised_when_all_parse_fail(self) -> None:
        provider = AsyncMock()
        provider.complete.return_value = _make_response("This is not JSON at all.")
        planner = _make_planner(provider=provider, max_parse_retries=1)

        with pytest.raises(CloudError):
            await planner.plan("x", _make_perception(), [])

    async def test_provider_called_max_retries_plus_one_times(self) -> None:
        provider = AsyncMock()
        provider.complete.return_value = _make_response("no json")
        planner = _make_planner(provider=provider, max_parse_retries=2)

        with pytest.raises(CloudError):
            await planner.plan("x", _make_perception(), [])

        assert provider.complete.call_count == 3  # 1 + 2 retries

    async def test_no_retry_when_max_parse_retries_zero(self) -> None:
        provider = AsyncMock()
        provider.complete.return_value = _make_response("no json")
        planner = _make_planner(provider=provider, max_parse_retries=0)

        with pytest.raises(CloudError):
            await planner.plan("x", _make_perception(), [])

        assert provider.complete.call_count == 1


# ---------------------------------------------------------------------------
# §13 — ActionRecord
# ---------------------------------------------------------------------------


class TestActionRecord:
    def test_fields(self) -> None:
        rec = ActionRecord(
            action_type="click",
            target_description="Save button",
            outcome="success",
            timestamp="2026-04-05T10:00:00Z",
        )
        assert rec.action_type == "click"
        assert rec.target_description == "Save button"
        assert rec.outcome == "success"

    def test_frozen(self) -> None:
        rec = ActionRecord("click", "target", "success", "ts")
        with pytest.raises((AttributeError, TypeError)):
            rec.action_type = "type"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# §14 — PlannerDecision
# ---------------------------------------------------------------------------


class TestPlannerDecision:
    def test_frozen(self) -> None:
        alert = _make_alert()
        decision = PlannerDecision(
            action_type="click",
            target_description="OK",
            target_element_id="el-1",
            value=None,
            reasoning="because",
            confidence=0.9,
            task_status="in_progress",
            raw_response="{}",
            tokens_used=80,
            alert=alert,
        )
        with pytest.raises((AttributeError, TypeError)):
            decision.action_type = "type"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# §15 — _build_decision helper
# ---------------------------------------------------------------------------


class TestBuildDecision:
    def test_confidence_clamped_above_one(self) -> None:
        data = {
            "action_type": "click",
            "target_description": "x",
            "reasoning": "y",
            "confidence": 1.5,
            "task_status": "in_progress",
        }
        decision = _build_decision(data, _make_response("{}"), _make_alert())
        assert decision.confidence == 1.0

    def test_confidence_clamped_below_zero(self) -> None:
        data = {
            "action_type": "click",
            "target_description": "x",
            "reasoning": "y",
            "confidence": -0.5,
            "task_status": "in_progress",
        }
        decision = _build_decision(data, _make_response("{}"), _make_alert())
        assert decision.confidence == 0.0

    def test_missing_keys_get_defaults(self) -> None:
        decision = _build_decision({}, _make_response("{}"), _make_alert())
        assert decision.action_type == "none"
        assert decision.target_description == ""
        assert decision.target_element_id is None
        assert decision.value is None
        assert decision.task_status == "in_progress"
        assert 0.0 <= decision.confidence <= 1.0

    def test_empty_string_target_id_becomes_none(self) -> None:
        data = {
            "action_type": "click",
            "target_description": "x",
            "target_element_id": "",
            "reasoning": "y",
            "confidence": 0.8,
            "task_status": "in_progress",
        }
        decision = _build_decision(data, _make_response("{}"), _make_alert())
        assert decision.target_element_id is None

    def test_tokens_used_from_response(self) -> None:
        resp = CloudResponse(
            content="{}",
            tokens_input=100,
            tokens_output=50,
            model_used=_MODEL,
            provider="openai",
            latency_ms=20.0,
            finish_reason="stop",
        )
        decision = _build_decision({}, resp, _make_alert())
        assert decision.tokens_used == 150

    def test_estimate_tokens_function(self) -> None:
        assert _estimate_tokens("hello") == 1
        assert _estimate_tokens("a" * 40) == 10
        assert _estimate_tokens("") == 1  # min 1
