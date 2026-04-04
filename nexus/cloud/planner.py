"""
nexus/cloud/planner.py
Cloud Planner — orchestrates one LLM planning cycle per agent step.

CloudPlanner.plan()
-------------------
  1. PromptBuilder.build()     → [system, user] CloudMessage pair
  2. CloudProvider.complete()  → CloudResponse with LLM text
  3. CostTracker.record()      → AlertResult (budget status)
  4. JSON extraction + parse   → dict
     On JSONDecodeError: append fix-request message and retry once.
  5. Build PlannerDecision and return.

PlannerDecision
---------------
  Fields map 1-to-1 to the LLM JSON schema:
    action_type, target_description, target_element_id, value,
    reasoning, confidence, task_status.
  Plus metadata: raw_response, tokens_used, alert.

JSON extraction
---------------
  The raw LLM output may contain markdown fences or surrounding prose.
  _extract_json() finds the first ``{...}`` block (possibly wrapped in
  ``` or ```json fences) and returns the inner JSON string.
  Values for mandatory keys receive safe defaults when absent.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from nexus.cloud.prompt_builder import ActionRecord, BuiltPrompt, PromptBuilder
from nexus.cloud.providers import CloudMessage, CloudProvider, CloudResponse
from nexus.core.errors import CloudError
from nexus.core.types import Rect
from nexus.infra.cost_tracker import AlertResult, CostTracker
from nexus.infra.logger import get_logger
from nexus.perception.orchestrator import PerceptionResult

_log = get_logger(__name__)

_DEFAULT_MODEL: str = "gpt-4o"
_DEFAULT_MAX_TOKENS: int = 512
_DEFAULT_TIMEOUT: float = 30.0
_DEFAULT_CONTEXT_BUDGET: int = 3000
_DEFAULT_MAX_PARSE_RETRIES: int = 1

_JSON_FIX_MESSAGE: str = (
    "Your previous response did not contain valid JSON. "
    "Please respond with ONLY a valid JSON object matching the required format. "
    "Do not include any markdown fences, explanation, or extra text."
)


# ---------------------------------------------------------------------------
# PlannerDecision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerDecision:
    """
    The result of one CloudPlanner.plan() call.

    Attributes
    ----------
    action_type:
        Verb for the next action: ``"click"``, ``"type"``, ``"scroll"``,
        ``"press_key"``, ``"wait"``, or ``"none"``.
    target_description:
        Human-readable description of the target UI element.
    target_element_id:
        Element ID from perception data, or ``None``.
    value:
        Text to type or key name, or ``None``.
    reasoning:
        LLM's step-by-step reasoning.
    confidence:
        LLM self-assessed confidence in [0.0, 1.0].
    task_status:
        ``"in_progress"``, ``"complete"``, or ``"need_help"``.
    raw_response:
        The full text from the LLM (for debugging / logging).
    tokens_used:
        ``tokens_input + tokens_output`` from the final provider call.
    alert:
        Budget alert returned by CostTracker.
    """

    action_type: str
    target_description: str
    target_element_id: str | None
    value: str | None
    reasoning: str
    confidence: float
    task_status: str
    raw_response: str
    tokens_used: int
    alert: AlertResult


# ---------------------------------------------------------------------------
# CloudPlanner
# ---------------------------------------------------------------------------


class CloudPlanner:
    """
    Orchestrates prompt building, LLM calls, cost tracking, and JSON parsing.

    Parameters
    ----------
    provider:
        LLM provider (OpenAIProvider, AnthropicProvider, or FallbackProvider).
    cost_tracker:
        Records token costs per call.
    prompt_builder:
        Builds the message list from goal + perception + history.
    task_id:
        Identifier for the current task (used by CostTracker).
    model:
        Model string recognised by both the provider and CostTracker.
    max_tokens:
        Maximum completion tokens requested from the provider.
    timeout:
        Provider network timeout in seconds.
    context_budget_tokens:
        Soft token budget forwarded to PromptBuilder.
    max_parse_retries:
        How many times to retry after JSON parse failure (default: 1).
    """

    def __init__(
        self,
        provider: CloudProvider,
        cost_tracker: CostTracker,
        prompt_builder: PromptBuilder,
        *,
        task_id: str,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout: float = _DEFAULT_TIMEOUT,
        context_budget_tokens: int = _DEFAULT_CONTEXT_BUDGET,
        max_parse_retries: int = _DEFAULT_MAX_PARSE_RETRIES,
    ) -> None:
        self._provider = provider
        self._tracker = cost_tracker
        self._builder = prompt_builder
        self._task_id = task_id
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._context_budget = context_budget_tokens
        self._max_parse_retries = max_parse_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def plan(
        self,
        goal: str,
        perception: PerceptionResult,
        history: Sequence[ActionRecord],
        *,
        screenshot: np.ndarray | None = None,
        sensitive_regions: list[Rect] | None = None,
    ) -> PlannerDecision:
        """
        Run one planning cycle and return a PlannerDecision.

        Parameters
        ----------
        goal:
            The agent's current objective.
        perception:
            Current screen perception (spatial graph + screen state).
        history:
            Completed actions so far.  PromptBuilder uses the last 5.
        screenshot:
            Raw HxWx3 uint8 RGB array; sensitive regions are masked and
            the image is resized before encoding.
        sensitive_regions:
            Screen rects to black-fill before sending.

        Returns
        -------
        PlannerDecision

        Raises
        ------
        CloudError
            When JSON parsing fails after all retries, or on provider error.
        """
        built: BuiltPrompt = self._builder.build(
            goal,
            perception,
            history,
            context_budget_tokens=self._context_budget,
            screenshot=screenshot,
            sensitive_regions=sensitive_regions,
        )

        messages: list[CloudMessage] = list(built.messages)
        last_response: CloudResponse | None = None

        for attempt in range(self._max_parse_retries + 1):
            response: CloudResponse = await self._provider.complete(
                messages,
                self._model,
                self._max_tokens,
                self._timeout,
            )
            last_response = response

            # Record cost every attempt.
            alert = self._tracker.record(
                self._task_id,
                self._model,
                input_tokens=response.tokens_input,
                output_tokens=response.tokens_output,
            )

            _log.debug(
                "planner_response",
                attempt=attempt,
                tokens_input=response.tokens_input,
                tokens_output=response.tokens_output,
                finish_reason=response.finish_reason,
            )

            # Try to parse JSON.
            parsed = _try_parse(response.content)
            if parsed is not None:
                decision = _build_decision(parsed, response, alert)
                _log.debug(
                    "planner_decision",
                    action_type=decision.action_type,
                    confidence=decision.confidence,
                    task_status=decision.task_status,
                )
                return decision

            # Parse failed — add fix request and retry.
            if attempt < self._max_parse_retries:
                _log.warning(
                    "planner_json_parse_failed",
                    attempt=attempt,
                    raw=response.content[:200],
                )
                messages = messages + [
                    CloudMessage(role="assistant", content=response.content),
                    CloudMessage(role="user", content=_JSON_FIX_MESSAGE),
                ]

        # All attempts exhausted with no valid JSON.
        raw = last_response.content if last_response else ""
        raise CloudError(
            f"Failed to parse valid JSON from LLM after "
            f"{self._max_parse_retries + 1} attempt(s). "
            f"Last response (first 200 chars): {raw[:200]!r}",
        )


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str | None:
    """
    Extract the first JSON object ``{...}`` from *text*.

    Handles:
    - Raw JSON: ``{"key": "value"}``
    - Fenced blocks: ````json\\n{...}\\n````
    """
    # Try fenced code block first
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL
    )
    if fence_match:
        return fence_match.group(1)
    # Fall back to first {...} in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    return None


def _try_parse(text: str) -> dict[str, Any] | None:
    """Return parsed JSON dict or None on any failure."""
    raw = _extract_json(text)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return None


def _build_decision(
    data: dict[str, Any],
    response: CloudResponse,
    alert: AlertResult,
) -> PlannerDecision:
    """Build a PlannerDecision from parsed JSON + response metadata."""
    raw_conf = data.get("confidence", 0.5)
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return PlannerDecision(
        action_type=str(data.get("action_type", "none")),
        target_description=str(data.get("target_description", "")),
        target_element_id=data.get("target_element_id") or None,
        value=data.get("value") or None,
        reasoning=str(data.get("reasoning", "")),
        confidence=confidence,
        task_status=str(data.get("task_status", "in_progress")),
        raw_response=response.content,
        tokens_used=response.tokens_input + response.tokens_output,
        alert=alert,
    )
