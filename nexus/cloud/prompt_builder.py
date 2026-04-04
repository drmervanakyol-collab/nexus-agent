"""
nexus/cloud/prompt_builder.py
Prompt builder for CloudPlanner.

Responsibilities
----------------
  - Compose a system + user message pair from goal, perception, and history.
  - Enforce a token budget via a three-stage trim sequence:
      1. Remove oldest action-history items.
      2. Collapse screen summary to a compact form.
      3. Drop the screenshot entirely (last resort).
  - Mask sensitive screen regions (black fill) before encoding the screenshot.

Token estimation
----------------
  Text tokens are estimated as ``len(text) // 4`` (≈ 4 chars/token average).
  Screenshots are counted as a fixed ``_SCREENSHOT_TOKEN_ESTIMATE`` tokens
  regardless of resolution (vision APIs apply their own tile-based pricing).

Public API
----------
  ActionRecord    Lightweight record of one completed action.
  BuiltPrompt     Output of PromptBuilder.build().
  PromptBuilder   Stateless builder; call build() per planning cycle.
"""
from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from nexus.cloud.providers import CloudMessage
from nexus.core.types import Rect
from nexus.infra.logger import get_logger

if TYPE_CHECKING:
    from nexus.perception.orchestrator import PerceptionResult

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed token estimate for any screenshot regardless of resolution.
_SCREENSHOT_TOKEN_ESTIMATE: int = 800

# Maximum screenshot dimension (pixels) before encoding.  Keeps the base64
# payload small; vision models downsample anyway.
_SCREENSHOT_MAX_DIM: int = 512

# Characters reserved for the system prompt (well below actual length for safety).
_SYSTEM_PROMPT_TOKEN_RESERVE: int = 120

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = (
    "You are Nexus Agent, an AI assistant that autonomously controls"
    " Windows desktop applications.\n\n"
    "Your task: analyse the current screen state and decide the single"
    " best next action towards the goal.\n\n"
    "Rules:\n"
    "1. Respond ONLY with a valid JSON object — no markdown fences,"
    " no text outside the JSON.\n"
    "2. Use the exact element ID strings from the perception data"
    " for target_element_id.\n"
    "3. Turkish UI awareness — common button labels:"
    " Kaydet (Save), İptal (Cancel), Sil (Delete),"
    " Evet (Yes), Hayır (No), Tamam (OK),"
    " Yeni (New), Aç (Open), Kapat (Close), Güncelle (Update).\n"
    "4. Never perform destructive actions (Sil, Delete, Remove)"
    " unless confidence >= 0.90.\n"
    "5. Set task_status to \"need_help\" when you are uncertain or blocked.\n"
    "6. Allowed action_type values:"
    " click, type, scroll, press_key, wait, none.\n\n"
    "Required JSON format (respond with this object and nothing else):\n"
    "{\n"
    '  "action_type": "click",\n'
    '  "target_description": "human-readable description of the target",\n'
    '  "target_element_id": "element-id-string or null",\n'
    '  "value": "text to type or key name, or null",\n'
    '  "reasoning": "step-by-step explanation",\n'
    '  "confidence": 0.85,\n'
    '  "task_status": "in_progress"\n'
    "}"
)


# ---------------------------------------------------------------------------
# ActionRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionRecord:
    """
    A single completed action in the agent's history.

    Attributes
    ----------
    action_type:   Action verb (e.g. ``"click"``, ``"type"``).
    target_description: Human-readable target (e.g. ``"OK button"``).
    outcome:       Result of the action: ``"success"``, ``"failed"``, or
                   ``"skipped"``.
    timestamp:     ISO-8601 UTC timestamp string.
    """

    action_type: str
    target_description: str
    outcome: str
    timestamp: str


# ---------------------------------------------------------------------------
# BuiltPrompt
# ---------------------------------------------------------------------------


@dataclass
class BuiltPrompt:
    """
    Output of :meth:`PromptBuilder.build`.

    Attributes
    ----------
    messages:
        Ready-to-send ``[system, user]`` CloudMessage pair.
    estimated_tokens:
        Rough token count for the full prompt (text + screenshot estimate).
    screenshot_included:
        True when a screenshot was included in the user message.
    history_kept:
        Number of ActionRecord items included (after trimming).
    """

    messages: list[CloudMessage]
    estimated_tokens: int
    screenshot_included: bool
    history_kept: int


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """
    Builds ``[system, user]`` CloudMessage pairs for the planning LLM call.

    This class is stateless; create one instance and reuse it across calls.
    """

    def build(
        self,
        goal: str,
        perception: PerceptionResult,
        action_history: Sequence[ActionRecord],
        *,
        context_budget_tokens: int = 3000,
        screenshot: np.ndarray | None = None,
        sensitive_regions: list[Rect] | None = None,
    ) -> BuiltPrompt:
        """
        Build a prompt pair for one planning cycle.

        Parameters
        ----------
        goal:
            The agent's current objective.
        perception:
            Current perception state (spatial graph + screen state).
        action_history:
            All completed actions for this task.  Only the last 5 are used.
        context_budget_tokens:
            Soft token budget.  Trim sequence is applied when exceeded.
        screenshot:
            Raw HxWx3 uint8 RGB numpy array.  Sensitive regions are masked
            and the image is resized before encoding.
        sensitive_regions:
            Screen rectangles to fill with black before encoding.

        Returns
        -------
        BuiltPrompt
        """
        # Work with the last 5 history items to start.
        history_window: list[ActionRecord] = list(action_history)[-5:]
        include_screenshot: bool = screenshot is not None
        full_summary: bool = True

        # Iterative trimming — at most 20 passes (bounded loop).
        for _ in range(20):
            summary_text = (
                _build_screen_summary(perception)
                if full_summary
                else _build_compact_screen_summary(perception)
            )
            hist_text = _format_history(history_window)
            user_text = _assemble_user_text(goal, summary_text, hist_text)

            shot_tokens = _SCREENSHOT_TOKEN_ESTIMATE if include_screenshot else 0
            total = (
                _SYSTEM_PROMPT_TOKEN_RESERVE
                + _estimate_tokens(user_text)
                + shot_tokens
            )

            if total <= context_budget_tokens:
                break  # fits within budget

            # Trim order ①: drop oldest history entry
            if history_window:
                history_window.pop(0)
                _log.debug("prompt_history_trimmed", remaining=len(history_window))
                continue

            # Trim order ②: collapse screen summary
            if full_summary:
                full_summary = False
                _log.debug("prompt_summary_collapsed")
                continue

            # Trim order ③: drop screenshot (last resort)
            if include_screenshot:
                include_screenshot = False
                _log.debug("prompt_screenshot_dropped")
                continue

            break  # nothing more to drop

        # Recompute final content with settled decisions.
        summary_text = (
            _build_screen_summary(perception)
            if full_summary
            else _build_compact_screen_summary(perception)
        )
        hist_text = _format_history(history_window)
        user_text = _assemble_user_text(goal, summary_text, hist_text)

        # Encode screenshot (with masking) if still included.
        image_bytes: bytes | None = None
        if include_screenshot and screenshot is not None:
            image_bytes = _encode_screenshot(screenshot, sensitive_regions)

        shot_tokens = _SCREENSHOT_TOKEN_ESTIMATE if include_screenshot else 0
        total_tokens = (
            _SYSTEM_PROMPT_TOKEN_RESERVE
            + _estimate_tokens(user_text)
            + shot_tokens
        )

        _log.debug(
            "prompt_built",
            estimated_tokens=total_tokens,
            history_kept=len(history_window),
            screenshot=include_screenshot,
        )

        messages = [
            CloudMessage(role="system", content=_SYSTEM_PROMPT),
            CloudMessage(role="user", content=user_text, image=image_bytes),
        ]

        return BuiltPrompt(
            messages=messages,
            estimated_tokens=total_tokens,
            screenshot_included=include_screenshot,
            history_kept=len(history_window),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Approximate token count: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def _build_screen_summary(perception: PerceptionResult) -> str:
    """Build a full, element-per-line screen summary."""
    graph_dict: dict[str, Any] = perception.spatial_graph.to_summary_dict()
    state_name: str = perception.screen_state.state_type.name
    confidence: float = perception.screen_state.confidence

    lines: list[str] = [
        "=== SCREEN STATE ===",
        f"State: {state_name}  |  Confidence: {confidence:.2f}",
        f"Elements detected: {graph_dict['node_count']}",
        "",
        "UI Elements:",
    ]
    for node in graph_dict.get("nodes", [])[:60]:
        eid = node.get("id", "?")
        etype = node.get("type", "")
        aff = node.get("affordance", "")
        label = node.get("label", "")
        text = node.get("text", "")
        bbox = node.get("bbox", [])
        dest = " [DESTRUCTIVE]" if node.get("is_destructive") else ""
        conf = node.get("confidence", 0.0)
        text_part = f' text="{text}"' if text else ""
        lines.append(
            f"  [{eid}] {etype}({aff}) label={label!r}{text_part}"
            f" conf={conf:.2f} bbox={bbox}{dest}"
        )
    return "\n".join(lines)


def _build_compact_screen_summary(perception: PerceptionResult) -> str:
    """Compact fallback: state + counts only (no per-element lines)."""
    graph_dict: dict[str, Any] = perception.spatial_graph.to_summary_dict()
    state_name: str = perception.screen_state.state_type.name

    # Count by type
    type_counts: dict[str, int] = {}
    for node in graph_dict.get("nodes", []):
        t = node.get("type", "UNKNOWN")
        type_counts[t] = type_counts.get(t, 0) + 1

    parts = [f"{v}×{k}" for k, v in sorted(type_counts.items())]
    return (
        f"Screen: {state_name}, {graph_dict['node_count']} elements "
        f"({', '.join(parts) or 'none'})"
    )


def _format_history(history: list[ActionRecord]) -> str:
    """Format history items into readable lines."""
    if not history:
        return "No previous actions."
    lines = ["Recent actions (oldest→newest):"]
    for i, rec in enumerate(history, 1):
        lines.append(
            f"  {i}. {rec.action_type} on '{rec.target_description}'"
            f" → {rec.outcome} [{rec.timestamp}]"
        )
    return "\n".join(lines)


def _assemble_user_text(goal: str, summary: str, history: str) -> str:
    """Combine goal, screen summary, and history into one user message."""
    return (
        f"GOAL: {goal}\n\n"
        f"{summary}\n\n"
        f"{history}\n\n"
        f"Based on the above, decide the next action."
    )


def _encode_screenshot(
    data: np.ndarray,
    sensitive_regions: list[Rect] | None,
) -> bytes:
    """
    Apply sensitive-region masking, resize to max dim, encode as PNG bytes.

    Parameters
    ----------
    data:
        HxWx3 uint8 RGB array.
    sensitive_regions:
        Rectangles to fill with black (0).
    """
    masked: np.ndarray = data.copy()

    # Black-fill sensitive regions
    h, w = masked.shape[:2]
    for rect in sensitive_regions or []:
        x1 = max(0, rect.x)
        y1 = max(0, rect.y)
        x2 = min(w, rect.x + rect.width)
        y2 = min(h, rect.y + rect.height)
        if x2 > x1 and y2 > y1:
            masked[y1:y2, x1:x2] = 0

    # Resize to fit within _SCREENSHOT_MAX_DIM
    if max(h, w) > _SCREENSHOT_MAX_DIM:
        scale = _SCREENSHOT_MAX_DIM / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        from PIL import Image  # noqa: PLC0415

        img = Image.fromarray(masked, mode="RGB")
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        masked = np.asarray(img, dtype=np.uint8)

    from PIL import Image  # noqa: PLC0415

    buf = io.BytesIO()
    Image.fromarray(masked, mode="RGB").save(
        buf, format="PNG", optimize=False, compress_level=1
    )
    return buf.getvalue()


def _screen_summary_as_json_str(perception: PerceptionResult) -> str:
    """JSON string of the spatial graph summary (for debugging / tests)."""
    return json.dumps(
        perception.spatial_graph.to_summary_dict(), ensure_ascii=False, indent=2
    )
