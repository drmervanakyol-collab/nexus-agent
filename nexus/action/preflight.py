"""
nexus/action/preflight.py
Preflight Checker — validates an ActionSpec before execution.

Every action passes through PreflightChecker.check() before being handed
to an ActionHandler.  Checks run in priority order; the first failing check
short-circuits the chain and returns a failed PreflightResult.

Checks (in evaluation order)
-----------------------------
CHECK_ELEMENT_VISIBLE     — target element_id must be present in the graph.
CHECK_NOT_OCCLUDED        — screen must not be in a perception-blocking state.
CHECK_COORDS_ON_SCREEN    — coordinates must lie within the screen bounds.
CHECK_POLICY              — PolicyEngine must not block or abort the action.
CHECK_DESTRUCTIVE_CONFIRM — destructive actions require explicit confirmation.
CHECK_SENSITIVE_AREA      — target must not overlap a block-level sensitive region.
CHECK_TRANSPORT_AVAILABLE — preferred transport must be available, or fallback
                            must be explicitly permitted.

Design notes
------------
- Each check is a private method returning ``PreflightResult | None``.
  ``None`` means "this check passed; continue to the next one."
- PreflightChecker is stateless after construction (thread-safe).
- Injecting ``policy=None`` or ``sensitive_detector=None`` in the context
  skips those checks; this is intentional for tests and lightweight callers.
"""
from __future__ import annotations

from dataclasses import dataclass

from nexus.action.registry import ActionSpec
from nexus.core.policy import ActionContext, PolicyEngine
from nexus.core.sensitive_regions import SensitiveRegionDetector
from nexus.core.types import ElementId, Rect
from nexus.infra.logger import get_logger
from nexus.perception.orchestrator import PerceptionResult

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Check name constants
# ---------------------------------------------------------------------------

CHECK_ELEMENT_VISIBLE = "CHECK_ELEMENT_VISIBLE"
CHECK_NOT_OCCLUDED = "CHECK_NOT_OCCLUDED"
CHECK_COORDS_ON_SCREEN = "CHECK_COORDS_ON_SCREEN"
CHECK_POLICY = "CHECK_POLICY"
CHECK_DESTRUCTIVE_CONFIRM = "CHECK_DESTRUCTIVE_CONFIRM"
CHECK_SENSITIVE_AREA = "CHECK_SENSITIVE_AREA"
CHECK_TRANSPORT_AVAILABLE = "CHECK_TRANSPORT_AVAILABLE"

# Transports that do not require availability checking (always present).
_OS_TRANSPORTS: frozenset[str] = frozenset({"mouse", "keyboard"})

# ---------------------------------------------------------------------------
# PreflightResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreflightResult:
    """
    Result of a single PreflightChecker.check() call.

    Attributes
    ----------
    passed:
        True when all checks passed.
    failed_check:
        Name of the first check that failed, or ``None`` on success.
    reason:
        Human-readable explanation of the failure, or ``None`` on success.
    """

    passed: bool
    failed_check: str | None = None
    reason: str | None = None


_PASS = PreflightResult(passed=True)


# ---------------------------------------------------------------------------
# PreflightContext
# ---------------------------------------------------------------------------


@dataclass
class PreflightContext:
    """
    Caller-supplied snapshot of execution environment and task state.

    Attributes
    ----------
    screen_width / screen_height:
        Current screen resolution in pixels.  Used by CHECK_COORDS_ON_SCREEN.
    uia_available:
        True when the UIA transport is operational.
    dom_available:
        True when the DOM (CDP) transport is operational.
    allow_transport_fallback:
        When True, a missing preferred transport is demoted to a warning and
        the action proceeds.  When False, CHECK_TRANSPORT_AVAILABLE fails.
    actions_so_far:
        Actions already executed this task (forwarded to PolicyEngine).
    elapsed_seconds:
        Seconds since the task started (forwarded to PolicyEngine).
    task_cost_usd:
        Accumulated LLM cost this task in USD (forwarded to PolicyEngine).
    daily_cost_usd:
        Accumulated LLM cost today in USD (forwarded to PolicyEngine).
    destructive_confirmed:
        True when the caller has obtained explicit user confirmation for a
        destructive action.  Required by CHECK_DESTRUCTIVE_CONFIRM.
    sensitive_detector:
        Optional SensitiveRegionDetector.  When None, CHECK_SENSITIVE_AREA
        is skipped.
    policy:
        Optional PolicyEngine.  When None, CHECK_POLICY is skipped.
    """

    screen_width: int = 1920
    screen_height: int = 1080
    uia_available: bool = True
    dom_available: bool = True
    allow_transport_fallback: bool = True
    actions_so_far: int = 0
    elapsed_seconds: float = 0.0
    task_cost_usd: float = 0.0
    daily_cost_usd: float = 0.0
    destructive_confirmed: bool = False
    sensitive_detector: SensitiveRegionDetector | None = None
    policy: PolicyEngine | None = None


# ---------------------------------------------------------------------------
# PreflightChecker
# ---------------------------------------------------------------------------


class PreflightChecker:
    """
    Stateless preflight validator for action specifications.

    Instantiate once and call :meth:`check` for each action before
    handing it to an :class:`~nexus.action.registry.ActionHandler`.
    """

    def check(
        self,
        spec: ActionSpec,
        perception: PerceptionResult,
        context: PreflightContext,
    ) -> PreflightResult:
        """
        Run all preflight checks against *spec* in priority order.

        Returns the first failed :class:`PreflightResult`, or the singleton
        ``_PASS`` when every check passes.

        Parameters
        ----------
        spec:
            The action to be validated.
        perception:
            Current perception result (spatial graph, screen state, source).
        context:
            Execution environment snapshot (screen size, transport state,
            policy engine, sensitive-region detector, etc.).

        Returns
        -------
        PreflightResult
        """
        for check_fn in (
            self._check_element_visible,
            self._check_not_occluded,
            self._check_coords_on_screen,
            self._check_policy,
            self._check_destructive_confirm,
            self._check_sensitive_area,
            self._check_transport_available,
        ):
            result = check_fn(spec, perception, context)
            if result is not None:
                _log.debug(
                    "preflight_failed",
                    check=result.failed_check,
                    reason=result.reason,
                    action_type=spec.action_type,
                )
                return result

        _log.debug("preflight_passed", action_type=spec.action_type)
        return _PASS

    # ------------------------------------------------------------------
    # Individual checks — return None when the check passes
    # ------------------------------------------------------------------

    def _check_element_visible(
        self,
        spec: ActionSpec,
        perception: PerceptionResult,
        context: PreflightContext,
    ) -> PreflightResult | None:
        """Fail if a target element_id is specified but absent from the graph."""
        if spec.target_element_id is None:
            return None
        node = perception.spatial_graph.get_node(ElementId(spec.target_element_id))
        if node is None:
            return PreflightResult(
                passed=False,
                failed_check=CHECK_ELEMENT_VISIBLE,
                reason=(
                    f"Element '{spec.target_element_id}' was not found in the "
                    "spatial graph — it may not be visible on screen."
                ),
            )
        return None

    def _check_not_occluded(
        self,
        spec: ActionSpec,
        perception: PerceptionResult,
        context: PreflightContext,
    ) -> PreflightResult | None:
        """Fail if the screen is in a perception-blocking state."""
        if perception.screen_state.blocks_perception:
            return PreflightResult(
                passed=False,
                failed_check=CHECK_NOT_OCCLUDED,
                reason=(
                    f"Screen is in a perception-blocking state "
                    f"({perception.screen_state.reason}); "
                    "the target element may be occluded or still loading."
                ),
            )
        return None

    def _check_coords_on_screen(
        self,
        spec: ActionSpec,
        perception: PerceptionResult,
        context: PreflightContext,
    ) -> PreflightResult | None:
        """Fail if the resolved coordinates fall outside the screen bounds."""
        coords = _resolve_coordinates(spec, perception)
        if coords is None:
            return None
        cx, cy = coords
        if (
            cx < 0
            or cy < 0
            or cx >= context.screen_width
            or cy >= context.screen_height
        ):
            return PreflightResult(
                passed=False,
                failed_check=CHECK_COORDS_ON_SCREEN,
                reason=(
                    f"Coordinates ({cx}, {cy}) are outside the screen bounds "
                    f"({context.screen_width}×{context.screen_height})."
                ),
            )
        return None

    def _check_policy(
        self,
        spec: ActionSpec,
        perception: PerceptionResult,
        context: PreflightContext,
    ) -> PreflightResult | None:
        """Fail if PolicyEngine returns a block or abort verdict."""
        if context.policy is None:
            return None
        transport = spec.preferred_transport or perception.source_result.source_type
        target_rect = _resolve_rect(spec, perception)
        action_ctx = ActionContext(
            action_type=spec.action_type,
            transport=str(transport),
            is_destructive=spec.is_destructive,
            target_rect=target_rect,
            actions_so_far=context.actions_so_far,
            elapsed_seconds=context.elapsed_seconds,
            task_cost_usd=context.task_cost_usd,
            daily_cost_usd=context.daily_cost_usd,
        )
        policy_result = context.policy.check_action(action_ctx)
        if policy_result.verdict in ("block", "abort"):
            return PreflightResult(
                passed=False,
                failed_check=CHECK_POLICY,
                reason=policy_result.message,
            )
        return None

    def _check_destructive_confirm(
        self,
        spec: ActionSpec,
        perception: PerceptionResult,
        context: PreflightContext,
    ) -> PreflightResult | None:
        """Fail if the action is destructive and confirmation is missing."""
        if not spec.is_destructive:
            return None
        if context.destructive_confirmed:
            return None
        return PreflightResult(
            passed=False,
            failed_check=CHECK_DESTRUCTIVE_CONFIRM,
            reason=(
                f"Action '{spec.action_type}' is marked destructive but "
                "no confirmation has been provided. "
                "Set PreflightContext.destructive_confirmed=True to proceed."
            ),
        )

    def _check_sensitive_area(
        self,
        spec: ActionSpec,
        perception: PerceptionResult,
        context: PreflightContext,
    ) -> PreflightResult | None:
        """Fail if the target overlaps a block-level sensitive region."""
        if context.sensitive_detector is None:
            return None
        rect = _resolve_rect(spec, perception)
        if rect is None:
            return None
        hits = context.sensitive_detector.detect_rect(rect)
        block_hits = [r for r in hits if r.severity == "block"]
        if block_hits:
            labels = ", ".join(r.label for r in block_hits)
            return PreflightResult(
                passed=False,
                failed_check=CHECK_SENSITIVE_AREA,
                reason=(
                    f"Target overlaps block-level sensitive region(s): {labels}."
                ),
            )
        return None

    def _check_transport_available(
        self,
        spec: ActionSpec,
        perception: PerceptionResult,
        context: PreflightContext,
    ) -> PreflightResult | None:
        """
        Fail if the preferred transport is unavailable and fallback is
        not permitted.

        OS-level transports (mouse, keyboard) are always considered available.
        When ``allow_transport_fallback=True``, the check passes even if the
        preferred native transport is down — the caller is expected to fall
        back to mouse/keyboard.
        """
        transport = spec.preferred_transport
        if transport is None or transport in _OS_TRANSPORTS:
            return None

        available = (
            (transport == "uia" and context.uia_available)
            or (transport == "dom" and context.dom_available)
        )
        if available:
            return None

        if context.allow_transport_fallback:
            _log.debug(
                "preflight_transport_fallback_allowed",
                transport=transport,
                action_type=spec.action_type,
            )
            return None

        return PreflightResult(
            passed=False,
            failed_check=CHECK_TRANSPORT_AVAILABLE,
            reason=(
                f"Preferred transport '{transport}' is not available and "
                "transport fallback is not permitted "
                "(PreflightContext.allow_transport_fallback=False)."
            ),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_coordinates(
    spec: ActionSpec,
    perception: PerceptionResult,
) -> tuple[int, int] | None:
    """
    Return the best available (cx, cy) for *spec*.

    Priority:
      1. spec.coordinates (explicit)
      2. Centre of the element bounding box from the spatial graph.
    """
    if spec.coordinates is not None:
        return spec.coordinates
    if spec.target_element_id is not None:
        node = perception.spatial_graph.get_node(ElementId(spec.target_element_id))
        if node is not None:
            bb = node.element.bounding_box
            return (bb.x + bb.width // 2, bb.y + bb.height // 2)
    return None


def _resolve_rect(
    spec: ActionSpec,
    perception: PerceptionResult,
) -> Rect | None:
    """
    Return the best available Rect for *spec*.

    Priority:
      1. Element bounding box from the spatial graph (most accurate).
      2. 1×1 Rect at spec.coordinates (point-only spec).
    """
    if spec.target_element_id is not None:
        node = perception.spatial_graph.get_node(ElementId(spec.target_element_id))
        if node is not None:
            return node.element.bounding_box
    if spec.coordinates is not None:
        cx, cy = spec.coordinates
        return Rect(cx, cy, 1, 1)
    return None
