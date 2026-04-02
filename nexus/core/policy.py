"""
nexus/core/policy.py
Policy engine for Nexus Agent action safety checks.

Every action passes through PolicyEngine.check_action() before execution.
Rules are evaluated in priority order; the first triggered rule determines
the final verdict.

Rules (in evaluation order)
---------------------------
RULE_DRY_RUN              — block any destructive action in dry-run mode
RULE_MAX_ACTIONS          — block when per-task action cap is reached
RULE_MAX_DURATION         — block when per-task wall-clock limit is reached
RULE_TASK_BUDGET          — block when per-task cost cap is reached
RULE_DAILY_BUDGET         — block when daily cost cap is reached
RULE_SENSITIVE_COORDS     — warn/block based on sensitive region severity
RULE_NATIVE_ACTION_SAFETY — warn on native destructive; block if dry-run
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nexus.core.sensitive_regions import SensitiveRegionDetector
from nexus.core.settings import NexusSettings
from nexus.core.types import Rect

# ---------------------------------------------------------------------------
# Rule name constants
# ---------------------------------------------------------------------------

RULE_DRY_RUN = "RULE_DRY_RUN"
RULE_MAX_ACTIONS = "RULE_MAX_ACTIONS"
RULE_MAX_DURATION = "RULE_MAX_DURATION"
RULE_TASK_BUDGET = "RULE_TASK_BUDGET"
RULE_DAILY_BUDGET = "RULE_DAILY_BUDGET"
RULE_SENSITIVE_COORDS = "RULE_SENSITIVE_COORDS"
RULE_NATIVE_ACTION_SAFETY = "RULE_NATIVE_ACTION_SAFETY"

_NATIVE_TRANSPORTS: frozenset[str] = frozenset({"uia", "dom", "file"})

Verdict = Literal["allow", "warn", "block", "abort"]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionContext:
    """
    Caller-supplied snapshot of the action and its surrounding state.

    Parameters
    ----------
    action_type:
        Semantic action name, e.g. ``"click"``, ``"delete"``.
    transport:
        Delivery mechanism: ``"uia"``, ``"dom"``, ``"file"``,
        ``"mouse"``, or ``"keyboard"``.
    is_destructive:
        True if the action irreversibly modifies data (delete, overwrite…).
    target_rect:
        Screen region the action targets.  Used for sensitive-coord checks.
    actions_so_far:
        Number of actions already executed in the current task.
    elapsed_seconds:
        Seconds elapsed since the task started.
    task_cost_usd:
        Accumulated LLM spend for the current task (USD).
    daily_cost_usd:
        Accumulated LLM spend for today across all tasks (USD).
    """

    action_type: str
    transport: str
    is_destructive: bool = False
    target_rect: Rect | None = None
    actions_so_far: int = 0
    elapsed_seconds: float = 0.0
    task_cost_usd: float = 0.0
    daily_cost_usd: float = 0.0


@dataclass(frozen=True)
class PolicyResult:
    """
    Result of a policy check.

    Parameters
    ----------
    verdict:
        ``"allow"`` — proceed normally.
        ``"warn"``  — proceed, but log a warning.
        ``"block"`` — stop this action; task may continue.
        ``"abort"`` — stop the entire task immediately.
    rule:
        Name of the rule that triggered, or ``None`` when verdict is
        ``"allow"``.
    severity:
        Same granularity as verdict for downstream consumers.
    message:
        Human-readable explanation.
    """

    verdict: Verdict
    rule: str | None
    severity: str
    message: str


# Singleton for the "all clear" case.
_ALLOW = PolicyResult(
    verdict="allow",
    rule=None,
    severity="info",
    message="Action permitted.",
)


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------


class PolicyEngine:
    """
    Stateless policy evaluator.  Thread-safe (no mutable state after init).

    Parameters
    ----------
    settings:
        NexusSettings instance (uses .safety and .budget sub-settings).
    detector:
        Optional SensitiveRegionDetector.  When None, RULE_SENSITIVE_COORDS
        is skipped.
    """

    def __init__(
        self,
        settings: NexusSettings,
        detector: SensitiveRegionDetector | None = None,
    ) -> None:
        self._s = settings
        self._detector = detector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_action(self, ctx: ActionContext) -> PolicyResult:
        """
        Evaluate all rules against *ctx* in priority order.

        Returns the first non-allow PolicyResult, or ``_ALLOW`` if all
        rules pass.
        """
        for rule_fn in (
            self._rule_dry_run,
            self._rule_max_actions,
            self._rule_max_duration,
            self._rule_task_budget,
            self._rule_daily_budget,
            self._rule_sensitive_coords,
            self._rule_native_action_safety,
        ):
            result = rule_fn(ctx)
            if result is not None:
                return result
        return _ALLOW

    # ------------------------------------------------------------------
    # Individual rules — return None when the rule does not apply
    # ------------------------------------------------------------------

    def _rule_dry_run(self, ctx: ActionContext) -> PolicyResult | None:
        if not self._s.safety.dry_run_mode:
            return None
        if not ctx.is_destructive:
            return None
        return PolicyResult(
            verdict="block",
            rule=RULE_DRY_RUN,
            severity="block",
            message=(
                f"Action '{ctx.action_type}' is destructive and the agent is "
                "running in dry-run mode — execution blocked."
            ),
        )

    def _rule_max_actions(self, ctx: ActionContext) -> PolicyResult | None:
        limit = self._s.safety.max_actions_per_task
        if ctx.actions_so_far < limit:
            return None
        return PolicyResult(
            verdict="block",
            rule=RULE_MAX_ACTIONS,
            severity="block",
            message=(
                f"Per-task action cap reached "
                f"({ctx.actions_so_far} >= {limit})."
            ),
        )

    def _rule_max_duration(self, ctx: ActionContext) -> PolicyResult | None:
        limit_s = self._s.safety.max_task_duration_minutes * 60.0
        if ctx.elapsed_seconds < limit_s:
            return None
        elapsed_min = ctx.elapsed_seconds / 60.0
        return PolicyResult(
            verdict="block",
            rule=RULE_MAX_DURATION,
            severity="block",
            message=(
                f"Per-task duration limit exceeded "
                f"({elapsed_min:.1f} min >= "
                f"{self._s.safety.max_task_duration_minutes} min)."
            ),
        )

    def _rule_task_budget(self, ctx: ActionContext) -> PolicyResult | None:
        cap = self._s.budget.max_cost_per_task_usd
        if ctx.task_cost_usd < cap:
            return None
        return PolicyResult(
            verdict="block",
            rule=RULE_TASK_BUDGET,
            severity="block",
            message=(
                f"Per-task cost cap reached "
                f"(${ctx.task_cost_usd:.4f} >= ${cap:.4f})."
            ),
        )

    def _rule_daily_budget(self, ctx: ActionContext) -> PolicyResult | None:
        cap = self._s.budget.max_cost_per_day_usd
        if ctx.daily_cost_usd < cap:
            return None
        return PolicyResult(
            verdict="block",
            rule=RULE_DAILY_BUDGET,
            severity="block",
            message=(
                f"Daily cost cap reached "
                f"(${ctx.daily_cost_usd:.4f} >= ${cap:.4f})."
            ),
        )

    def _rule_sensitive_coords(self, ctx: ActionContext) -> PolicyResult | None:
        if self._detector is None or ctx.target_rect is None:
            return None
        hits = self._detector.detect_rect(ctx.target_rect)
        if not hits:
            return None
        # Most restrictive severity wins.
        block_hits = [r for r in hits if r.severity == "block"]
        labels = ", ".join(r.label for r in hits)
        if block_hits:
            return PolicyResult(
                verdict="block",
                rule=RULE_SENSITIVE_COORDS,
                severity="block",
                message=(
                    f"Target rect overlaps block-level sensitive region(s): {labels}."
                ),
            )
        return PolicyResult(
            verdict="warn",
            rule=RULE_SENSITIVE_COORDS,
            severity="warn",
            message=(
                f"Target rect overlaps warn-level sensitive region(s): {labels}."
            ),
        )

    def _rule_native_action_safety(self, ctx: ActionContext) -> PolicyResult | None:
        if ctx.transport not in _NATIVE_TRANSPORTS:
            return None
        if not ctx.is_destructive:
            return None
        # Destructive native action in dry-run → block
        if self._s.safety.dry_run_mode:
            return PolicyResult(
                verdict="block",
                rule=RULE_NATIVE_ACTION_SAFETY,
                severity="block",
                message=(
                    f"Native transport '{ctx.transport}' attempted a destructive "
                    f"action '{ctx.action_type}' in dry-run mode — blocked."
                ),
            )
        # Destructive native action outside dry-run → warn
        return PolicyResult(
            verdict="warn",
            rule=RULE_NATIVE_ACTION_SAFETY,
            severity="warn",
            message=(
                f"Native transport '{ctx.transport}' is executing destructive "
                f"action '{ctx.action_type}' — review recommended."
            ),
        )
