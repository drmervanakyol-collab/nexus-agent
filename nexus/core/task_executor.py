"""
nexus/core/task_executor.py
TaskExecutor — the top-level agent loop that wires all V1-core subsystems.

TaskContext
-----------
Mutable task-level state carried across every loop iteration.

  task_id          : str
  goal             : str
  started_at       : float           time.monotonic() baseline
  action_count     : int             actions executed so far
  total_cost_usd   : float           accumulated LLM cost
  status           : TaskStatus
  action_history   : list[ActionRecord]

TaskResult
----------
Immutable summary returned after the loop exits.

  task_id, success, steps_completed, total_cost_usd, duration_ms,
  status, summary, error, transport_stats

TransportStats
--------------
  native_count : int   actions delivered via UIA / DOM / File
  fallback_count: int  actions delivered via mouse / keyboard fallback

TaskExecutor.execute(goal, task_id?) → TaskResult
--------------------------------------------------
Injectability
~~~~~~~~~~~~~
Every heavy I/O stage is replaceable with a lightweight callable:

  source_fn      ()                        -> SourceResult
  capture_fn     ()                        -> Frame
  perceive_fn    (frame, source)           -> PerceptionResult
  done_fn        (decision)               -> bool   (task complete?)
  verifier_fn    (before, after, action_type) -> VerificationResult
  progress_fn    (msg: str)               -> None   (UI progress)

The rest (DecisionEngine, TransportResolver, PreflightChecker, etc.) are
passed as concrete instances — or left None to skip that stage entirely,
which is useful for narrow unit tests.

Loop phases (13 steps)
~~~~~~~~~~~~~~~~~~~~~~
  1.  HealthChecker.run_all()    — skip if None; abort on "fail"
  2.  source_fn()                — resolve source
  3.  capture_fn()               — acquire stable frame
  4.  perceive_fn()              — build PerceptionResult
  5.  DecisionEngine.decide()    — determine next action
      • source=="hitl"    → HITLManager.request(); re-loop or break
      • source=="suspend" → SuspendManager.suspend(); break
  6.  PolicyEngine check         — via DecisionContext (inside decide())
  7.  PreflightChecker.check()   — structural validation
  8.  TransportResolver.execute()— deliver action
  9.  verifier_fn()              — post-action verification
  10. CostTracker.record()       — budget accounting
  11. DB: action row INSERT       — via ActionRepository
  12. FingerprintStore.record_outcome()
  13. done_fn(decision)          — exit loop if True
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from nexus.action.preflight import PreflightChecker, PreflightContext
from nexus.action.registry import ActionSpec as RegistryActionSpec
from nexus.capture.frame import Frame
from nexus.cloud.prompt_builder import ActionRecord
from nexus.core.hitl_manager import HITLManager, HITLRequest
from nexus.core.settings import NexusSettings
from nexus.core.suspend_manager import SuspendManager
from nexus.decision.engine import Decision, DecisionContext, DecisionEngine
from nexus.infra.cost_tracker import CostTracker
from nexus.infra.database import Database
from nexus.infra.logger import get_logger
from nexus.infra.repositories import ActionRepository, TaskRepository
from nexus.memory.fingerprint_store import FingerprintStore
from nexus.perception.orchestrator import PerceptionResult
from nexus.source.resolver import SourceResult
from nexus.source.transport.resolver import (
    ActionSpec as TransportActionSpec,
)
from nexus.source.transport.resolver import (
    TransportResolver,
    TransportResult,
)
from nexus.verification import VerificationPolicy, VerificationResult

_log = get_logger(__name__)


def _uia_elem_id(elem: Any) -> str:
    """
    Generate the element ID consistent with _uia_to_spatial_graph.
    Uses automation_id when non-empty, otherwise falls back to name.
    """
    return (elem.automation_id or elem.name or "").strip()


def _resolve_uia_target(source_result: SourceResult, target: Any) -> Any:
    """
    Find the best matching UIAElement from UIA source data.

    Tries, in order:
      1. Generated element-ID match (same logic as _uia_to_spatial_graph).
      2. Name / description substring match (visible elements only).
      3. Coordinate containment (bounding rect).
    Returns None when no match is found or when the source is not UIA.
    """
    if source_result.source_type != "uia":
        return None
    elements = source_result.data
    if not isinstance(elements, list) or not elements:
        return None

    # 1. Generated element-ID match
    if target.element_id:
        tid = target.element_id.strip()
        for elem in elements:
            if _uia_elem_id(elem) == tid:
                return elem

    # 2. Name / description substring match (visible elements only)
    if target.description:
        desc = target.description.lower().strip()
        candidates = [e for e in elements if e.is_visible and e.name]
        # elem.name is a prefix/substring of description
        for elem in candidates:
            if elem.name.lower() in desc:
                return elem
        # description is a prefix/substring of elem.name
        for elem in candidates:
            if desc in elem.name.lower():
                return elem

    # 3. Coordinate containment (bounding rect)
    if target.coordinates:
        x, y = target.coordinates
        for elem in elements:
            if elem.bounding_rect and elem.is_visible:
                r = elem.bounding_rect
                if r.x <= x <= r.x + r.width and r.y <= y <= r.y + r.height:
                    return elem

    return None


# ---------------------------------------------------------------------------
# Type aliases for injectable callables
# ---------------------------------------------------------------------------

TaskStatus = Literal["running", "completed", "failed", "suspended", "cancelled"]

SourceFn = Callable[[], Awaitable[SourceResult]]
CaptureFn = Callable[[], Awaitable[Frame]]
PerceiveFn = Callable[[Frame, SourceResult], Awaitable[PerceptionResult]]
DoneFn = Callable[[Decision], bool]
VerifierFn = Callable[[Frame, Frame, str], Awaitable[VerificationResult]]
ProgressFn = Callable[[str], None]

# Action types that should not trigger verification
_SKIP_VERIFY_ACTIONS: frozenset[str] = frozenset({"done", "complete", "finish"})

# Action types (and task_status values) that signal task completion
_DONE_ACTION_TYPES: frozenset[str] = frozenset({"done", "complete", "finish"})

# Native transport methods (for transport_stats counting)
_NATIVE_METHODS: frozenset[str] = frozenset({"uia", "dom", "file"})


# ---------------------------------------------------------------------------
# TransportStats
# ---------------------------------------------------------------------------


@dataclass
class TransportStats:
    """Transport usage summary for a completed task."""

    native_count: int = 0
    fallback_count: int = 0

    @property
    def total(self) -> int:
        return self.native_count + self.fallback_count

    @property
    def native_ratio(self) -> float:
        if self.total == 0:
            return 0.0
        return self.native_count / self.total


# ---------------------------------------------------------------------------
# TaskContext
# ---------------------------------------------------------------------------


@dataclass
class TaskContext:
    """Mutable loop-level state for one task execution."""

    task_id: str
    goal: str
    started_at: float
    action_count: int = 0
    total_cost_usd: float = 0.0
    status: TaskStatus = "running"
    action_history: list[ActionRecord] = field(default_factory=list)
    transport_stats: TransportStats = field(default_factory=TransportStats)


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskResult:
    """Immutable summary of a completed task execution."""

    task_id: str
    success: bool
    steps_completed: int
    total_cost_usd: float
    duration_ms: float
    status: TaskStatus
    summary: str
    error: str | None
    transport_stats: TransportStats


# ---------------------------------------------------------------------------
# TaskExecutor
# ---------------------------------------------------------------------------


class TaskExecutor:
    """
    Top-level agent loop.  Wires all V1-core subsystems together.

    All heavy I/O stages are injectable via callables so the executor is
    fully unit-testable without real hardware.

    Parameters
    ----------
    db:
        Database for persisting task and action rows.
    settings:
        NexusSettings — drives policy, budget, and verification config.
    source_fn:
        Async callable ``() -> SourceResult``.
    capture_fn:
        Async callable ``() -> Frame``.
    perceive_fn:
        Async callable ``(frame, source) -> PerceptionResult``.
    decision_engine:
        DecisionEngine instance (or None to skip).
    transport_resolver:
        TransportResolver instance (or None to skip).
    preflight:
        PreflightChecker instance (or None to skip).
    health_checker:
        HealthChecker instance (or None to skip).
    hitl_manager:
        HITLManager for human-in-the-loop prompts (or None to skip).
    suspend_manager:
        SuspendManager for task suspension (or None to skip).
    cost_tracker:
        CostTracker for budget accounting (or None to skip).
    fingerprint_store:
        FingerprintStore for transport-outcome learning (or None to skip).
    verifier_fn:
        Async callable ``(before_frame, after_frame, action_type) ->
        VerificationResult`` (or None to skip verification).
    done_fn:
        Callable ``(decision) -> bool`` — returns True when the task is
        complete.  Default: ``decision.action_type in {"done", "complete"}``.
    progress_fn:
        Callable ``(msg: str) -> None`` for real-time progress reporting.
        Default: no-op.
    max_steps:
        Hard cap on loop iterations (overrides policy when lower).
    """

    def __init__(
        self,
        db: Database,
        settings: NexusSettings,
        *,
        source_fn: SourceFn | None = None,
        capture_fn: CaptureFn | None = None,
        perceive_fn: PerceiveFn | None = None,
        decision_engine: DecisionEngine | None = None,
        transport_resolver: TransportResolver | None = None,
        preflight: PreflightChecker | None = None,
        health_checker: Any = None,
        hitl_manager: HITLManager | None = None,
        suspend_manager: SuspendManager | None = None,
        cost_tracker: CostTracker | None = None,
        fingerprint_store: FingerprintStore | None = None,
        verifier_fn: VerifierFn | None = None,
        done_fn: DoneFn | None = None,
        progress_fn: ProgressFn | None = None,
        max_steps: int | None = None,
    ) -> None:
        self._db = db
        self._settings = settings

        self._source_fn = source_fn or _default_source_fn
        self._capture_fn = capture_fn or _default_capture_fn
        self._perceive_fn = perceive_fn or _default_perceive_fn
        self._decision_engine = decision_engine
        self._transport_resolver = transport_resolver
        self._preflight = preflight
        self._health_checker = health_checker
        self._hitl_manager = hitl_manager
        self._suspend_manager = suspend_manager
        self._cost_tracker = cost_tracker
        self._fingerprint_store = fingerprint_store
        self._verifier_fn = verifier_fn
        self._done_fn = done_fn or _default_done_fn
        self._progress_fn = progress_fn or (lambda _msg: None)

        policy_cap = settings.safety.max_actions_per_task
        self._max_steps: int = (
            min(max_steps, policy_cap) if max_steps is not None else policy_cap
        )

        self._cancelled: bool = False

        self._task_repo = TaskRepository()
        self._action_repo = ActionRepository()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        goal: str,
        task_id: str | None = None,
    ) -> TaskResult:
        """
        Run the agent loop for *goal* and return a TaskResult.

        Parameters
        ----------
        goal:
            High-level natural-language objective.
        task_id:
            Stable identifier for DB persistence; generated if omitted.
        """
        self._cancelled = False
        task_id = task_id or str(uuid.uuid4())
        ctx = TaskContext(
            task_id=task_id,
            goal=goal,
            started_at=time.monotonic(),
        )

        self._progress(ctx, "Starting task execution")

        # Persist task row
        await self._persist_task(ctx)

        error: str | None = None
        last_decision: Decision | None = None
        before_frame: Frame | None = None

        try:
            while not self._cancelled and ctx.action_count < self._max_steps:
                # 1. Health check (first step only)
                if ctx.action_count == 0 and self._health_checker is not None:
                    report = self._health_checker.run_all()
                    if report.overall == "fail":
                        ctx.status = "failed"
                        error = "HealthChecker: critical failure before first step"
                        _log.error("task_executor.health_fail", task_id=task_id)
                        break

                step = ctx.action_count + 1
                self._progress(ctx, f"Step {step}: resolving source")

                # 2. Resolve source
                source_result: SourceResult = await self._source_fn()

                # 3. Capture frame
                self._progress(ctx, f"Step {step}: capturing frame")
                current_frame: Frame = await self._capture_fn()

                # 4. Perceive
                self._progress(ctx, f"Step {step}: perceiving screen")
                perception: PerceptionResult = await self._perceive_fn(
                    current_frame, source_result
                )

                # 5 + 6. Decide (policy check is inside DecisionEngine.decide)
                if self._decision_engine is None:
                    ctx.status = "failed"
                    error = "No DecisionEngine configured"
                    break

                # Determine if the last transport used a fallback channel
                _last_transport = ctx.action_history[-1] if ctx.action_history else None
                _used_fallback = (
                    _last_transport is not None
                    and getattr(_last_transport, "outcome", None) == "fallback"
                ) or (source_result.source_type == "visual")

                # UIA/DOM sources give reliable structured data — treat screen as
                # "previously seen" to avoid inflating the ambiguity score with
                # the new_screen_pattern factor (which assumes visual uncertainty).
                _screen_seen = (
                    source_result.source_type in ("uia", "dom")
                    or self._fingerprint_store is not None
                )

                dec_ctx = DecisionContext(
                    task_id=task_id,
                    actions_so_far=ctx.action_count,
                    elapsed_seconds=time.monotonic() - ctx.started_at,
                    task_cost_usd=ctx.total_cost_usd,
                    daily_cost_usd=ctx.total_cost_usd,  # simplified
                    screen_previously_seen=_screen_seen,
                    used_fallback_transport=_used_fallback,
                    screenshot=current_frame.data,
                )
                self._progress(ctx, f"Step {step}: deciding next action")
                decision: Decision = await self._decision_engine.decide(
                    goal, perception, ctx.action_history, dec_ctx
                )
                last_decision = decision

                # -- HITL branch --
                if decision.source == "hitl":
                    ctx.status = "running"
                    if self._hitl_manager is not None:
                        self._progress(ctx, f"Step {step}: HITL prompt")
                        hitl_req = HITLRequest(
                            task_id=task_id,
                            question=decision.reasoning,
                            options=["continue", "skip", "abort"],
                            default_index=0,
                        )
                        hitl_resp = await self._hitl_manager.request(hitl_req)
                        if hitl_resp.chosen_option == "abort":
                            ctx.status = "cancelled"
                            break
                        # "continue" or "skip" → re-loop
                        continue
                    else:
                        # No HITL manager → treat as suspend
                        ctx.status = "suspended"
                        break

                # -- Suspend branch --
                if decision.source == "suspend":
                    if self._suspend_manager is not None:
                        await self._suspend_manager.suspend(
                            task_id, decision.reasoning, {"step": step}
                        )
                    ctx.status = "suspended"
                    self._progress(  # noqa: E501
                        ctx, f"Step {step}: task suspended — {decision.reasoning}"
                    )
                    break

                # -- Done check (early) --
                if self._done_fn(decision):
                    ctx.action_count += 1
                    ctx.status = "completed"
                    self._progress(ctx, f"Task completed in {ctx.action_count} step(s)")
                    break

                # 7. Preflight
                if self._preflight is not None:
                    pf_spec = RegistryActionSpec(
                        action_type=decision.action_type,
                        preferred_transport=decision.transport_hint,
                    )
                    pf_ctx = PreflightContext(
                        uia_available=source_result.source_type == "uia",
                        dom_available=source_result.source_type == "dom",
                        allow_transport_fallback=True,
                    )
                    pf_result = self._preflight.check(pf_spec, perception, pf_ctx)
                    if not pf_result.passed:
                        _log.warning(
                            "task_executor.preflight_failed",
                            check=pf_result.failed_check,
                            msg=pf_result.reason,
                        )
                        ctx.status = "failed"
                        error = f"Preflight failed: {pf_result.failed_check}"
                        break

                # 8. Transport execute
                transport_result: TransportResult | None = None
                if self._transport_resolver is not None:
                    self._progress(
                        ctx,
                        f"Step {step}: executing {decision.action_type} "
                        f"via {decision.transport_hint or 'auto'}",
                    )
                    t_spec = TransportActionSpec(
                        action_type=decision.action_type,  # type: ignore[arg-type]
                        text=decision.value,
                        coordinates=decision.target.coordinates,
                        task_id=task_id,
                        action_id=str(step),
                    )
                    target_element = _resolve_uia_target(
                        source_result, decision.target
                    )
                    transport_result = await self._transport_resolver.execute(
                        t_spec, source_result, target_element
                    )
                    # Record transport stats
                    if transport_result.method_used in _NATIVE_METHODS:
                        ctx.transport_stats.native_count += 1
                    else:
                        ctx.transport_stats.fallback_count += 1

                    if self._cost_tracker is not None:
                        self._cost_tracker.record_transport(
                            task_id, transport_result.method_used
                        )

                    transport_label = (
                        f"{transport_result.method_used}"
                        f"{'(fallback)' if transport_result.fallback_used else ''}"
                    )
                    self._progress(ctx, f"Step {step}: transport={transport_label}")

                # 9. Verify
                v_result: VerificationResult | None = None
                if (
                    self._verifier_fn is not None
                    and before_frame is not None
                    and decision.action_type not in _SKIP_VERIFY_ACTIONS
                ):
                    v_result = await self._verifier_fn(
                        before_frame, current_frame, decision.action_type
                    )
                    if v_result and not v_result.success:
                        _log.warning(
                            "task_executor.verification_failed",
                            action=decision.action_type,
                            mode=v_result.mode_used.name,
                            confidence=round(v_result.confidence, 3),
                        )

                before_frame = current_frame

                # 10. Cost tracking (LLM cost already recorded by planner)
                ctx.total_cost_usd += decision.cost_incurred

                # 10b. Cost cap check
                cost_cap = self._settings.budget.max_cost_per_task_usd
                if cost_cap > 0 and ctx.total_cost_usd >= cost_cap:
                    ctx.status = "failed"
                    error = (
                        f"Cost cap exceeded: ${ctx.total_cost_usd:.4f} >= "
                        f"${cost_cap:.4f}"
                    )
                    _log.warning(
                        "task_executor.cost_cap",
                        task_id=task_id,
                        cost=ctx.total_cost_usd,
                        cap=cost_cap,
                    )
                    break

                # 11. Persist action row
                action_id = f"{task_id}-step-{step}"
                outcome = (
                    "success"
                    if (transport_result is None or transport_result.success)
                    else "failed"
                )
                await self._persist_action(ctx, action_id, decision, outcome)

                # 12. Fingerprint store outcome
                if (
                    self._fingerprint_store is not None
                    and transport_result is not None
                ):
                    fp_match = await self._fingerprint_store.find_similar(
                        layout_hash=str(perception.frame_sequence),
                        element_signature="",
                    )
                    if fp_match is not None:
                        await self._fingerprint_store.record_outcome(
                            fp_match.id,
                            success=transport_result.success,
                            transport_used=transport_result.method_used,
                            strategy=f"{decision.action_type}+{transport_result.method_used}",
                        )

                # Append to action history
                ctx.action_history.append(
                    ActionRecord(
                        action_type=decision.action_type,
                        target_description=str(decision.target),
                        outcome=outcome,
                        timestamp=perception.timestamp,
                    )
                )

                ctx.action_count += 1

            else:
                # while condition became False — either _cancelled or max_steps
                if self._cancelled:
                    if ctx.status == "running":
                        ctx.status = "cancelled"
                elif ctx.status == "running":
                    ctx.status = "failed"
                    error = f"Max steps ({self._max_steps}) reached without completion"
                    _log.warning(
                        "task_executor.max_steps",
                        task_id=task_id,
                        steps=ctx.action_count,
                    )

        except Exception as exc:  # noqa: BLE001
            _log.exception("task_executor.unexpected_error", task_id=task_id)
            ctx.status = "failed"
            error = str(exc)

        if self._cancelled and ctx.status == "running":  # type: ignore[comparison-overlap]
            ctx.status = "cancelled"

        duration_ms = (time.monotonic() - ctx.started_at) * 1000.0
        success = ctx.status == "completed"

        # Update DB task status
        await self._update_task_status(ctx)

        summary = _build_summary(ctx, last_decision, error)
        self._progress(ctx, summary)

        return TaskResult(
            task_id=task_id,
            success=success,
            steps_completed=ctx.action_count,
            total_cost_usd=ctx.total_cost_usd,
            duration_ms=duration_ms,
            status=ctx.status,
            summary=summary,
            error=error,
            transport_stats=ctx.transport_stats,
        )

    def cancel(self) -> None:
        """
        Signal the executor to stop after the current step completes.

        The next iteration of the loop will see ``_cancelled=True`` and
        exit gracefully, setting status to "cancelled".
        """
        self._cancelled = True
        _log.info("task_executor.cancel_requested")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _progress(self, ctx: TaskContext, msg: str) -> None:
        """Emit a progress message with cost and transport context."""
        full = (
            f"[{ctx.task_id[:8]}] "
            f"step={ctx.action_count} "
            f"cost=${ctx.total_cost_usd:.4f} "
            f"native={ctx.transport_stats.native_count} "
            f"fallback={ctx.transport_stats.fallback_count} "
            f"| {msg}"
        )
        _log.info("task_executor.progress", msg=full)
        self._progress_fn(full)

    async def _persist_task(self, ctx: TaskContext) -> None:
        """Insert task row (best-effort; never raises)."""
        try:
            async with self._db.connection() as conn:
                await self._task_repo.create(
                    conn, id=ctx.task_id, goal=ctx.goal, status="running"
                )
        except Exception as exc:  # noqa: BLE001
            _log.warning("task_executor.persist_task_failed", error=str(exc))

    async def _update_task_status(self, ctx: TaskContext) -> None:
        """Update task row status (best-effort)."""
        try:
            db_status = _map_status(ctx.status)
            async with self._db.connection() as conn:
                await self._task_repo.update_status(conn, ctx.task_id, db_status)
        except Exception as exc:  # noqa: BLE001
            _log.warning("task_executor.update_task_failed", error=str(exc))

    async def _persist_action(
        self,
        ctx: TaskContext,
        action_id: str,
        decision: Decision,
        outcome: str,
    ) -> None:
        """Insert action row (best-effort)."""
        import json

        try:
            async with self._db.connection() as conn:
                await self._action_repo.create(
                    conn,
                    id=action_id,
                    task_id=ctx.task_id,
                    type=decision.action_type,
                    payload=json.dumps(
                        {"target": str(decision.target), "value": decision.value}
                    ),
                    status=outcome if outcome in ("success", "failed") else "pending",
                )
        except Exception as exc:  # noqa: BLE001
            _log.warning("task_executor.persist_action_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Defaults for injectable callables
# ---------------------------------------------------------------------------


async def _default_source_fn() -> SourceResult:
    """Fall-back: visual source (always available)."""
    return SourceResult(
        source_type="visual",
        data={"visual_pending": True},
        confidence=0.70,
        latency_ms=0.0,
    )


async def _default_capture_fn() -> Frame:
    """Fall-back: blank 1×1 frame — only used when no real capture is wired."""
    import numpy as np

    data = np.zeros((1, 1, 3), dtype=np.uint8)
    return Frame(
        data=data,
        width=1,
        height=1,
        captured_at_monotonic=time.monotonic(),
        captured_at_utc="1970-01-01T00:00:00+00:00",
        sequence_number=0,
    )


async def _default_perceive_fn(frame: Frame, source: SourceResult) -> PerceptionResult:
    """Fall-back: minimal perception with empty graph."""
    from nexus.perception.arbitration.arbitrator import ArbitrationResult
    from nexus.perception.orchestrator import PerceptionResult
    from nexus.perception.spatial_graph import SpatialGraph
    from nexus.perception.temporal.temporal_expert import ScreenState, StateType

    return PerceptionResult(
        spatial_graph=SpatialGraph([], [], {}),
        screen_state=ScreenState(
            state_type=StateType.STABLE,
            confidence=1.0,
            blocks_perception=False,
            reason="default",
            retry_after_ms=0,
        ),
        arbitration=ArbitrationResult(
            resolved_elements=(),
            resolved_labels=(),
            conflicts_detected=0,
            conflicts_resolved=0,
            temporal_blocked=False,
            overall_confidence=1.0,
        ),
        source_result=source,
        perception_ms=0.0,
        frame_sequence=1,
        timestamp="1970-01-01T00:00:00+00:00",
    )


def _default_done_fn(decision: Decision) -> bool:
    """Consider the task done when action_type is a completion verb or task_status is complete."""
    if decision.action_type in _DONE_ACTION_TYPES:
        return True
    task_status = getattr(decision, "task_status", "in_progress")
    return task_status == "complete"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _map_status(status: TaskStatus) -> str:
    """Map TaskStatus to DB-allowed task status string."""
    return {
        "running": "running",
        "completed": "success",
        "failed": "failed",
        "suspended": "aborted",
        "cancelled": "aborted",
    }.get(status, "failed")


def _build_summary(
    ctx: TaskContext,
    last_decision: Decision | None,
    error: str | None,
) -> str:
    status_label = ctx.status.upper()
    steps = ctx.action_count
    cost = f"${ctx.total_cost_usd:.4f}"
    last = f" last_action={last_decision.action_type}" if last_decision else ""
    err = f" error={error!r}" if error else ""
    return f"{status_label} steps={steps} cost={cost}{last}{err}"


# ---------------------------------------------------------------------------
# Verification policy lookup
# ---------------------------------------------------------------------------


def get_verification_policy(
    action_type: str,
    settings: NexusSettings,
) -> VerificationPolicy:
    """
    Return the VerificationPolicy for *action_type* based on settings.

    Maps VerificationSettings method strings to VerificationPolicy objects.
    """
    if action_type in _SKIP_VERIFY_ACTIONS:
        return VerificationPolicy.skip()

    method_map: dict[str, str] = {
        "click": settings.verification.click_method,
        "type": settings.verification.type_method,
        "select": settings.verification.click_method,
        "sheet_write": settings.verification.sheet_write_method,
        "row_write": settings.verification.row_write_method,
        "field_replace": settings.verification.field_replace_method,
        "form_submit": settings.verification.form_submit_method,
        "navigate": settings.verification.navigate_method,
    }
    method = method_map.get(action_type, "visual")

    if "source" in method:
        return VerificationPolicy.source()
    if method == "none":
        return VerificationPolicy.skip()
    return VerificationPolicy.visual()
