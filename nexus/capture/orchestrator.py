"""
nexus/capture/orchestrator.py
Capture Orchestrator — ties together session detection, stabilization,
dirty-region analysis, and the frozen-frame watchdog into one callable API.

Flow for get_stable_frame()
---------------------------
1. Session check  — PolicyBlockedError on LOCKED / SECURE_DESKTOP
2. Memory check   — log warning + call _on_pressure_fn when over budget
3. Stabilization  — StabilizationGate.wait_for_stable()
4. Frame fetch    — _get_frame_fn() → Frame
5. Dirty regions  — DirtyRegionDetector.detect() (skipped on first call or
                    force_full_refresh=True)
6. Frozen check   — FrozenFrameWatchdog.check() → FrozenScreenError
7. Assemble       — return StableFrame

All platform and timing dependencies are injectable so the orchestrator is
fully testable without real hardware.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from nexus.capture.dirty_region import DirtyRegionDetector, DirtyRegions
from nexus.capture.frame import Frame
from nexus.capture.session_detector import (
    CaptureDecision,
    CapturePolicy,
    FrozenFrameWatchdog,
    FrozenScreenError as _LocalFrozenScreenError,
    SessionDetector,
    SessionInfo,
)
from nexus.capture.stabilization import StabilizationGate, StabilizationResult
from nexus.core.errors import (
    CaptureError,
    FrozenScreenError,
    PolicyBlockedError,
)
from nexus.core.settings import CaptureSettings
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default memory probe — uses psutil when available
# ---------------------------------------------------------------------------

_DEFAULT_MAX_MEMORY_BYTES: int = 512 * 1024 * 1024  # 512 MB


def _default_memory_bytes() -> int:
    """Return current process RSS in bytes.  Returns 0 when psutil is absent."""
    try:
        import psutil  # type: ignore[import-untyped]

        return psutil.Process().memory_info().rss
    except Exception:
        return 0


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# StableFrame value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StableFrame:
    """
    A fully qualified stable screen frame ready for downstream processing.

    Attributes
    ----------
    frame:
        The stable screen frame.
    prev_frame:
        The immediately preceding frame, or ``None`` on the first capture.
    dirty_regions:
        Changed regions between *prev_frame* and *frame*.  ``None`` when no
        previous frame is available or ``force_full_refresh=True`` was passed.
    session:
        Session snapshot at capture time.
    stabilization_result:
        Detailed output from the stabilization gate (stable flag, wait time,
        reason, final change ratio).
    captured_at:
        ISO-8601 UTC timestamp string recorded when the frame was acquired.
    """

    frame: Frame
    prev_frame: Frame | None
    dirty_regions: DirtyRegions | None
    session: SessionInfo
    stabilization_result: StabilizationResult
    captured_at: str


# ---------------------------------------------------------------------------
# CaptureOrchestrator
# ---------------------------------------------------------------------------


class CaptureOrchestrator:
    """
    Coordinates the full capture pipeline for one stable-frame request.

    Parameters
    ----------
    settings:
        ``CaptureSettings`` section from the root config.
    _get_frame_fn:
        ``() -> Frame | None`` — returns the latest raw frame.  Typically
        wired to ``CaptureWorkerClient.get_latest_frame``.
    _session_detector:
        Provides ``SessionInfo``.  Defaults to ``SessionDetector()`` (real
        Windows API calls).
    _dirty_detector:
        Performs block-level dirty-region analysis between frames.  Defaults
        to ``DirtyRegionDetector(block_size=settings.dirty_region_block_size)``.
    _stabilization_gate:
        Waits until the screen settles.  Defaults to
        ``StabilizationGate(_get_frame_fn=_get_frame_fn)``.
    _frozen_watchdog:
        Raises ``FrozenScreenError`` when the screen freezes in a NORMAL
        session.  Defaults to ``FrozenFrameWatchdog()``.
    _memory_fn:
        ``() -> int`` — current process RAM in bytes.  Defaults to a
        psutil-based probe (returns 0 when psutil is absent).
    _max_memory_bytes:
        Memory cap in bytes.  When exceeded, ``_on_pressure_fn`` is called
        and a warning is logged.  Default: 512 MB.
    _on_pressure_fn:
        ``() -> None`` — called when memory exceeds the cap.  Typically
        reduces the frame buffer.  No-op when ``None``.
    _utc_now_fn:
        ``() -> str`` — returns the current UTC time as an ISO-8601 string.
        Defaults to ``datetime.now(timezone.utc).isoformat()``.
    """

    def __init__(
        self,
        settings: CaptureSettings,
        *,
        _get_frame_fn: Callable[[], Frame | None] | None = None,
        _session_detector: SessionDetector | None = None,
        _dirty_detector: DirtyRegionDetector | None = None,
        _stabilization_gate: StabilizationGate | None = None,
        _frozen_watchdog: FrozenFrameWatchdog | None = None,
        _memory_fn: Callable[[], int] | None = None,
        _max_memory_bytes: int = _DEFAULT_MAX_MEMORY_BYTES,
        _on_pressure_fn: Callable[[], None] | None = None,
        _utc_now_fn: Callable[[], str] | None = None,
    ) -> None:
        self._settings = settings
        self._get_frame: Callable[[], Frame | None] = _get_frame_fn or (lambda: None)
        self._session_detector: SessionDetector = _session_detector or SessionDetector()
        self._dirty_detector: DirtyRegionDetector = _dirty_detector or DirtyRegionDetector(
            block_size=settings.dirty_region_block_size
        )
        self._stabilization_gate: StabilizationGate = (
            _stabilization_gate
            or StabilizationGate(_get_frame_fn=self._get_frame)
        )
        self._frozen_watchdog: FrozenFrameWatchdog = _frozen_watchdog or FrozenFrameWatchdog()
        self._memory_fn: Callable[[], int] = _memory_fn or _default_memory_bytes
        self._max_memory_bytes: int = _max_memory_bytes
        self._on_pressure_fn: Callable[[], None] | None = _on_pressure_fn
        self._utc_now: Callable[[], str] = _utc_now_fn or _utc_iso_now

        self._capture_policy = CapturePolicy()
        self._prev_frame: Frame | None = None

        self._metrics: dict[str, Any] = {
            "total_calls": 0,
            "stable_frames": 0,
            "policy_blocked": 0,
            "frozen_errors": 0,
            "last_change_ratio": 0.0,
            "last_waited_ms": 0.0,
            "memory_bytes": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stable_frame(self, force_full_refresh: bool = False) -> StableFrame:
        """
        Run the full capture pipeline and return a stable frame.

        Parameters
        ----------
        force_full_refresh:
            Skip dirty-region computation; ``StableFrame.dirty_regions``
            will be ``None`` regardless of the previous frame.

        Returns
        -------
        StableFrame
            Always returned when the session is capturable and the screen
            is not frozen.

        Raises
        ------
        PolicyBlockedError
            When the session is LOCKED or SECURE_DESKTOP.
        FrozenScreenError
            When the screen has been unchanged for ≥ the watchdog threshold
            in a NORMAL session.
        CaptureError
            When no frame is available after stabilization.
        """
        self._metrics["total_calls"] += 1

        # ----------------------------------------------------------------
        # Step 1 — Session check
        # ----------------------------------------------------------------
        session = self._session_detector.get_session_info()
        decision = self._capture_policy.should_capture(session)

        if decision in (CaptureDecision.SUSPEND, CaptureDecision.SUSPEND_NOTIFY):
            self._metrics["policy_blocked"] += 1
            _log.info(
                "capture_policy_blocked",
                session_type=session.session_type.value,
                decision=decision.value,
            )
            raise PolicyBlockedError(
                f"Capture suspended: {session.session_type.value}",
                rule=f"session_{session.session_type.value}",
                severity="block",
            )

        # ----------------------------------------------------------------
        # Step 2 — Memory check
        # ----------------------------------------------------------------
        mem = self._memory_fn()
        self._metrics["memory_bytes"] = mem
        if mem > self._max_memory_bytes:
            _log.warning(
                "memory_pressure",
                memory_bytes=mem,
                max_bytes=self._max_memory_bytes,
            )
            if self._on_pressure_fn is not None:
                self._on_pressure_fn()

        # ----------------------------------------------------------------
        # Step 3 — Stabilization
        # ----------------------------------------------------------------
        stab_result = self._stabilization_gate.wait_for_stable(
            timeout_ms=float(self._settings.stabilization_timeout_ms),
            poll_ms=float(self._settings.stabilization_poll_ms),
        )
        self._metrics["last_change_ratio"] = stab_result.change_ratio_final
        self._metrics["last_waited_ms"] = stab_result.waited_ms

        # ----------------------------------------------------------------
        # Step 4 — Frame fetch
        # ----------------------------------------------------------------
        frame = self._get_frame()
        if frame is None:
            raise CaptureError("No frame available after stabilization")

        # ----------------------------------------------------------------
        # Step 5 — Dirty regions
        # ----------------------------------------------------------------
        dirty_regions: DirtyRegions | None = None
        if self._prev_frame is not None and not force_full_refresh:
            dirty_regions = self._dirty_detector.detect(self._prev_frame, frame)

        # ----------------------------------------------------------------
        # Step 6 — Frozen watchdog check
        # ----------------------------------------------------------------
        try:
            self._frozen_watchdog.check(frame, session)
        except _LocalFrozenScreenError as exc:
            self._metrics["frozen_errors"] += 1
            _log.warning(
                "frozen_screen_in_orchestrator",
                frozen_ms=exc.frozen_ms,
            )
            raise FrozenScreenError(
                f"Screen frozen for {exc.frozen_ms:.0f} ms",
                context={"frozen_ms": exc.frozen_ms},
            ) from exc

        # ----------------------------------------------------------------
        # Step 7 — Assemble StableFrame
        # ----------------------------------------------------------------
        stable_frame = StableFrame(
            frame=frame,
            prev_frame=self._prev_frame,
            dirty_regions=dirty_regions,
            session=session,
            stabilization_result=stab_result,
            captured_at=self._utc_now(),
        )

        self._prev_frame = frame
        self._metrics["stable_frames"] += 1

        _log.debug(
            "stable_frame_captured",
            stable=stab_result.stable,
            reason=stab_result.reason,
            change_ratio=round(stab_result.change_ratio_final, 4),
            waited_ms=round(stab_result.waited_ms, 1),
            dirty_blocks=len(dirty_regions.blocks) if dirty_regions else 0,
        )

        return stable_frame

    def get_frame_for_debug(self) -> Frame:
        """
        Return the latest raw frame without policy checks or stabilization.

        Raises
        ------
        CaptureError
            When no frame is currently available.
        """
        frame = self._get_frame()
        if frame is None:
            raise CaptureError("No frame available for debug")
        return frame

    def get_metrics(self) -> dict[str, Any]:
        """
        Return a snapshot of internal counters.

        Keys
        ----
        total_calls:       Number of ``get_stable_frame`` invocations.
        stable_frames:     Successful StableFrame returns.
        policy_blocked:    PolicyBlockedError raises.
        frozen_errors:     FrozenScreenError raises.
        last_change_ratio: Pixel-change ratio from the last stabilization.
        last_waited_ms:    Wait time from the last stabilization.
        memory_bytes:      RAM usage measured on the last call.
        """
        return dict(self._metrics)
