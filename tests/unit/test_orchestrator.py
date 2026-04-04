"""
tests/unit/test_orchestrator.py
Unit tests for nexus/capture/orchestrator.py — Faz 21.

All platform dependencies (frame source, session, stabilization gate,
watchdog, memory probe) are injected via stubs so no real hardware or
OS calls are made.

Sections:
  1. StableFrame value object
  2. Normal flow — happy path
  3. Policy blocked — LOCKED and SECURE_DESKTOP sessions
  4. Frozen screen — FrozenScreenError propagation
  5. Dirty regions — computation and force_full_refresh flag
  6. Memory budget — pressure callback and metrics
  7. get_frame_for_debug
  8. get_metrics
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest

from nexus.capture.frame import Frame
from nexus.capture.orchestrator import (
    CaptureOrchestrator,
    StableFrame,
    _DEFAULT_MAX_MEMORY_BYTES,
)
from nexus.capture.session_detector import (
    FrozenScreenError as _LocalFrozenScreenError,
    SessionDetector,
    SessionInfo,
    SessionType,
)
from nexus.capture.stabilization import StabilizationGate, StabilizationResult
from nexus.core.errors import CaptureError, FrozenScreenError, PolicyBlockedError
from nexus.core.settings import CaptureSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC_STAMP = "2026-04-04T00:00:00+00:00"
_DEFAULT_SETTINGS = CaptureSettings()


def _make_frame(
    color: tuple[int, int, int] = (100, 100, 100),
    width: int = 16,
    height: int = 16,
    seq: int = 1,
) -> Frame:
    data = np.full((height, width, 3), color, dtype=np.uint8)
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=0.0,
        captured_at_utc=_UTC_STAMP,
        sequence_number=seq,
    )


def _make_session(
    session_type: SessionType = SessionType.NORMAL,
    foreground_window: str = "TestApp",
) -> SessionInfo:
    return SessionInfo(
        session_type=session_type,
        is_minimized=session_type is SessionType.RDP_MINIMIZED,
        is_locked=session_type is SessionType.LOCKED,
        is_secure_desktop=session_type is SessionType.SECURE_DESKTOP,
        foreground_window=foreground_window,
    )


class _StubGate:
    """StabilizationGate stub that returns a fixed StabilizationResult."""

    def __init__(
        self,
        stable: bool = True,
        reason: str = "stable",
        change_ratio: float = 0.0,
        waited_ms: float = 5.0,
    ) -> None:
        self._result = StabilizationResult(
            stable=stable,
            waited_ms=waited_ms,
            reason=reason,
            change_ratio_final=change_ratio,
        )

    def wait_for_stable(self, **_: object) -> StabilizationResult:
        return self._result


class _RaisingWatchdog:
    """FrozenFrameWatchdog stub that always raises LocalFrozenScreenError."""

    def __init__(self, frozen_ms: float = 5001.0) -> None:
        self._ms = frozen_ms

    def check(self, frame: Frame, session: SessionInfo) -> None:  # noqa: ARG002
        raise _LocalFrozenScreenError(self._ms)


class _PassingWatchdog:
    """FrozenFrameWatchdog stub that never raises."""

    def check(self, frame: Frame, session: SessionInfo) -> None:  # noqa: ARG002
        pass


_MISSING = object()  # sentinel for "no frame argument provided"


def _make_orchestrator(
    *,
    frame: Frame | None = _MISSING,  # type: ignore[assignment]
    frames: list[Frame | None] | None = None,
    session_type: SessionType = SessionType.NORMAL,
    stable: bool = True,
    reason: str = "stable",
    change_ratio: float = 0.0,
    watchdog: object | None = None,
    memory_bytes: int = 0,
    max_memory_bytes: int = _DEFAULT_MAX_MEMORY_BYTES,
    on_pressure: Callable[[], None] | None = None,
    settings: CaptureSettings | None = None,
) -> CaptureOrchestrator:
    """
    Build a CaptureOrchestrator with all platform calls stubbed out.

    Pass *frame* for a constant frame source (use ``frame=None`` to simulate
    no frame), or *frames* for a sequential source (last item repeated when
    exhausted).  When *frame* is omitted, a default frame is used.
    """
    if frames is not None:
        _iter = iter(frames)
        _sentinel = frames[-1] if frames else None

        def _get_frame() -> Frame | None:
            try:
                return next(_iter)
            except StopIteration:
                return _sentinel

    elif frame is _MISSING:
        # No argument supplied — use a default frame
        _default = _make_frame()
        _get_frame = lambda: _default  # noqa: E731
    else:
        # Explicit frame value (may be None to simulate no-frame condition)
        _get_frame = lambda: frame  # noqa: E731

    _session = _make_session(session_type)
    _session_det = SessionDetector(
        _get_title_fn=lambda: _session.foreground_window,
        _is_locked_fn=lambda: _session.is_locked,
        _is_secure_desktop_fn=lambda: _session.is_secure_desktop,
        _is_minimized_fn=lambda: _session.is_minimized,
    )

    _wdog: object = watchdog if watchdog is not None else _PassingWatchdog()

    return CaptureOrchestrator(
        settings=settings or _DEFAULT_SETTINGS,
        _get_frame_fn=_get_frame,
        _session_detector=_session_det,
        _stabilization_gate=_StubGate(stable=stable, reason=reason, change_ratio=change_ratio),
        _frozen_watchdog=_wdog,  # type: ignore[arg-type]
        _memory_fn=lambda: memory_bytes,
        _max_memory_bytes=max_memory_bytes,
        _on_pressure_fn=on_pressure,
        _utc_now_fn=lambda: _UTC_STAMP,
    )


# ---------------------------------------------------------------------------
# 1. StableFrame value object
# ---------------------------------------------------------------------------


class TestStableFrameValueObject:
    def _make_stable(self) -> StableFrame:
        f = _make_frame()
        stab = StabilizationResult(
            stable=True, waited_ms=5.0, reason="stable", change_ratio_final=0.0
        )
        return StableFrame(
            frame=f,
            prev_frame=None,
            dirty_regions=None,
            session=_make_session(),
            stabilization_result=stab,
            captured_at=_UTC_STAMP,
        )

    def test_frozen(self) -> None:
        sf = self._make_stable()
        with pytest.raises((AttributeError, TypeError)):
            sf.frame = _make_frame()  # type: ignore[misc]

    def test_fields(self) -> None:
        sf = self._make_stable()
        assert isinstance(sf.frame, Frame)
        assert sf.prev_frame is None
        assert sf.dirty_regions is None
        assert isinstance(sf.session, SessionInfo)
        assert isinstance(sf.stabilization_result, StabilizationResult)
        assert isinstance(sf.captured_at, str)

    def test_stabilization_result_embedded(self) -> None:
        sf = self._make_stable()
        assert sf.stabilization_result.stable is True
        assert sf.stabilization_result.reason == "stable"


# ---------------------------------------------------------------------------
# 2. Normal flow — happy path
# ---------------------------------------------------------------------------


class TestNormalFlow:
    def test_returns_stable_frame(self) -> None:
        orch = _make_orchestrator()
        result = orch.get_stable_frame()
        assert isinstance(result, StableFrame)

    def test_frame_is_from_get_frame_fn(self) -> None:
        f = _make_frame(color=(10, 20, 30))
        orch = _make_orchestrator(frame=f)
        result = orch.get_stable_frame()
        assert result.frame is f

    def test_session_populated(self) -> None:
        orch = _make_orchestrator()
        result = orch.get_stable_frame()
        assert result.session.session_type is SessionType.NORMAL

    def test_stabilization_result_passed_through(self) -> None:
        orch = _make_orchestrator(stable=True, reason="stable", change_ratio=0.01)
        result = orch.get_stable_frame()
        assert result.stabilization_result.stable is True
        assert result.stabilization_result.reason == "stable"
        assert result.stabilization_result.change_ratio_final == pytest.approx(0.01)

    def test_captured_at_is_injected_timestamp(self) -> None:
        orch = _make_orchestrator()
        result = orch.get_stable_frame()
        assert result.captured_at == _UTC_STAMP

    def test_first_call_no_prev_frame(self) -> None:
        orch = _make_orchestrator()
        result = orch.get_stable_frame()
        assert result.prev_frame is None

    def test_second_call_has_prev_frame(self) -> None:
        f1 = _make_frame(seq=1)
        f2 = _make_frame(seq=2)
        orch = _make_orchestrator(frames=[f1, f2])
        orch.get_stable_frame()
        result = orch.get_stable_frame()
        assert result.prev_frame is f1

    def test_rdp_minimized_session_does_not_raise(self) -> None:
        """RDP_MINIMIZED → START_WATCHDOG → proceed (not blocked)."""
        orch = _make_orchestrator(session_type=SessionType.RDP_MINIMIZED)
        result = orch.get_stable_frame()
        assert isinstance(result, StableFrame)

    def test_unstable_result_still_returns_stable_frame(self) -> None:
        """Timeout from stabilization gate still yields a StableFrame."""
        orch = _make_orchestrator(stable=False, reason="timeout")
        result = orch.get_stable_frame()
        assert isinstance(result, StableFrame)
        assert result.stabilization_result.stable is False

    def test_no_frame_raises_capture_error(self) -> None:
        orch = _make_orchestrator(frame=None)
        with pytest.raises(CaptureError):
            orch.get_stable_frame()


# ---------------------------------------------------------------------------
# 3. Policy blocked — LOCKED and SECURE_DESKTOP sessions
# ---------------------------------------------------------------------------


class TestPolicyBlocked:
    def test_locked_raises_policy_blocked(self) -> None:
        orch = _make_orchestrator(session_type=SessionType.LOCKED)
        with pytest.raises(PolicyBlockedError):
            orch.get_stable_frame()

    def test_secure_desktop_raises_policy_blocked(self) -> None:
        orch = _make_orchestrator(session_type=SessionType.SECURE_DESKTOP)
        with pytest.raises(PolicyBlockedError):
            orch.get_stable_frame()

    def test_locked_rule_identifier(self) -> None:
        orch = _make_orchestrator(session_type=SessionType.LOCKED)
        with pytest.raises(PolicyBlockedError) as exc_info:
            orch.get_stable_frame()
        assert "locked" in exc_info.value.rule

    def test_secure_desktop_rule_identifier(self) -> None:
        orch = _make_orchestrator(session_type=SessionType.SECURE_DESKTOP)
        with pytest.raises(PolicyBlockedError) as exc_info:
            orch.get_stable_frame()
        assert "secure" in exc_info.value.rule

    def test_policy_blocked_is_not_recoverable(self) -> None:
        orch = _make_orchestrator(session_type=SessionType.LOCKED)
        with pytest.raises(PolicyBlockedError) as exc_info:
            orch.get_stable_frame()
        assert exc_info.value.recoverable is False

    def test_policy_blocked_severity(self) -> None:
        orch = _make_orchestrator(session_type=SessionType.LOCKED)
        with pytest.raises(PolicyBlockedError) as exc_info:
            orch.get_stable_frame()
        assert exc_info.value.severity == "block"

    def test_metrics_count_policy_blocks(self) -> None:
        orch = _make_orchestrator(session_type=SessionType.LOCKED)
        for _ in range(3):
            with pytest.raises(PolicyBlockedError):
                orch.get_stable_frame()
        assert orch.get_metrics()["policy_blocked"] == 3

    def test_total_calls_incremented_before_block(self) -> None:
        """total_calls must be incremented even when PolicyBlockedError is raised."""
        orch = _make_orchestrator(session_type=SessionType.LOCKED)
        with pytest.raises(PolicyBlockedError):
            orch.get_stable_frame()
        assert orch.get_metrics()["total_calls"] == 1


# ---------------------------------------------------------------------------
# 4. Frozen screen — FrozenScreenError propagation
# ---------------------------------------------------------------------------


class TestFrozenScreen:
    def test_frozen_watchdog_raises_frozen_screen_error(self) -> None:
        orch = _make_orchestrator(watchdog=_RaisingWatchdog(frozen_ms=5001.0))
        with pytest.raises(FrozenScreenError):
            orch.get_stable_frame()

    def test_frozen_screen_error_is_capture_error(self) -> None:
        """FrozenScreenError(CaptureError) must satisfy isinstance(CaptureError)."""
        orch = _make_orchestrator(watchdog=_RaisingWatchdog())
        with pytest.raises(FrozenScreenError) as exc_info:
            orch.get_stable_frame()
        assert isinstance(exc_info.value, CaptureError)

    def test_frozen_screen_error_context_has_frozen_ms(self) -> None:
        orch = _make_orchestrator(watchdog=_RaisingWatchdog(frozen_ms=7500.0))
        with pytest.raises(FrozenScreenError) as exc_info:
            orch.get_stable_frame()
        assert exc_info.value.context.get("frozen_ms", 0) >= 7000.0

    def test_frozen_error_increments_metric(self) -> None:
        orch = _make_orchestrator(watchdog=_RaisingWatchdog())
        for _ in range(2):
            with pytest.raises(FrozenScreenError):
                orch.get_stable_frame()
        assert orch.get_metrics()["frozen_errors"] == 2

    def test_passing_watchdog_does_not_raise(self) -> None:
        orch = _make_orchestrator(watchdog=_PassingWatchdog())
        result = orch.get_stable_frame()
        assert isinstance(result, StableFrame)


# ---------------------------------------------------------------------------
# 5. Dirty regions — computation and force_full_refresh flag
# ---------------------------------------------------------------------------


class TestDirtyRegions:
    def test_first_call_no_dirty_regions(self) -> None:
        orch = _make_orchestrator()
        result = orch.get_stable_frame()
        assert result.dirty_regions is None

    def test_second_call_computes_dirty_regions(self) -> None:
        orch = _make_orchestrator()
        orch.get_stable_frame()
        result = orch.get_stable_frame()
        assert result.dirty_regions is not None

    def test_identical_frames_zero_dirty_blocks(self) -> None:
        f = _make_frame(color=(128, 128, 128))
        orch = _make_orchestrator(frame=f)
        orch.get_stable_frame()
        result = orch.get_stable_frame()
        assert result.dirty_regions is not None
        assert result.dirty_regions.change_ratio == pytest.approx(0.0)
        assert result.dirty_regions.blocks == ()

    def test_force_full_refresh_skips_dirty_regions(self) -> None:
        orch = _make_orchestrator()
        orch.get_stable_frame()  # establishes prev_frame
        result = orch.get_stable_frame(force_full_refresh=True)
        assert result.dirty_regions is None

    def test_fully_changed_frames_full_refresh(self) -> None:
        f1 = _make_frame(color=(0, 0, 0))
        f2 = _make_frame(color=(255, 255, 255))
        orch = _make_orchestrator(frames=[f1, f2])
        orch.get_stable_frame()
        result = orch.get_stable_frame()
        assert result.dirty_regions is not None
        assert result.dirty_regions.full_refresh is True

    def test_dirty_regions_blocks_are_rects(self) -> None:
        from nexus.core.types import Rect

        f1 = _make_frame(color=(0, 0, 0))
        f2 = _make_frame(color=(255, 255, 255))
        orch = _make_orchestrator(frames=[f1, f2])
        orch.get_stable_frame()
        result = orch.get_stable_frame()
        assert all(isinstance(b, Rect) for b in result.dirty_regions.blocks)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 6. Memory budget — pressure callback and metrics
# ---------------------------------------------------------------------------


class TestMemoryBudget:
    _LIMIT = 256 * 1024 * 1024  # 256 MB

    def test_under_budget_no_callback(self) -> None:
        called: list[bool] = []
        orch = _make_orchestrator(
            memory_bytes=100 * 1024 * 1024,
            max_memory_bytes=self._LIMIT,
            on_pressure=lambda: called.append(True),
        )
        orch.get_stable_frame()
        assert called == []

    def test_over_budget_triggers_callback(self) -> None:
        called: list[bool] = []
        orch = _make_orchestrator(
            memory_bytes=400 * 1024 * 1024,
            max_memory_bytes=self._LIMIT,
            on_pressure=lambda: called.append(True),
        )
        orch.get_stable_frame()
        assert called == [True]

    def test_over_budget_still_returns_stable_frame(self) -> None:
        """Memory pressure must not block capture — graceful degradation."""
        orch = _make_orchestrator(
            memory_bytes=400 * 1024 * 1024,
            max_memory_bytes=self._LIMIT,
        )
        result = orch.get_stable_frame()
        assert isinstance(result, StableFrame)

    def test_no_callback_provided_over_budget_no_error(self) -> None:
        orch = _make_orchestrator(
            memory_bytes=400 * 1024 * 1024,
            max_memory_bytes=self._LIMIT,
            on_pressure=None,
        )
        result = orch.get_stable_frame()  # must not raise
        assert isinstance(result, StableFrame)

    def test_metrics_record_memory_bytes(self) -> None:
        mb = 123 * 1024 * 1024
        orch = _make_orchestrator(memory_bytes=mb)
        orch.get_stable_frame()
        assert orch.get_metrics()["memory_bytes"] == mb

    def test_callback_called_each_over_budget_call(self) -> None:
        count: list[int] = [0]
        orch = _make_orchestrator(
            memory_bytes=400 * 1024 * 1024,
            max_memory_bytes=self._LIMIT,
            on_pressure=lambda: count.__setitem__(0, count[0] + 1),
        )
        orch.get_stable_frame()
        orch.get_stable_frame()
        assert count[0] == 2


# ---------------------------------------------------------------------------
# 7. get_frame_for_debug
# ---------------------------------------------------------------------------


class TestGetFrameForDebug:
    def test_returns_frame_when_available(self) -> None:
        f = _make_frame(color=(42, 42, 42))
        orch = _make_orchestrator(frame=f)
        assert orch.get_frame_for_debug() is f

    def test_raises_when_no_frame(self) -> None:
        orch = _make_orchestrator(frame=None)
        with pytest.raises(CaptureError):
            orch.get_frame_for_debug()

    def test_does_not_modify_metrics(self) -> None:
        orch = _make_orchestrator()
        orch.get_frame_for_debug()
        assert orch.get_metrics()["total_calls"] == 0

    def test_bypasses_session_policy(self) -> None:
        """get_frame_for_debug must not check session policy."""
        orch = _make_orchestrator(
            session_type=SessionType.LOCKED,
            frame=_make_frame(),
        )
        result = orch.get_frame_for_debug()
        assert isinstance(result, Frame)


# ---------------------------------------------------------------------------
# 8. get_metrics
# ---------------------------------------------------------------------------


class TestGetMetrics:
    def test_initial_all_zero(self) -> None:
        orch = _make_orchestrator()
        m = orch.get_metrics()
        assert m["total_calls"] == 0
        assert m["stable_frames"] == 0
        assert m["policy_blocked"] == 0
        assert m["frozen_errors"] == 0

    def test_successful_call_increments_stable_frames(self) -> None:
        orch = _make_orchestrator()
        orch.get_stable_frame()
        m = orch.get_metrics()
        assert m["total_calls"] == 1
        assert m["stable_frames"] == 1

    def test_returns_copy_not_reference(self) -> None:
        orch = _make_orchestrator()
        m = orch.get_metrics()
        m["total_calls"] = 999
        assert orch.get_metrics()["total_calls"] == 0

    def test_last_change_ratio_updated(self) -> None:
        orch = _make_orchestrator(change_ratio=0.123)
        orch.get_stable_frame()
        assert orch.get_metrics()["last_change_ratio"] == pytest.approx(0.123)

    def test_multiple_calls_tracked(self) -> None:
        orch = _make_orchestrator()
        for _ in range(5):
            orch.get_stable_frame()
        m = orch.get_metrics()
        assert m["total_calls"] == 5
        assert m["stable_frames"] == 5

    def test_all_expected_keys_present(self) -> None:
        orch = _make_orchestrator()
        m = orch.get_metrics()
        expected_keys = {
            "total_calls",
            "stable_frames",
            "policy_blocked",
            "frozen_errors",
            "last_change_ratio",
            "last_waited_ms",
            "memory_bytes",
        }
        assert expected_keys.issubset(m.keys())
