"""
tests/unit/test_session_detector.py
Unit tests for nexus/capture/session_detector.py — Faz 20.

Sections:
  1. SessionInfo value object
  2. FrozenScreenError
  3. CapturePolicy — all session type → decision mappings
  4. SessionDetector — injectable platform functions
  5. FrozenFrameWatchdog — frozen detection and timer reset
"""
from __future__ import annotations

import numpy as np
import pytest

from nexus.capture.frame import Frame
from nexus.capture.session_detector import (
    CaptureDecision,
    CapturePolicy,
    FrozenFrameWatchdog,
    FrozenScreenError,
    SessionDetector,
    SessionInfo,
    SessionType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(
    color: tuple[int, int, int] = (100, 100, 100),
    width: int = 10,
    height: int = 10,
    seq: int = 1,
) -> Frame:
    data = np.full((height, width, 3), color, dtype=np.uint8)
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=0.0,
        captured_at_utc="",
        sequence_number=seq,
    )


def _session(
    session_type: SessionType = SessionType.NORMAL,
    *,
    is_minimized: bool = False,
    is_locked: bool = False,
    is_secure_desktop: bool = False,
    foreground_window: str = "",
) -> SessionInfo:
    return SessionInfo(
        session_type=session_type,
        is_minimized=is_minimized,
        is_locked=is_locked,
        is_secure_desktop=is_secure_desktop,
        foreground_window=foreground_window,
    )


class _FakeTimer:
    """Controllable monotonic clock for watchdog tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


# ---------------------------------------------------------------------------
# 1. SessionInfo value object
# ---------------------------------------------------------------------------


class TestSessionInfo:
    def test_frozen(self) -> None:
        si = _session()
        with pytest.raises((AttributeError, TypeError)):
            si.session_type = SessionType.LOCKED  # type: ignore[misc]

    def test_fields(self) -> None:
        si = SessionInfo(
            session_type=SessionType.LOCKED,
            is_minimized=True,
            is_locked=True,
            is_secure_desktop=False,
            foreground_window="LockApp",
        )
        assert si.session_type is SessionType.LOCKED
        assert si.is_minimized is True
        assert si.is_locked is True
        assert si.is_secure_desktop is False
        assert si.foreground_window == "LockApp"

    def test_normal_session_defaults(self) -> None:
        si = _session()
        assert si.session_type is SessionType.NORMAL
        assert si.is_locked is False
        assert si.is_secure_desktop is False
        assert si.is_minimized is False


# ---------------------------------------------------------------------------
# 2. FrozenScreenError
# ---------------------------------------------------------------------------


class TestFrozenScreenError:
    def test_attributes(self) -> None:
        err = FrozenScreenError(5123.4)
        assert err.frozen_ms == pytest.approx(5123.4)

    def test_str_contains_ms(self) -> None:
        err = FrozenScreenError(5000.0)
        assert "5000" in str(err)

    def test_is_exception(self) -> None:
        assert isinstance(FrozenScreenError(0.0), Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(FrozenScreenError) as exc_info:
            raise FrozenScreenError(1234.5)
        assert exc_info.value.frozen_ms == pytest.approx(1234.5)


# ---------------------------------------------------------------------------
# 3. CapturePolicy — all session type → decision mappings
# ---------------------------------------------------------------------------


class TestCapturePolicy:
    _policy = CapturePolicy()

    def test_normal_returns_capture(self) -> None:
        assert self._policy.should_capture(_session()) is CaptureDecision.CAPTURE

    def test_locked_returns_suspend_notify(self) -> None:
        assert (
            self._policy.should_capture(
                _session(SessionType.LOCKED, is_locked=True)
            )
            is CaptureDecision.SUSPEND_NOTIFY
        )

    def test_secure_desktop_returns_suspend(self) -> None:
        assert (
            self._policy.should_capture(
                _session(SessionType.SECURE_DESKTOP, is_secure_desktop=True)
            )
            is CaptureDecision.SUSPEND
        )

    def test_rdp_minimized_returns_start_watchdog(self) -> None:
        assert (
            self._policy.should_capture(
                _session(SessionType.RDP_MINIMIZED, is_minimized=True)
            )
            is CaptureDecision.START_WATCHDOG
        )

    def test_locked_is_not_capture(self) -> None:
        decision = self._policy.should_capture(
            _session(SessionType.LOCKED, is_locked=True)
        )
        assert decision is not CaptureDecision.CAPTURE

    def test_secure_desktop_is_not_capture(self) -> None:
        decision = self._policy.should_capture(
            _session(SessionType.SECURE_DESKTOP, is_secure_desktop=True)
        )
        assert decision is not CaptureDecision.CAPTURE

    def test_locked_is_not_silent_suspend(self) -> None:
        """LOCKED must produce SUSPEND_NOTIFY, not the silent SUSPEND."""
        decision = self._policy.should_capture(
            _session(SessionType.LOCKED, is_locked=True)
        )
        assert decision is not CaptureDecision.SUSPEND

    def test_all_decisions_are_enum_members(self) -> None:
        for st in SessionType:
            si = _session(st)
            decision = self._policy.should_capture(si)
            assert isinstance(decision, CaptureDecision)


# ---------------------------------------------------------------------------
# 4. SessionDetector — injectable platform functions
# ---------------------------------------------------------------------------


class TestSessionDetector:
    def _make_detector(
        self,
        *,
        title: str = "App",
        locked: bool = False,
        secure: bool = False,
        minimized: bool = False,
    ) -> SessionDetector:
        return SessionDetector(
            _get_title_fn=lambda: title,
            _is_locked_fn=lambda: locked,
            _is_secure_desktop_fn=lambda: secure,
            _is_minimized_fn=lambda: minimized,
        )

    def test_normal_session(self) -> None:
        det = self._make_detector()
        info = det.get_session_info()
        assert info.session_type is SessionType.NORMAL
        assert info.is_locked is False
        assert info.is_secure_desktop is False
        assert info.is_minimized is False

    def test_locked_session(self) -> None:
        det = self._make_detector(locked=True, title="LockApp")
        info = det.get_session_info()
        assert info.session_type is SessionType.LOCKED
        assert info.is_locked is True
        assert info.foreground_window == "LockApp"

    def test_secure_desktop_session(self) -> None:
        det = self._make_detector(secure=True)
        info = det.get_session_info()
        assert info.session_type is SessionType.SECURE_DESKTOP
        assert info.is_secure_desktop is True

    def test_minimized_session(self) -> None:
        det = self._make_detector(minimized=True)
        info = det.get_session_info()
        assert info.session_type is SessionType.RDP_MINIMIZED
        assert info.is_minimized is True

    def test_locked_takes_priority_over_minimized(self) -> None:
        """When both locked and minimized, result must be LOCKED."""
        det = self._make_detector(locked=True, minimized=True)
        info = det.get_session_info()
        assert info.session_type is SessionType.LOCKED

    def test_locked_takes_priority_over_secure_desktop(self) -> None:
        det = self._make_detector(locked=True, secure=True)
        info = det.get_session_info()
        assert info.session_type is SessionType.LOCKED

    def test_secure_desktop_takes_priority_over_minimized(self) -> None:
        det = self._make_detector(secure=True, minimized=True)
        info = det.get_session_info()
        assert info.session_type is SessionType.SECURE_DESKTOP

    def test_foreground_window_propagated(self) -> None:
        det = self._make_detector(title="My Window")
        info = det.get_session_info()
        assert info.foreground_window == "My Window"

    def test_returns_session_info_instance(self) -> None:
        det = self._make_detector()
        assert isinstance(det.get_session_info(), SessionInfo)


# ---------------------------------------------------------------------------
# 5. FrozenFrameWatchdog — frozen detection and timer reset
# ---------------------------------------------------------------------------


class TestFrozenFrameWatchdog:
    def test_first_frame_never_raises(self) -> None:
        """The very first frame must never trigger FrozenScreenError."""
        timer = _FakeTimer()
        wdog = FrozenFrameWatchdog(threshold_ms=0.0, _time_fn=timer)
        wdog.check(_make_frame(), _session())  # must not raise

    def test_raises_after_threshold(self) -> None:
        timer = _FakeTimer(0.0)
        wdog = FrozenFrameWatchdog(threshold_ms=5000.0, _time_fn=timer)
        f = _make_frame()
        normal = _session()

        wdog.check(f, normal)       # frame 1 — sets baseline at t=0
        timer.advance(4.999)
        wdog.check(f, normal)       # still under threshold → no error
        timer.advance(0.002)        # total 5.001 s
        with pytest.raises(FrozenScreenError) as exc:
            wdog.check(f, normal)
        assert exc.value.frozen_ms >= 5000.0

    def test_no_raise_just_below_threshold(self) -> None:
        timer = _FakeTimer(0.0)
        wdog = FrozenFrameWatchdog(threshold_ms=5000.0, _time_fn=timer)
        f = _make_frame()
        normal = _session()

        wdog.check(f, normal)
        timer.advance(4.999)
        wdog.check(f, normal)  # must not raise

    def test_changing_frames_reset_frozen_timer(self) -> None:
        timer = _FakeTimer(0.0)
        wdog = FrozenFrameWatchdog(threshold_ms=5000.0, _time_fn=timer)
        normal = _session()

        # Feed different frames — timer resets on each change
        for i in range(10):
            timer.advance(1.0)
            wdog.check(_make_frame(color=(i * 20, 0, 0)), normal)
        # 10 s elapsed but frames changed every 1 s → no error

    def test_locked_session_resets_frozen_timer(self) -> None:
        """Non-NORMAL session must reset the timer; no error after unlock."""
        timer = _FakeTimer(0.0)
        wdog = FrozenFrameWatchdog(threshold_ms=5000.0, _time_fn=timer)
        f = _make_frame()
        normal = _session()
        locked = _session(SessionType.LOCKED, is_locked=True)

        wdog.check(f, normal)   # t=0, sets baseline
        timer.advance(6.0)
        wdog.check(f, locked)   # t=6 — locked resets timer, no error
        timer.advance(1.0)
        wdog.check(f, normal)   # t=7 — frozen_s = 1.0 < 5.0, no error

    def test_secure_desktop_resets_timer(self) -> None:
        timer = _FakeTimer(0.0)
        wdog = FrozenFrameWatchdog(threshold_ms=5000.0, _time_fn=timer)
        f = _make_frame()
        normal = _session()
        secure = _session(SessionType.SECURE_DESKTOP, is_secure_desktop=True)

        wdog.check(f, normal)
        timer.advance(6.0)
        wdog.check(f, secure)   # resets timer
        timer.advance(4.0)
        wdog.check(f, normal)   # 4 s since reset < threshold → no error

    def test_rdp_minimized_resets_timer(self) -> None:
        timer = _FakeTimer(0.0)
        wdog = FrozenFrameWatchdog(threshold_ms=5000.0, _time_fn=timer)
        f = _make_frame()
        wdog.check(f, _session())
        timer.advance(6.0)
        wdog.check(f, _session(SessionType.RDP_MINIMIZED, is_minimized=True))
        timer.advance(4.0)
        wdog.check(f, _session())  # no error

    def test_error_message_contains_frozen_ms(self) -> None:
        timer = _FakeTimer(0.0)
        wdog = FrozenFrameWatchdog(threshold_ms=1000.0, _time_fn=timer)
        f = _make_frame()
        normal = _session()
        wdog.check(f, normal)
        timer.advance(1.001)
        with pytest.raises(FrozenScreenError) as exc:
            wdog.check(f, normal)
        assert exc.value.frozen_ms >= 1000.0
        assert "1001" in str(exc.value) or "1000" in str(exc.value)

    def test_frozen_ms_attribute(self) -> None:
        timer = _FakeTimer(0.0)
        wdog = FrozenFrameWatchdog(threshold_ms=5000.0, _time_fn=timer)
        f = _make_frame()
        normal = _session()
        wdog.check(f, normal)
        timer.advance(7.5)
        with pytest.raises(FrozenScreenError) as exc:
            wdog.check(f, normal)
        # frozen_ms should reflect actual time elapsed (~7500 ms)
        assert exc.value.frozen_ms >= 7000.0

    def test_multiple_frames_then_freeze(self) -> None:
        """Screen changes for a while then freezes — error fires at right time."""
        timer = _FakeTimer(0.0)
        wdog = FrozenFrameWatchdog(threshold_ms=5000.0, _time_fn=timer)
        normal = _session()

        # Changing frames for 3 s
        for i in range(3):
            timer.advance(1.0)
            wdog.check(_make_frame(color=(i * 50, 0, 0)), normal)

        # Now freeze
        frozen_frame = _make_frame(color=(150, 0, 0))
        wdog.check(frozen_frame, normal)  # t=3 — new frame, starts frozen timer
        timer.advance(4.999)
        wdog.check(frozen_frame, normal)  # t=7.999 — frozen_s ≈ 4.999 < 5
        timer.advance(0.002)
        with pytest.raises(FrozenScreenError):
            wdog.check(frozen_frame, normal)  # t=8.001 — frozen_s ≈ 5.001 >= 5
