"""
nexus/capture/session_detector.py
Windows session state detection and capture policy.

SessionDetector queries the Windows desktop/session state and returns a
``SessionInfo`` value object.  All platform calls are injectable so the
module is fully testable without a real display.

CapturePolicy translates a ``SessionInfo`` into a ``CaptureDecision``:

  NORMAL          → CAPTURE          (proceed normally)
  LOCKED          → SUSPEND_NOTIFY   (suspend + surface a notification)
  SECURE_DESKTOP  → SUSPEND          (silent suspend — UAC/credential prompts)
  RDP_MINIMIZED   → START_WATCHDOG   (start frozen-frame watchdog)

FrozenFrameWatchdog tracks consecutive unchanged frames.  When the screen
has not changed for ≥ ``threshold_ms`` while the session is NORMAL it
raises ``FrozenScreenError``.  Non-NORMAL sessions reset the frozen timer
so the watchdog does not fire spuriously after an unlock or window restore.
"""
from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nexus.capture.frame import Frame
from nexus.infra.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Session type
# ---------------------------------------------------------------------------


class SessionType(enum.Enum):
    """High-level classification of the Windows desktop session."""

    NORMAL = "normal"
    LOCKED = "locked"
    SECURE_DESKTOP = "secure_desktop"
    RDP_MINIMIZED = "rdp_minimized"


# ---------------------------------------------------------------------------
# SessionInfo value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionInfo:
    """
    Snapshot of the current Windows session state.

    Attributes
    ----------
    session_type:
        High-level session classification (see ``SessionType``).
    is_minimized:
        True when the foreground application is minimised / the desktop
        is not visible (e.g. RDP session minimised on the client side).
    is_locked:
        True when the Windows lock screen is active.
    is_secure_desktop:
        True when a secure desktop is active (UAC prompt, credential
        dialog, Ctrl+Alt+Del screen).
    foreground_window:
        Title of the foreground window, or an empty string when none is
        available.
    """

    session_type: SessionType
    is_minimized: bool
    is_locked: bool
    is_secure_desktop: bool
    foreground_window: str


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FrozenScreenError(Exception):
    """
    Raised when the screen has not changed for longer than the configured
    threshold while the session is NORMAL.

    Attributes
    ----------
    frozen_ms:
        Milliseconds the screen has been frozen when the error was raised.
    """

    def __init__(self, frozen_ms: float) -> None:
        super().__init__(f"Screen frozen for {frozen_ms:.0f} ms")
        self.frozen_ms = frozen_ms


# ---------------------------------------------------------------------------
# Platform helpers (production; injectable in tests)
# ---------------------------------------------------------------------------


def _default_get_foreground_title() -> str:
    """Return the title of the foreground window (Windows ctypes)."""
    try:
        import ctypes

        hwnd = ctypes.windll.user32.GetForegroundWindow()  # type: ignore[attr-defined]
        if not hwnd:
            return ""
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)  # type: ignore[attr-defined]
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)  # type: ignore[attr-defined]
        return buf.value
    except Exception as exc:
        _log.debug("get_foreground_title_failed", error=str(exc))
        return ""


def _default_is_locked() -> bool:
    """Detect Windows lock screen via GetForegroundWindow heuristic."""
    try:
        import ctypes

        hwnd = ctypes.windll.user32.GetForegroundWindow()  # type: ignore[attr-defined]
        if not hwnd:
            return True  # no foreground window → likely locked
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)  # type: ignore[attr-defined]
        buf = ctypes.create_unicode_buffer(max(length + 1, 1))
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)  # type: ignore[attr-defined]
        title = buf.value.lower()
        return "lockapp" in title or "winlogon" in title
    except Exception as exc:
        _log.debug("is_locked_check_failed", error=str(exc))
        return False


def _default_is_secure_desktop() -> bool:
    """
    Detect secure desktop by opening the input desktop and comparing
    its name against ``"Default"``.
    """
    try:
        import ctypes

        _ACCESS_DESKTOP = 0x0001
        hdesk = ctypes.windll.user32.OpenInputDesktop(0, False, _ACCESS_DESKTOP)  # type: ignore[attr-defined]
        if not hdesk:
            return True  # cannot open → likely secure desktop
        buf = ctypes.create_unicode_buffer(256)
        ok = ctypes.windll.user32.GetUserObjectInformationW(  # type: ignore[attr-defined]
            hdesk, 2, buf, ctypes.sizeof(buf), None
        )
        ctypes.windll.user32.CloseDesktop(hdesk)  # type: ignore[attr-defined]
        if not ok:
            return False
        return buf.value.lower() != "default"
    except Exception as exc:
        _log.debug("is_secure_desktop_check_failed", error=str(exc))
        return False


def _default_is_minimized() -> bool:
    """Return True when the foreground window is minimised (iconic)."""
    try:
        import ctypes

        hwnd = ctypes.windll.user32.GetForegroundWindow()  # type: ignore[attr-defined]
        if not hwnd:
            return False
        return bool(ctypes.windll.user32.IsIconic(hwnd))  # type: ignore[attr-defined]
    except Exception as exc:
        _log.debug("is_minimized_check_failed", error=str(exc))
        return False


# ---------------------------------------------------------------------------
# SessionDetector
# ---------------------------------------------------------------------------

_GetTitleFn = Callable[[], str]
_BoolFn = Callable[[], bool]


class SessionDetector:
    """
    Queries the current Windows session state and returns a ``SessionInfo``.

    Parameters
    ----------
    _get_title_fn:
        ``() -> str`` — returns the foreground window title.
    _is_locked_fn:
        ``() -> bool`` — returns True when the lock screen is active.
    _is_secure_desktop_fn:
        ``() -> bool`` — returns True on a secure desktop.
    _is_minimized_fn:
        ``() -> bool`` — returns True when the desktop is minimised.

    All parameters default to real Windows API calls and may be replaced
    with stubs in tests.
    """

    def __init__(
        self,
        *,
        _get_title_fn: _GetTitleFn | None = None,
        _is_locked_fn: _BoolFn | None = None,
        _is_secure_desktop_fn: _BoolFn | None = None,
        _is_minimized_fn: _BoolFn | None = None,
    ) -> None:
        self._get_title: _GetTitleFn = _get_title_fn or _default_get_foreground_title
        self._is_locked: _BoolFn = _is_locked_fn or _default_is_locked
        self._is_secure_desktop: _BoolFn = (
            _is_secure_desktop_fn or _default_is_secure_desktop
        )
        self._is_minimized: _BoolFn = _is_minimized_fn or _default_is_minimized

    def get_session_info(self) -> SessionInfo:
        """
        Sample the current session state and return a ``SessionInfo``.

        Priority order for ``session_type``:
          ``is_locked`` > ``is_secure_desktop`` > ``is_minimized`` > NORMAL
        """
        is_locked = self._is_locked()
        is_secure = self._is_secure_desktop()
        is_minimized = self._is_minimized()
        title = self._get_title()

        if is_locked:
            session_type = SessionType.LOCKED
        elif is_secure:
            session_type = SessionType.SECURE_DESKTOP
        elif is_minimized:
            session_type = SessionType.RDP_MINIMIZED
        else:
            session_type = SessionType.NORMAL

        info = SessionInfo(
            session_type=session_type,
            is_minimized=is_minimized,
            is_locked=is_locked,
            is_secure_desktop=is_secure,
            foreground_window=title,
        )
        _log.debug("session_info_sampled", session_type=info.session_type.value)
        return info


# ---------------------------------------------------------------------------
# CaptureDecision
# ---------------------------------------------------------------------------


class CaptureDecision(enum.Enum):
    """Action returned by ``CapturePolicy.should_capture``."""

    CAPTURE = "capture"
    SUSPEND = "suspend"
    SUSPEND_NOTIFY = "suspend_notify"
    START_WATCHDOG = "start_watchdog"


# ---------------------------------------------------------------------------
# CapturePolicy
# ---------------------------------------------------------------------------


class CapturePolicy:
    """
    Stateless policy that maps a ``SessionInfo`` to a ``CaptureDecision``.

    Rules
    -----
    NORMAL          → CAPTURE
    LOCKED          → SUSPEND_NOTIFY  (suspend + surface notification)
    SECURE_DESKTOP  → SUSPEND         (silent; security-sensitive context)
    RDP_MINIMIZED   → START_WATCHDOG  (start frozen-frame watchdog)
    """

    def should_capture(self, session: SessionInfo) -> CaptureDecision:
        match session.session_type:
            case SessionType.NORMAL:
                return CaptureDecision.CAPTURE
            case SessionType.LOCKED:
                return CaptureDecision.SUSPEND_NOTIFY
            case SessionType.SECURE_DESKTOP:
                return CaptureDecision.SUSPEND
            case SessionType.RDP_MINIMIZED:
                return CaptureDecision.START_WATCHDOG
            case _:  # pragma: no cover
                return CaptureDecision.CAPTURE


# ---------------------------------------------------------------------------
# FrozenFrameWatchdog
# ---------------------------------------------------------------------------

_FROZEN_THRESHOLD_MS: float = 5000.0  # 5 seconds


class FrozenFrameWatchdog:
    """
    Raises ``FrozenScreenError`` when the screen has been frozen for
    ≥ ``threshold_ms`` milliseconds in a NORMAL session.

    A frame is considered "frozen" (unchanged) when its raw pixel data is
    identical to the previous frame (``numpy.array_equal``).

    Parameters
    ----------
    threshold_ms:
        How long (ms) the screen must be unchanged before raising.
        Default: 5 000 ms (5 s).
    _time_fn:
        ``() -> float`` — monotonic time in seconds.  Defaults to
        ``time.monotonic``.  Inject a stub in tests for instant execution.
    """

    def __init__(
        self,
        threshold_ms: float = _FROZEN_THRESHOLD_MS,
        *,
        _time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._threshold_s: float = threshold_ms / 1000.0
        self._time: Callable[[], float] = _time_fn or time.monotonic
        self._last_change_time: float = self._time()
        self._prev_data: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, frame: Frame, session: SessionInfo) -> None:
        """
        Update internal state with *frame* and raise ``FrozenScreenError``
        when the screen has been frozen too long in a NORMAL session.

        Non-NORMAL sessions reset the frozen timer so that the watchdog
        does not fire spuriously after an unlock or window restore.

        Parameters
        ----------
        frame:
            The latest captured frame.
        session:
            Current session snapshot (from ``SessionDetector``).

        Raises
        ------
        FrozenScreenError
            When ``session.session_type`` is NORMAL and the screen has
            not changed for ≥ ``threshold_ms``.
        """
        now = self._time()

        if session.session_type != SessionType.NORMAL:
            # Non-normal state: reset the timer; no frozen check.
            self._last_change_time = now
            self._prev_data = frame.data.copy()
            return

        if self._prev_data is None:
            self._prev_data = frame.data.copy()
            self._last_change_time = now
            return

        changed = not np.array_equal(frame.data, self._prev_data)
        if changed:
            self._last_change_time = now
            self._prev_data = frame.data.copy()
        else:
            frozen_s = now - self._last_change_time
            if frozen_s >= self._threshold_s:
                frozen_ms = frozen_s * 1000.0
                _log.warning(
                    "frozen_screen_detected",
                    frozen_ms=round(frozen_ms, 1),
                    threshold_ms=self._threshold_s * 1000.0,
                )
                raise FrozenScreenError(frozen_ms)
