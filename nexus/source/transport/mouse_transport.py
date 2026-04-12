"""
nexus/source/transport/mouse_transport.py
Full-featured mouse transport — Windows SendInput, DPI-aware, system-param aware.

Used as the OS-level fallback when native UIA / DOM transport fails or is
unavailable.  All Windows API calls are injectable so the module is fully
unit-testable without a live display device.

Architecture
------------
Every public method converts the caller-supplied **logical** (DPI-unaware)
coordinates to **physical** pixels via ScreenMetricsProvider, then encodes
them as ABSOLUTE SendInput events in the [0, 65535] range required by
MOUSEEVENTF_ABSOLUTE.

Injectable callables
--------------------
_send_input_fn:
    ``(events: list[_MouseEvent | _KeyEvent]) -> int``
    Receives high-level event dicts and returns the count delivered.
    Default: converts to ctypes INPUT and calls user32.SendInput.

_get_double_click_time_fn:
    ``() -> int``  — returns Windows double-click interval in ms.
    Default: user32.GetDoubleClickTime().

_get_scroll_lines_fn:
    ``() -> int``  — returns scroll lines per WHEEL_DELTA notch.
    Default: SystemParametersInfo(SPI_GETWHEELSCROLLLINES).

metrics_provider:
    ScreenMetricsProvider for DPI-aware coordinate conversion.
    Default: a new ScreenMetricsProvider() (performs real DPI detection).

Windows API usage
-----------------
  SendInput                 — all pointer and wheel events
  GetDoubleClickTime        — double-click timing
  SystemParametersInfo      — wheel scroll lines (SPI_GETWHEELSCROLLLINES)
  ScreenMetricsProvider     — DPI / scale_factor / physical screen size

Event format (internal)
-----------------------
Events passed to _send_input_fn are plain dataclasses (not ctypes), so test
code can inspect them without importing ctypes structures:

  _MouseEvent(phys_x, phys_y, flags, mouse_data=0)
  _KeyEvent(vk, scan, flags)         # only used internally for scroll
"""
from __future__ import annotations

import asyncio
import ctypes
import ctypes.wintypes
import time
from collections.abc import Callable
from dataclasses import dataclass

from nexus.capture.screen_metrics import ScreenMetricsProvider
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Windows constants
# ---------------------------------------------------------------------------

_INPUT_MOUSE: int = 0
_INPUT_KEYBOARD: int = 1

# MOUSEEVENTF flags
_MOUSEEVENTF_MOVE: int = 0x0001
_MOUSEEVENTF_LEFTDOWN: int = 0x0002
_MOUSEEVENTF_LEFTUP: int = 0x0004
_MOUSEEVENTF_RIGHTDOWN: int = 0x0008
_MOUSEEVENTF_RIGHTUP: int = 0x0010
_MOUSEEVENTF_MIDDLEDOWN: int = 0x0020
_MOUSEEVENTF_MIDDLEUP: int = 0x0040
_MOUSEEVENTF_WHEEL: int = 0x0800
_MOUSEEVENTF_ABSOLUTE: int = 0x8000

# One scroll notch = 120 WHEEL_DELTA units
_WHEEL_DELTA: int = 120

# SystemParametersInfo action codes
_SPI_GETWHEELSCROLLLINES: int = 0x0068

# ---------------------------------------------------------------------------
# Event dataclasses (injectable / testable format)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MouseEvent:
    """
    High-level description of one Windows mouse input event.

    ``phys_x`` / ``phys_y`` are in physical screen pixels (not the 0-65535
    normalised range).  The real _send_input_fn normalises them internally.
    """

    phys_x: int
    phys_y: int
    flags: int
    mouse_data: int = 0


@dataclass(frozen=True)
class _KeyEvent:
    """High-level description of one Windows keyboard input event."""

    vk: int    # virtual-key code (0 when using scan / Unicode)
    scan: int  # scan code or Unicode codepoint
    flags: int  # KEYEVENTF_* flags


# Union type for the injectable send_input function
_AnyEvent = _MouseEvent | _KeyEvent
_SendInputFn = Callable[[list[_AnyEvent]], int]
_GetDoubleClickTimeFn = Callable[[], int]
_GetScrollLinesFn = Callable[[], int]
_GetCursorPosFn = Callable[[], "tuple[int, int] | None"]

# ---------------------------------------------------------------------------
# ctypes structures (used by the default real implementation only)
# ---------------------------------------------------------------------------


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("_input", _INPUTUNION),
    ]


# ---------------------------------------------------------------------------
# Default Windows API implementations
# ---------------------------------------------------------------------------


def _default_send_input(events: list[_AnyEvent]) -> int:
    """Convert high-level events to ctypes INPUT and call user32.SendInput."""
    if not events:
        return 0
    try:
        n = len(events)
        arr = (_INPUT * n)()
        for i, ev in enumerate(events):
            if isinstance(ev, _MouseEvent):
                arr[i].type = _INPUT_MOUSE
                arr[i]._input.mi.dx = ev.phys_x
                arr[i]._input.mi.dy = ev.phys_y
                arr[i]._input.mi.mouseData = ev.mouse_data
                arr[i]._input.mi.dwFlags = ev.flags
            else:  # _KeyEvent
                arr[i].type = _INPUT_KEYBOARD
                arr[i]._input.ki.wVk = ev.vk
                arr[i]._input.ki.wScan = ev.scan
                arr[i]._input.ki.dwFlags = ev.flags
        result = ctypes.windll.user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
        return int(result)
    except Exception as exc:
        _log.debug("send_input_failed", error=str(exc))
        return 0


def _default_get_double_click_time() -> int:
    """Return Windows double-click interval in ms (default 500)."""
    try:
        result = ctypes.windll.user32.GetDoubleClickTime()
        return int(result) if result > 0 else 500
    except Exception:
        return 500


def _default_get_scroll_lines() -> int:
    """Return scroll lines per WHEEL_DELTA notch via SystemParametersInfo."""
    try:
        lines = ctypes.c_uint(3)
        ok = ctypes.windll.user32.SystemParametersInfoW(
            _SPI_GETWHEELSCROLLLINES, 0, ctypes.byref(lines), 0
        )
        return int(lines.value) if ok else 3
    except Exception:
        return 3


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _default_get_cursor_pos() -> tuple[int, int] | None:
    """Return current physical cursor position, or None on failure."""
    try:
        pt = _POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return (int(pt.x), int(pt.y))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# MouseTransport
# ---------------------------------------------------------------------------


class MouseTransport:
    """
    Delivers mouse actions via Windows SendInput.

    All coordinates accepted by public methods are **logical** (DPI-unaware)
    screen pixels.  Internally they are converted to physical pixels using
    the ScreenMetricsProvider before being passed to SendInput as ABSOLUTE
    events.

    Parameters
    ----------
    metrics_provider:
        ScreenMetricsProvider for DPI scale_factor and physical screen size.
        When None a fresh provider is created (real DPI detection).
    _send_input_fn:
        Sync callable ``(events: list[_MouseEvent | _KeyEvent]) -> int``.
        Inject a recording stub in tests; uses ctypes.windll in production.
    _get_double_click_time_fn:
        Sync callable ``() -> int``.  Returns Windows double-click interval.
    _get_scroll_lines_fn:
        Sync callable ``() -> int``.  Returns scroll lines per wheel notch.
    """

    def __init__(
        self,
        *,
        metrics_provider: ScreenMetricsProvider | None = None,
        _send_input_fn: _SendInputFn | None = None,
        _get_double_click_time_fn: _GetDoubleClickTimeFn | None = None,
        _get_scroll_lines_fn: _GetScrollLinesFn | None = None,
        _get_cursor_pos_fn: _GetCursorPosFn | None = None,
    ) -> None:
        self._metrics = metrics_provider or ScreenMetricsProvider()
        self._send = _send_input_fn or _default_send_input
        self._get_dbl_click_time = (
            _get_double_click_time_fn or _default_get_double_click_time
        )
        self._get_scroll_lines = _get_scroll_lines_fn or _default_get_scroll_lines
        self._get_cursor_pos = _get_cursor_pos_fn or _default_get_cursor_pos

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def click(
        self,
        coordinates: tuple[int, int],
        button: str = "left",
    ) -> bool:
        """
        Click at *coordinates* with the specified *button*.

        Parameters
        ----------
        coordinates:
            Logical (x, y) in screen pixels.
        button:
            ``"left"`` (default), ``"right"``, or ``"middle"``.
        """
        return await asyncio.to_thread(self._click_sync, coordinates, button)

    async def double_click(self, coordinates: tuple[int, int]) -> bool:
        """Double-click at *coordinates* with OS-configured timing."""
        return await asyncio.to_thread(self._double_click_sync, coordinates)

    async def right_click(self, coordinates: tuple[int, int]) -> bool:
        """Right-click at *coordinates*."""
        return await self.click(coordinates, button="right")

    async def scroll(
        self,
        coordinates: tuple[int, int],
        direction: str,
        amount: int = 1,
    ) -> bool:
        """
        Scroll at *coordinates*.

        Parameters
        ----------
        coordinates:
            Logical (x, y) target position.
        direction:
            ``"up"`` or ``"down"``.
        amount:
            Number of scroll *notches* (each notch = WHEEL_DELTA × scroll_lines).
        """
        return await asyncio.to_thread(
            self._scroll_sync, coordinates, direction, amount
        )

    async def drag(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> bool:
        """
        Click-drag from *start* to *end* with the left button held.

        Parameters
        ----------
        start:
            Logical (x, y) for the drag origin.
        end:
            Logical (x, y) for the drag destination.
        """
        return await asyncio.to_thread(self._drag_sync, start, end)

    # ------------------------------------------------------------------
    # Synchronous implementation helpers
    # ------------------------------------------------------------------

    def _click_sync(
        self,
        coordinates: tuple[int, int],
        button: str,
    ) -> bool:
        phys_x, phys_y = self._to_physical(*coordinates)
        down_flag, up_flag = _button_flags(button)
        events: list[_AnyEvent] = [
            _MouseEvent(phys_x, phys_y, _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE),
            _MouseEvent(phys_x, phys_y, down_flag | _MOUSEEVENTF_ABSOLUTE),
            _MouseEvent(phys_x, phys_y, up_flag | _MOUSEEVENTF_ABSOLUTE),
        ]
        sent = self._send(events)
        events_ok = sent == len(events)

        # Verify cursor actually reached the target — detects UIPI-blocked input
        # where SendInput reports success but events are silently dropped.
        cursor_ok = True
        if events_ok:
            actual = self._get_cursor_pos()
            if actual is not None:
                dx = abs(actual[0] - phys_x)
                dy = abs(actual[1] - phys_y)
                cursor_ok = dx <= 2 and dy <= 2
                if not cursor_ok:
                    _log.warning(
                        "mouse_click_cursor_mismatch",
                        expected=(phys_x, phys_y),
                        actual=actual,
                        logical=coordinates,
                    )

        ok = events_ok and cursor_ok
        _log.debug(
            "mouse_click",
            logical=coordinates,
            physical=(phys_x, phys_y),
            button=button,
            events_ok=events_ok,
            cursor_ok=cursor_ok,
            ok=ok,
        )
        return ok

    def _double_click_sync(self, coordinates: tuple[int, int]) -> bool:
        interval_ms = self._get_dbl_click_time()
        phys_x, phys_y = self._to_physical(*coordinates)
        events: list[_AnyEvent] = [
            _MouseEvent(phys_x, phys_y, _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE),
            _MouseEvent(phys_x, phys_y, _MOUSEEVENTF_LEFTDOWN | _MOUSEEVENTF_ABSOLUTE),
            _MouseEvent(phys_x, phys_y, _MOUSEEVENTF_LEFTUP | _MOUSEEVENTF_ABSOLUTE),
        ]
        sent1 = self._send(events)
        # Pause for just under the double-click threshold
        time.sleep(max(0.0, (interval_ms - 50) / 1000.0))
        sent2 = self._send(events)
        ok = (sent1 + sent2) == len(events) * 2
        _log.debug("mouse_double_click", logical=coordinates, ok=ok)
        return ok

    def _scroll_sync(
        self,
        coordinates: tuple[int, int],
        direction: str,
        amount: int,
    ) -> bool:
        phys_x, phys_y = self._to_physical(*coordinates)
        lines = self._get_scroll_lines()
        # Positive WHEEL_DELTA = scroll up; negative = scroll down
        sign = 1 if direction == "up" else -1
        delta = sign * _WHEEL_DELTA * lines * amount
        events: list[_AnyEvent] = [
            _MouseEvent(phys_x, phys_y, _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE),
            _MouseEvent(
                phys_x,
                phys_y,
                _MOUSEEVENTF_WHEEL | _MOUSEEVENTF_ABSOLUTE,
                mouse_data=delta & 0xFFFFFFFF,  # ensure unsigned DWORD
            ),
        ]
        sent = self._send(events)
        ok = sent == len(events)
        _log.debug(
            "mouse_scroll",
            direction=direction,
            amount=amount,
            delta=delta,
            ok=ok,
        )
        return ok

    def _drag_sync(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> bool:
        sx, sy = self._to_physical(*start)
        ex, ey = self._to_physical(*end)
        events: list[_AnyEvent] = [
            _MouseEvent(sx, sy, _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE),
            _MouseEvent(sx, sy, _MOUSEEVENTF_LEFTDOWN | _MOUSEEVENTF_ABSOLUTE),
            _MouseEvent(ex, ey, _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE),
            _MouseEvent(ex, ey, _MOUSEEVENTF_LEFTUP | _MOUSEEVENTF_ABSOLUTE),
        ]
        sent = self._send(events)
        ok = sent == len(events)
        _log.debug("mouse_drag", start=start, end=end, ok=ok)
        return ok

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_physical(self, lx: int, ly: int) -> tuple[int, int]:
        """Convert logical coordinates to physical pixels via DPI scale_factor."""
        return self._metrics.logical_to_physical(lx, ly)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _button_flags(button: str) -> tuple[int, int]:
    """Return (down_flag, up_flag) for a named mouse button."""
    if button == "right":
        return _MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP
    if button == "middle":
        return _MOUSEEVENTF_MIDDLEDOWN, _MOUSEEVENTF_MIDDLEUP
    return _MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP
