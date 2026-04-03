"""
nexus/capture/screen_metrics.py
Screen Metrics and DPI Awareness for Nexus Agent.

Architecture
------------
ScreenMetrics is a frozen value object describing a single monitor's
dimensions, DPI, and work area.  ScreenMetricsProvider wraps Windows
APIs via injectable callables so every method is unit-testable without
a live display device.

DPI Awareness
-------------
SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
is called once at construction via ``_set_dpi_awareness_fn``.  On 96-DPI
displays logical coordinates equal physical pixels (scale_factor == 1.0).
On a 144-DPI (150 %) display scale_factor == 1.5 and logical-to-physical
conversion multiplies by that factor.

Display Change
--------------
Handlers registered via ``add_display_change_handler`` are called when
the display configuration changes.  Tests (or a real WM_DISPLAYCHANGE
listener) call ``notify_display_change()`` directly to trigger them.

Windows APIs used
-----------------
  SetProcessDpiAwarenessContext  — user32
  GetDpiForMonitor               — shcore
  MonitorFromPoint               — user32
  GetMonitorInfo (GetMonitorInfoW) — user32
  EnumDisplayMonitors            — user32
"""
from __future__ import annotations

import contextlib
import ctypes
import ctypes.wintypes
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nexus.core.types import Rect
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Windows API constants
# ---------------------------------------------------------------------------

_MDT_EFFECTIVE_DPI: int = 0
_MONITOR_DEFAULTTOPRIMARY: int = 1
_MONITOR_DEFAULTTONEAREST: int = 2
_MONITORINFOF_PRIMARY: int = 0x00000001
# SetProcessDpiAwarenessContext value for PER_MONITOR_AWARE_V2
_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2: int = -4

_BASE_DPI: int = 96  # logical-equals-physical baseline

# ---------------------------------------------------------------------------
# ctypes helper structures (only used by the default Windows implementations)
# ---------------------------------------------------------------------------


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _MONITORINFOEX(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint32),
        ("rcMonitor", _RECT),
        ("rcWork", _RECT),
        ("dwFlags", ctypes.c_uint32),
        ("szDevice", ctypes.c_wchar * 32),
    ]


# ---------------------------------------------------------------------------
# Public value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreenMetrics:
    """
    Immutable snapshot of a single monitor's display metrics.

    Attributes
    ----------
    width, height:
        Physical pixel dimensions of the monitor.
    dpi:
        Effective DPI as reported by Windows (e.g. 96, 120, 144, 192).
    scale_factor:
        ``dpi / 96``.  At 96 DPI → 1.0; at 144 DPI → 1.5.
    is_primary:
        True when this is the primary Windows monitor.
    monitor_handle:
        Windows HMONITOR value (0 in test / non-Windows environments).
    work_area:
        Usable monitor area excluding the system taskbar, expressed as a
        ``Rect`` with its origin in virtual-screen coordinates.
    """

    width: int
    height: int
    dpi: int
    scale_factor: float
    is_primary: bool
    monitor_handle: int
    work_area: Rect


# ---------------------------------------------------------------------------
# Type aliases for injectable callables
# ---------------------------------------------------------------------------

_SetDpiAwarenessFn = Callable[[], bool]
_GetDpiForMonitorFn = Callable[[int], int]           # (hmonitor) -> dpi
_MonitorFromPointFn = Callable[[int, int], int]       # (x, y) -> hmonitor
_GetMonitorInfoFn = Callable[[int], "dict[str, Any] | None"]
_EnumDisplayMonitorsFn = Callable[[], "list[int]"]    # () -> [hmonitor, ...]

DisplayChangeHandler = Callable[[], None]

# ---------------------------------------------------------------------------
# Default Windows API wrapper implementations
# ---------------------------------------------------------------------------


def _default_set_dpi_awareness() -> bool:
    """Call SetProcessDpiAwarenessContext; return True on success."""
    try:
        user32 = ctypes.windll.user32
        result = user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        )
        return bool(result)
    except Exception as exc:
        _log.debug("set_dpi_awareness_failed", error=str(exc))
        return False


def _default_get_dpi_for_monitor(hmonitor: int) -> int:
    """Return effective DPI for *hmonitor* via shcore.GetDpiForMonitor."""
    try:
        shcore = ctypes.windll.shcore
        dpi_x = ctypes.c_uint(0)
        dpi_y = ctypes.c_uint(0)
        hr = shcore.GetDpiForMonitor(
            ctypes.c_void_p(hmonitor),
            _MDT_EFFECTIVE_DPI,
            ctypes.byref(dpi_x),
            ctypes.byref(dpi_y),
        )
        if hr == 0:  # S_OK
            return int(dpi_x.value)
    except Exception as exc:
        _log.debug("get_dpi_for_monitor_failed", error=str(exc))
    return _BASE_DPI


def _default_monitor_from_point(x: int, y: int) -> int:
    """Return HMONITOR for the monitor nearest to the point (x, y)."""
    try:
        user32 = ctypes.windll.user32
        pt = ctypes.wintypes.POINT(x, y)
        handle = user32.MonitorFromPoint(pt, _MONITOR_DEFAULTTONEAREST)
        return int(handle) if handle else 0
    except Exception as exc:
        _log.debug("monitor_from_point_failed", error=str(exc))
        return 0


def _default_get_monitor_info(hmonitor: int) -> dict[str, Any] | None:
    """
    Wrap GetMonitorInfoW and return a plain dict:
    ``{rcMonitor, rcWork, is_primary, device_name}``.
    Returns None on failure.
    """
    try:
        user32 = ctypes.windll.user32
        mi = _MONITORINFOEX()
        mi.cbSize = ctypes.sizeof(_MONITORINFOEX)
        if user32.GetMonitorInfoW(ctypes.c_void_p(hmonitor), ctypes.byref(mi)):
            return {
                "rcMonitor": mi.rcMonitor,
                "rcWork": mi.rcWork,
                "is_primary": bool(mi.dwFlags & _MONITORINFOF_PRIMARY),
                "device_name": mi.szDevice,
            }
    except Exception as exc:
        _log.debug("get_monitor_info_failed", error=str(exc))
    return None


def _default_enum_display_monitors() -> list[int]:
    """Return HMONITOR handles for every connected monitor."""
    handles: list[int] = []
    try:
        user32 = ctypes.windll.user32
        _monitor_enum_proc = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(_RECT),
            ctypes.c_long,
        )

        def _cb(hmon: int, _hdc: int, _lprect: Any, _lparam: int) -> bool:
            handles.append(hmon)
            return True

        user32.EnumDisplayMonitors(None, None, _monitor_enum_proc(_cb), 0)
    except Exception as exc:
        _log.debug("enum_display_monitors_failed", error=str(exc))
    return handles


# ---------------------------------------------------------------------------
# ScreenMetricsProvider
# ---------------------------------------------------------------------------


class ScreenMetricsProvider:
    """
    Provides per-monitor screen metrics with DPI-aware coordinate conversion.

    All Windows API calls are injectable via constructor parameters so the
    class can be tested without a real display.

    Parameters
    ----------
    _set_dpi_awareness_fn:
        ``() -> bool``.  Sets DPI awareness context once at startup.
    _get_dpi_for_monitor_fn:
        ``(hmonitor: int) -> int``.  Returns effective DPI.
    _monitor_from_point_fn:
        ``(x: int, y: int) -> int``.  Returns HMONITOR nearest to point.
    _get_monitor_info_fn:
        ``(hmonitor: int) -> dict | None``.  Returns monitor geometry.
    _enum_display_monitors_fn:
        ``() -> list[int]``.  Returns all HMONITOR handles.
    """

    def __init__(
        self,
        *,
        _set_dpi_awareness_fn: _SetDpiAwarenessFn | None = None,
        _get_dpi_for_monitor_fn: _GetDpiForMonitorFn | None = None,
        _monitor_from_point_fn: _MonitorFromPointFn | None = None,
        _get_monitor_info_fn: _GetMonitorInfoFn | None = None,
        _enum_display_monitors_fn: _EnumDisplayMonitorsFn | None = None,
    ) -> None:
        self._set_dpi_awareness_api = (
            _set_dpi_awareness_fn or _default_set_dpi_awareness
        )
        self._get_dpi_for_monitor_api = (
            _get_dpi_for_monitor_fn or _default_get_dpi_for_monitor
        )
        self._monitor_from_point_api = (
            _monitor_from_point_fn or _default_monitor_from_point
        )
        self._get_monitor_info_api = (
            _get_monitor_info_fn or _default_get_monitor_info
        )
        self._enum_display_monitors_api = (
            _enum_display_monitors_fn or _default_enum_display_monitors
        )

        # Maps HMONITOR → full bounds Rect in virtual-screen coordinates.
        # Populated lazily by _build_metrics; used by is_on_screen.
        self._monitor_bounds: dict[int, Rect] = {}
        self._lock = threading.Lock()

        self._display_change_handlers: list[DisplayChangeHandler] = []

        ok = self._set_dpi_awareness_api()
        _log.debug("dpi_awareness_context_set", success=ok)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_active_monitor(self, x: int = 0, y: int = 0) -> ScreenMetrics:
        """
        Return metrics for the monitor that contains (x, y).

        Falls back to the nearest monitor when the point is outside all
        connected monitors.
        """
        hmonitor = self._monitor_from_point_api(x, y)
        return self._build_metrics(hmonitor)

    def get_all_monitors(self) -> list[ScreenMetrics]:
        """Return ScreenMetrics for every connected monitor."""
        handles = self._enum_display_monitors_api()
        if not handles:
            return [self.get_active_monitor(0, 0)]
        return [self._build_metrics(h) for h in handles]

    def logical_to_physical(self, x: int, y: int) -> tuple[int, int]:
        """
        Convert logical (DPI-unaware) coordinates to physical pixel coords.

        At 96 DPI (scale_factor == 1.0) the result equals the input.
        At 144 DPI (scale_factor == 1.5) logical (100, 100) → (150, 150).
        """
        scale = self.get_active_monitor(x, y).scale_factor
        return (round(x * scale), round(y * scale))

    def physical_to_logical(self, x: int, y: int) -> tuple[int, int]:
        """
        Convert physical pixel coordinates to logical (DPI-unaware) coords.

        Inverse of logical_to_physical.
        """
        scale = self.get_active_monitor(x, y).scale_factor
        return (round(x / scale), round(y / scale))

    def is_on_screen(self, x: int, y: int, epsilon: int = 0) -> bool:
        """
        Return True when (x, y) lies within any connected monitor's bounds.

        Parameters
        ----------
        epsilon:
            Tolerance in pixels.  Positive values expand the acceptance
            region (off-by-one / near-edge); negative values shrink it.
        """
        self.get_all_monitors()  # ensure _monitor_bounds is populated
        with self._lock:
            bounds_snapshot = dict(self._monitor_bounds)

        for bounds in bounds_snapshot.values():
            if (
                bounds.x - epsilon <= x < bounds.x + bounds.width + epsilon
                and bounds.y - epsilon <= y < bounds.y + bounds.height + epsilon
            ):
                return True
        return False

    def add_display_change_handler(self, handler: DisplayChangeHandler) -> None:
        """Register *handler* to be called on display configuration changes."""
        with self._lock:
            self._display_change_handlers.append(handler)

    def remove_display_change_handler(self, handler: DisplayChangeHandler) -> None:
        """Unregister a previously registered *handler*."""
        with self._lock, contextlib.suppress(ValueError):
            self._display_change_handlers.remove(handler)

    def notify_display_change(self) -> None:
        """
        Fire all registered display-change handlers.

        Called by a real WM_DISPLAYCHANGE listener, or directly from
        tests to simulate the event.  Handler exceptions are swallowed
        so that one bad handler does not block the others.
        """
        with self._lock:
            handlers = list(self._display_change_handlers)

        _log.info("display_change_detected", handler_count=len(handlers))
        for handler in handlers:
            try:
                handler()
            except Exception as exc:
                _log.debug("display_change_handler_error", error=str(exc))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_metrics(self, hmonitor: int) -> ScreenMetrics:
        """
        Build a ScreenMetrics from an HMONITOR handle.

        Also updates the internal _monitor_bounds cache used by
        is_on_screen.  Falls back to a synthetic 1920×1080 primary
        monitor when GetMonitorInfo returns None (test environments
        that inject only a DPI function).
        """
        info = self._get_monitor_info_api(hmonitor)
        dpi = self._get_dpi_for_monitor_api(hmonitor)
        scale = round(dpi / _BASE_DPI, 10)  # avoid float rounding drift

        if info is not None:
            rc = info["rcMonitor"]
            rw = info["rcWork"]
            # rcMonitor / rcWork are _RECT-like objects with left/top/right/bottom
            width = rc.right - rc.left
            height = rc.bottom - rc.top
            work_area = Rect(
                rw.left,
                rw.top,
                rw.right - rw.left,
                rw.bottom - rw.top,
            )
            bounds = Rect(rc.left, rc.top, width, height)
            is_primary = info["is_primary"]
        else:
            # Fallback used when only a DPI stub is injected (no monitor info)
            width, height = 1920, 1080
            work_area = Rect(0, 0, 1920, 1040)
            bounds = Rect(0, 0, 1920, 1080)
            is_primary = True

        with self._lock:
            self._monitor_bounds[hmonitor] = bounds

        return ScreenMetrics(
            width=width,
            height=height,
            dpi=dpi,
            scale_factor=scale,
            is_primary=is_primary,
            monitor_handle=hmonitor,
            work_area=work_area,
        )
