"""
tests/unit/test_screen_metrics.py
Unit tests for nexus/capture/screen_metrics.py — Faz 17.

All Windows API calls are injected so tests run on any platform without
a live display.  A plain namespace object stands in for the _RECT ctypes
struct (left / top / right / bottom attributes).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.capture.screen_metrics import ScreenMetrics, ScreenMetricsProvider
from nexus.core.types import Rect

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_HANDLE_PRIMARY = 1001
_HANDLE_SECONDARY = 1002


def _make_rect(left: int, top: int, right: int, bottom: int) -> Any:
    """Minimal rect-like object matching the _RECT ctypes struct interface."""
    return SimpleNamespace(left=left, top=top, right=right, bottom=bottom)


def _monitor_info(
    left: int,
    top: int,
    right: int,
    bottom: int,
    *,
    work_inset: int = 40,
    is_primary: bool = True,
) -> dict[str, Any]:
    """Build a mock GetMonitorInfo result dict."""
    return {
        "rcMonitor": _make_rect(left, top, right, bottom),
        "rcWork": _make_rect(left, top, right, bottom - work_inset),
        "is_primary": is_primary,
        "device_name": r"\\.\DISPLAY1",
    }


def _provider_96dpi() -> ScreenMetricsProvider:
    """Single 1920×1080 primary monitor at 96 DPI."""
    info = _monitor_info(0, 0, 1920, 1080)
    return ScreenMetricsProvider(
        _set_dpi_awareness_fn=lambda: True,
        _get_dpi_for_monitor_fn=lambda h: 96,
        _monitor_from_point_fn=lambda x, y: _HANDLE_PRIMARY,
        _get_monitor_info_fn=lambda h: info,
        _enum_display_monitors_fn=lambda: [_HANDLE_PRIMARY],
    )


def _provider_144dpi() -> ScreenMetricsProvider:
    """Single 1920×1080 monitor at 144 DPI (150 % scale)."""
    info = _monitor_info(0, 0, 1920, 1080)
    return ScreenMetricsProvider(
        _set_dpi_awareness_fn=lambda: True,
        _get_dpi_for_monitor_fn=lambda h: 144,
        _monitor_from_point_fn=lambda x, y: _HANDLE_PRIMARY,
        _get_monitor_info_fn=lambda h: info,
        _enum_display_monitors_fn=lambda: [_HANDLE_PRIMARY],
    )


# ---------------------------------------------------------------------------
# ScreenMetrics value object
# ---------------------------------------------------------------------------


class TestScreenMetricsDataclass:
    def test_frozen(self) -> None:
        m = ScreenMetrics(
            width=1920,
            height=1080,
            dpi=96,
            scale_factor=1.0,
            is_primary=True,
            monitor_handle=0,
            work_area=Rect(0, 0, 1920, 1040),
        )
        with pytest.raises((AttributeError, TypeError)):
            m.dpi = 144  # type: ignore[misc]

    def test_scale_factor_stored(self) -> None:
        m = ScreenMetrics(
            width=2560,
            height=1440,
            dpi=144,
            scale_factor=1.5,
            is_primary=False,
            monitor_handle=999,
            work_area=Rect(1920, 0, 2560, 1400),
        )
        assert m.scale_factor == pytest.approx(1.5)
        assert m.dpi == 144
        assert m.is_primary is False


# ---------------------------------------------------------------------------
# DPI scale factor computation
# ---------------------------------------------------------------------------


class TestDpiScaleFactor:
    """96 DPI → 1.0x; 144 DPI → 1.5x; 192 DPI → 2.0x."""

    def test_96_dpi_scale_is_1(self) -> None:
        p = _provider_96dpi()
        m = p.get_active_monitor()
        assert m.dpi == 96
        assert m.scale_factor == pytest.approx(1.0)

    def test_144_dpi_scale_is_1_5(self) -> None:
        p = _provider_144dpi()
        m = p.get_active_monitor()
        assert m.dpi == 144
        assert m.scale_factor == pytest.approx(1.5)

    def test_192_dpi_scale_is_2(self) -> None:
        info = _monitor_info(0, 0, 1920, 1080)
        p = ScreenMetricsProvider(
            _set_dpi_awareness_fn=lambda: True,
            _get_dpi_for_monitor_fn=lambda h: 192,
            _monitor_from_point_fn=lambda x, y: _HANDLE_PRIMARY,
            _get_monitor_info_fn=lambda h: info,
            _enum_display_monitors_fn=lambda: [_HANDLE_PRIMARY],
        )
        m = p.get_active_monitor()
        assert m.scale_factor == pytest.approx(2.0)

    def test_120_dpi_scale_is_1_25(self) -> None:
        info = _monitor_info(0, 0, 1920, 1080)
        p = ScreenMetricsProvider(
            _set_dpi_awareness_fn=lambda: True,
            _get_dpi_for_monitor_fn=lambda h: 120,
            _monitor_from_point_fn=lambda x, y: _HANDLE_PRIMARY,
            _get_monitor_info_fn=lambda h: info,
            _enum_display_monitors_fn=lambda: [_HANDLE_PRIMARY],
        )
        m = p.get_active_monitor()
        assert m.scale_factor == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# Logical ↔ Physical coordinate conversion
# ---------------------------------------------------------------------------


class TestCoordinateConversion:
    """At 96 DPI logical == physical; at 144 DPI multiply / divide by 1.5."""

    def test_96dpi_logical_equals_physical(self) -> None:
        p = _provider_96dpi()
        assert p.logical_to_physical(100, 200) == (100, 200)
        assert p.physical_to_logical(100, 200) == (100, 200)

    def test_96dpi_zero_stays_zero(self) -> None:
        p = _provider_96dpi()
        assert p.logical_to_physical(0, 0) == (0, 0)
        assert p.physical_to_logical(0, 0) == (0, 0)

    def test_144dpi_logical_to_physical_multiplies_by_1_5(self) -> None:
        p = _provider_144dpi()
        px, py = p.logical_to_physical(100, 200)
        assert px == 150
        assert py == 300

    def test_144dpi_physical_to_logical_divides_by_1_5(self) -> None:
        p = _provider_144dpi()
        lx, ly = p.physical_to_logical(150, 300)
        assert lx == 100
        assert ly == 200

    def test_144dpi_roundtrip(self) -> None:
        p = _provider_144dpi()
        x, y = 320, 480
        px, py = p.logical_to_physical(x, y)
        lx, ly = p.physical_to_logical(px, py)
        assert lx == x
        assert ly == y

    def test_144dpi_large_coords(self) -> None:
        p = _provider_144dpi()
        px, py = p.logical_to_physical(1000, 500)
        assert px == 1500
        assert py == 750


# ---------------------------------------------------------------------------
# get_active_monitor / get_all_monitors
# ---------------------------------------------------------------------------


class TestGetMonitors:
    def test_active_monitor_returns_screen_metrics(self) -> None:
        p = _provider_96dpi()
        m = p.get_active_monitor(500, 300)
        assert isinstance(m, ScreenMetrics)
        assert m.width == 1920
        assert m.height == 1080
        assert m.is_primary is True
        assert m.monitor_handle == _HANDLE_PRIMARY

    def test_get_all_monitors_single(self) -> None:
        p = _provider_96dpi()
        monitors = p.get_all_monitors()
        assert len(monitors) == 1
        assert monitors[0].width == 1920

    def test_get_all_monitors_dual(self) -> None:
        """Two monitors side by side."""
        info_map = {
            _HANDLE_PRIMARY: _monitor_info(0, 0, 1920, 1080, is_primary=True),
            _HANDLE_SECONDARY: _monitor_info(1920, 0, 3840, 1080, is_primary=False),
        }
        p = ScreenMetricsProvider(
            _set_dpi_awareness_fn=lambda: True,
            _get_dpi_for_monitor_fn=lambda h: 96,
            _monitor_from_point_fn=lambda x, y: _HANDLE_PRIMARY,
            _get_monitor_info_fn=lambda h: info_map.get(h),
            _enum_display_monitors_fn=lambda: [_HANDLE_PRIMARY, _HANDLE_SECONDARY],
        )
        monitors = p.get_all_monitors()
        assert len(monitors) == 2
        primary = next(m for m in monitors if m.is_primary)
        secondary = next(m for m in monitors if not m.is_primary)
        assert primary.monitor_handle == _HANDLE_PRIMARY
        assert secondary.monitor_handle == _HANDLE_SECONDARY

    def test_work_area_excludes_taskbar(self) -> None:
        p = _provider_96dpi()
        m = p.get_active_monitor()
        assert m.work_area.height < m.height  # taskbar inset of 40 px

    def test_enum_returns_empty_fallback_to_active(self) -> None:
        """When EnumDisplayMonitors returns nothing, fall back to active monitor."""
        info = _monitor_info(0, 0, 1280, 720)
        p = ScreenMetricsProvider(
            _set_dpi_awareness_fn=lambda: True,
            _get_dpi_for_monitor_fn=lambda h: 96,
            _monitor_from_point_fn=lambda x, y: _HANDLE_PRIMARY,
            _get_monitor_info_fn=lambda h: info,
            _enum_display_monitors_fn=lambda: [],  # empty!
        )
        monitors = p.get_all_monitors()
        assert len(monitors) == 1
        assert monitors[0].width == 1280


# ---------------------------------------------------------------------------
# is_on_screen — boundary cases
# ---------------------------------------------------------------------------


class TestIsOnScreen:
    """Boundary cases for is_on_screen with a 1920×1080 monitor at origin."""

    @pytest.fixture()
    def provider(self) -> ScreenMetricsProvider:
        return _provider_96dpi()

    def test_origin_is_on_screen(self, provider: ScreenMetricsProvider) -> None:
        assert provider.is_on_screen(0, 0) is True

    def test_inside_is_on_screen(self, provider: ScreenMetricsProvider) -> None:
        assert provider.is_on_screen(960, 540) is True

    def test_top_left_corner(self, provider: ScreenMetricsProvider) -> None:
        assert provider.is_on_screen(0, 0) is True

    def test_bottom_right_last_pixel(self, provider: ScreenMetricsProvider) -> None:
        # is_on_screen uses half-open interval [0, width), so 1919, 1079 is in
        assert provider.is_on_screen(1919, 1079) is True

    def test_exactly_at_width_is_off_screen(self, provider: ScreenMetricsProvider) -> None:
        assert provider.is_on_screen(1920, 0) is False

    def test_exactly_at_height_is_off_screen(self, provider: ScreenMetricsProvider) -> None:
        assert provider.is_on_screen(0, 1080) is False

    def test_negative_x_is_off_screen(self, provider: ScreenMetricsProvider) -> None:
        assert provider.is_on_screen(-1, 0) is False

    def test_negative_y_is_off_screen(self, provider: ScreenMetricsProvider) -> None:
        assert provider.is_on_screen(0, -1) is False

    def test_far_outside_is_off_screen(self, provider: ScreenMetricsProvider) -> None:
        assert provider.is_on_screen(9999, 9999) is False

    def test_epsilon_expands_boundary(self, provider: ScreenMetricsProvider) -> None:
        # Without epsilon: (1920, 0) is off screen
        assert provider.is_on_screen(1920, 0) is False
        # With epsilon=1 the boundary expands by 1 pixel → now on screen
        assert provider.is_on_screen(1920, 0, epsilon=1) is True

    def test_epsilon_negative_shrinks_boundary(
        self, provider: ScreenMetricsProvider
    ) -> None:
        # (1919, 1079) is the last valid pixel without epsilon
        assert provider.is_on_screen(1919, 1079) is True
        # With epsilon=-1 the boundary shrinks → (1919, 1079) becomes off screen
        assert provider.is_on_screen(1919, 1079, epsilon=-1) is False

    def test_epsilon_zero_is_default(self, provider: ScreenMetricsProvider) -> None:
        assert provider.is_on_screen(960, 540, epsilon=0) is True

    def test_dual_monitor_point_on_secondary(self) -> None:
        """Point on secondary monitor (offset at x=1920) is on screen."""
        info_map = {
            _HANDLE_PRIMARY: _monitor_info(0, 0, 1920, 1080, is_primary=True),
            _HANDLE_SECONDARY: _monitor_info(1920, 0, 3840, 1080, is_primary=False),
        }
        p = ScreenMetricsProvider(
            _set_dpi_awareness_fn=lambda: True,
            _get_dpi_for_monitor_fn=lambda h: 96,
            _monitor_from_point_fn=lambda x, y: _HANDLE_PRIMARY,
            _get_monitor_info_fn=lambda h: info_map.get(h),
            _enum_display_monitors_fn=lambda: [_HANDLE_PRIMARY, _HANDLE_SECONDARY],
        )
        assert p.is_on_screen(2000, 500) is True

    def test_gap_between_dual_monitors_is_off_screen(self) -> None:
        """If monitors are not adjacent, the gap is off screen."""
        info_map = {
            _HANDLE_PRIMARY: _monitor_info(0, 0, 1920, 1080, is_primary=True),
            # secondary starts at 2000, leaving a gap of 80 px
            _HANDLE_SECONDARY: _monitor_info(2000, 0, 3920, 1080, is_primary=False),
        }
        p = ScreenMetricsProvider(
            _set_dpi_awareness_fn=lambda: True,
            _get_dpi_for_monitor_fn=lambda h: 96,
            _monitor_from_point_fn=lambda x, y: _HANDLE_PRIMARY,
            _get_monitor_info_fn=lambda h: info_map.get(h),
            _enum_display_monitors_fn=lambda: [_HANDLE_PRIMARY, _HANDLE_SECONDARY],
        )
        # x=1950 is in the gap
        assert p.is_on_screen(1950, 500) is False


# ---------------------------------------------------------------------------
# Display change event
# ---------------------------------------------------------------------------


class TestDisplayChangeEvent:
    def test_handler_called_on_notify(self) -> None:
        p = _provider_96dpi()
        calls: list[int] = []
        p.add_display_change_handler(lambda: calls.append(1))
        p.notify_display_change()
        assert calls == [1]

    def test_multiple_handlers_all_called(self) -> None:
        p = _provider_96dpi()
        log: list[str] = []
        p.add_display_change_handler(lambda: log.append("a"))
        p.add_display_change_handler(lambda: log.append("b"))
        p.notify_display_change()
        assert sorted(log) == ["a", "b"]

    def test_handler_called_multiple_times(self) -> None:
        p = _provider_96dpi()
        calls: list[int] = []
        p.add_display_change_handler(lambda: calls.append(1))
        p.notify_display_change()
        p.notify_display_change()
        assert len(calls) == 2

    def test_removed_handler_not_called(self) -> None:
        p = _provider_96dpi()
        calls: list[int] = []

        def handler() -> None:
            calls.append(1)

        p.add_display_change_handler(handler)
        p.remove_display_change_handler(handler)
        p.notify_display_change()
        assert calls == []

    def test_remove_nonexistent_handler_is_silent(self) -> None:
        p = _provider_96dpi()
        p.remove_display_change_handler(lambda: None)  # must not raise

    def test_faulty_handler_does_not_block_others(self) -> None:
        p = _provider_96dpi()
        log: list[str] = []

        def bad_handler() -> None:
            raise RuntimeError("boom")

        p.add_display_change_handler(bad_handler)
        p.add_display_change_handler(lambda: log.append("ok"))
        p.notify_display_change()  # must not raise
        assert log == ["ok"]

    def test_no_handlers_notify_is_silent(self) -> None:
        p = _provider_96dpi()
        p.notify_display_change()  # must not raise


# ---------------------------------------------------------------------------
# DPI awareness setup
# ---------------------------------------------------------------------------


class TestDpiAwarenessSetup:
    def test_awareness_fn_called_at_construction(self) -> None:
        calls: list[int] = []

        ScreenMetricsProvider(
            _set_dpi_awareness_fn=lambda: calls.append(1) or True,
            _get_dpi_for_monitor_fn=lambda h: 96,
            _monitor_from_point_fn=lambda x, y: 0,
            _get_monitor_info_fn=lambda h: None,
            _enum_display_monitors_fn=lambda: [],
        )
        assert len(calls) == 1

    def test_awareness_fn_failure_does_not_raise(self) -> None:
        """Constructor must not raise even when the DPI-awareness call fails."""
        p = ScreenMetricsProvider(
            _set_dpi_awareness_fn=lambda: False,
            _get_dpi_for_monitor_fn=lambda h: 96,
            _monitor_from_point_fn=lambda x, y: 0,
            _get_monitor_info_fn=lambda h: None,
            _enum_display_monitors_fn=lambda: [],
        )
        assert p is not None


# ---------------------------------------------------------------------------
# Fallback when GetMonitorInfo returns None
# ---------------------------------------------------------------------------


class TestMonitorInfoFallback:
    def test_fallback_metrics_when_info_is_none(self) -> None:
        """When _get_monitor_info_fn returns None, provider uses 1920×1080 defaults."""
        p = ScreenMetricsProvider(
            _set_dpi_awareness_fn=lambda: True,
            _get_dpi_for_monitor_fn=lambda h: 96,
            _monitor_from_point_fn=lambda x, y: 0,
            _get_monitor_info_fn=lambda h: None,
            _enum_display_monitors_fn=lambda: [0],
        )
        m = p.get_active_monitor()
        assert m.width == 1920
        assert m.height == 1080
        assert m.is_primary is True

    def test_fallback_is_primary_and_handle_zero(self) -> None:
        p = ScreenMetricsProvider(
            _set_dpi_awareness_fn=lambda: True,
            _get_dpi_for_monitor_fn=lambda h: 144,
            _monitor_from_point_fn=lambda x, y: 0,
            _get_monitor_info_fn=lambda h: None,
            _enum_display_monitors_fn=lambda: [0],
        )
        m = p.get_active_monitor()
        assert m.monitor_handle == 0
        assert m.dpi == 144
        assert m.scale_factor == pytest.approx(1.5)
