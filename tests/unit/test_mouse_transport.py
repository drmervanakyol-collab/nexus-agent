"""
tests/unit/test_mouse_transport.py
Unit tests for nexus/source/transport/mouse_transport.py.

Coverage
--------
  MouseTransport.click       — left/right/middle, DPI-aware coordinates
  MouseTransport.double_click — timing + two event bursts
  MouseTransport.right_click  — delegates to click(button="right")
  MouseTransport.scroll       — direction, system scroll-lines, wheel_delta
  MouseTransport.drag         — start/end coords, left-button hold sequence
  DPI-awareness               — scale_factor applied before SendInput
  Event flags                 — ABSOLUTE flag always set, button flags correct
  Failure path                — send returns 0 → method returns False
"""
from __future__ import annotations

from unittest.mock import MagicMock

from nexus.source.transport.mouse_transport import (
    _MOUSEEVENTF_ABSOLUTE,
    _MOUSEEVENTF_LEFTDOWN,
    _MOUSEEVENTF_LEFTUP,
    _MOUSEEVENTF_MIDDLEDOWN,
    _MOUSEEVENTF_MIDDLEUP,
    _MOUSEEVENTF_MOVE,
    _MOUSEEVENTF_RIGHTDOWN,
    _MOUSEEVENTF_RIGHTUP,
    _MOUSEEVENTF_WHEEL,
    _WHEEL_DELTA,
    MouseTransport,
    _MouseEvent,
)

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _make_metrics(scale: float = 1.0, width: int = 1920, height: int = 1080):
    """Return a MagicMock ScreenMetricsProvider with given scale."""
    m = MagicMock()
    m.logical_to_physical.side_effect = lambda lx, ly: (
        round(lx * scale),
        round(ly * scale),
    )
    return m


class _Recorder:
    """Records all events passed to _send_input_fn and returns len(events)."""

    def __init__(self, return_count: int | None = None) -> None:
        self.calls: list[list] = []
        self._return_count = return_count

    def __call__(self, events: list) -> int:
        self.calls.append(list(events))
        if self._return_count is not None:
            return self._return_count
        return len(events)  # simulate all events delivered

    @property
    def all_events(self) -> list:
        return [ev for call in self.calls for ev in call]


def _make_transport(
    scale: float = 1.0,
    return_count: int | None = None,
    dbl_click_time_ms: int = 500,
    scroll_lines: int = 3,
) -> tuple[MouseTransport, _Recorder]:
    recorder = _Recorder(return_count=return_count)

    # Cursor stub: reports the last physical position that SendInput moved to.
    # Reads the final MOVE event from the recorder so it always matches.
    def _fake_cursor_pos() -> tuple[int, int] | None:
        for call in reversed(recorder.calls):
            for ev in reversed(call):
                if isinstance(ev, _MouseEvent) and (ev.flags & _MOUSEEVENTF_MOVE):
                    return (ev.phys_x, ev.phys_y)
        return None  # no events yet → skip cursor check

    transport = MouseTransport(
        metrics_provider=_make_metrics(scale=scale),
        _send_input_fn=recorder,
        _get_double_click_time_fn=lambda: dbl_click_time_ms,
        _get_scroll_lines_fn=lambda: scroll_lines,
        _get_cursor_pos_fn=_fake_cursor_pos,
    )
    return transport, recorder


# ---------------------------------------------------------------------------
# TestClick
# ---------------------------------------------------------------------------


class TestClick:
    async def test_left_click_returns_true(self) -> None:
        t, _ = _make_transport()
        assert await t.click((100, 200)) is True

    async def test_click_calls_send_input(self) -> None:
        t, rec = _make_transport()
        await t.click((100, 200))
        assert len(rec.calls) == 1

    async def test_click_sends_three_events_move_down_up(self) -> None:
        t, rec = _make_transport()
        await t.click((100, 200))
        events = rec.all_events
        assert len(events) == 3
        flags = [e.flags for e in events]
        assert flags[0] & _MOUSEEVENTF_MOVE
        assert flags[1] & _MOUSEEVENTF_LEFTDOWN
        assert flags[2] & _MOUSEEVENTF_LEFTUP

    async def test_click_all_events_are_absolute(self) -> None:
        t, rec = _make_transport()
        await t.click((100, 200))
        for ev in rec.all_events:
            assert ev.flags & _MOUSEEVENTF_ABSOLUTE, f"Event missing ABSOLUTE flag: {ev}"

    async def test_right_click_uses_right_button_flags(self) -> None:
        t, rec = _make_transport()
        await t.click((50, 50), button="right")
        flags = [e.flags for e in rec.all_events]
        assert any(f & _MOUSEEVENTF_RIGHTDOWN for f in flags)
        assert any(f & _MOUSEEVENTF_RIGHTUP for f in flags)

    async def test_middle_click_uses_middle_button_flags(self) -> None:
        t, rec = _make_transport()
        await t.click((50, 50), button="middle")
        flags = [e.flags for e in rec.all_events]
        assert any(f & _MOUSEEVENTF_MIDDLEDOWN for f in flags)
        assert any(f & _MOUSEEVENTF_MIDDLEUP for f in flags)

    async def test_click_returns_false_when_send_input_fails(self) -> None:
        t, _ = _make_transport(return_count=0)
        assert await t.click((100, 200)) is False

    async def test_right_click_helper_returns_true(self) -> None:
        t, _ = _make_transport()
        assert await t.right_click((100, 200)) is True


# ---------------------------------------------------------------------------
# TestDpiAwareness
# ---------------------------------------------------------------------------


class TestDpiAwareness:
    async def test_click_physical_coords_are_scaled(self) -> None:
        """scale=2.0 → SendInput should receive 2× the logical coords."""
        t, rec = _make_transport(scale=2.0)
        await t.click((100, 200))  # logical
        move_event: _MouseEvent = rec.all_events[0]
        assert move_event.phys_x == 200   # 100 × 2
        assert move_event.phys_y == 400   # 200 × 2

    async def test_click_unit_scale_passes_through(self) -> None:
        """scale=1.0 → physical == logical."""
        t, rec = _make_transport(scale=1.0)
        await t.click((300, 150))
        move_event: _MouseEvent = rec.all_events[0]
        assert move_event.phys_x == 300
        assert move_event.phys_y == 150

    async def test_scroll_coords_are_also_scaled(self) -> None:
        t, rec = _make_transport(scale=1.5)
        await t.scroll((100, 100), "up", amount=1)
        move_event: _MouseEvent = rec.all_events[0]
        assert move_event.phys_x == 150
        assert move_event.phys_y == 150

    async def test_drag_start_and_end_both_scaled(self) -> None:
        t, rec = _make_transport(scale=2.0)
        await t.drag((10, 20), (50, 60))
        phys_coords = [(e.phys_x, e.phys_y) for e in rec.all_events]
        # First two events are at start, last two at end
        assert phys_coords[0] == (20, 40)   # start × 2
        assert phys_coords[-1] == (100, 120)  # end × 2


# ---------------------------------------------------------------------------
# TestDoubleClick
# ---------------------------------------------------------------------------


class TestDoubleClick:
    async def test_double_click_returns_true(self) -> None:
        t, _ = _make_transport()
        assert await t.double_click((100, 100)) is True

    async def test_double_click_sends_six_events_total(self) -> None:
        """Two click bursts of 3 events each = 6 total."""
        t, rec = _make_transport()
        await t.double_click((100, 100))
        assert len(rec.all_events) == 6

    async def test_double_click_consults_system_timing(self) -> None:
        """The get_double_click_time_fn should be called once."""
        called = []
        def _dbl(events: list) -> int:
            return len(events)

        dbl_time_calls: list[int] = []
        def _get_time() -> int:
            dbl_time_calls.append(1)
            return 500

        t = MouseTransport(
            metrics_provider=_make_metrics(),
            _send_input_fn=_dbl,
            _get_double_click_time_fn=_get_time,
            _get_scroll_lines_fn=lambda: 3,
        )
        await t.double_click((50, 50))
        assert len(dbl_time_calls) == 1


# ---------------------------------------------------------------------------
# TestScroll
# ---------------------------------------------------------------------------


class TestScroll:
    async def test_scroll_up_returns_true(self) -> None:
        t, _ = _make_transport()
        assert await t.scroll((100, 100), "up") is True

    async def test_scroll_down_returns_true(self) -> None:
        t, _ = _make_transport()
        assert await t.scroll((100, 100), "down") is True

    async def test_scroll_sends_move_and_wheel_events(self) -> None:
        t, rec = _make_transport()
        await t.scroll((100, 100), "up")
        flags = [e.flags for e in rec.all_events]
        assert any(f & _MOUSEEVENTF_MOVE for f in flags)
        assert any(f & _MOUSEEVENTF_WHEEL for f in flags)

    async def test_scroll_up_has_positive_wheel_delta(self) -> None:
        t, rec = _make_transport(scroll_lines=3)
        await t.scroll((100, 100), "up", amount=1)
        wheel_event = next(
            e for e in rec.all_events if e.flags & _MOUSEEVENTF_WHEEL
        )
        # delta positive means up; stored as unsigned DWORD but <2^31 means positive
        delta = wheel_event.mouse_data
        # delta == WHEEL_DELTA * scroll_lines = 120 * 3 = 360
        assert delta == _WHEEL_DELTA * 3

    async def test_scroll_down_has_negative_wheel_delta(self) -> None:
        t, rec = _make_transport(scroll_lines=3)
        await t.scroll((100, 100), "down", amount=1)
        wheel_event = next(
            e for e in rec.all_events if e.flags & _MOUSEEVENTF_WHEEL
        )
        # Negative delta stored as unsigned 32-bit: -360 → 4294966936
        delta_signed = wheel_event.mouse_data
        if delta_signed > 0x7FFFFFFF:
            delta_signed -= 0x100000000
        assert delta_signed == -(_WHEEL_DELTA * 3)

    async def test_scroll_uses_system_scroll_lines(self) -> None:
        """Scroll delta respects the system scroll lines setting."""
        t, rec = _make_transport(scroll_lines=5)
        await t.scroll((100, 100), "up", amount=1)
        wheel_event = next(
            e for e in rec.all_events if e.flags & _MOUSEEVENTF_WHEEL
        )
        assert wheel_event.mouse_data == _WHEEL_DELTA * 5

    async def test_scroll_amount_multiplier(self) -> None:
        t, rec = _make_transport(scroll_lines=3)
        await t.scroll((100, 100), "up", amount=2)
        wheel_event = next(
            e for e in rec.all_events if e.flags & _MOUSEEVENTF_WHEEL
        )
        assert wheel_event.mouse_data == _WHEEL_DELTA * 3 * 2


# ---------------------------------------------------------------------------
# TestDrag
# ---------------------------------------------------------------------------


class TestDrag:
    async def test_drag_returns_true(self) -> None:
        t, _ = _make_transport()
        assert await t.drag((10, 10), (200, 200)) is True

    async def test_drag_sends_four_events(self) -> None:
        t, rec = _make_transport()
        await t.drag((10, 10), (200, 200))
        assert len(rec.all_events) == 4

    async def test_drag_event_sequence_is_move_down_move_up(self) -> None:
        t, rec = _make_transport()
        await t.drag((10, 10), (200, 200))
        flags = [e.flags for e in rec.all_events]
        assert flags[0] & _MOUSEEVENTF_MOVE       # move to start
        assert flags[1] & _MOUSEEVENTF_LEFTDOWN   # press
        assert flags[2] & _MOUSEEVENTF_MOVE       # move to end
        assert flags[3] & _MOUSEEVENTF_LEFTUP     # release

    async def test_drag_start_coords_in_first_two_events(self) -> None:
        t, rec = _make_transport(scale=1.0)
        await t.drag((10, 20), (300, 400))
        events = rec.all_events
        assert events[0].phys_x == 10
        assert events[1].phys_x == 10

    async def test_drag_end_coords_in_last_two_events(self) -> None:
        t, rec = _make_transport(scale=1.0)
        await t.drag((10, 20), (300, 400))
        events = rec.all_events
        assert events[2].phys_x == 300
        assert events[3].phys_x == 300

    async def test_drag_returns_false_on_send_failure(self) -> None:
        t, _ = _make_transport(return_count=0)
        assert await t.drag((0, 0), (100, 100)) is False
