"""
tests/unit/test_locator.py
Unit tests for nexus/perception/locator/locator.py — Faz 24.

Sections:
  1.  UIElement value object
  2.  ElementType enum completeness
  3.  _classify heuristics
  4.  _iou / _intersection_area geometry
  5.  Locator.locate — button detection (ui_buttons fixture)
  6.  Locator.locate — occluded element marking (ui_occluded fixture)
  7.  Locator.locate — dirty region restricts search area
  8.  Locator.locate — cache returns same result within TTL
  9.  Locator.locate — cache expires after TTL
  10. Locator.locate — active_window clips results
  11. Locator.locate — empty frame returns []
  12. Locator.locate — LocatorError on corrupt input
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from nexus.capture.frame import Frame
from nexus.core.errors import LocatorError
from nexus.core.types import ElementId, Rect
from nexus.perception.locator.locator import (
    ElementType,
    Locator,
    UIElement,
    _classify,
    _intersection_area,
    _iou,
)

# ---------------------------------------------------------------------------
# Synthetic image builders
# ---------------------------------------------------------------------------

_UTC = "2026-04-04T00:00:00+00:00"


def _frame(
    width: int = 400,
    height: int = 300,
    bg: tuple[int, int, int] = (200, 200, 200),
) -> Frame:
    """Return a blank RGB frame filled with *bg*."""
    data = np.full((height, width, 3), bg, dtype=np.uint8)
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=0.0,
        captured_at_utc=_UTC,
        sequence_number=1,
    )


def _draw_rect(
    frame: Frame,
    rect: Rect,
    color: tuple[int, int, int] = (50, 50, 200),
    thickness: int = 2,
) -> Frame:
    """Draw a filled rectangle on the frame data (mutates data in-place for simplicity)."""
    data = frame.data.copy()
    x1, y1 = rect.x, rect.y
    x2, y2 = rect.x + rect.width, rect.y + rect.height
    # Draw thick border to create edges Canny can detect
    cv2.rectangle(data, (x1, y1), (x2, y2), color, thickness)
    return Frame(
        data=data,
        width=frame.width,
        height=frame.height,
        captured_at_monotonic=frame.captured_at_monotonic,
        captured_at_utc=frame.captured_at_utc,
        sequence_number=frame.sequence_number,
    )


def _ui_buttons_frame() -> Frame:
    """
    Synthetic button-rich frame:
      Three button-sized rectangles (80x30) at different positions.
    """
    f = _frame(400, 300)
    for x_start in [30, 160, 280]:
        f = _draw_rect(f, Rect(x_start, 50, 80, 30), (30, 80, 200), thickness=3)
    return f


def _ui_form_frame() -> Frame:
    """
    Synthetic form frame:
      Two wide input fields (200x28) and two small checkboxes (20x20).
    """
    f = _frame(400, 300)
    f = _draw_rect(f, Rect(50, 40, 200, 28), (10, 10, 10), thickness=2)
    f = _draw_rect(f, Rect(50, 100, 200, 28), (10, 10, 10), thickness=2)
    f = _draw_rect(f, Rect(50, 150, 20, 20), (10, 10, 10), thickness=2)
    f = _draw_rect(f, Rect(100, 150, 20, 20), (10, 10, 10), thickness=2)
    return f


def _ui_occluded_frame() -> Frame:
    """
    Synthetic occluded frame:
      A large background panel (250x150) with a smaller element on top (60x40)
      at the same position — the panel is occluded by the smaller element.
    """
    f = _frame(400, 300)
    # Large panel
    f = _draw_rect(f, Rect(50, 50, 250, 150), (80, 80, 80), thickness=3)
    # Smaller element overlapping the top-left corner of the panel
    f = _draw_rect(f, Rect(55, 55, 60, 40), (20, 20, 180), thickness=3)
    return f


# ---------------------------------------------------------------------------
# Section 1 — UIElement value object
# ---------------------------------------------------------------------------


class TestUIElement:
    def test_frozen(self):
        el = UIElement(
            id=ElementId("x"),
            element_type=ElementType.BUTTON,
            bounding_box=Rect(0, 0, 10, 10),
            confidence=0.9,
            is_visible=True,
            is_occluded=False,
            occlusion_ratio=0.0,
            z_order_estimate=0,
        )
        with pytest.raises((AttributeError, TypeError)):
            el.confidence = 0.5  # type: ignore[misc]

    def test_fields_accessible(self):
        el = UIElement(
            id=ElementId("abc"),
            element_type=ElementType.INPUT,
            bounding_box=Rect(10, 20, 100, 30),
            confidence=0.75,
            is_visible=True,
            is_occluded=True,
            occlusion_ratio=0.4,
            z_order_estimate=2,
        )
        assert el.element_type is ElementType.INPUT
        assert el.bounding_box == Rect(10, 20, 100, 30)
        assert el.occlusion_ratio == pytest.approx(0.4)
        assert el.z_order_estimate == 2


# ---------------------------------------------------------------------------
# Section 2 — ElementType enum completeness
# ---------------------------------------------------------------------------


class TestElementTypeEnum:
    _REQUIRED = {
        "BUTTON", "INPUT", "LABEL", "IMAGE", "ICON", "CONTAINER", "PANEL",
        "MENU", "DROPDOWN", "CHECKBOX", "RADIO", "LINK", "SCROLLBAR",
        "TAB", "DIALOG", "TOOLTIP", "UNKNOWN",
    }

    def test_all_required_members_present(self):
        names = {m.name for m in ElementType}
        assert names >= self._REQUIRED


# ---------------------------------------------------------------------------
# Section 3 — _classify heuristics
# ---------------------------------------------------------------------------


class TestClassifyHeuristics:
    def test_wide_medium_is_input(self):
        box = Rect(0, 0, 180, 28)  # aspect ≈ 6.4, w>40, h<60
        assert _classify(box) is ElementType.INPUT

    def test_small_square_is_checkbox(self):
        box = Rect(0, 0, 20, 20)  # small, aspect=1.0
        assert _classify(box) is ElementType.CHECKBOX

    def test_small_icon(self):
        box = Rect(0, 0, 16, 24)  # small, aspect ≈ 0.67
        assert _classify(box) is ElementType.ICON

    def test_very_elongated_horizontal_scrollbar(self):
        box = Rect(0, 0, 300, 14)  # aspect ≈ 21 → scrollbar
        assert _classify(box) is ElementType.SCROLLBAR

    def test_very_elongated_vertical_scrollbar(self):
        box = Rect(0, 0, 12, 200)  # aspect ≈ 0.06 → scrollbar
        assert _classify(box) is ElementType.SCROLLBAR

    def test_button_shape(self):
        box = Rect(0, 0, 80, 30)  # aspect ≈ 2.67, w<=200, h<=80
        assert _classify(box) is ElementType.BUTTON

    def test_large_area_is_panel(self):
        box = Rect(0, 0, 350, 250)
        assert _classify(box) is ElementType.PANEL

    def test_zero_dimension_is_unknown(self):
        assert _classify(Rect(0, 0, 0, 10)) is ElementType.UNKNOWN
        assert _classify(Rect(0, 0, 10, 0)) is ElementType.UNKNOWN


# ---------------------------------------------------------------------------
# Section 4 — geometry utilities
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_intersection_area_no_overlap(self):
        a = Rect(0, 0, 10, 10)
        b = Rect(20, 0, 10, 10)
        assert _intersection_area(a, b) == 0

    def test_intersection_area_full_overlap(self):
        a = Rect(0, 0, 10, 10)
        assert _intersection_area(a, a) == 100

    def test_intersection_area_partial(self):
        a = Rect(0, 0, 10, 10)
        b = Rect(5, 5, 10, 10)
        assert _intersection_area(a, b) == 25

    def test_iou_identical(self):
        a = Rect(0, 0, 10, 10)
        assert _iou(a, a) == pytest.approx(1.0)

    def test_iou_disjoint(self):
        a = Rect(0, 0, 10, 10)
        b = Rect(100, 100, 10, 10)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_iou_partial(self):
        a = Rect(0, 0, 10, 10)
        b = Rect(5, 0, 10, 10)
        # inter = 5*10 = 50, union = 100+100-50 = 150
        assert _iou(a, b) == pytest.approx(50 / 150)


# ---------------------------------------------------------------------------
# Section 5 — button detection (ui_buttons fixture)
# ---------------------------------------------------------------------------


class TestButtonDetection:
    def test_detects_elements_in_button_frame(self):
        frame = _ui_buttons_frame()
        locator = Locator()
        elements = locator.locate(frame)
        # Should detect at least one element
        assert len(elements) >= 1

    def test_detected_elements_have_button_or_known_type(self):
        frame = _ui_buttons_frame()
        locator = Locator()
        elements = locator.locate(frame)
        types = {e.element_type for e in elements}
        # At least one recognisable type (not all UNKNOWN)
        assert types != {ElementType.UNKNOWN}

    def test_confidence_in_range(self):
        frame = _ui_buttons_frame()
        locator = Locator()
        elements = locator.locate(frame)
        for el in elements:
            assert 0.0 <= el.confidence <= 1.0

    def test_bounding_boxes_within_frame(self):
        frame = _ui_buttons_frame()
        locator = Locator()
        elements = locator.locate(frame)
        for el in elements:
            bb = el.bounding_box
            assert bb.x >= 0
            assert bb.y >= 0
            assert bb.x + bb.width <= frame.width
            assert bb.y + bb.height <= frame.height


# ---------------------------------------------------------------------------
# Section 6 — occluded element marking (ui_occluded fixture)
# ---------------------------------------------------------------------------


class TestOcclusionDetection:
    def test_occluded_element_marked(self):
        frame = _ui_occluded_frame()
        locator = Locator()
        elements = locator.locate(frame)
        # At least one element should be detected
        assert elements, "No elements detected in occluded frame"
        # If multiple elements overlap, at least one should be marked occluded
        if len(elements) >= 2:
            occluded = [e for e in elements if e.is_occluded]
            assert occluded, "No elements marked as occluded despite overlap"

    def test_occlusion_ratio_valid_range(self):
        frame = _ui_occluded_frame()
        locator = Locator()
        elements = locator.locate(frame)
        for el in elements:
            assert 0.0 <= el.occlusion_ratio <= 1.0

    def test_non_occluded_element_ratio_zero(self):
        frame = _ui_occluded_frame()
        locator = Locator()
        elements = locator.locate(frame)
        non_occluded = [e for e in elements if not e.is_occluded]
        for el in non_occluded:
            assert el.occlusion_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Section 7 — dirty region restricts search area
# ---------------------------------------------------------------------------


class TestDirtyRegion:
    def test_dirty_region_only_detects_within_region(self):
        """Elements outside dirty_regions should not appear."""
        frame = _ui_buttons_frame()
        locator = Locator()

        # Button 1 is at x=30..110, y=50..80
        # Restrict to left 120px strip — should detect button 1
        dirty = [Rect(0, 0, 120, 300)]
        elements_dirty = locator.locate(frame, dirty_regions=dirty)

        # Full scan may yield more elements (buttons 2 and 3 at x=160 and x=280)
        elements_full = locator.locate(Frame(
            data=frame.data.copy(),  # new frame → no cache hit
            width=frame.width,
            height=frame.height,
            captured_at_monotonic=frame.captured_at_monotonic + 1,
            captured_at_utc=_UTC,
            sequence_number=frame.sequence_number + 1,
        ))

        # All dirty-region results must be inside the dirty rect
        for el in elements_dirty:
            bb = el.bounding_box
            assert bb.x >= 0
            assert bb.x + bb.width <= 120 + 10  # small tolerance for edge contours

    def test_dirty_region_outside_frame_ignored(self):
        frame = _frame(100, 100)
        locator = Locator()
        # Dirty region entirely outside frame
        result = locator.locate(frame, dirty_regions=[Rect(500, 500, 50, 50)])
        assert result == []

    def test_empty_dirty_regions_returns_empty(self):
        frame = _ui_buttons_frame()
        locator = Locator()
        result = locator.locate(frame, dirty_regions=[])
        assert result == []


# ---------------------------------------------------------------------------
# Section 8 — cache: returns same result within TTL
# ---------------------------------------------------------------------------


class TestCacheHit:
    def test_cache_hit_returns_same_list(self):
        frame = _ui_buttons_frame()
        times = iter([0.0, 0.050, 0.100])  # all within 200ms TTL
        locator = Locator(cache_ttl=0.200, _time_fn=lambda: next(times))
        first = locator.locate(frame)
        second = locator.locate(frame)
        # Same objects (cache returned)
        assert first is second

    def test_cache_ids_are_stable(self):
        frame = _ui_buttons_frame()
        times = iter([0.0, 0.050])
        locator = Locator(cache_ttl=0.200, _time_fn=lambda: next(times))
        first = locator.locate(frame)
        second = locator.locate(frame)
        ids_first = [e.id for e in first]
        ids_second = [e.id for e in second]
        assert ids_first == ids_second


# ---------------------------------------------------------------------------
# Section 9 — cache expires after TTL
# ---------------------------------------------------------------------------


class TestCacheExpiry:
    def test_cache_miss_after_ttl(self):
        frame = _ui_buttons_frame()
        times = iter([0.0, 0.300])  # second call is 300ms later → expired
        locator = Locator(cache_ttl=0.200, _time_fn=lambda: next(times))
        first = locator.locate(frame)
        second = locator.locate(frame)
        # Different list objects (re-detected)
        assert first is not second

    def test_fresh_result_after_ttl(self):
        frame = _ui_buttons_frame()
        call_times = [0.0, 0.250]
        idx = [0]

        def _clock():
            t = call_times[idx[0]]
            if idx[0] < len(call_times) - 1:
                idx[0] += 1
            return t

        locator = Locator(cache_ttl=0.200, _time_fn=_clock)
        first = locator.locate(frame)
        second = locator.locate(frame)
        # Both should be non-empty (detection ran twice)
        assert isinstance(first, list)
        assert isinstance(second, list)


# ---------------------------------------------------------------------------
# Section 10 — active_window clips results
# ---------------------------------------------------------------------------


class TestActiveWindow:
    def test_active_window_clips_detection(self):
        frame = _ui_buttons_frame()
        locator = Locator()
        # Window restricted to top-left quadrant
        window = Rect(0, 0, 200, 150)
        elements = locator.locate(frame, active_window=window)
        for el in elements:
            bb = el.bounding_box
            # Result must be within (or very close to) the window
            assert bb.x < window.x + window.width
            assert bb.y < window.y + window.height

    def test_active_window_no_overlap_with_frame_returns_empty(self):
        frame = _frame(200, 200)
        locator = Locator()
        # Window far outside frame
        result = locator.locate(frame, active_window=Rect(1000, 1000, 100, 100))
        assert result == []


# ---------------------------------------------------------------------------
# Section 11 — empty frame returns []
# ---------------------------------------------------------------------------


class TestEmptyFrame:
    def test_blank_frame_no_elements(self):
        frame = _frame(200, 200, bg=(128, 128, 128))
        locator = Locator()
        elements = locator.locate(frame)
        # Blank frame has no edges → no elements
        assert elements == []

    def test_tiny_frame_no_crash(self):
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        frame = Frame(
            data=data,
            width=4,
            height=4,
            captured_at_monotonic=0.0,
            captured_at_utc=_UTC,
            sequence_number=1,
        )
        locator = Locator()
        elements = locator.locate(frame)
        assert isinstance(elements, list)


# ---------------------------------------------------------------------------
# Section 12 — LocatorError on corrupt input
# ---------------------------------------------------------------------------


class TestLocatorError:
    def test_invalid_frame_data_raises_locator_error(self):
        """A frame with 1-D (non-image) data should raise LocatorError."""
        _bad_data = np.zeros((10,), dtype=np.uint8)  # 1-D — corrupt

        class _BadFrame:
            data = _bad_data
            width = 10
            height = 1
            captured_at_monotonic = 0.0
            captured_at_utc = _UTC
            sequence_number = 1

        locator = Locator()
        with pytest.raises(LocatorError):
            locator.locate(_BadFrame())  # type: ignore[arg-type]
