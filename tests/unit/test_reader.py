"""
tests/unit/test_reader.py
Unit tests for nexus/perception/reader/reader.py — Faz 25.

Sections:
  1.  ReaderOutput value object
  2.  LayoutRegion / TableData value objects
  3.  _find_gaps helper
  4.  Reader.read — basic text extraction
  5.  Reader.read — reading order (single column)
  6.  Reader.read — multi-column reading order
  7.  Reader.read — table detection
  8.  Reader.read — selective OCR cache (cache hit)
  9.  Reader.read — selective OCR cache (dirty region bypasses cache)
  10. Reader.read — dirty region filters elements
  11. Reader.read — invisible elements skipped
  12. Reader.read — empty element list
  13. Reader.read — OCR failure is non-fatal
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from nexus.capture.frame import Frame
from nexus.core.types import ElementId, Rect
from nexus.perception.locator.locator import ElementType, UIElement
from nexus.perception.reader.ocr_engine import OCRResult
from nexus.perception.reader.reader import (
    LayoutRegion,
    LayoutRegionType,
    Reader,
    ReaderOutput,
    TableData,
    _find_gaps,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_UTC = "2026-04-05T00:00:00+00:00"


def _frame(width: int = 600, height: int = 400) -> Frame:
    data = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=0.0,
        captured_at_utc=_UTC,
        sequence_number=1,
    )


def _element(
    x: int,
    y: int,
    w: int = 80,
    h: int = 30,
    el_type: ElementType = ElementType.BUTTON,
    is_visible: bool = True,
) -> UIElement:
    return UIElement(
        id=ElementId(str(uuid.uuid4())),
        element_type=el_type,
        bounding_box=Rect(x, y, w, h),
        confidence=0.9,
        is_visible=is_visible,
        is_occluded=False,
        occlusion_ratio=0.0,
        z_order_estimate=0,
    )


def _ocr(text: str, x: int, y: int, w: int = 60, h: int = 14) -> OCRResult:
    return OCRResult(
        text=text,
        confidence=0.95,
        bounding_box=Rect(x, y, w, h),
        language="eng",
    )


class _MockOCR:
    """
    Configurable mock OCR engine.

    Pass a mapping of Rect → list[OCRResult] to control what is returned
    for each region, or a single list for any region.
    """

    def __init__(
        self,
        results: dict[tuple[int, int, int, int], list[OCRResult]] | list[OCRResult]
        | None = None,
        side_effect: Exception | None = None,
    ) -> None:
        self._results = results or {}
        self._side_effect = side_effect
        self.call_count = 0
        self.call_regions: list[Rect | None] = []

    def extract(
        self,
        image: np.ndarray,
        region: Rect | None = None,
        languages: list[str] | None = None,
    ) -> list[OCRResult]:
        self.call_count += 1
        self.call_regions.append(region)

        if self._side_effect is not None:
            raise self._side_effect

        if isinstance(self._results, list):
            return self._results

        if region is not None:
            key = (region.x, region.y, region.width, region.height)
            return self._results.get(key, [])
        return []


# ---------------------------------------------------------------------------
# Section 1 — ReaderOutput value object
# ---------------------------------------------------------------------------


class TestReaderOutput:
    def test_fields_accessible(self):
        out = ReaderOutput(
            element_texts={},
            text_blocks=[],
            layout_regions=[],
            reading_order=[],
            table_data=None,
        )
        assert out.element_texts == {}
        assert out.table_data is None

    def test_frozen(self):
        out = ReaderOutput(
            element_texts={},
            text_blocks=[],
            layout_regions=[],
            reading_order=[],
            table_data=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            out.table_data = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Section 2 — LayoutRegion / TableData value objects
# ---------------------------------------------------------------------------


class TestValueObjects:
    def test_layout_region_frozen(self):
        lr = LayoutRegion(
            bounding_box=Rect(0, 0, 100, 200),
            region_type=LayoutRegionType.COLUMN,
            column_index=0,
        )
        with pytest.raises((AttributeError, TypeError)):
            lr.column_index = 1  # type: ignore[misc]

    def test_table_data_rows_accessible(self):
        td = TableData(
            bounding_box=Rect(0, 0, 300, 100),
            rows=(("A", "B"), ("1", "2")),
            header_row=("A", "B"),
        )
        assert td.rows[0] == ("A", "B")
        assert td.header_row == ("A", "B")

    def test_table_data_frozen(self):
        td = TableData(
            bounding_box=Rect(0, 0, 100, 50),
            rows=(("x",),),
            header_row=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            td.header_row = ("x",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Section 3 — _find_gaps helper
# ---------------------------------------------------------------------------


class TestFindGaps:
    def test_no_gaps(self):
        cov = np.ones(100, dtype=np.int32)
        assert _find_gaps(cov, min_gap=5) == []

    def test_single_gap(self):
        cov = np.ones(100, dtype=np.int32)
        cov[40:60] = 0
        gaps = _find_gaps(cov, min_gap=5)
        assert gaps == [(40, 60)]

    def test_gap_too_small_ignored(self):
        cov = np.ones(100, dtype=np.int32)
        cov[40:43] = 0  # only 3 wide
        assert _find_gaps(cov, min_gap=5) == []

    def test_two_gaps(self):
        cov = np.ones(200, dtype=np.int32)
        cov[40:60] = 0
        cov[120:150] = 0
        gaps = _find_gaps(cov, min_gap=5)
        assert len(gaps) == 2
        assert gaps[0] == (40, 60)
        assert gaps[1] == (120, 150)

    def test_gap_at_start(self):
        cov = np.zeros(20, dtype=np.int32)
        cov[10:] = 1
        gaps = _find_gaps(cov, min_gap=5)
        assert gaps == [(0, 10)]

    def test_gap_at_end(self):
        cov = np.zeros(20, dtype=np.int32)
        cov[:10] = 1
        gaps = _find_gaps(cov, min_gap=5)
        assert gaps == [(10, 20)]


# ---------------------------------------------------------------------------
# Section 4 — basic text extraction
# ---------------------------------------------------------------------------


class TestBasicExtraction:
    def test_text_returned_per_element(self):
        el = _element(10, 10)
        ocr = _MockOCR(
            {(10, 10, 80, 30): [_ocr("Hello", 10, 10), _ocr("World", 70, 10)]}
        )
        reader = Reader(ocr)
        out = reader.read(_frame(), [el])
        assert out.element_texts[el.id] == "Hello World"

    def test_text_blocks_collected(self):
        el1 = _element(10, 10)
        el2 = _element(10, 60)
        ocr = _MockOCR(
            {
                (10, 10, 80, 30): [_ocr("A", 10, 10)],
                (10, 60, 80, 30): [_ocr("B", 10, 60)],
            }
        )
        reader = Reader(ocr)
        out = reader.read(_frame(), [el1, el2])
        texts = [r.text for r in out.text_blocks]
        assert "A" in texts
        assert "B" in texts

    def test_empty_ocr_gives_empty_string(self):
        el = _element(0, 0)
        reader = Reader(_MockOCR({}))
        out = reader.read(_frame(), [el])
        assert out.element_texts[el.id] == ""


# ---------------------------------------------------------------------------
# Section 5 — reading order (single column)
# ---------------------------------------------------------------------------


class TestReadingOrderSingleColumn:
    def test_top_to_bottom(self):
        els = [
            _element(20, 200),
            _element(20, 50),
            _element(20, 125),
        ]
        reader = Reader(_MockOCR([]))
        out = reader.read(_frame(400, 400), els)
        ys = [
            next(e.bounding_box.y for e in els if e.id == eid)
            for eid in out.reading_order
        ]
        assert ys == sorted(ys)

    def test_same_row_left_to_right(self):
        els = [
            _element(300, 50),
            _element(10, 50),
            _element(150, 50),
        ]
        reader = Reader(_MockOCR([]))
        out = reader.read(_frame(400, 200), els)
        xs = [
            next(e.bounding_box.x for e in els if e.id == eid)
            for eid in out.reading_order
        ]
        assert xs == sorted(xs)

    def test_all_elements_in_order(self):
        els = [_element(0, i * 40) for i in range(5)]
        reader = Reader(_MockOCR([]))
        out = reader.read(_frame(), els)
        assert set(out.reading_order) == {e.id for e in els}
        assert len(out.reading_order) == 5


# ---------------------------------------------------------------------------
# Section 6 — multi-column reading order
# ---------------------------------------------------------------------------


class TestMultiColumnReadingOrder:
    """
    Two-column layout:
      Left column  (x=10..110):  elements at y=50, y=100, y=150
      Right column (x=310..410): elements at y=30, y=80, y=130

    Expected reading order: all left-column elements (sorted by y) THEN
    all right-column elements (sorted by y).
    """

    def _make_two_column_elements(self) -> list[UIElement]:
        left = [_element(10, y, w=100, h=30) for y in [50, 100, 150]]
        right = [_element(310, y, w=100, h=30) for y in [30, 80, 130]]
        return left + right

    def test_left_column_precedes_right(self):
        els = self._make_two_column_elements()
        frame = _frame(500, 300)
        reader = Reader(_MockOCR([]))
        out = reader.read(frame, els)

        order_ids = out.reading_order
        left_ids = {e.id for e in els if e.bounding_box.x < 200}
        right_ids = {e.id for e in els if e.bounding_box.x >= 200}

        left_positions = [i for i, eid in enumerate(order_ids) if eid in left_ids]
        right_positions = [i for i, eid in enumerate(order_ids) if eid in right_ids]

        assert max(left_positions) < min(right_positions), (
            "All left-column elements must appear before right-column elements"
        )

    def test_within_column_top_to_bottom(self):
        els = self._make_two_column_elements()
        frame = _frame(500, 300)
        reader = Reader(_MockOCR([]))
        out = reader.read(frame, els)

        order_ids = out.reading_order
        left_els = sorted(
            [e for e in els if e.bounding_box.x < 200],
            key=lambda e: e.bounding_box.y,
        )
        right_els = sorted(
            [e for e in els if e.bounding_box.x >= 200],
            key=lambda e: e.bounding_box.y,
        )

        # Left-column order in reading_order must be top-to-bottom
        left_order = [eid for eid in order_ids if eid in {e.id for e in left_els}]
        assert left_order == [e.id for e in left_els]

        # Right-column order must also be top-to-bottom
        right_order = [eid for eid in order_ids if eid in {e.id for e in right_els}]
        assert right_order == [e.id for e in right_els]

    def test_column_layout_regions_detected(self):
        els = self._make_two_column_elements()
        frame = _frame(500, 300)
        reader = Reader(_MockOCR([]))
        out = reader.read(frame, els)
        col_regions = [
            r for r in out.layout_regions
            if r.region_type is LayoutRegionType.COLUMN
        ]
        assert len(col_regions) >= 2


# ---------------------------------------------------------------------------
# Section 7 — table detection
# ---------------------------------------------------------------------------


class TestTableDetection:
    def _table_blocks(self) -> list[OCRResult]:
        """
        2-row × 3-column table:
          Row 0 (y≈10): "Name", "Age", "City"
          Row 1 (y≈30): "Alice", "30", "London"
        """
        return [
            _ocr("Name", 10, 10, w=50, h=14),
            _ocr("Age", 70, 10, w=30, h=14),
            _ocr("City", 110, 10, w=40, h=14),
            _ocr("Alice", 10, 30, w=50, h=14),
            _ocr("30", 70, 30, w=20, h=14),
            _ocr("London", 110, 30, w=60, h=14),
        ]

    def test_table_detected(self):
        blocks = self._table_blocks()
        el = _element(0, 0, w=200, h=60)
        ocr = _MockOCR({(0, 0, 200, 60): blocks})
        reader = Reader(ocr)
        out = reader.read(_frame(), [el])
        assert out.table_data is not None
        assert len(out.table_data) == 1

    def test_table_has_two_rows(self):
        blocks = self._table_blocks()
        el = _element(0, 0, w=200, h=60)
        ocr = _MockOCR({(0, 0, 200, 60): blocks})
        reader = Reader(ocr)
        out = reader.read(_frame(), [el])
        assert out.table_data is not None
        td = out.table_data[0]
        assert len(td.rows) == 2

    def test_table_header_row(self):
        blocks = self._table_blocks()
        el = _element(0, 0, w=200, h=60)
        ocr = _MockOCR({(0, 0, 200, 60): blocks})
        reader = Reader(ocr)
        out = reader.read(_frame(), [el])
        assert out.table_data is not None
        td = out.table_data[0]
        assert td.header_row is not None
        assert "Name" in td.header_row

    def test_table_cells_sorted_left_to_right(self):
        blocks = self._table_blocks()
        el = _element(0, 0, w=200, h=60)
        ocr = _MockOCR({(0, 0, 200, 60): blocks})
        reader = Reader(ocr)
        out = reader.read(_frame(), [el])
        assert out.table_data is not None
        header = out.table_data[0].header_row
        assert header is not None
        assert header[0] == "Name"
        assert header[1] == "Age"
        assert header[2] == "City"

    def test_single_row_not_a_table(self):
        blocks = [_ocr("A", 0, 10), _ocr("B", 60, 10), _ocr("C", 120, 10)]
        el = _element(0, 0, w=200, h=30)
        ocr = _MockOCR({(0, 0, 200, 30): blocks})
        reader = Reader(ocr)
        out = reader.read(_frame(), [el])
        assert out.table_data is None

    def test_single_col_per_row_not_a_table(self):
        # Each row has only one word → not a table
        blocks = [
            _ocr("Hello", 10, 10),
            _ocr("World", 10, 40),
            _ocr("Foo", 10, 70),
        ]
        el = _element(0, 0, w=100, h=100)
        ocr = _MockOCR({(0, 0, 100, 100): blocks})
        reader = Reader(ocr)
        out = reader.read(_frame(), [el])
        assert out.table_data is None


# ---------------------------------------------------------------------------
# Section 8 — selective OCR cache (hit)
# ---------------------------------------------------------------------------


class TestOCRCacheHit:
    def test_second_read_uses_cache(self):
        """
        Two reads of the same elements with no dirty regions.
        First read: OCR called once per element.
        Second read: OCR NOT called (cache serves result).
        """
        els = [_element(10, 10), _element(10, 60)]
        mock_ocr = _MockOCR([_ocr("X", 10, 10)])
        times = iter([0.0, 0.1, 0.2, 0.3])
        reader = Reader(mock_ocr, cache_ttl=5.0, _time_fn=lambda: next(times))

        reader.read(_frame(), els)
        count_after_first = mock_ocr.call_count

        reader.read(_frame(), els)
        count_after_second = mock_ocr.call_count

        # Second read must not add any new OCR calls
        assert count_after_second == count_after_first

    def test_cache_returns_same_text(self):
        el = _element(20, 20)
        mock_ocr = _MockOCR({(20, 20, 80, 30): [_ocr("Cached", 20, 20)]})
        times = iter([0.0, 0.5, 1.0, 1.5])
        reader = Reader(mock_ocr, cache_ttl=5.0, _time_fn=lambda: next(times))

        out1 = reader.read(_frame(), [el])
        out2 = reader.read(_frame(), [el])

        assert out1.element_texts[el.id] == "Cached"
        assert out2.element_texts[el.id] == "Cached"

    def test_cache_expiry_triggers_reocr(self):
        """After TTL expires, OCR runs again."""
        el = _element(5, 5)
        mock_ocr = _MockOCR([_ocr("Fresh", 5, 5)])
        # _time_fn is called once per read(); first at t=0, second at t=10
        times = iter([0.0, 10.0])
        reader = Reader(mock_ocr, cache_ttl=5.0, _time_fn=lambda: next(times))

        reader.read(_frame(), [el])
        count_after_first = mock_ocr.call_count

        reader.read(_frame(), [el])
        count_after_second = mock_ocr.call_count

        assert count_after_second > count_after_first


# ---------------------------------------------------------------------------
# Section 9 — dirty region bypasses cache
# ---------------------------------------------------------------------------


class TestDirtyRegionBypassesCache:
    def test_dirty_element_re_ocrd(self):
        """
        Second read marks the element's region as dirty → OCR runs again.
        """
        el = _element(10, 10, w=80, h=30)
        mock_ocr = _MockOCR([_ocr("Z", 10, 10)])
        times = iter([0.0, 0.1, 0.2, 0.3])
        reader = Reader(mock_ocr, cache_ttl=5.0, _time_fn=lambda: next(times))

        reader.read(_frame(), [el])
        count_after_first = mock_ocr.call_count

        # Mark the element's region dirty
        reader.read(_frame(), [el], dirty_regions=[Rect(0, 0, 200, 100)])
        count_after_second = mock_ocr.call_count

        assert count_after_second > count_after_first

    def test_non_dirty_element_not_re_ocrd(self):
        """
        Two elements; only one is dirty. The clean one should use cache.
        """
        el_clean = _element(10, 10, w=80, h=30)   # y=10..40
        el_dirty = _element(10, 200, w=80, h=30)  # y=200..230

        mock_ocr = _MockOCR([_ocr("T", 10, 10)])
        times = iter([0.0, 0.0, 1.0, 1.0, 1.0])
        reader = Reader(mock_ocr, cache_ttl=5.0, _time_fn=lambda: next(times))

        # First read — both elements OCR'd
        reader.read(_frame(), [el_clean, el_dirty])
        count_after_first = mock_ocr.call_count

        # Second read — only dirty region (y=150..250) is marked dirty
        reader.read(
            _frame(),
            [el_clean, el_dirty],
            dirty_regions=[Rect(0, 150, 600, 100)],
        )
        count_after_second = mock_ocr.call_count

        # Only el_dirty should be re-OCR'd → exactly 1 extra call
        assert count_after_second == count_after_first + 1


# ---------------------------------------------------------------------------
# Section 10 — dirty region filters elements
# ---------------------------------------------------------------------------


class TestDirtyRegionFiltering:
    def test_elements_outside_dirty_region_use_cache(self):
        """
        Elements completely outside all dirty regions are served from cache.
        On first read (no cache), they are OCR'd. On second read with a dirty
        region that doesn't overlap them, cache is used.
        """
        el = _element(400, 300, w=80, h=30)  # bottom-right area
        mock_ocr = _MockOCR([_ocr("Far", 400, 300)])
        times = iter([0.0, 0.1, 0.2, 0.3])
        reader = Reader(mock_ocr, cache_ttl=5.0, _time_fn=lambda: next(times))

        reader.read(_frame(600, 400), [el])
        count1 = mock_ocr.call_count

        # Dirty region in top-left, far from el
        reader.read(
            _frame(600, 400), [el], dirty_regions=[Rect(0, 0, 50, 50)]
        )
        count2 = mock_ocr.call_count

        assert count2 == count1  # no extra OCR call

    def test_empty_dirty_regions_all_from_cache(self):
        el = _element(10, 10)
        mock_ocr = _MockOCR([_ocr("K", 10, 10)])
        times = iter([0.0, 1.0, 2.0])
        reader = Reader(mock_ocr, cache_ttl=5.0, _time_fn=lambda: next(times))

        reader.read(_frame(), [el])
        count1 = mock_ocr.call_count

        # Empty dirty_regions list → no element is dirty → all from cache
        reader.read(_frame(), [el], dirty_regions=[])
        count2 = mock_ocr.call_count

        assert count2 == count1


# ---------------------------------------------------------------------------
# Section 11 — invisible elements skipped
# ---------------------------------------------------------------------------


class TestInvisibleElements:
    def test_invisible_element_not_ocrd(self):
        visible = _element(0, 0, is_visible=True)
        invisible = _element(100, 100, is_visible=False)
        mock_ocr = _MockOCR([_ocr("V", 0, 0)])
        reader = Reader(mock_ocr)
        out = reader.read(_frame(), [visible, invisible])

        assert visible.id in out.element_texts
        assert invisible.id not in out.element_texts
        assert mock_ocr.call_count == 1

    def test_invisible_not_in_reading_order(self):
        visible = _element(0, 0, is_visible=True)
        invisible = _element(0, 50, is_visible=False)
        reader = Reader(_MockOCR([]))
        out = reader.read(_frame(), [visible, invisible])
        assert invisible.id not in out.reading_order
        assert visible.id in out.reading_order


# ---------------------------------------------------------------------------
# Section 12 — empty element list
# ---------------------------------------------------------------------------


class TestEmptyElements:
    def test_empty_elements_returns_empty_output(self):
        reader = Reader(_MockOCR([]))
        out = reader.read(_frame(), [])
        assert out.element_texts == {}
        assert out.text_blocks == []
        assert out.reading_order == []
        assert out.table_data is None

    def test_no_ocr_calls_for_empty_list(self):
        mock_ocr = _MockOCR([])
        reader = Reader(mock_ocr)
        reader.read(_frame(), [])
        assert mock_ocr.call_count == 0


# ---------------------------------------------------------------------------
# Section 13 — OCR failure is non-fatal
# ---------------------------------------------------------------------------


class TestOCRFailureNonFatal:
    def test_ocr_exception_returns_empty_text(self):
        el = _element(0, 0)
        mock_ocr = _MockOCR(side_effect=RuntimeError("OCR boom"))
        reader = Reader(mock_ocr)
        out = reader.read(_frame(), [el])
        # Should not raise; element gets empty text
        assert out.element_texts[el.id] == ""

    def test_ocr_failure_element_in_reading_order(self):
        el = _element(0, 0)
        mock_ocr = _MockOCR(side_effect=ValueError("bad"))
        reader = Reader(mock_ocr)
        out = reader.read(_frame(), [el])
        assert el.id in out.reading_order
