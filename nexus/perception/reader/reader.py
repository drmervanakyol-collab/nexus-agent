"""
nexus/perception/reader/reader.py
Reader — extracts and structures text from detected UI elements.

ReaderOutput
------------
  element_texts   : dict[ElementId, str]         OCR text per element
  text_blocks     : list[OCRResult]               all raw word-level results
  layout_regions  : list[LayoutRegion]            detected columns / regions
  reading_order   : list[ElementId]               top→bottom, left→right
  table_data      : list[TableData] | None        detected grid structures

Reader.read(frame, elements, dirty_regions)
-------------------------------------------
Pipeline:
  1. For each element, check dirty_regions to decide re-OCR vs cache.
  2. Run OCR on dirty (or un-cached) element regions via OCREngine.
  3. Cache results keyed on region coordinates; serve cache for unchanged regions.
  4. Detect layout regions — find horizontal gaps → column boundaries.
  5. Assign reading order — column-first, then top-to-bottom within each column.
  6. Detect tables — cluster text blocks into y-bands; check column alignment.
  7. Return ReaderOutput.

Selective OCR cache
-------------------
  Key   : (x, y, width, height) of the element bounding box.
  Policy: dirty element → always re-OCR (cache overwritten).
          non-dirty element → cache hit returned; miss → OCR once then cache.
  TTL   : configurable, default 5 seconds.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from nexus.capture.frame import Frame
from nexus.core.types import ElementId, Rect
from nexus.infra.logger import get_logger
from nexus.perception.locator.locator import UIElement
from nexus.perception.reader.ocr_engine import OCREngine, OCRResult

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_CACHE_TTL_S: float = 5.0
_COLUMN_GAP_MIN_RATIO: float = 0.05   # gap must be >= 5% of frame width
_TABLE_Y_TOLERANCE: int = 12           # pixels — words within this y-band = same row
_TABLE_MIN_COLS: int = 2
_TABLE_MIN_ROWS: int = 2


# ---------------------------------------------------------------------------
# LayoutRegionType
# ---------------------------------------------------------------------------


class LayoutRegionType(Enum):
    BODY = auto()
    COLUMN = auto()
    HEADER = auto()
    FOOTER = auto()
    TABLE = auto()


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutRegion:
    """
    A detected layout zone within the frame.

    Attributes
    ----------
    bounding_box:
        Pixel rectangle of this region.
    region_type:
        Semantic classification.
    column_index:
        0-based column index when region_type is COLUMN; None otherwise.
    """

    bounding_box: Rect
    region_type: LayoutRegionType
    column_index: int | None = None


@dataclass(frozen=True)
class TableData:
    """
    A detected table / grid structure.

    Attributes
    ----------
    bounding_box:
        Pixel rectangle enclosing the full table.
    rows:
        List of rows; each row is a list of cell text strings, sorted
        left-to-right.
    header_row:
        First row treated as a header, or None when the table has no
        recognisable header.
    """

    bounding_box: Rect
    rows: tuple[tuple[str, ...], ...]
    header_row: tuple[str, ...] | None


@dataclass(frozen=True)
class ReaderOutput:
    """
    Full structured result of a Reader.read() call.

    Attributes
    ----------
    element_texts:
        Mapping of element ID → joined OCR text for that element's region.
    text_blocks:
        All word-level OCRResult objects collected across all elements.
    layout_regions:
        Detected layout columns / regions; empty when structure is unknown.
    reading_order:
        Element IDs sorted into the natural reading order for this layout.
    table_data:
        Detected table structures, or None when no table is found.
    """

    element_texts: dict[ElementId, str]
    text_blocks: list[OCRResult]
    layout_regions: list[LayoutRegion]
    reading_order: list[ElementId]
    table_data: list[TableData] | None


# ---------------------------------------------------------------------------
# Cache entry (internal)
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    text: str
    results: list[OCRResult]
    expires_at: float


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class Reader:
    """
    Extracts and structures text from a set of UI elements.

    Parameters
    ----------
    ocr_engine:
        Any object that satisfies the OCREngine protocol.
    cache_ttl:
        Seconds to keep an OCR result in the selective cache.
    _time_fn:
        Injectable clock; defaults to ``time.monotonic``.
    """

    def __init__(
        self,
        ocr_engine: OCREngine,
        cache_ttl: float = _CACHE_TTL_S,
        *,
        _time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._ocr = ocr_engine
        self._cache_ttl = cache_ttl
        self._time: Callable[[], float] = _time_fn or time.monotonic
        # Key: (x, y, width, height); value: _CacheEntry
        self._ocr_cache: dict[tuple[int, int, int, int], _CacheEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(
        self,
        frame: Frame,
        elements: Sequence[UIElement],
        dirty_regions: Sequence[Rect] | None = None,
    ) -> ReaderOutput:
        """
        Extract text from *elements* found in *frame*.

        Parameters
        ----------
        frame:
            The captured screen frame.
        elements:
            UI elements whose text should be extracted; typically from
            ``Locator.locate()``.
        dirty_regions:
            When provided, only elements that overlap a dirty region are
            re-OCR'd; other elements are served from the cache (or OCR'd
            once on first encounter).

        Returns
        -------
        ReaderOutput
        """
        now = self._time()
        element_texts: dict[ElementId, str] = {}
        text_blocks: list[OCRResult] = []

        for el in elements:
            if not el.is_visible:
                continue

            bb = el.bounding_box
            cache_key = (bb.x, bb.y, bb.width, bb.height)

            # An element is forced to re-OCR only when it overlaps an explicit
            # dirty region.  dirty_regions=None means "no forced re-OCR".
            force_reocr = dirty_regions is not None and any(
                bb.overlaps(dr) for dr in dirty_regions
            )

            if not force_reocr:
                cached = self._ocr_cache.get(cache_key)
                if cached is not None and cached.expires_at > now:
                    _log.debug(
                        "reader_ocr_cache_hit",
                        element_id=str(el.id)[:8],
                        rect=cache_key,
                    )
                    element_texts[el.id] = cached.text
                    text_blocks.extend(cached.results)
                    continue

            # Run OCR (forced by dirty region, or no valid cache entry)
            try:
                ocr_results = self._ocr.extract(frame.data, region=bb)
            except Exception as exc:
                _log.warning(
                    "reader_ocr_failed",
                    element_id=str(el.id)[:8],
                    error=str(exc),
                )
                ocr_results = []

            joined = " ".join(r.text for r in ocr_results if r.text.strip())
            element_texts[el.id] = joined
            text_blocks.extend(ocr_results)

            self._ocr_cache[cache_key] = _CacheEntry(
                text=joined,
                results=ocr_results,
                expires_at=now + self._cache_ttl,
            )

        visible_els = [e for e in elements if e.is_visible]

        layout_regions = self._detect_layout(visible_els, frame.width, frame.height)
        reading_order = self._reading_order(visible_els, layout_regions)
        tables = self._detect_tables(text_blocks)

        _log.debug(
            "reader_done",
            elements=len(visible_els),
            text_blocks=len(text_blocks),
            columns=sum(
                1 for r in layout_regions
                if r.region_type is LayoutRegionType.COLUMN
            ),
            tables=len(tables),
        )

        return ReaderOutput(
            element_texts=element_texts,
            text_blocks=text_blocks,
            layout_regions=layout_regions,
            reading_order=reading_order,
            table_data=tables if tables else None,
        )

    # ------------------------------------------------------------------
    # Layout detection
    # ------------------------------------------------------------------

    def _detect_layout(
        self,
        elements: list[UIElement],
        frame_width: int,
        frame_height: int,
    ) -> list[LayoutRegion]:
        """
        Detect column layout by projecting element x-coverage onto a 1-D mask
        and finding significant horizontal gaps.
        """
        if not elements or frame_width <= 0:
            return []

        # Build horizontal coverage mask
        coverage = np.zeros(frame_width, dtype=np.int32)
        for el in elements:
            bb = el.bounding_box
            x1 = max(0, bb.x)
            x2 = min(frame_width, bb.x + bb.width)
            if x2 > x1:
                coverage[x1:x2] += 1

        min_gap = max(1, int(frame_width * _COLUMN_GAP_MIN_RATIO))
        gaps = _find_gaps(coverage, min_gap)

        if not gaps:
            return [
                LayoutRegion(
                    bounding_box=Rect(0, 0, frame_width, frame_height),
                    region_type=LayoutRegionType.BODY,
                    column_index=None,
                )
            ]

        # Build column regions between gaps
        regions: list[LayoutRegion] = []
        boundaries: list[int] = [0]
        for gap_start, gap_end in gaps:
            boundaries.append(gap_start)
            boundaries.append(gap_end)
        boundaries.append(frame_width)

        col_idx = 0
        for i in range(0, len(boundaries) - 1, 2):
            x1 = boundaries[i]
            x2 = boundaries[i + 1]
            if x2 <= x1:
                continue
            regions.append(
                LayoutRegion(
                    bounding_box=Rect(x1, 0, x2 - x1, frame_height),
                    region_type=LayoutRegionType.COLUMN,
                    column_index=col_idx,
                )
            )
            col_idx += 1

        return regions

    # ------------------------------------------------------------------
    # Reading order
    # ------------------------------------------------------------------

    @staticmethod
    def _reading_order(
        elements: list[UIElement],
        layout_regions: list[LayoutRegion],
    ) -> list[ElementId]:
        """
        Return element IDs in natural reading order.

        Single column / no layout: top-to-bottom, left-to-right.
        Multi-column: left column first, each column top-to-bottom.
        """
        if not elements:
            return []

        col_regions = [
            r for r in layout_regions
            if r.region_type is LayoutRegionType.COLUMN
        ]

        if not col_regions:
            sorted_els = sorted(
                elements,
                key=lambda e: (e.bounding_box.y, e.bounding_box.x),
            )
            return [e.id for e in sorted_els]

        def _col_index(el: UIElement) -> int:
            cx = el.bounding_box.x + el.bounding_box.width // 2
            for col in col_regions:
                cb = col.bounding_box
                if cb.x <= cx < cb.x + cb.width:
                    return col.column_index or 0
            # Fallback: use the nearest column by distance
            return min(
                range(len(col_regions)),
                key=lambda i: abs(
                    (col_regions[i].bounding_box.x
                     + col_regions[i].bounding_box.width // 2)
                    - cx
                ),
            )

        sorted_els = sorted(
            elements,
            key=lambda e: (_col_index(e), e.bounding_box.y, e.bounding_box.x),
        )
        return [e.id for e in sorted_els]

    # ------------------------------------------------------------------
    # Table detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_tables(text_blocks: list[OCRResult]) -> list[TableData]:
        """
        Detect grid / table structures in *text_blocks*.

        Clusters words into y-bands (rows), then checks whether multiple
        rows share similar x-column alignment.  Returns one TableData per
        detected grid that has >= 2 rows × 2 columns.
        """
        if len(text_blocks) < _TABLE_MIN_COLS * _TABLE_MIN_ROWS:
            return []

        sorted_blocks = sorted(text_blocks, key=lambda b: b.bounding_box.y)

        # Cluster into y-bands
        rows: list[list[OCRResult]] = []
        current_row: list[OCRResult] = [sorted_blocks[0]]
        current_y = _mid_y(sorted_blocks[0])

        for block in sorted_blocks[1:]:
            by = _mid_y(block)
            if abs(by - current_y) <= _TABLE_Y_TOLERANCE:
                current_row.append(block)
            else:
                rows.append(current_row)
                current_row = [block]
                current_y = by
        rows.append(current_row)

        # Keep only rows with >= 2 cells
        grid_rows = [r for r in rows if len(r) >= _TABLE_MIN_COLS]
        if len(grid_rows) < _TABLE_MIN_ROWS:
            return []

        # Bounding box of the table
        all_blocks = [b for row in grid_rows for b in row]
        min_x = min(b.bounding_box.x for b in all_blocks)
        min_y = min(b.bounding_box.y for b in all_blocks)
        max_x = max(b.bounding_box.x + b.bounding_box.width for b in all_blocks)
        max_y = max(b.bounding_box.y + b.bounding_box.height for b in all_blocks)

        table_rows: list[tuple[str, ...]] = []
        for row in grid_rows:
            cells = sorted(row, key=lambda b: b.bounding_box.x)
            table_rows.append(tuple(b.text for b in cells))

        rows_tuple = tuple(table_rows)
        header = rows_tuple[0] if rows_tuple else None

        return [
            TableData(
                bounding_box=Rect(min_x, min_y, max_x - min_x, max_y - min_y),
                rows=rows_tuple,
                header_row=header,
            )
        ]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _mid_y(block: OCRResult) -> int:
    """Y-centre of a text block's bounding box."""
    return block.bounding_box.y + block.bounding_box.height // 2


def _find_gaps(
    coverage: np.ndarray,
    min_gap: int,
) -> list[tuple[int, int]]:
    """
    Find runs of zeros in *coverage* that are at least *min_gap* wide.

    Returns list of (gap_start, gap_end) tuples (exclusive end).
    """
    gaps: list[tuple[int, int]] = []
    n = len(coverage)
    in_gap = False
    gap_start = 0

    for i in range(n):
        if coverage[i] == 0:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                in_gap = False
                if i - gap_start >= min_gap:
                    gaps.append((gap_start, i))

    if in_gap and n - gap_start >= min_gap:
        gaps.append((gap_start, n))

    return gaps
