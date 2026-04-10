"""
tests/unit/test_dirty_region.py
Unit tests for nexus/capture/dirty_region.py — Faz 20.

Sections:
  1. DirtyRegions value object
  2. detect() — identical frames
  3. detect() — full-change frames
  4. detect() — FULL_REFRESH_THRESHOLD boundary
  5. detect() — block coordinate correctness
  6. detect() — size mismatch
  7. DirtyRegionDetector constructor validation
  8. Hypothesis — change_ratio always in [0.0, 1.0]
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nexus.capture.dirty_region import (
    BLOCK_SIZE,
    FULL_REFRESH_THRESHOLD,
    DirtyRegionDetector,
    DirtyRegions,
)
from nexus.capture.frame import Frame
from nexus.core.types import Rect

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(
    width: int = 32,
    height: int = 32,
    color: tuple[int, int, int] = (0, 0, 0),
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


def _frame_from_array(arr: np.ndarray, seq: int = 1) -> Frame:
    h, w = arr.shape[:2]
    return Frame(
        data=arr.copy(),
        width=w,
        height=h,
        captured_at_monotonic=0.0,
        captured_at_utc="",
        sequence_number=seq,
    )


def _make_pair_with_dirty_cols(
    num_dirty: int,
    total_cols: int,
    block_size: int = 10,
) -> tuple[Frame, Frame]:
    """
    Return (prev, curr) pair where *num_dirty* column-blocks differ.

    Frame: (total_cols * block_size) wide, block_size tall →
    exactly *total_cols* blocks of size *block_size* × *block_size*.
    Dirty blocks occupy the leftmost *num_dirty* columns.
    """
    w = total_cols * block_size
    h = block_size
    prev_data = np.zeros((h, w, 3), dtype=np.uint8)
    curr_data = prev_data.copy()
    for col_idx in range(num_dirty):
        curr_data[:, col_idx * block_size : (col_idx + 1) * block_size] = 255
    return _frame_from_array(prev_data), _frame_from_array(curr_data)


# ---------------------------------------------------------------------------
# 1. DirtyRegions value object
# ---------------------------------------------------------------------------


class TestDirtyRegionsValueObject:
    def test_frozen(self) -> None:
        dr = DirtyRegions(blocks=(), full_refresh=False, change_ratio=0.0)
        with pytest.raises((AttributeError, TypeError)):
            dr.full_refresh = True  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        r = Rect(0, 0, 32, 32)
        dr = DirtyRegions(blocks=(r,), full_refresh=True, change_ratio=1.0)
        assert dr.blocks == (r,)
        assert dr.full_refresh is True
        assert dr.change_ratio == pytest.approx(1.0)

    def test_empty_blocks(self) -> None:
        dr = DirtyRegions(blocks=(), full_refresh=False, change_ratio=0.0)
        assert len(dr.blocks) == 0


# ---------------------------------------------------------------------------
# 2. detect() — identical frames
# ---------------------------------------------------------------------------


class TestIdenticalFrames:
    _det = DirtyRegionDetector(block_size=32)

    def test_no_dirty_blocks(self) -> None:
        f = _make_frame(color=(128, 64, 32))
        result = self._det.detect(f, f)
        assert result.blocks == ()

    def test_change_ratio_zero(self) -> None:
        f = _make_frame(color=(200, 200, 200))
        result = self._det.detect(f, f)
        assert result.change_ratio == pytest.approx(0.0)

    def test_full_refresh_false(self) -> None:
        f = _make_frame()
        result = self._det.detect(f, f)
        assert result.full_refresh is False

    def test_copy_of_same_data_is_identical(self) -> None:
        data = np.full((32, 32, 3), 77, dtype=np.uint8)
        f1 = _frame_from_array(data)
        f2 = _frame_from_array(data)
        result = self._det.detect(f1, f2)
        assert result.blocks == ()
        assert result.change_ratio == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. detect() — fully changed frames
# ---------------------------------------------------------------------------


class TestFullyChangedFrames:
    _det = DirtyRegionDetector(block_size=32)

    def test_all_blocks_dirty(self) -> None:
        f1 = _make_frame(color=(0, 0, 0))
        f2 = _make_frame(color=(255, 255, 255))
        result = self._det.detect(f1, f2)
        assert result.change_ratio == pytest.approx(1.0)

    def test_full_refresh_true(self) -> None:
        f1 = _make_frame(color=(0, 0, 0))
        f2 = _make_frame(color=(255, 255, 255))
        result = self._det.detect(f1, f2)
        assert result.full_refresh is True

    def test_block_count_matches_frame_layout(self) -> None:
        """A 64×64 frame with block_size=32 has exactly 4 blocks."""
        det = DirtyRegionDetector(block_size=32)
        f1 = _make_frame(width=64, height=64, color=(0, 0, 0))
        f2 = _make_frame(width=64, height=64, color=(255, 255, 255))
        result = det.detect(f1, f2)
        assert len(result.blocks) == 4

    def test_symmetric(self) -> None:
        """detect(a, b) and detect(b, a) give the same change_ratio."""
        f1 = _make_frame(color=(10, 20, 30))
        f2 = _make_frame(color=(200, 150, 100))
        det = DirtyRegionDetector(block_size=32)
        r1 = det.detect(f1, f2)
        r2 = det.detect(f2, f1)
        assert r1.change_ratio == pytest.approx(r2.change_ratio)
        assert r1.full_refresh == r2.full_refresh


# ---------------------------------------------------------------------------
# 4. detect() — FULL_REFRESH_THRESHOLD boundary
# ---------------------------------------------------------------------------


class TestFullRefreshThreshold:
    """
    Uses a detector with block_size=10.
    Frames are (total_cols * 10) wide, 10 tall → exactly *total_cols* blocks.
    """

    _det = DirtyRegionDetector(block_size=10)

    def test_exactly_80_percent_triggers_full_refresh(self) -> None:
        prev, curr = _make_pair_with_dirty_cols(8, 10, block_size=10)
        result = self._det.detect(prev, curr)
        assert result.full_refresh is True
        assert result.change_ratio == pytest.approx(0.8)

    def test_below_80_percent_no_full_refresh(self) -> None:
        prev, curr = _make_pair_with_dirty_cols(7, 10, block_size=10)
        result = self._det.detect(prev, curr)
        assert result.full_refresh is False
        assert result.change_ratio == pytest.approx(0.7)

    def test_100_percent_dirty_is_full_refresh(self) -> None:
        prev, curr = _make_pair_with_dirty_cols(10, 10, block_size=10)
        result = self._det.detect(prev, curr)
        assert result.full_refresh is True
        assert result.change_ratio == pytest.approx(1.0)

    def test_0_percent_dirty_no_full_refresh(self) -> None:
        prev, curr = _make_pair_with_dirty_cols(0, 10, block_size=10)
        result = self._det.detect(prev, curr)
        assert result.full_refresh is False
        assert result.change_ratio == pytest.approx(0.0)
        assert result.blocks == ()

    def test_threshold_constant_is_reasonable(self) -> None:
        assert 0.0 < FULL_REFRESH_THRESHOLD <= 1.0


# ---------------------------------------------------------------------------
# 5. detect() — block coordinate correctness
# ---------------------------------------------------------------------------


class TestBlockCoordinates:
    """
    Frame: 30×10 with block_size=10 → 3 blocks:
      Rect(0,0,10,10), Rect(10,0,10,10), Rect(20,0,10,10)
    """

    _det = DirtyRegionDetector(block_size=10)

    def _make_30x10(self, dirty_col: int | None = None) -> tuple[Frame, Frame]:
        """Return (prev, curr); optionally mark one 10-pixel column-block dirty."""
        w, h = 30, 10
        prev_data = np.zeros((h, w, 3), dtype=np.uint8)
        curr_data = prev_data.copy()
        if dirty_col is not None:
            curr_data[:, dirty_col * 10 : dirty_col * 10 + 10] = 128
        return _frame_from_array(prev_data), _frame_from_array(curr_data)

    def test_first_block_coordinates(self) -> None:
        prev, curr = self._make_30x10(dirty_col=0)
        result = self._det.detect(prev, curr)
        assert Rect(x=0, y=0, width=10, height=10) in result.blocks

    def test_middle_block_coordinates(self) -> None:
        prev, curr = self._make_30x10(dirty_col=1)
        result = self._det.detect(prev, curr)
        assert Rect(x=10, y=0, width=10, height=10) in result.blocks

    def test_last_block_coordinates(self) -> None:
        prev, curr = self._make_30x10(dirty_col=2)
        result = self._det.detect(prev, curr)
        assert Rect(x=20, y=0, width=10, height=10) in result.blocks

    def test_only_dirty_block_reported(self) -> None:
        """Only the changed block must appear; the other two must not."""
        prev, curr = self._make_30x10(dirty_col=1)
        result = self._det.detect(prev, curr)
        assert len(result.blocks) == 1
        assert result.blocks[0] == Rect(x=10, y=0, width=10, height=10)

    def test_partial_last_block_correct_size(self) -> None:
        """
        Frame 35×10 with block_size=10 → last block is 5 px wide, not 10.
        """
        w, h = 35, 10
        prev_data = np.zeros((h, w, 3), dtype=np.uint8)
        curr_data = prev_data.copy()
        curr_data[:, 30:35] = 200  # mark last (partial) block dirty
        prev = _frame_from_array(prev_data)
        curr = _frame_from_array(curr_data)
        result = self._det.detect(prev, curr)
        last_block = max(result.blocks, key=lambda r: r.x)
        assert last_block.x == 30
        assert last_block.width == 5

    def test_multi_row_block_y_coordinate(self) -> None:
        """
        Frame 10×20 with block_size=10 → 2 row blocks.
        Dirty the second row block; expect y=10.
        """
        w, h = 10, 20
        prev_data = np.zeros((h, w, 3), dtype=np.uint8)
        curr_data = prev_data.copy()
        curr_data[10:20, :] = 99  # second row dirty
        prev = _frame_from_array(prev_data)
        curr = _frame_from_array(curr_data)
        result = self._det.detect(prev, curr)
        assert len(result.blocks) == 1
        assert result.blocks[0].y == 10


# ---------------------------------------------------------------------------
# 6. detect() — size mismatch
# ---------------------------------------------------------------------------


class TestSizeMismatch:
    _det = DirtyRegionDetector(block_size=32)

    def test_full_refresh_on_width_mismatch(self) -> None:
        f1 = _make_frame(width=32, height=32)
        f2 = _make_frame(width=64, height=32)
        result = self._det.detect(f1, f2)
        assert result.full_refresh is True

    def test_change_ratio_one_on_mismatch(self) -> None:
        f1 = _make_frame(width=32, height=32)
        f2 = _make_frame(width=32, height=64)
        result = self._det.detect(f1, f2)
        assert result.change_ratio == pytest.approx(1.0)

    def test_blocks_cover_current_frame_on_mismatch(self) -> None:
        """All blocks in the result must be within *curr* bounds."""
        f1 = _make_frame(width=32, height=32)
        f2 = _make_frame(width=64, height=64)
        result = self._det.detect(f1, f2)
        for block in result.blocks:
            assert block.x + block.width <= 64
            assert block.y + block.height <= 64


# ---------------------------------------------------------------------------
# 7. DirtyRegionDetector constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_zero_block_size_raises(self) -> None:
        with pytest.raises(ValueError, match="block_size"):
            DirtyRegionDetector(block_size=0)

    def test_negative_block_size_raises(self) -> None:
        with pytest.raises(ValueError, match="block_size"):
            DirtyRegionDetector(block_size=-1)

    def test_one_is_valid(self) -> None:
        det = DirtyRegionDetector(block_size=1)
        f1 = _make_frame(width=4, height=4, color=(0, 0, 0))
        f2 = _make_frame(width=4, height=4, color=(255, 0, 0))
        result = det.detect(f1, f2)
        assert result.change_ratio == pytest.approx(1.0)

    def test_default_block_size(self) -> None:
        det = DirtyRegionDetector()
        assert det._block_size == BLOCK_SIZE


# ---------------------------------------------------------------------------
# 8. Hypothesis — change_ratio always in [0.0, 1.0]
# ---------------------------------------------------------------------------


class TestHypothesisChangeRatio:
    """Property-based tests for the change_ratio invariant."""

    _det = DirtyRegionDetector(block_size=8)

    @given(
        arrays(dtype=np.dtype("uint8"), shape=(32, 64, 3)),
        arrays(dtype=np.dtype("uint8"), shape=(32, 64, 3)),
    )
    @settings(max_examples=200)
    def test_change_ratio_in_0_1(
        self, arr1: np.ndarray, arr2: np.ndarray
    ) -> None:
        """For any two same-shape arrays the change_ratio must be in [0, 1]."""
        f1 = _frame_from_array(arr1)
        f2 = _frame_from_array(arr2)
        result = self._det.detect(f1, f2)
        assert 0.0 <= result.change_ratio <= 1.0, f"Got {result.change_ratio!r}"

    @given(arrays(dtype=np.dtype("uint8"), shape=(16, 16, 3)))
    @settings(max_examples=100)
    def test_identical_arrays_have_zero_ratio(self, arr: np.ndarray) -> None:
        """A frame compared with itself must always give change_ratio=0."""
        f = _frame_from_array(arr)
        result = self._det.detect(f, f)
        assert result.change_ratio == pytest.approx(0.0)
        assert result.blocks == ()

    @given(
        st.integers(1, 3),
        st.integers(1, 3),
        st.integers(1, 3),
        st.integers(1, 3),
    )
    @settings(max_examples=50)
    def test_size_mismatch_returns_full_refresh(
        self, w1: int, w2: int, h1: int, h2: int
    ) -> None:
        """Frames with different sizes always return full_refresh=True."""
        if w1 == w2 and h1 == h2:
            w2 = w2 + 1  # force mismatch
        f1 = _make_frame(width=w1 * 8, height=h1 * 8)
        f2 = _make_frame(width=w2 * 8, height=h2 * 8)
        result = self._det.detect(f1, f2)
        assert result.full_refresh is True
        assert result.change_ratio == pytest.approx(1.0)
