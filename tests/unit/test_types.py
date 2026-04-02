"""Unit tests for nexus/core/types.py — geometry and ID types."""
from __future__ import annotations

import math

import pytest

from nexus.core.types import (
    ElementId,
    FingerprintId,
    Point,
    Rect,
    TaskId,
    TraceId,
)


# ---------------------------------------------------------------------------
# Point
# ---------------------------------------------------------------------------


class TestPoint:
    def test_distance_to_same_point(self) -> None:
        p = Point(3, 4)
        assert p.distance_to(p) == 0.0

    def test_distance_to_axis_aligned(self) -> None:
        assert Point(0, 0).distance_to(Point(3, 0)) == pytest.approx(3.0)
        assert Point(0, 0).distance_to(Point(0, 4)) == pytest.approx(4.0)

    def test_distance_to_diagonal(self) -> None:
        assert Point(0, 0).distance_to(Point(3, 4)) == pytest.approx(5.0)

    def test_distance_to_negative_coords(self) -> None:
        assert Point(-1, -1).distance_to(Point(2, 3)) == pytest.approx(5.0)

    def test_immutable(self) -> None:
        p = Point(1, 2)
        with pytest.raises((AttributeError, TypeError)):
            p.x = 99  # type: ignore[misc]

    def test_hashable(self) -> None:
        assert hash(Point(1, 2)) == hash(Point(1, 2))
        assert hash(Point(1, 2)) != hash(Point(2, 1))

    def test_equality(self) -> None:
        assert Point(5, 10) == Point(5, 10)
        assert Point(5, 10) != Point(10, 5)


# ---------------------------------------------------------------------------
# Rect — construction
# ---------------------------------------------------------------------------


class TestRectConstruction:
    def test_valid_rect(self) -> None:
        r = Rect(1, 2, 10, 20)
        assert r.x == 1 and r.y == 2 and r.width == 10 and r.height == 20

    def test_zero_dimensions_ok(self) -> None:
        r = Rect(0, 0, 0, 0)
        assert r.area() == 0

    def test_negative_width_raises(self) -> None:
        with pytest.raises(ValueError, match="width"):
            Rect(0, 0, -1, 10)

    def test_negative_height_raises(self) -> None:
        with pytest.raises(ValueError, match="height"):
            Rect(0, 0, 10, -1)

    def test_immutable(self) -> None:
        r = Rect(0, 0, 10, 10)
        with pytest.raises((AttributeError, TypeError)):
            r.x = 99  # type: ignore[misc]

    def test_hashable(self) -> None:
        assert hash(Rect(0, 0, 10, 10)) == hash(Rect(0, 0, 10, 10))


# ---------------------------------------------------------------------------
# Rect — derived properties
# ---------------------------------------------------------------------------


class TestRectProperties:
    def test_right(self) -> None:
        assert Rect(5, 0, 10, 0).right == 15

    def test_bottom(self) -> None:
        assert Rect(0, 5, 0, 10).bottom == 15

    def test_right_zero_width(self) -> None:
        assert Rect(3, 0, 0, 0).right == 3

    def test_bottom_zero_height(self) -> None:
        assert Rect(0, 7, 0, 0).bottom == 7


# ---------------------------------------------------------------------------
# Rect — area
# ---------------------------------------------------------------------------


class TestRectArea:
    def test_area_normal(self) -> None:
        assert Rect(0, 0, 4, 5).area() == 20

    def test_area_zero_width(self) -> None:
        assert Rect(0, 0, 0, 10).area() == 0

    def test_area_zero_height(self) -> None:
        assert Rect(0, 0, 10, 0).area() == 0

    def test_area_unit(self) -> None:
        assert Rect(0, 0, 1, 1).area() == 1


# ---------------------------------------------------------------------------
# Rect — center
# ---------------------------------------------------------------------------


class TestRectCenter:
    def test_center_even(self) -> None:
        assert Rect(0, 0, 10, 10).center() == Point(5, 5)

    def test_center_odd_truncates(self) -> None:
        # Integer division: 7//2 = 3
        assert Rect(0, 0, 7, 7).center() == Point(3, 3)

    def test_center_offset(self) -> None:
        assert Rect(10, 20, 4, 6).center() == Point(12, 23)

    def test_center_zero_size(self) -> None:
        assert Rect(5, 5, 0, 0).center() == Point(5, 5)


# ---------------------------------------------------------------------------
# Rect — contains
# ---------------------------------------------------------------------------


class TestRectContains:
    R = Rect(10, 10, 20, 20)  # right=30, bottom=30

    def test_inside(self) -> None:
        assert self.R.contains(Point(15, 15))

    def test_top_left_corner(self) -> None:
        assert self.R.contains(Point(10, 10))

    def test_bottom_right_corner(self) -> None:
        assert self.R.contains(Point(30, 30))

    def test_outside_left(self) -> None:
        assert not self.R.contains(Point(9, 15))

    def test_outside_right(self) -> None:
        assert not self.R.contains(Point(31, 15))

    def test_outside_top(self) -> None:
        assert not self.R.contains(Point(15, 9))

    def test_outside_bottom(self) -> None:
        assert not self.R.contains(Point(15, 31))

    def test_zero_size_rect_contains_own_corner(self) -> None:
        r = Rect(5, 5, 0, 0)
        assert r.contains(Point(5, 5))

    def test_zero_size_rect_not_contains_neighbor(self) -> None:
        r = Rect(5, 5, 0, 0)
        assert not r.contains(Point(6, 5))


# ---------------------------------------------------------------------------
# Rect — overlaps
# ---------------------------------------------------------------------------


class TestRectOverlaps:
    A = Rect(0, 0, 10, 10)

    def test_overlapping_rects(self) -> None:
        assert self.A.overlaps(Rect(5, 5, 10, 10))

    def test_contained_rect(self) -> None:
        assert self.A.overlaps(Rect(2, 2, 5, 5))

    def test_adjacent_right_no_overlap(self) -> None:
        # A.right == 10, B.x == 10 → touching edge, strict inequality
        assert not self.A.overlaps(Rect(10, 0, 10, 10))

    def test_adjacent_bottom_no_overlap(self) -> None:
        assert not self.A.overlaps(Rect(0, 10, 10, 10))

    def test_disjoint_right(self) -> None:
        assert not self.A.overlaps(Rect(20, 0, 5, 5))

    def test_disjoint_below(self) -> None:
        assert not self.A.overlaps(Rect(0, 20, 5, 5))

    def test_disjoint_left(self) -> None:
        assert not self.A.overlaps(Rect(-20, 0, 5, 5))

    def test_same_rect(self) -> None:
        assert self.A.overlaps(self.A)

    def test_symmetry(self) -> None:
        B = Rect(5, 5, 10, 10)
        assert self.A.overlaps(B) == B.overlaps(self.A)

    def test_zero_area_rects_do_not_overlap(self) -> None:
        a = Rect(0, 0, 0, 10)
        b = Rect(0, 0, 10, 10)
        # Zero-width strip: a.right == 0 == b.x → strict < fails
        assert not a.overlaps(b)


# ---------------------------------------------------------------------------
# Rect — clip_to
# ---------------------------------------------------------------------------


class TestRectClipTo:
    BOUNDS = Rect(0, 0, 100, 100)

    def test_fully_inside(self) -> None:
        r = Rect(10, 10, 20, 20)
        assert r.clip_to(self.BOUNDS) == r

    def test_fully_outside_right(self) -> None:
        r = Rect(110, 10, 20, 20)
        clipped = r.clip_to(self.BOUNDS)
        assert clipped.area() == 0

    def test_fully_outside_below(self) -> None:
        r = Rect(10, 110, 20, 20)
        clipped = r.clip_to(self.BOUNDS)
        assert clipped.area() == 0

    def test_partial_overlap_right(self) -> None:
        r = Rect(90, 10, 20, 20)  # extends to x=110
        clipped = r.clip_to(self.BOUNDS)
        assert clipped == Rect(90, 10, 10, 20)

    def test_partial_overlap_bottom(self) -> None:
        r = Rect(10, 90, 20, 20)  # extends to y=110
        clipped = r.clip_to(self.BOUNDS)
        assert clipped == Rect(10, 90, 20, 10)

    def test_partial_overlap_top_left(self) -> None:
        r = Rect(-5, -5, 15, 15)
        clipped = r.clip_to(self.BOUNDS)
        assert clipped == Rect(0, 0, 10, 10)

    def test_clip_returns_rect_type(self) -> None:
        r = Rect(10, 10, 5, 5)
        assert isinstance(r.clip_to(self.BOUNDS), Rect)

    def test_clip_entirely_outside_left(self) -> None:
        r = Rect(-50, 10, 20, 20)
        clipped = r.clip_to(self.BOUNDS)
        assert clipped.area() == 0

    def test_clip_to_itself(self) -> None:
        r = Rect(10, 10, 30, 30)
        assert r.clip_to(r) == r


# ---------------------------------------------------------------------------
# Rect — to_tuple
# ---------------------------------------------------------------------------


class TestRectToTuple:
    def test_to_tuple(self) -> None:
        assert Rect(1, 2, 3, 4).to_tuple() == (1, 2, 3, 4)

    def test_to_tuple_zero(self) -> None:
        assert Rect(0, 0, 0, 0).to_tuple() == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Opaque ID types
# ---------------------------------------------------------------------------


class TestOpaqueIds:
    def test_task_id_is_str_subtype(self) -> None:
        tid = TaskId("task-001")
        assert isinstance(tid, str)
        assert tid == "task-001"

    def test_element_id(self) -> None:
        eid = ElementId("elem-abc")
        assert isinstance(eid, str)

    def test_fingerprint_id(self) -> None:
        fid = FingerprintId("fp-xyz")
        assert isinstance(fid, str)

    def test_trace_id(self) -> None:
        trid = TraceId("trace-999")
        assert isinstance(trid, str)

    def test_ids_are_distinct_types(self) -> None:
        # NewType creates distinct runtime types for mypy but same str at runtime;
        # confirm at least they hold correct values independently.
        tid = TaskId("x")
        eid = ElementId("x")
        assert tid == eid  # same underlying value
        assert type(tid) is type(eid)  # both are str at runtime
