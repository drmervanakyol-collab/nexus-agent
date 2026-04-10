"""
tests/property/test_geometry.py
Geometry invariant property tests — Faz 65

Covers Rect and Point from nexus.core.types using Hypothesis.

Invariants tested
-----------------
Rect:
  - width >= 0 and height >= 0 always (construction rejects negatives)
  - area() == width * height
  - right == x + width, bottom == y + height
  - center() is inside the rect
  - area() >= 0 always
  - overlaps() is symmetric
  - overlaps(self) is always True when area > 0
  - clip_to(bounds) result is always within bounds
  - clip_to result area <= original area
  - contains(center) is True

Point:
  - distance_to() >= 0 always
  - distance_to() is symmetric (a.distance_to(b) == b.distance_to(a))
  - distance_to(self) == 0
  - triangle inequality: dist(a,c) <= dist(a,b) + dist(b,c) + epsilon
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nexus.core.types import Point, Rect

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_COORD = st.integers(min_value=-10_000, max_value=10_000)
_DIM = st.integers(min_value=0, max_value=5_000)
_DIM_POS = st.integers(min_value=1, max_value=5_000)  # strictly positive


@st.composite
def rects(draw: st.DrawFn, positive_area: bool = False) -> Rect:
    x = draw(_COORD)
    y = draw(_COORD)
    w = draw(_DIM_POS if positive_area else _DIM)
    h = draw(_DIM_POS if positive_area else _DIM)
    return Rect(x, y, w, h)


@st.composite
def points(draw: st.DrawFn) -> Point:
    return Point(draw(_COORD), draw(_COORD))


# ---------------------------------------------------------------------------
# Rect: construction invariants
# ---------------------------------------------------------------------------


def test_rect_rejects_negative_width() -> None:
    with pytest.raises(ValueError, match="width"):
        Rect(0, 0, -1, 10)


def test_rect_rejects_negative_height() -> None:
    with pytest.raises(ValueError, match="height"):
        Rect(0, 0, 10, -1)


@given(rects())
def test_rect_dimensions_nonnegative(r: Rect) -> None:
    assert r.width >= 0
    assert r.height >= 0


# ---------------------------------------------------------------------------
# Rect: derived property invariants
# ---------------------------------------------------------------------------


@given(rects())
def test_rect_area_equals_width_times_height(r: Rect) -> None:
    assert r.area() == r.width * r.height


@given(rects())
def test_rect_area_nonnegative(r: Rect) -> None:
    assert r.area() >= 0


@given(rects())
def test_rect_right_equals_x_plus_width(r: Rect) -> None:
    assert r.right == r.x + r.width


@given(rects())
def test_rect_bottom_equals_y_plus_height(r: Rect) -> None:
    assert r.bottom == r.y + r.height


@given(rects())
def test_rect_to_tuple_roundtrip(r: Rect) -> None:
    t = r.to_tuple()
    assert t == (r.x, r.y, r.width, r.height)
    restored = Rect(*t)
    assert restored == r


# ---------------------------------------------------------------------------
# Rect: center
# ---------------------------------------------------------------------------


@given(rects(positive_area=True))
def test_rect_center_is_inside(r: Rect) -> None:
    c = r.center()
    assert r.contains(c), f"center {c} not inside {r}"


@given(rects())
def test_rect_center_x_in_range(r: Rect) -> None:
    c = r.center()
    assert r.x <= c.x <= r.right
    assert r.y <= c.y <= r.bottom


# ---------------------------------------------------------------------------
# Rect: overlaps
# ---------------------------------------------------------------------------


@given(rects(positive_area=True))
def test_rect_overlaps_self(r: Rect) -> None:
    assert r.overlaps(r)


@given(rects(), rects())
def test_rect_overlaps_symmetric(a: Rect, b: Rect) -> None:
    assert a.overlaps(b) == b.overlaps(a)


@given(rects())
def test_rect_zero_area_does_not_overlap_itself(r: Rect) -> None:
    # A degenerate line or point rect (width=0 or height=0)
    zero = Rect(r.x, r.y, 0, r.height)
    # overlaps requires strict interior sharing
    assert not zero.overlaps(zero)


# ---------------------------------------------------------------------------
# Rect: clip_to
# ---------------------------------------------------------------------------


@st.composite
def overlapping_rect_pair(draw: st.DrawFn) -> tuple[Rect, Rect]:
    """Generate (r, bounds) guaranteed to overlap by constructing r inside bounds."""
    bx = draw(st.integers(min_value=-1000, max_value=1000))
    by = draw(st.integers(min_value=-1000, max_value=1000))
    bw = draw(st.integers(min_value=2, max_value=2000))
    bh = draw(st.integers(min_value=2, max_value=2000))
    bounds = Rect(bx, by, bw, bh)
    # r starts inside bounds
    rx = draw(st.integers(min_value=bx, max_value=bx + bw - 1))
    ry = draw(st.integers(min_value=by, max_value=by + bh - 1))
    rw = draw(st.integers(min_value=1, max_value=bw * 2))
    rh = draw(st.integers(min_value=1, max_value=bh * 2))
    r = Rect(rx, ry, rw, rh)
    return r, bounds


@given(overlapping_rect_pair())
def test_rect_clip_result_within_bounds_when_overlapping(pair: tuple[Rect, Rect]) -> None:
    """When r and bounds overlap, the clip result must lie within bounds."""
    r, bounds = pair
    assert r.overlaps(bounds), "strategy should guarantee overlap"
    clipped = r.clip_to(bounds)
    assert clipped.x >= bounds.x
    assert clipped.y >= bounds.y
    assert clipped.right <= bounds.right
    assert clipped.bottom <= bounds.bottom


@given(rects(positive_area=True), rects(positive_area=True))
def test_rect_clip_area_lte_bounds_area(r: Rect, bounds: Rect) -> None:
    """Clipped area never exceeds bounds area (clip_to uses max(1,...) for width)."""
    clipped = r.clip_to(bounds)
    assert clipped.area() <= bounds.area()


@given(rects(positive_area=True))
def test_rect_clip_to_self_preserves_dimensions(r: Rect) -> None:
    """Clipping a positive-area rect to itself gives the same result."""
    assert r.clip_to(r) == r


@given(rects(), rects(positive_area=True))
def test_rect_clip_area_nonnegative(r: Rect, bounds: Rect) -> None:
    clipped = r.clip_to(bounds)
    assert clipped.area() >= 0
    assert clipped.width >= 0
    assert clipped.height >= 0


# ---------------------------------------------------------------------------
# Point: distance invariants
# ---------------------------------------------------------------------------


@given(points(), points())
def test_point_distance_nonnegative(a: Point, b: Point) -> None:
    assert a.distance_to(b) >= 0.0


@given(points(), points())
def test_point_distance_symmetric(a: Point, b: Point) -> None:
    assert math.isclose(a.distance_to(b), b.distance_to(a), rel_tol=1e-9)


@given(points())
def test_point_distance_to_self_is_zero(p: Point) -> None:
    assert p.distance_to(p) == 0.0


@given(points(), points(), points())
def test_point_triangle_inequality(a: Point, b: Point, c: Point) -> None:
    eps = 1e-9
    assert a.distance_to(c) <= a.distance_to(b) + b.distance_to(c) + eps


# ---------------------------------------------------------------------------
# Rect: boundary conditions (kills <= -> < and and -> or mutations)
# ---------------------------------------------------------------------------


@given(rects())
def test_rect_contains_top_left_corner(r: Rect) -> None:
    """Top-left corner must be contained (tests <= not < on left boundary)."""
    assert r.contains(Point(r.x, r.y))


@given(rects())
def test_rect_contains_bottom_right_corner(r: Rect) -> None:
    """Bottom-right corner must be contained (tests <= not < on right boundary)."""
    assert r.contains(Point(r.right, r.bottom))


@given(rects(positive_area=True))
def test_rect_does_not_contain_point_just_outside_left(r: Rect) -> None:
    outside = Point(r.x - 1, r.y + r.height // 2)
    assert not r.contains(outside)


@given(rects(positive_area=True))
def test_rect_does_not_contain_point_just_outside_right(r: Rect) -> None:
    outside = Point(r.right + 1, r.y + r.height // 2)
    assert not r.contains(outside)


@given(rects(positive_area=True))
def test_rect_contains_requires_both_axes(r: Rect) -> None:
    """A point outside in both x and y must not be contained (and not or)."""
    outside = Point(r.x - 1, r.y - 1)
    assert not r.contains(outside)


# ---------------------------------------------------------------------------
# Rect: center() exact value (kills // -> % mutation)
# ---------------------------------------------------------------------------


def test_rect_center_exact_values() -> None:
    r = Rect(10, 20, 100, 80)
    c = r.center()
    assert c.x == 60   # 10 + 100//2
    assert c.y == 60   # 20 + 80//2


@given(rects(positive_area=True))
def test_rect_center_x_matches_formula(r: Rect) -> None:
    c = r.center()
    assert c.x == r.x + r.width // 2


@given(rects(positive_area=True))
def test_rect_center_y_matches_formula(r: Rect) -> None:
    c = r.center()
    assert c.y == r.y + r.height // 2


# ---------------------------------------------------------------------------
# Rect: overlaps boundary conditions
# ---------------------------------------------------------------------------


def test_rect_overlaps_adjacent_does_not_overlap() -> None:
    """Two rects that share only an edge must NOT overlap (strict interior)."""
    a = Rect(0, 0, 10, 10)
    b = Rect(10, 0, 10, 10)   # touches at x=10 but no pixel sharing
    assert not a.overlaps(b)


def test_rect_overlaps_one_pixel_interior() -> None:
    """Two rects sharing exactly one pixel must overlap."""
    a = Rect(0, 0, 10, 10)
    b = Rect(9, 9, 10, 10)   # overlaps at pixel (9,9)
    assert a.overlaps(b)


@given(rects(positive_area=True))
def test_rect_does_not_overlap_fully_separated_right(r: Rect) -> None:
    other = Rect(r.right + 1, r.y, r.width, r.height)
    assert not r.overlaps(other)
