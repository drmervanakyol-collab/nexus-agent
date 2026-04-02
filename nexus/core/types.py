"""
nexus/core/types.py
Shared value types and opaque ID types used across all Nexus layers.
All dataclasses are frozen (immutable) and hashable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NewType

# ---------------------------------------------------------------------------
# Opaque ID types — prevent accidental mixing of different ID kinds
# ---------------------------------------------------------------------------

TaskId = NewType("TaskId", str)
ElementId = NewType("ElementId", str)
FingerprintId = NewType("FingerprintId", str)
TraceId = NewType("TraceId", str)

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """A 2-D integer coordinate."""

    x: int
    y: int

    def distance_to(self, other: Point) -> float:
        """Euclidean distance to *other*."""
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Rect:
    """
    An axis-aligned bounding rectangle defined by its top-left corner,
    width, and height.  All values are in screen pixels (integers).
    Width and height must be non-negative.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 0:
            raise ValueError(f"Rect.width must be >= 0, got {self.width}")
        if self.height < 0:
            raise ValueError(f"Rect.height must be >= 0, got {self.height}")

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def area(self) -> int:
        """Return width * height."""
        return self.width * self.height

    def center(self) -> Point:
        """Return the centre pixel (integer arithmetic)."""
        return Point(self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, point: Point) -> bool:
        """Return True if *point* lies strictly inside or on the boundary."""
        return self.x <= point.x <= self.right and self.y <= point.y <= self.bottom

    def overlaps(self, other: Rect) -> bool:
        """Return True if this rect and *other* share at least one pixel."""
        return (
            self.x < other.right
            and other.x < self.right
            and self.y < other.bottom
            and other.y < self.bottom
        )

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def clip_to(self, bounds: Rect) -> Rect:
        """
        Return the intersection of this rect with *bounds*.
        If there is no intersection, return a zero-area rect at the
        nearest corner of *bounds*.
        """
        cx = max(self.x, bounds.x)
        cy = max(self.y, bounds.y)
        cr = min(self.right, bounds.right)
        cb = min(self.bottom, bounds.bottom)
        cw = max(0, cr - cx)
        ch = max(0, cb - cy)
        return Rect(cx, cy, cw, ch)

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Return ``(x, y, width, height)``."""
        return (self.x, self.y, self.width, self.height)
