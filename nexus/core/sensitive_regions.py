"""
nexus/core/sensitive_regions.py
Sensitive screen-region registry used by PolicyEngine and ScreenshotMasker.

A SensitiveRegion marks an area of the screen that must be treated with
extra caution — either warned about or fully blocked from automated access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nexus.core.types import Point, Rect

Severity = Literal["warn", "block"]


@dataclass(frozen=True)
class SensitiveRegion:
    """A labelled, severity-tagged bounding rectangle."""

    rect: Rect
    label: str
    severity: Severity = "warn"


class SensitiveRegionDetector:
    """
    Matches screen coordinates against a collection of SensitiveRegions.

    Parameters
    ----------
    regions:
        Pre-defined sensitive areas (e.g. password fields, payment widgets).
    """

    def __init__(self, regions: list[SensitiveRegion] | None = None) -> None:
        self._regions: list[SensitiveRegion] = list(regions or [])

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, region: SensitiveRegion) -> None:
        """Append a region at runtime."""
        self._regions.append(region)

    def clear(self) -> None:
        """Remove all registered regions."""
        self._regions.clear()

    @property
    def regions(self) -> list[SensitiveRegion]:
        return list(self._regions)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, point: Point) -> list[SensitiveRegion]:
        """Return all regions whose bounding rect contains *point*."""
        return [r for r in self._regions if r.rect.contains(point)]

    def detect_rect(self, rect: Rect) -> list[SensitiveRegion]:
        """Return all regions that overlap with *rect*."""
        return [r for r in self._regions if r.rect.overlaps(rect)]

    def is_blocked(self, point: Point) -> bool:
        """Return True if any matching region has severity == 'block'."""
        return any(r.severity == "block" for r in self.detect(point))

    def is_blocked_rect(self, rect: Rect) -> bool:
        """Return True if any overlapping region has severity == 'block'."""
        return any(r.severity == "block" for r in self.detect_rect(rect))
