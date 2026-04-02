"""
nexus/core/screenshot_masker.py
Applies pixel-level masking to screenshots before they are stored or sent
to a cloud LLM, ensuring sensitive regions are never leaked.

Masking is performed in-place on a copy of the numpy array.
Every masking event is logged via structlog for telemetry / audit trails.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nexus.core.types import Rect
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# Default fill colour: solid black (BGR or RGB — doesn't matter for masking)
_MASK_FILL: int = 0


@dataclass
class MaskingResult:
    """Summary of one masking operation."""

    regions_masked: int
    pixels_masked: int
    region_labels: list[str] = field(default_factory=list)


class ScreenshotMasker:
    """
    Blacks-out rectangular regions in a screenshot (numpy array).

    Parameters
    ----------
    fill_value:
        Pixel value used to fill masked regions.  Default: 0 (black).
    """

    def __init__(self, fill_value: int = _MASK_FILL) -> None:
        self._fill = fill_value

    def mask(
        self,
        image: np.ndarray,
        regions: list[Rect],
        *,
        labels: list[str] | None = None,
    ) -> tuple[np.ndarray, MaskingResult]:
        """
        Return a copy of *image* with *regions* blacked-out.

        Parameters
        ----------
        image:
            H×W or H×W×C numpy array (uint8).
        regions:
            Screen rectangles to erase.
        labels:
            Optional human-readable names for each region (for logging).

        Returns
        -------
        (masked_image, MaskingResult)
        """
        if image.dtype != np.uint8:
            raise ValueError(
                f"ScreenshotMasker expects uint8 image, got {image.dtype}"
            )

        out = image.copy()
        h, w = out.shape[:2]
        pixels_masked = 0
        effective_labels: list[str] = (
            labels or [f"region_{i}" for i in range(len(regions))]
        )

        for rect in regions:
            # Clip rect to actual image bounds
            x1 = max(0, rect.x)
            y1 = max(0, rect.y)
            x2 = min(w, rect.x + rect.width)
            y2 = min(h, rect.y + rect.height)
            if x2 <= x1 or y2 <= y1:
                continue  # region fully outside image
            out[y1:y2, x1:x2] = self._fill
            pixels_masked += (x2 - x1) * (y2 - y1)

        result = MaskingResult(
            regions_masked=len(regions),
            pixels_masked=pixels_masked,
            region_labels=effective_labels,
        )

        _log.info(
            "screenshot_masked",
            regions_masked=result.regions_masked,
            pixels_masked=result.pixels_masked,
            labels=result.region_labels,
        )

        return out, result

    def mask_from_sensitive(
        self,
        image: np.ndarray,
        sensitive_regions: list[object],  # list[SensitiveRegion] — avoid circular
    ) -> tuple[np.ndarray, MaskingResult]:
        """
        Convenience wrapper: accepts SensitiveRegion objects directly.

        Parameters
        ----------
        sensitive_regions:
            List of SensitiveRegion instances (duck-typed: .rect, .label).
        """
        rects = [sr.rect for sr in sensitive_regions]  # type: ignore[attr-defined]
        labels = [sr.label for sr in sensitive_regions]  # type: ignore[attr-defined]
        return self.mask(image, rects, labels=labels)
