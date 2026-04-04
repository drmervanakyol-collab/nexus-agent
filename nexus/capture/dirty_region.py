"""
nexus/capture/dirty_region.py
Dirty region detection — finds which screen blocks changed between frames.

DirtyRegionDetector divides both frames into fixed-size blocks and compares
their raw pixel data.  A block is "dirty" when any pixel in it differs
between the two frames.

If ≥ FULL_REFRESH_THRESHOLD of all blocks are dirty the result sets
``full_refresh=True``, signalling that a targeted repaint is pointless
and a full-screen action should be taken instead.

Block coordinates
-----------------
Each ``Rect`` in ``DirtyRegions.blocks`` is expressed in frame-local pixel
coordinates with the top-left corner at (x, y).  The last column / row of
blocks may be narrower / shorter than ``block_size`` when the frame
dimensions are not perfectly divisible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nexus.capture.frame import Frame
from nexus.core.types import Rect
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

BLOCK_SIZE: int = 32  # pixels per block edge (square)
FULL_REFRESH_THRESHOLD: float = 0.80  # fraction of dirty blocks → full refresh


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirtyRegions:
    """
    Result of a dirty-region comparison between two frames.

    Attributes
    ----------
    blocks:
        Tuple of ``Rect`` objects describing each dirty block in frame
        pixel coordinates.  Empty when frames are identical.
    full_refresh:
        True when ``change_ratio >= FULL_REFRESH_THRESHOLD`` or when
        the frames have incompatible sizes.
    change_ratio:
        Fraction of total blocks that are dirty: ``len(blocks) / total``.
        Always in [0.0, 1.0].
    """

    blocks: tuple[Rect, ...]
    full_refresh: bool
    change_ratio: float


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class DirtyRegionDetector:
    """
    Detects changed regions between two consecutive frames using block hashing.

    Parameters
    ----------
    block_size:
        Edge length (pixels) of each square comparison block.  Default: 32.
    """

    def __init__(self, block_size: int = BLOCK_SIZE) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be ≥ 1, got {block_size}")
        self._block_size = block_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, prev: Frame, curr: Frame) -> DirtyRegions:
        """
        Compare *prev* and *curr* and return dirty blocks.

        When the frames have different dimensions every block in *curr*
        is considered dirty and ``full_refresh=True`` is returned.
        """
        if prev.width != curr.width or prev.height != curr.height:
            _log.debug(
                "dirty_region_size_mismatch",
                prev=(prev.width, prev.height),
                curr=(curr.width, curr.height),
            )
            all_blocks = tuple(self._enumerate_blocks(curr))
            return DirtyRegions(
                blocks=all_blocks,
                full_refresh=True,
                change_ratio=1.0,
            )

        bs = self._block_size
        h, w = curr.height, curr.width
        dirty: list[Rect] = []
        total = 0

        for row_start in range(0, h, bs):
            row_end = min(row_start + bs, h)
            for col_start in range(0, w, bs):
                col_end = min(col_start + bs, w)
                total += 1
                if not np.array_equal(
                    prev.data[row_start:row_end, col_start:col_end],
                    curr.data[row_start:row_end, col_start:col_end],
                ):
                    dirty.append(
                        Rect(
                            x=col_start,
                            y=row_start,
                            width=col_end - col_start,
                            height=row_end - row_start,
                        )
                    )

        if total == 0:
            return DirtyRegions(blocks=(), full_refresh=False, change_ratio=0.0)

        change_ratio = len(dirty) / total
        full_refresh = change_ratio >= FULL_REFRESH_THRESHOLD

        _log.debug(
            "dirty_region_detected",
            dirty_blocks=len(dirty),
            total_blocks=total,
            change_ratio=round(change_ratio, 4),
            full_refresh=full_refresh,
        )
        return DirtyRegions(
            blocks=tuple(dirty),
            full_refresh=full_refresh,
            change_ratio=change_ratio,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enumerate_blocks(self, frame: Frame) -> list[Rect]:
        """Return all block rects for *frame* in row-major order."""
        bs = self._block_size
        blocks: list[Rect] = []
        for row_start in range(0, frame.height, bs):
            row_end = min(row_start + bs, frame.height)
            for col_start in range(0, frame.width, bs):
                col_end = min(col_start + bs, frame.width)
                blocks.append(
                    Rect(
                        x=col_start,
                        y=row_start,
                        width=col_end - col_start,
                        height=row_end - row_start,
                    )
                )
        return blocks
