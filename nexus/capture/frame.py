"""
nexus/capture/frame.py
Immutable screen-frame value object.

Frame carries raw HxWx3 uint8 RGB pixel data together with capture
timestamps and a monotonic sequence number.  It intentionally has no
dependency on the capture backend (dxcam, PIL, etc.) so that it can
be created cheaply in tests.

PNG encoding and geometric transformations (crop / resize) are provided
as convenience methods but are *not* on the hot path — they use PIL
internally via lazy imports.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from nexus.core.types import Rect


@dataclass
class Frame:
    """
    A single captured screen frame.

    Attributes
    ----------
    data:
        HxWx3 uint8 numpy array in RGB order.  Callers that need to
        keep the array alive after the Frame is discarded should call
        ``data.copy()``; the array is not guaranteed to own its memory
        (it may be a view over shared memory).
    width, height:
        Pixel dimensions.  Always equal to ``data.shape[1]`` and
        ``data.shape[0]`` respectively.
    captured_at_monotonic:
        ``time.monotonic()`` value recorded at capture time.  Suitable
        for measuring frame intervals; not meaningful across restarts.
    captured_at_utc:
        ISO-8601 UTC timestamp string (e.g. ``"2026-04-03T12:00:00+00:00"``).
    sequence_number:
        Monotonically increasing integer; resets to 1 when the capture
        worker is (re-)started.
    """

    data: np.ndarray
    width: int
    height: int
    captured_at_monotonic: float
    captured_at_utc: str
    sequence_number: int

    # ------------------------------------------------------------------
    # Encoding (recording / debugging only — never on the hot path)
    # ------------------------------------------------------------------

    def to_png_bytes(self) -> bytes:
        """
        Encode the frame as PNG bytes.

        For logging, recording, and debugging only.  compress_level=1
        keeps encoding fast at the cost of larger files.
        """
        from PIL import Image  # lazy import — not on the hot path

        img = Image.fromarray(self.data, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False, compress_level=1)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Geometric transformations
    # ------------------------------------------------------------------

    def crop(self, rect: Rect) -> Frame:
        """
        Return a new Frame cropped to *rect* in frame-local pixel
        coordinates.

        The crop is silently clipped to valid frame bounds so an
        out-of-range rect never raises; the result may be smaller than
        *rect* if *rect* extends beyond the frame edge.
        """
        x0 = max(0, rect.x)
        y0 = max(0, rect.y)
        x1 = min(self.width, rect.x + rect.width)
        y1 = min(self.height, rect.y + rect.height)
        cropped = self.data[y0:y1, x0:x1].copy()
        h, w = cropped.shape[:2]
        return Frame(
            data=cropped,
            width=w,
            height=h,
            captured_at_monotonic=self.captured_at_monotonic,
            captured_at_utc=self.captured_at_utc,
            sequence_number=self.sequence_number,
        )

    def resize(self, scale: float) -> Frame:
        """
        Return a new Frame uniformly scaled by *scale*.

        Parameters
        ----------
        scale:
            Positive float multiplier (e.g. ``0.5`` = half size,
            ``2.0`` = double size).  Uses LANCZOS resampling.

        Raises
        ------
        ValueError
            When *scale* is <= 0.
        """
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale!r}")

        from PIL import Image  # lazy import

        new_w = max(1, round(self.width * scale))
        new_h = max(1, round(self.height * scale))
        img = Image.fromarray(self.data, mode="RGB")
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        arr = np.asarray(resized, dtype=np.uint8).copy()
        return Frame(
            data=arr,
            width=new_w,
            height=new_h,
            captured_at_monotonic=self.captured_at_monotonic,
            captured_at_utc=self.captured_at_utc,
            sequence_number=self.sequence_number,
        )
