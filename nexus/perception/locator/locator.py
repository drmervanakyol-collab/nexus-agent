"""
nexus/perception/locator/locator.py
UI Element Locator — detects and classifies UI elements in screen frames.

UIElement
---------
  id, element_type, bounding_box, confidence, is_visible,
  is_occluded, occlusion_ratio, z_order_estimate

ElementType
-----------
  BUTTON, INPUT, LABEL, IMAGE, ICON, CONTAINER, PANEL, MENU,
  DROPDOWN, CHECKBOX, RADIO, LINK, SCROLLBAR, TAB, DIALOG,
  TOOLTIP, UNKNOWN

Locator.locate(frame, dirty_regions, active_window)
---------------------------------------------------
Pipeline:
  1. Cache lookup — 200 ms TTL keyed on frame hash + active_window
  2. Region filtering — when dirty_regions given, process only those areas
  3. Grayscale + Canny edge detection
  4. Contour extraction and bounding-box computation
  5. Type heuristics — classify each contour by shape features
  6. Occlusion computation — overlapping boxes → occluded flag + ratio
  7. Z-order estimation — smaller/topmost elements get higher z-order
  8. Cache store

Raises LocatorError only on unrecoverable internal failures.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np

from nexus.capture.frame import Frame
from nexus.core.errors import LocatorError
from nexus.core.types import ElementId, Rect
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_CACHE_TTL_S: float = 0.200           # 200 ms
_CANNY_LOW: int = 50
_CANNY_HIGH: int = 150
_MIN_CONTOUR_AREA: int = 50           # px² — noise floor
_MAX_CONTOUR_AREA_RATIO: float = 0.90 # ignore contours covering >90% of frame

# Element size heuristics (in pixels)
_SMALL_MAX: int = 32       # checkbox / radio / icon
_SCROLLBAR_RATIO: float = 5.0    # width/height or height/width
_SCROLLBAR_THIN_MAX: int = 20    # scrollbar small dimension must be <= this
_BUTTON_ASPECT_MIN: float = 0.3
_BUTTON_ASPECT_MAX: float = 5.0
_INPUT_ASPECT_MIN: float = 3.5
_INPUT_MIN_W: int = 40
_INPUT_MAX_H: int = 60


# ---------------------------------------------------------------------------
# ElementType
# ---------------------------------------------------------------------------


class ElementType(Enum):
    BUTTON = auto()
    INPUT = auto()
    LABEL = auto()
    IMAGE = auto()
    ICON = auto()
    CONTAINER = auto()
    PANEL = auto()
    MENU = auto()
    DROPDOWN = auto()
    CHECKBOX = auto()
    RADIO = auto()
    LINK = auto()
    SCROLLBAR = auto()
    TAB = auto()
    DIALOG = auto()
    TOOLTIP = auto()
    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# UIElement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UIElement:
    """
    A detected UI element.

    Attributes
    ----------
    id:
        Unique opaque identifier for this detection instance.
    element_type:
        Inferred semantic type.
    bounding_box:
        Pixel rectangle in frame-local coordinates.
    confidence:
        Heuristic confidence score in [0.0, 1.0].
    is_visible:
        True when the element is not fully hidden.
    is_occluded:
        True when another element overlaps this one.
    occlusion_ratio:
        Fraction of this element's area that is covered by higher-z elements.
        0.0 when not occluded.
    z_order_estimate:
        Estimated stacking order; higher value = closer to the viewer.
    """

    id: ElementId
    element_type: ElementType
    bounding_box: Rect
    confidence: float
    is_visible: bool
    is_occluded: bool
    occlusion_ratio: float
    z_order_estimate: int


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    elements: list[UIElement]
    expires_at: float  # monotonic time


# ---------------------------------------------------------------------------
# Locator
# ---------------------------------------------------------------------------


class Locator:
    """
    Detects UI elements in captured screen frames using computer vision.

    Parameters
    ----------
    canny_low, canny_high:
        Thresholds for Canny edge detection.
    min_contour_area:
        Minimum contour area in pixels²; smaller contours are noise.
    cache_ttl:
        How long (seconds) a detection result is valid for the same frame.
    _time_fn:
        Injectable clock; defaults to ``time.monotonic``.
    """

    def __init__(
        self,
        canny_low: int = _CANNY_LOW,
        canny_high: int = _CANNY_HIGH,
        min_contour_area: int = _MIN_CONTOUR_AREA,
        cache_ttl: float = _CACHE_TTL_S,
        *,
        _time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._canny_low = canny_low
        self._canny_high = canny_high
        self._min_contour_area = min_contour_area
        self._cache_ttl = cache_ttl
        self._time: Callable[[], float] = _time_fn or time.monotonic
        self._cache: dict[str, _CacheEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def locate(
        self,
        frame: Frame,
        dirty_regions: Sequence[Rect] | None = None,
        active_window: Rect | None = None,
    ) -> list[UIElement]:
        """
        Detect UI elements in *frame*.

        Parameters
        ----------
        frame:
            The captured screen frame to analyse.
        dirty_regions:
            When provided, only regions that changed are analysed.  This
            restricts detection to the union of those rectangles.
        active_window:
            When provided, detection is clipped to this window boundary.

        Returns
        -------
        list[UIElement]
            Detected elements, possibly empty.

        Raises
        ------
        LocatorError
            On unrecoverable internal failures (e.g. corrupt frame data).
        """
        now = self._time()
        cache_key = self._cache_key(frame, active_window)

        # 1. Cache lookup
        cached = self._cache.get(cache_key)
        if cached is not None and cached.expires_at > now:
            _log.debug("locator_cache_hit", key=cache_key[:8])
            return cached.elements

        try:
            elements = self._detect(frame, dirty_regions, active_window)
        except Exception as exc:
            raise LocatorError(
                "Element detection failed",
                context={"error": str(exc)},
            ) from exc

        self._cache[cache_key] = _CacheEntry(
            elements=elements,
            expires_at=now + self._cache_ttl,
        )
        _log.debug(
            "locator_detected",
            count=len(elements),
            cached=False,
        )
        return elements

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _detect(
        self,
        frame: Frame,
        dirty_regions: Sequence[Rect] | None,
        active_window: Rect | None,
    ) -> list[UIElement]:
        image = frame.data  # HxWx3 RGB uint8
        frame_h, frame_w = image.shape[:2]
        frame_area = frame_h * frame_w

        # Determine the set of search rectangles
        search_rects = self._build_search_rects(
            frame_w, frame_h, dirty_regions, active_window
        )

        raw: list[tuple[Rect, float]] = []  # (bounding_box, raw_confidence)

        for region in search_rects:
            # Clip region to frame bounds
            clip = region.clip_to(Rect(0, 0, frame_w, frame_h))
            if clip.area() == 0:
                continue

            patch = image[
                clip.y : clip.y + clip.height,
                clip.x : clip.x + clip.width,
            ]

            # BGR for OpenCV (frame data is RGB)
            bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            # Canny edges
            edges = cv2.Canny(gray, self._canny_low, self._canny_high)

            # Contour extraction
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self._min_contour_area:
                    continue
                if area > frame_area * _MAX_CONTOUR_AREA_RATIO:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                # Re-offset to frame coordinates
                abs_box = Rect(clip.x + x, clip.y + y, w, h)
                conf = self._contour_confidence(cnt, area)
                raw.append((abs_box, conf))

        if not raw:
            return []

        # Deduplicate heavily overlapping boxes
        raw = self._nms(raw)

        # Assign z-order by area (smaller on top)
        sorted_by_area = sorted(raw, key=lambda t: t[0].area(), reverse=True)

        # Compute occlusion
        elements = self._build_elements(sorted_by_area, frame_area)
        return elements

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_search_rects(
        frame_w: int,
        frame_h: int,
        dirty_regions: Sequence[Rect] | None,
        active_window: Rect | None,
    ) -> list[Rect]:
        """Return the list of rectangles to scan."""
        if dirty_regions is None:
            regions = [Rect(0, 0, frame_w, frame_h)]
        else:
            regions = list(dirty_regions)  # empty list → nothing to scan

        if active_window is not None:
            regions = [
                r.clip_to(active_window)
                for r in regions
                if r.overlaps(active_window)
            ]

        return [r for r in regions if r.area() > 0]

    @staticmethod
    def _contour_confidence(cnt: np.ndarray, area: float) -> float:
        """
        Heuristic confidence: how well the contour approximates a rectangle.
        Perfect rectangle → 1.0; noisy polygon → lower.
        """
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            return 0.5
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        # Rectangles have 4 vertices; score falls off with vertex count
        n = len(approx)
        if n == 4:
            return 0.9
        elif n <= 6:
            return 0.75
        elif n <= 10:
            return 0.6
        return 0.45

    @staticmethod
    def _nms(
        raw: list[tuple[Rect, float]],
        iou_threshold: float = 0.5,
    ) -> list[tuple[Rect, float]]:
        """
        Non-maximum suppression: keep only the highest-confidence box
        when two boxes overlap by more than *iou_threshold*.
        """
        if len(raw) <= 1:
            return raw

        # Sort descending by confidence
        items = sorted(raw, key=lambda t: t[1], reverse=True)
        kept: list[tuple[Rect, float]] = []

        for item in items:
            box, conf = item
            dominated = False
            for kept_box, _ in kept:
                if _iou(box, kept_box) > iou_threshold:
                    dominated = True
                    break
            if not dominated:
                kept.append(item)

        return kept

    def _build_elements(
        self,
        sorted_raw: list[tuple[Rect, float]],
        frame_area: int,
    ) -> list[UIElement]:
        """
        Build UIElement objects with type classification, occlusion, and
        z-order fields.  *sorted_raw* must be ordered largest-area-first
        (i.e. background to foreground).
        """
        elements: list[UIElement] = []
        n = len(sorted_raw)

        for z_index, (box, conf) in enumerate(sorted_raw):
            element_type = _classify(box)

            # Occlusion: check all higher-z (smaller index in reverse)
            occlusion_area = 0
            for higher_z in range(z_index):
                higher_box = sorted_raw[higher_z][0]
                occlusion_area += _intersection_area(box, higher_box)

            box_area = box.area()
            occlusion_ratio = (
                min(1.0, occlusion_area / box_area) if box_area > 0 else 0.0
            )
            is_occluded = occlusion_ratio > 0.0

            elements.append(
                UIElement(
                    id=ElementId(str(uuid.uuid4())),
                    element_type=element_type,
                    bounding_box=box,
                    confidence=round(conf, 3),
                    is_visible=True,
                    is_occluded=is_occluded,
                    occlusion_ratio=round(occlusion_ratio, 3),
                    z_order_estimate=n - 1 - z_index,  # higher z = on top
                )
            )

        return elements

    @staticmethod
    def _cache_key(frame: Frame, active_window: Rect | None) -> str:
        """Stable key based on frame pixel content hash + window."""
        # Sample up to 4 KB of frame data for speed
        data_sample = frame.data.tobytes()[:4096]
        shape_str = f"{frame.width}x{frame.height}"
        window_str = str(active_window)
        raw = data_sample + shape_str.encode() + window_str.encode()
        return hashlib.md5(raw, usedforsecurity=False).hexdigest()


# ---------------------------------------------------------------------------
# Type classification heuristics
# ---------------------------------------------------------------------------


def _classify(box: Rect) -> ElementType:
    """Classify a bounding box into an ElementType using shape heuristics."""
    w, h = box.width, box.height
    if w == 0 or h == 0:
        return ElementType.UNKNOWN

    aspect = w / h  # > 1 → wide, < 1 → tall

    # Scrollbar: very elongated AND thin (inputs can also be wide but aren't thin)
    small_dim = min(w, h)
    elongated = aspect >= _SCROLLBAR_RATIO or aspect <= 1.0 / _SCROLLBAR_RATIO
    if elongated and small_dim <= _SCROLLBAR_THIN_MAX:
        return ElementType.SCROLLBAR

    # Checkbox / radio: small and roughly square
    if w <= _SMALL_MAX and h <= _SMALL_MAX:
        if 0.75 <= aspect <= 1.33:
            return ElementType.CHECKBOX
        return ElementType.ICON

    # Input: wide, not too tall
    if aspect >= _INPUT_ASPECT_MIN and w >= _INPUT_MIN_W and h <= _INPUT_MAX_H:
        return ElementType.INPUT

    # Button: reasonable aspect ratio, medium size
    if _BUTTON_ASPECT_MIN <= aspect <= _BUTTON_ASPECT_MAX and w <= 200 and h <= 80:
        return ElementType.BUTTON

    # Large areas → container / panel
    if w >= 300 or h >= 200:
        return ElementType.PANEL

    return ElementType.UNKNOWN


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def _intersection_area(a: Rect, b: Rect) -> int:
    """Pixel area of the intersection of two rectangles; 0 if disjoint."""
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _iou(a: Rect, b: Rect) -> float:
    """Intersection-over-Union for two rectangles."""
    inter = _intersection_area(a, b)
    if inter == 0:
        return 0.0
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0
