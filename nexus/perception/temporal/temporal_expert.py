"""
nexus/perception/temporal/temporal_expert.py
Temporal Expert — classifies the current screen state from a frame history.

ScreenState
-----------
  state_type       : StateType
                     STABLE|LOADING|ANIMATING|TRANSITIONING|FROZEN|UNKNOWN
  confidence       : float          [0.0, 1.0]
  blocks_perception: bool           True when perception should wait
  reason           : str            human-readable token
  retry_after_ms   : int            suggested wait before re-analysis (0 = no wait)

TemporalExpert.analyze(frame_history)
--------------------------------------
Requires at least 3 frames; returns UNKNOWN when fewer are provided.

Detection pipeline (evaluated in priority order):
  1. UNKNOWN     — fewer than 3 frames, or all frames identical dimension-mismatch
  2. FROZEN      — elapsed time >= FROZEN_THRESHOLD_S and no pixel change
  3. LOADING     — loading text found  OR  spinner pattern detected
  4. TRANSITIONING — very high recent change ratio (> TRANS_THRESHOLD)
  5. ANIMATING   — moderate recent change (> ANIM_THRESHOLD), localised
  6. STABLE      — all recent ratios below STABLE_THRESHOLD
  7. UNKNOWN     — fallback

Pixel-change metric
-------------------
  A pixel is "changed" when any channel diff > PIXEL_CHANGE_THRESHOLD (10/255).
  change_ratio = changed_pixels / total_pixels   ∈ [0.0, 1.0]

Spinner heuristic
-----------------
  The last SPINNER_WINDOW inter-frame change ratios must ALL lie in
  [SPINNER_MIN_RATIO, SPINNER_MAX_RATIO].  This captures small periodic
  motion (rotating icon) while excluding idle (≈ 0) and large activity.

Loading text
------------
  Matched as lower-case substrings (Turkish + English):
  "yükleniyor", "lütfen bekleyin", "bekleyiniz", "loading", "please wait"

Injectable dependencies
-----------------------
  _ocr_fn : (Frame) -> str   defaults to a pytesseract call; inject a stub in tests.

retry_after_ms defaults
-----------------------
  LOADING       → 500 ms
  ANIMATING     → 200 ms
  TRANSITIONING → 300 ms
  FROZEN        → 1000 ms
  STABLE        → 0 ms
  UNKNOWN       → 100 ms
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from nexus.capture.frame import Frame
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_MIN_FRAMES: int = 3

# Pixel change threshold (matches StabilizationGate)
_PIXEL_CHANGE_THRESHOLD: int = 10

# Change-ratio thresholds
_STABLE_THRESHOLD: float = 0.02     # below → stable
_ANIM_THRESHOLD: float = 0.05       # above → animating
_TRANS_THRESHOLD: float = 0.30      # above → transitioning (full-screen jump)

# Spinner detection band (matches StabilizationGate)
_SPINNER_MIN_RATIO: float = 3e-4
_SPINNER_MAX_RATIO: float = 0.05
_SPINNER_WINDOW: int = 4            # need this many consecutive ratios in band

# Frozen: no change for this many seconds
_FROZEN_THRESHOLD_S: float = 5.0

# How many trailing frames must all be stable to call STABLE
_STABLE_MIN_FRAMES: int = 3

# Loading text patterns (lower-case, substring match)
_LOADING_PATTERNS: frozenset[str] = frozenset([
    "yükleniyor",
    "lütfen bekleyin",
    "bekleyiniz",
    "loading",
    "please wait",
])

# Default retry suggestions per state (milliseconds)
_RETRY_MS: dict[str, int] = {
    "LOADING":       500,
    "ANIMATING":     200,
    "TRANSITIONING": 300,
    "FROZEN":        1000,
    "STABLE":        0,
    "UNKNOWN":       100,
}


# ---------------------------------------------------------------------------
# StateType
# ---------------------------------------------------------------------------


class StateType(Enum):
    STABLE       = auto()
    LOADING      = auto()
    ANIMATING    = auto()
    TRANSITIONING = auto()
    FROZEN       = auto()
    UNKNOWN      = auto()


# ---------------------------------------------------------------------------
# ScreenState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreenState:
    """
    Classification of the current screen state derived from frame history.

    Attributes
    ----------
    state_type:
        The detected screen state.
    confidence:
        Heuristic confidence in [0.0, 1.0].
    blocks_perception:
        True when the state should prevent further perception until it
        resolves (e.g. loading, animating, frozen).
    reason:
        Short human-readable token identifying the primary signal that
        produced this classification.
    retry_after_ms:
        Suggested minimum wait in milliseconds before re-analysing.
        0 when no wait is needed (STABLE).
    """

    state_type: StateType
    confidence: float
    blocks_perception: bool
    reason: str
    retry_after_ms: int


# ---------------------------------------------------------------------------
# Default OCR (injectable; production only)
# ---------------------------------------------------------------------------


def _default_ocr(frame: Frame) -> str:
    """Extract text from *frame* via pytesseract; returns '' on any error."""
    try:
        import pytesseract
        from PIL import Image

        img = Image.fromarray(frame.data, mode="RGB")
        return str(pytesseract.image_to_string(img, lang="eng+tur", config="--psm 11"))
    except Exception as exc:
        _log.debug("temporal_ocr_failed", error=str(exc))
        return ""


# ---------------------------------------------------------------------------
# TemporalExpert
# ---------------------------------------------------------------------------


class TemporalExpert:
    """
    Classifies the current screen state from a sequence of recent frames.

    Parameters
    ----------
    _ocr_fn:
        ``(Frame) -> str`` — injectable OCR; defaults to pytesseract.
        Inject ``lambda f: ""`` in tests to skip OCR.
    stable_threshold:
        Per-frame pixel-change ratio below which a frame is "stable".
    anim_threshold:
        Ratio above which a frame is "animating".
    trans_threshold:
        Ratio above which a frame is "transitioning" (full-screen change).
    frozen_threshold_s:
        Seconds with no change before declaring FROZEN.
    """

    def __init__(
        self,
        *,
        _ocr_fn: Callable[[Frame], str] | None = None,
        stable_threshold: float = _STABLE_THRESHOLD,
        anim_threshold: float = _ANIM_THRESHOLD,
        trans_threshold: float = _TRANS_THRESHOLD,
        frozen_threshold_s: float = _FROZEN_THRESHOLD_S,
    ) -> None:
        self._ocr: Callable[[Frame], str] = _ocr_fn or _default_ocr
        self._stable_threshold = stable_threshold
        self._anim_threshold = anim_threshold
        self._trans_threshold = trans_threshold
        self._frozen_threshold_s = frozen_threshold_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, frame_history: Sequence[Frame]) -> ScreenState:
        """
        Classify the current screen state from *frame_history*.

        Parameters
        ----------
        frame_history:
            Ordered list of recent frames (oldest first).  At least 3
            frames are required for a meaningful classification.

        Returns
        -------
        ScreenState
            Always returns a valid ScreenState; never raises.
        """
        frames = list(frame_history)

        if len(frames) < _MIN_FRAMES:
            return _state(StateType.UNKNOWN, 0.5, True, "insufficient_frames")

        # Compute inter-frame change ratios (N-1 ratios for N frames)
        ratios: list[float] = []
        for i in range(1, len(frames)):
            ratios.append(_change_ratio(frames[i - 1], frames[i]))

        latest = frames[-1]

        # 1. FROZEN — long elapsed time + no pixel change
        elapsed_s = _elapsed_s(frames)
        if (
            elapsed_s >= self._frozen_threshold_s
            and all(r < 1e-5 for r in ratios)
        ):
            return _state(StateType.FROZEN, 0.90, True, "frozen_no_change")

        # 2. LOADING — loading text or spinner
        try:
            ocr_text = self._ocr(latest).lower()
        except Exception:
            ocr_text = ""

        if _has_loading_text(ocr_text):
            return _state(StateType.LOADING, 0.85, True, "loading_text")

        if _spinner_detected(ratios):
            return _state(StateType.LOADING, 0.80, True, "spinner")

        # 3. TRANSITIONING — very high recent change
        recent = ratios[-min(3, len(ratios)):]
        max_recent = max(recent)
        mean_recent = sum(recent) / len(recent)

        if max_recent > self._trans_threshold:
            return _state(StateType.TRANSITIONING, 0.80, True, "high_change")

        # 4. ANIMATING — moderate change
        if mean_recent > self._anim_threshold:
            return _state(StateType.ANIMATING, 0.75, True, "animating")

        # 5. STABLE — all recent frames below threshold
        stable_window = ratios[-min(_STABLE_MIN_FRAMES, len(ratios)):]
        if all(r <= self._stable_threshold for r in stable_window):
            conf = 0.95 if len(stable_window) >= _STABLE_MIN_FRAMES else 0.75
            return _state(StateType.STABLE, conf, False, "stable")

        # 6. Fallback
        return _state(StateType.UNKNOWN, 0.40, True, "indeterminate")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _state(
    state_type: StateType,
    confidence: float,
    blocks: bool,
    reason: str,
) -> ScreenState:
    return ScreenState(
        state_type=state_type,
        confidence=confidence,
        blocks_perception=blocks,
        reason=reason,
        retry_after_ms=_RETRY_MS[state_type.name],
    )


def _change_ratio(a: Frame, b: Frame) -> float:
    """
    Fraction of pixels whose value changed by more than the threshold.
    Returns 1.0 when frames have different shapes (full change assumed).
    """
    if a.data.shape != b.data.shape:
        return 1.0
    h, w = a.data.shape[:2]
    total = h * w
    if total == 0:
        return 0.0
    diff = np.abs(
        a.data.astype(np.int32) - b.data.astype(np.int32)
    )
    changed = int(np.any(diff > _PIXEL_CHANGE_THRESHOLD, axis=-1).sum())
    return float(changed) / float(total)


def _elapsed_s(frames: list[Frame]) -> float:
    """Wall-clock duration (seconds) covered by *frames*."""
    if len(frames) < 2:
        return 0.0
    return frames[-1].captured_at_monotonic - frames[0].captured_at_monotonic


def _has_loading_text(text_lower: str) -> bool:
    """Return True when *text_lower* contains any known loading pattern."""
    return any(pattern in text_lower for pattern in _LOADING_PATTERNS)


def _spinner_detected(ratios: list[float]) -> bool:
    """
    Return True when the last SPINNER_WINDOW ratios all lie in
    [SPINNER_MIN_RATIO, SPINNER_MAX_RATIO] (small periodic motion).
    """
    if len(ratios) < _SPINNER_WINDOW:
        return False
    window = ratios[-_SPINNER_WINDOW:]
    return all(_SPINNER_MIN_RATIO <= r <= _SPINNER_MAX_RATIO for r in window)
