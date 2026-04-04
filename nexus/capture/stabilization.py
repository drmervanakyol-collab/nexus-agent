"""
nexus/capture/stabilization.py
Screen stabilization gate — waits until the display stops changing.

Architecture
------------
StabilizationGate polls a user-supplied ``_get_frame_fn`` at a
configurable interval and compares consecutive frames.  It declares the
screen stable when the pixel-change ratio drops below ``change_threshold``
and no loading indicators are present.

Three blocking conditions prevent a "stable" verdict:
  1. change_ratio > change_threshold — content is still animating
  2. spinner_detected — a small region cycles with constant motion
  3. loading_text_detected — OCR found a recognised loading phrase

All time-keeping (sleep) and OCR are injectable so the gate is fully
testable without a real display or Tesseract installation.

Pixel-change metric
-------------------
  change_ratio = (pixels with ANY channel diff > THRESHOLD) / total_pixels

THRESHOLD = 10 (out of 255) to ignore minor JPEG/anti-aliasing noise.
Result is always in [0.0, 1.0].

Spinner heuristic
-----------------
A spinner is characterised by a small region that cycles continuously.
Detection: the last SPINNER_WINDOW consecutive inter-frame change ratios
must ALL be in [SPINNER_MIN_RATIO, SPINNER_MAX_RATIO].
This band captures small periodic motion (e.g. a rotating icon) while
excluding idle screens (ratio ≈ 0) and large activity (ratio > 0.05).

Loading text patterns
---------------------
  Turkish: "yükleniyor", "lütfen bekleyin", "bekleyiniz"
  English: "loading", "please wait"
OCR is run only once per polled frame to keep CPU usage low.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from nexus.capture.frame import Frame
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# A pixel is considered "changed" when any channel differs by more than this.
PIXEL_CHANGE_THRESHOLD: int = 10

# Spinner detection: window size and change-ratio band.
_SPINNER_WINDOW: int = 4        # consecutive frames to analyse
_SPINNER_MIN_RATIO: float = 3e-4  # 0.03 % — at least some pixels move
_SPINNER_MAX_RATIO: float = 0.05  # 5 %  — bounded (spinner is a small widget)

# Loading text patterns (lower-case; matched via substring).
_LOADING_PATTERNS: frozenset[str] = frozenset(
    [
        "yükleniyor",
        "lütfen bekleyin",
        "bekleyiniz",
        "loading",
        "please wait",
    ]
)


# ---------------------------------------------------------------------------
# Public result value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilizationResult:
    """
    Outcome of a single ``wait_for_stable`` call.

    Attributes
    ----------
    stable:
        True when the screen was considered stable before the timeout.
    waited_ms:
        Wall-clock milliseconds elapsed inside ``wait_for_stable``.
    reason:
        Human-readable token explaining the outcome:
        ``"stable"`` | ``"timeout"`` | ``"spinner_detected"`` |
        ``"timeout_loading_text"``.
    change_ratio_final:
        The last computed inter-frame pixel-change ratio in [0.0, 1.0].
    """

    stable: bool
    waited_ms: float
    reason: str
    change_ratio_final: float


# ---------------------------------------------------------------------------
# OCR default implementation (production only; injectable in tests)
# ---------------------------------------------------------------------------


def _default_ocr(frame: Frame) -> str:
    """
    Extract text from *frame* using pytesseract (Tesseract OCR).

    Returns an empty string on any error so callers never raise.
    Uses --psm 11 (sparse text) to find text anywhere on the screen.
    """
    try:
        import pytesseract  # type: ignore[import-untyped]
        from PIL import Image  # lazy import

        img = Image.fromarray(frame.data, mode="RGB")
        return pytesseract.image_to_string(
            img,
            lang="eng+tur",
            config="--psm 11",
        )
    except Exception as exc:
        _log.debug("ocr_failed_in_stabilization", error=str(exc))
        return ""


# ---------------------------------------------------------------------------
# StabilizationGate
# ---------------------------------------------------------------------------

OcrFn = Callable[[Frame], str]
GetFrameFn = Callable[[], "Frame | None"]
SleepFn = Callable[[float], None]


class StabilizationGate:
    """
    Polls the screen and declares stability when change activity ceases.

    Parameters
    ----------
    _get_frame_fn:
        ``() -> Frame | None``.  Returns the current screen frame, or
        ``None`` when no frame is available yet.  Typically wired to
        ``CaptureWorkerClient.get_latest_frame``.
    _ocr_fn:
        ``(Frame) -> str``.  Extracts text from a frame for loading-text
        detection.  Defaults to ``_default_ocr`` (pytesseract).
        Inject a no-op (``lambda f: ""``) in tests to skip OCR.
    _sleep_fn:
        ``(seconds: float) -> None``.  Defaults to ``time.sleep``.
        Inject a no-op in tests for instant execution.
    """

    def __init__(
        self,
        _get_frame_fn: GetFrameFn,
        *,
        _ocr_fn: OcrFn | None = None,
        _sleep_fn: SleepFn | None = None,
    ) -> None:
        self._get_frame = _get_frame_fn
        self._ocr_fn: OcrFn = _ocr_fn or _default_ocr
        self._sleep: SleepFn = _sleep_fn or time.sleep

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wait_for_stable(
        self,
        timeout_ms: float = 3000.0,
        poll_ms: float = 100.0,
        change_threshold: float = 0.02,
    ) -> StabilizationResult:
        """
        Block until the screen stabilises or the timeout expires.

        Parameters
        ----------
        timeout_ms:
            Maximum wall-clock time to wait (milliseconds).
        poll_ms:
            Interval between frame captures (milliseconds).
        change_threshold:
            Maximum pixel-change ratio that counts as "stable".
            Range [0.0, 1.0]; default 0.02 (2 %).

        Returns
        -------
        StabilizationResult
            ``stable=True`` when the screen settled within the timeout.
        """
        t0 = time.monotonic()
        deadline = t0 + timeout_ms / 1000.0
        poll_s = poll_ms / 1000.0

        history: deque[Frame] = deque(maxlen=_SPINNER_WINDOW)
        prev_frame: Frame | None = None
        change_ratio: float = 0.0
        last_was_loading: bool = False

        while True:
            now = time.monotonic()
            waited_ms = (now - t0) * 1000.0

            if now >= deadline:
                reason = "timeout_loading_text" if last_was_loading else "timeout"
                return StabilizationResult(
                    stable=False,
                    waited_ms=waited_ms,
                    reason=reason,
                    change_ratio_final=change_ratio,
                )

            frame = self._get_frame()
            if frame is None:
                self._sleep(poll_s)
                continue

            history.append(frame)

            if prev_frame is not None:
                change_ratio = self._compare_frames(prev_frame, frame)

                # --- Spinner check ---
                if (
                    len(history) >= _SPINNER_WINDOW
                    and self._detect_spinner(list(history))
                ):
                    return StabilizationResult(
                        stable=False,
                        waited_ms=(time.monotonic() - t0) * 1000.0,
                        reason="spinner_detected",
                        change_ratio_final=change_ratio,
                    )

                # --- Loading text check ---
                last_was_loading = self._detect_loading_text(frame)

                # --- Stability verdict (only after enough history for spinner detection) ---
                if (
                    len(history) >= _SPINNER_WINDOW
                    and change_ratio <= change_threshold
                    and not last_was_loading
                ):
                    return StabilizationResult(
                        stable=True,
                        waited_ms=(time.monotonic() - t0) * 1000.0,
                        reason="stable",
                        change_ratio_final=change_ratio,
                    )

                _log.debug(
                    "stabilization_polling",
                    change_ratio=round(change_ratio, 5),
                    loading=last_was_loading,
                    waited_ms=round(waited_ms, 1),
                )

            prev_frame = frame
            self._sleep(poll_s)

    # ------------------------------------------------------------------
    # Frame analysis helpers
    # ------------------------------------------------------------------

    def _compare_frames(self, f1: Frame, f2: Frame) -> float:
        """
        Compute the pixel-change ratio between *f1* and *f2*.

        A pixel is counted as "changed" when any of its RGB channels
        differs by more than ``PIXEL_CHANGE_THRESHOLD``.

        Returns a float in [0.0, 1.0]:
          0.0 — frames are identical (within threshold)
          1.0 — every pixel has changed (or frames have different sizes)
        """
        if f1.width != f2.width or f1.height != f2.height:
            return 1.0  # incomparable sizes → treat as fully changed

        diff: Any = np.abs(
            f1.data.astype(np.int16) - f2.data.astype(np.int16)
        )
        changed = np.any(diff > PIXEL_CHANGE_THRESHOLD, axis=2)
        total = f1.width * f1.height
        if total == 0:
            return 0.0
        return float(np.sum(changed)) / total

    def _detect_spinner(self, frames: list[Frame]) -> bool:
        """
        Return True when *frames* exhibit spinner-like motion.

        Spinner signature: every consecutive inter-frame change ratio
        lies within [_SPINNER_MIN_RATIO, _SPINNER_MAX_RATIO].  This
        distinguishes the small periodic motion of a rotating icon from
        an idle screen (ratio ≈ 0) or large content changes (ratio ≫ 0.05).

        Requires at least _SPINNER_WINDOW frames; returns False otherwise.
        """
        if len(frames) < _SPINNER_WINDOW:
            return False
        # Use only the last _SPINNER_WINDOW frames for efficiency
        window = frames[-_SPINNER_WINDOW:]
        ratios = [
            self._compare_frames(window[i - 1], window[i])
            for i in range(1, len(window))
        ]
        return all(_SPINNER_MIN_RATIO <= r <= _SPINNER_MAX_RATIO for r in ratios)

    def _detect_loading_text(self, frame: Frame) -> bool:
        """
        Return True when OCR detects a recognised loading phrase in *frame*.

        Recognised patterns (case-insensitive, substring match):
          "yükleniyor", "lütfen bekleyin", "bekleyiniz",
          "loading", "please wait"

        Note: Turkish uppercase İ (U+0130) lowercases to i + U+0307 (combining
        dot above).  The combining character is stripped so that "YÜKLENİYOR"
        correctly matches "yükleniyor".
        """
        text = self._ocr_fn(frame).lower().replace("\u0307", "")
        return any(pattern in text for pattern in _LOADING_PATTERNS)
