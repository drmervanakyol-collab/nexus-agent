"""
nexus/verification/visual_verification.py
Visual Verifier — post-action verification via frame diff and optional OCR.

Strategy
--------
1. Compute a normalised pixel-change ratio between *before_frame* and
   *after_frame* (RMSE over all pixels, clipped to [0, 1]).
2. If *expected_text* is provided: run OCR on *after_frame* and check
   whether the expected string appears in the extracted text.  Combine
   the pixel-change signal with the OCR match signal.
3. Map the combined signal to a confidence score:
     - No expected_text  : confidence = min(change_ratio * 10, 1.0)
       (any visible change → 100 % when ratio ≥ 0.10)
     - expected_text hit : confidence = 0.9 + 0.1 * min(change_ratio * 5, 1.0)
     - expected_text miss: confidence = change_ratio * 0.5

Injectability
-------------
OCR is optional.  Pass ``ocr_fn=None`` to disable OCR-based confirmation.
``ocr_fn`` must accept a numpy ``ndarray`` (HxWx3 uint8) and return a
plain string (the concatenated text recognised in the frame).

All heavy I/O is confined to ``verify()`` so callers can await it from
an async context by wrapping with ``asyncio.to_thread``.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nexus.capture.frame import Frame
from nexus.infra.logger import get_logger
from nexus.verification.policy import VerificationMode, VerificationPolicy
from nexus.verification.result import VerificationResult

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Type alias for the injectable OCR function
# ---------------------------------------------------------------------------

# Accepts an HxWx3 uint8 ndarray, returns recognised text as a string.
OcrFn = Callable[[np.ndarray], str]


# ---------------------------------------------------------------------------
# VisualVerifier
# ---------------------------------------------------------------------------


@dataclass
class VisualVerifier:
    """
    Verifies action outcomes by comparing before/after screen frames.

    Parameters
    ----------
    ocr_fn:
        Optional callable that performs OCR on an image array.  When
        provided, text presence is used to augment the confidence score.
    """

    ocr_fn: OcrFn | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        before_frame: Frame,
        after_frame: Frame,
        policy: VerificationPolicy,
        *,
        expected_text: str | None = None,
    ) -> VerificationResult:
        """
        Compare *before_frame* and *after_frame* and return a
        :class:`~nexus.verification.result.VerificationResult`.

        Parameters
        ----------
        before_frame:
            Frame captured immediately before the action.
        after_frame:
            Frame captured after the action (and any post-action wait).
        policy:
            Verification policy providing timeout and threshold values.
        expected_text:
            Optional text string that should appear in *after_frame*.
            When supplied, OCR is attempted if ``self.ocr_fn`` is set.
        """
        t0 = time.perf_counter()

        change_ratio = self._pixel_change_ratio(before_frame, after_frame)
        ocr_match: bool | None = None
        ocr_text: str | None = None

        if expected_text is not None and self.ocr_fn is not None:
            try:
                ocr_text = self.ocr_fn(after_frame.data)
                ocr_match = expected_text.lower() in ocr_text.lower()
            except Exception as exc:  # noqa: BLE001
                _log.warning("visual_verifier.ocr_failed", error=str(exc))
                ocr_match = None

        confidence = self._compute_confidence(
            change_ratio=change_ratio,
            expected_text=expected_text,
            ocr_match=ocr_match,
        )

        # require_change: fail if the screen did not visibly update
        if policy.require_change and change_ratio < 0.001:
            confidence = 0.0

        success = confidence >= policy.confidence_threshold
        duration_ms = (time.perf_counter() - t0) * 1000.0

        detail = self._build_detail(
            change_ratio=change_ratio,
            expected_text=expected_text,
            ocr_match=ocr_match,
            confidence=confidence,
        )

        _log.debug(
            "visual_verifier.result",
            change_ratio=round(change_ratio, 4),
            ocr_match=ocr_match,
            confidence=round(confidence, 3),
            success=success,
        )

        return VerificationResult(
            success=success,
            mode_used=VerificationMode.VISUAL,
            confidence=confidence,
            duration_ms=duration_ms,
            detail=detail,
            expected_value=expected_text,
            observed_value=ocr_text,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pixel_change_ratio(before: Frame, after: Frame) -> float:
        """
        Return a normalised measure of pixel change between two frames.

        Uses mean absolute difference normalised to [0, 1].
        Returns 0.0 when frames have different shapes (conservative).
        """
        b = before.data.astype(np.float32)
        a = after.data.astype(np.float32)

        if b.shape != a.shape:
            _log.warning(
                "visual_verifier.shape_mismatch",
                before=b.shape,
                after=a.shape,
            )
            return 0.0

        mean_abs_diff = float(np.mean(np.abs(a - b)))
        # normalise: 255 = maximum possible per-channel diff
        return mean_abs_diff / 255.0

    @staticmethod
    def _compute_confidence(
        *,
        change_ratio: float,
        expected_text: str | None,
        ocr_match: bool | None,
    ) -> float:
        """Map change_ratio and OCR match to a [0, 1] confidence score."""
        if expected_text is None:
            # Pure pixel-diff path: any perceptible change → high confidence.
            # change_ratio ≥ 0.10 → 1.0; linear below that.
            return min(change_ratio * 10.0, 1.0)

        if ocr_match is None:
            # OCR was not attempted or failed — fall back to pixel-diff only.
            return min(change_ratio * 10.0, 1.0)

        if ocr_match:
            # Text found: high base + pixel-change bonus.
            return 0.9 + 0.1 * min(change_ratio * 5.0, 1.0)

        # Text not found: penalise heavily.
        return change_ratio * 0.5

    @staticmethod
    def _build_detail(
        *,
        change_ratio: float,
        expected_text: str | None,
        ocr_match: bool | None,
        confidence: float,
    ) -> str:
        parts = [f"change_ratio={change_ratio:.4f}"]
        if expected_text is not None:
            parts.append(f"expected_text={expected_text!r}")
            if ocr_match is True:
                parts.append("ocr=match")
            elif ocr_match is False:
                parts.append("ocr=no_match")
            else:
                parts.append("ocr=skipped")
        parts.append(f"confidence={confidence:.3f}")
        return " ".join(parts)
