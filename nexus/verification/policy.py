"""
nexus/verification/policy.py
Verification Policy — controls when and how post-action verification runs.

VerificationMode
----------------
  SKIP   — no verification (fastest; suitable for idempotent reads).
  VISUAL — compare before/after frames using pixel diff + OCR.
  SOURCE — re-read element value from structured source (UIA / DOM).
  AUTO   — SOURCE when a structured source is available, VISUAL otherwise.

VerificationPolicy
------------------
Immutable configuration object consumed by VisualVerifier and SourceVerifier.

  mode                : VerificationMode          (default AUTO)
  timeout_s           : float                     max wall-clock per attempt
  confidence_threshold: float                     minimum acceptable confidence
  max_retries         : int                       verification retry count
  require_change      : bool                      fail if before==after (visual)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


# ---------------------------------------------------------------------------
# VerificationMode
# ---------------------------------------------------------------------------


class VerificationMode(Enum):
    """Controls which verification strategy is applied after an action."""

    SKIP = auto()    # No verification — caller accepts the risk.
    VISUAL = auto()  # Frame diff + optional OCR text check.
    SOURCE = auto()  # Structured source (UIA / DOM) value read-back.
    AUTO = auto()    # SOURCE if available, VISUAL as fallback.


# ---------------------------------------------------------------------------
# VerificationPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationPolicy:
    """
    Immutable policy that governs post-action verification behaviour.

    Attributes
    ----------
    mode:
        The verification strategy to apply.
    timeout_s:
        Maximum seconds to spend on a single verification attempt.
    confidence_threshold:
        The minimum ``VerificationResult.confidence`` required for the
        result to be considered successful.  Values in [0.0, 1.0].
    max_retries:
        How many additional attempts to make when confidence falls below
        the threshold.  0 means a single attempt only.
    require_change:
        When True and mode is VISUAL, verification fails if the
        before/after frames are identical (i.e. no observable change).
        Useful for destructive writes where the screen *must* update.
    """

    mode: VerificationMode = VerificationMode.AUTO
    timeout_s: float = 2.0
    confidence_threshold: float = 0.80
    max_retries: int = 1
    require_change: bool = False

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def skip(cls) -> "VerificationPolicy":
        """Return a policy that performs no verification."""
        return cls(mode=VerificationMode.SKIP)

    @classmethod
    def visual(
        cls,
        *,
        timeout_s: float = 2.0,
        confidence_threshold: float = 0.80,
        require_change: bool = False,
    ) -> "VerificationPolicy":
        """Return a visual-only verification policy."""
        return cls(
            mode=VerificationMode.VISUAL,
            timeout_s=timeout_s,
            confidence_threshold=confidence_threshold,
            require_change=require_change,
        )

    @classmethod
    def source(
        cls,
        *,
        timeout_s: float = 2.0,
        confidence_threshold: float = 0.90,
    ) -> "VerificationPolicy":
        """Return a source-based verification policy."""
        return cls(
            mode=VerificationMode.SOURCE,
            timeout_s=timeout_s,
            confidence_threshold=confidence_threshold,
        )
