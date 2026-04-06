"""
nexus/verification/result.py
VerificationResult — the outcome of a post-action verification attempt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from nexus.verification.policy import VerificationMode


@dataclass
class VerificationResult:
    """
    Outcome of a single post-action verification check.

    Attributes
    ----------
    success:
        True when verification confirms the action had its intended effect
        with confidence >= policy.confidence_threshold.
    mode_used:
        The actual verification mode that produced this result.  Will differ
        from the requested mode when AUTO resolves to VISUAL or SOURCE.
    confidence:
        Estimated confidence in the outcome in [0.0, 1.0].
        - SOURCE path: 1.0 on exact match, 0.0 on mismatch.
        - VISUAL path: pixel-change ratio clipped and normalised.
    duration_ms:
        Wall-clock time spent on all verification attempts (milliseconds).
    detail:
        Human-readable summary of the verification outcome.  Included in
        trace logs and HITL prompts.
    retries:
        Number of additional attempts beyond the first.
    expected_value:
        The value the caller expected to observe (may be None for visual).
    observed_value:
        The value actually observed by the verifier (may be None).
    """

    success: bool
    mode_used: VerificationMode
    confidence: float
    duration_ms: float = 0.0
    detail: str = ""
    retries: int = 0
    expected_value: str | None = None
    observed_value: str | None = None
    side_effects: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience factory — SKIP
    # ------------------------------------------------------------------

    @classmethod
    def skipped(cls) -> "VerificationResult":
        """Return a result representing a deliberately skipped verification."""
        return cls(
            success=True,
            mode_used=VerificationMode.SKIP,
            confidence=1.0,
            detail="verification skipped by policy",
        )
