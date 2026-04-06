"""
nexus/verification/source_verification.py
Source Verifier — post-action verification via structured source read-back.

Strategy
--------
After an action completes, re-read the target element's value through the
structured source layer (UIA / DOM) and compare it against the expected
value.

The verifier is transport-agnostic: all I/O goes through a pluggable
*source_probe* callable so real adapters can be swapped out in tests.

SourceProbe protocol
---------------------
Any callable that accepts a ``dict[str, Any]`` context dict and returns
either a ``str`` (the observed value) or ``None`` (source unavailable).

Matching
--------
Comparison is case-insensitive and strips leading/trailing whitespace.
  - Exact strip-match → confidence 1.0.
  - Contains match    → confidence 0.85 (observed contains expected).
  - No match          → confidence 0.0.

Retry
-----
The verifier retries up to ``policy.max_retries`` times with a short
backoff when confidence is below ``policy.confidence_threshold``.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nexus.infra.logger import get_logger
from nexus.verification.policy import VerificationMode, VerificationPolicy
from nexus.verification.result import VerificationResult

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Injectable probe type
# ---------------------------------------------------------------------------

# Accepts a context dict, returns the observed string value or None.
SourceProbe = Callable[[dict[str, Any]], str | None]

# Short sleep between retries (seconds) — kept minimal to avoid flakiness.
_RETRY_SLEEP_S: float = 0.05


# ---------------------------------------------------------------------------
# SourceVerifier
# ---------------------------------------------------------------------------


@dataclass
class SourceVerifier:
    """
    Verifies action outcomes by reading element values from the structured
    source layer.

    Parameters
    ----------
    source_probe:
        Callable that resolves an element value from context.  Must return
        a string on success or None when the element is unavailable.
    """

    source_probe: SourceProbe

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        expected_value: str,
        policy: VerificationPolicy,
        context: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """
        Attempt to verify that the element now holds *expected_value*.

        Parameters
        ----------
        expected_value:
            The value the action was supposed to produce.
        policy:
            Verification policy; controls timeout, threshold, and retries.
        context:
            Arbitrary key/value mapping forwarded to the source probe.
            Typically includes ``element_id`` and/or ``coordinates``.
        """
        ctx: dict[str, Any] = context or {}
        t0 = time.perf_counter()
        retries = 0
        last_result: VerificationResult | None = None

        for attempt in range(policy.max_retries + 1):
            if attempt > 0:
                retries += 1
                time.sleep(_RETRY_SLEEP_S)

            elapsed = time.perf_counter() - t0
            if elapsed >= policy.timeout_s:
                _log.warning(
                    "source_verifier.timeout",
                    attempt=attempt,
                    elapsed_s=round(elapsed, 3),
                )
                break

            observed = self._probe(ctx)
            confidence, match_type = self._match(expected_value, observed)

            duration_ms = (time.perf_counter() - t0) * 1000.0
            success = confidence >= policy.confidence_threshold

            last_result = VerificationResult(
                success=success,
                mode_used=VerificationMode.SOURCE,
                confidence=confidence,
                duration_ms=duration_ms,
                detail=self._build_detail(expected_value, observed, match_type),
                retries=retries,
                expected_value=expected_value,
                observed_value=observed,
            )

            _log.debug(
                "source_verifier.attempt",
                attempt=attempt,
                match_type=match_type,
                confidence=round(confidence, 3),
                success=success,
            )

            if success:
                return last_result

        # All attempts exhausted — return the last result (failure).
        if last_result is None:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            last_result = VerificationResult(
                success=False,
                mode_used=VerificationMode.SOURCE,
                confidence=0.0,
                duration_ms=duration_ms,
                detail="source_probe returned no result within timeout",
                retries=retries,
                expected_value=expected_value,
                observed_value=None,
            )

        last_result.success = False
        return last_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe(self, context: dict[str, Any]) -> str | None:
        """Invoke the source probe, catching and logging any exception."""
        try:
            return self.source_probe(context)
        except Exception as exc:  # noqa: BLE001
            _log.warning("source_verifier.probe_error", error=str(exc))
            return None

    @staticmethod
    def _match(expected: str, observed: str | None) -> tuple[float, str]:
        """
        Compare *expected* and *observed* strings.

        Returns
        -------
        (confidence, match_type)
            match_type is one of: "exact", "contains", "none", "unavailable".
        """
        if observed is None:
            return 0.0, "unavailable"

        exp = expected.strip().lower()
        obs = observed.strip().lower()

        if obs == exp:
            return 1.0, "exact"
        if exp in obs:
            return 0.85, "contains"
        return 0.0, "none"

    @staticmethod
    def _build_detail(
        expected: str,
        observed: str | None,
        match_type: str,
    ) -> str:
        return (
            f"expected={expected!r} "
            f"observed={observed!r} "
            f"match={match_type}"
        )
