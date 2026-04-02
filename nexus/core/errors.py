"""
nexus/core/errors.py
Full Nexus Agent exception hierarchy.

Design rules:
  - Every exception carries: code (str), recoverable (bool), context (dict).
  - recoverable=True  → caller may retry or fall back automatically.
  - recoverable=False → human attention or abort required.
  - code is a stable snake_case string suitable for logging / telemetry.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class NexusError(Exception):
    """Root of all Nexus Agent exceptions."""

    code: str = "nexus_error"
    recoverable: bool = False

    def __init__(
        self,
        message: str = "",
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}


# ---------------------------------------------------------------------------
# Source layer
# ---------------------------------------------------------------------------


class SourceError(NexusError):
    """Raised when reading from any structured source fails."""

    code = "source_error"
    recoverable = True


class UIAError(SourceError):
    """Windows UI Automation access failed."""

    code = "uia_error"
    recoverable = True


class DOMError(SourceError):
    """Chrome DevTools Protocol / DOM access failed."""

    code = "dom_error"
    recoverable = True


class FileAdapterError(SourceError):
    """File-system or structured-file read failed."""

    code = "file_adapter_error"
    recoverable = False


# ---------------------------------------------------------------------------
# Transport layer
# ---------------------------------------------------------------------------


class TransportError(NexusError):
    """Raised when delivering an action via any transport fails."""

    code = "transport_error"
    recoverable = True


class NativeActionFailedError(TransportError):
    """UIA or CDP action was dispatched but did not complete successfully."""

    code = "native_action_failed"
    recoverable = True


class TransportFallbackError(TransportError):
    """All transports (native + mouse/keyboard fallback) failed."""

    code = "transport_fallback_error"
    recoverable = False


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


class CaptureError(NexusError):
    """Raised by the capture subprocess or capture management layer."""

    code = "capture_error"
    recoverable = True


class StabilizationTimeoutError(CaptureError):
    """Screen did not stabilise within the configured timeout."""

    code = "stabilization_timeout"
    recoverable = True


class FrozenScreenError(CaptureError):
    """Consecutive frames are identical — screen appears frozen."""

    code = "frozen_screen"
    recoverable = True


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------


class PerceptionError(NexusError):
    """Raised when perception processing fails."""

    code = "perception_error"
    recoverable = True


class OCRError(PerceptionError):
    """OCR engine returned an error or produced unusable output."""

    code = "ocr_error"
    recoverable = True


class LocatorError(PerceptionError):
    """Target element could not be located on screen."""

    code = "locator_error"
    recoverable = True


class MatcherError(PerceptionError):
    """Template or feature matcher failed to find a match."""

    code = "matcher_error"
    recoverable = True


class ArbitrationError(PerceptionError):
    """Perception sources produced irreconcilable results."""

    code = "arbitration_error"
    recoverable = False


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


class DecisionError(NexusError):
    """Raised when the decision layer cannot produce a valid action."""

    code = "decision_error"
    recoverable = False


class AmbiguityTooHighError(DecisionError):
    """Ambiguity score exceeded the configured threshold; HITL required."""

    code = "ambiguity_too_high"
    recoverable = True


class StuckDetectedError(DecisionError):
    """Agent has repeated the same action without progress."""

    code = "stuck_detected"
    recoverable = False


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class ActionError(NexusError):
    """Raised when action execution fails."""

    code = "action_error"
    recoverable = True


class PreflightFailedError(ActionError):
    """Pre-action precondition check failed; action was not attempted."""

    code = "preflight_failed"
    recoverable = True


class PartialActionError(ActionError):
    """
    A macro-action partially completed before failing.

    Attributes
    ----------
    completed:
        Number of atomic steps that succeeded.
    total:
        Total number of atomic steps in the macro-action.
    """

    code = "partial_action"
    recoverable = False

    def __init__(
        self,
        message: str = "",
        *,
        completed: int,
        total: int,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.completed = completed
        self.total = total


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class VerificationError(NexusError):
    """Raised when post-action verification fails."""

    code = "verification_error"
    recoverable = False


class FalsePositiveError(VerificationError):
    """Verification passed visually but semantic/source check contradicts it."""

    code = "false_positive"
    recoverable = False


class VerificationPolicyError(VerificationError):
    """The configured verification policy could not be applied."""

    code = "verification_policy_error"
    recoverable = False


# ---------------------------------------------------------------------------
# Cloud / LLM
# ---------------------------------------------------------------------------


class CloudError(NexusError):
    """Raised when communication with an LLM provider fails."""

    code = "cloud_error"
    recoverable = True


class CloudUnavailableError(CloudError):
    """Provider endpoint is unreachable or returned a 5xx error."""

    code = "cloud_unavailable"
    recoverable = True


class CloudQuotaExceededError(CloudError):
    """Provider returned a 429 / quota-exceeded response."""

    code = "cloud_quota_exceeded"
    recoverable = True


class CloudTimeoutError(CloudError):
    """Provider did not respond within the configured timeout."""

    code = "cloud_timeout"
    recoverable = True


class InvalidAPIKeyError(CloudError):
    """Provider rejected the API key (401 / 403)."""

    code = "invalid_api_key"
    recoverable = False


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class BudgetExceededError(NexusError):
    """
    Cost cap was reached; agent must pause.

    Attributes
    ----------
    limit_type:
        ``"task"`` or ``"daily"``.
    current:
        Spend accrued so far (USD).
    limit:
        Configured cap (USD).
    """

    code = "budget_exceeded"
    recoverable = False

    def __init__(
        self,
        message: str = "",
        *,
        limit_type: str,
        current: float,
        limit: float,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.limit_type = limit_type
        self.current = current
        self.limit = limit


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class StorageError(NexusError):
    """Raised when database or file-system persistence fails."""

    code = "storage_error"
    recoverable = False


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class PolicyBlockedError(NexusError):
    """
    An action was blocked by a safety or compliance policy.

    Attributes
    ----------
    rule:
        Identifier of the rule that triggered the block.
    severity:
        ``"warn"``, ``"block"``, or ``"abort"``.
    """

    code = "policy_blocked"
    recoverable = False

    def __init__(
        self,
        message: str = "",
        *,
        rule: str,
        severity: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.rule = rule
        self.severity = severity
