"""Unit tests for nexus/core/errors.py — hierarchy, flags, and attributes."""
from __future__ import annotations

import pytest

from nexus.core.errors import (
    ActionError,
    AmbiguityTooHighError,
    ArbitrationError,
    BudgetExceededError,
    CaptureError,
    CloudError,
    CloudQuotaExceededError,
    CloudTimeoutError,
    CloudUnavailableError,
    DecisionError,
    DOMError,
    FileAdapterError,
    FalsePositiveError,
    FrozenScreenError,
    InvalidAPIKeyError,
    LocatorError,
    MatcherError,
    NativeActionFailedError,
    NexusError,
    OCRError,
    PartialActionError,
    PerceptionError,
    PolicyBlockedError,
    PreflightFailedError,
    SourceError,
    StabilizationTimeoutError,
    StorageError,
    StuckDetectedError,
    TransportError,
    TransportFallbackError,
    UIAError,
    VerificationError,
    VerificationPolicyError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def raises_as(exc_class: type[BaseException], parent: type[BaseException]) -> None:
    """Assert exc_class is a subclass of parent."""
    assert issubclass(exc_class, parent), (
        f"{exc_class.__name__} should be a subclass of {parent.__name__}"
    )


# ---------------------------------------------------------------------------
# NexusError base
# ---------------------------------------------------------------------------


class TestNexusError:
    def test_is_exception(self) -> None:
        raises_as(NexusError, Exception)

    def test_default_message(self) -> None:
        e = NexusError()
        assert str(e) == ""

    def test_custom_message(self) -> None:
        e = NexusError("something broke")
        assert str(e) == "something broke"

    def test_context_default_empty(self) -> None:
        assert NexusError().context == {}

    def test_context_stored(self) -> None:
        e = NexusError(context={"key": "val"})
        assert e.context["key"] == "val"

    def test_code(self) -> None:
        assert NexusError.code == "nexus_error"

    def test_recoverable_false(self) -> None:
        assert NexusError.recoverable is False


# ---------------------------------------------------------------------------
# Source layer hierarchy
# ---------------------------------------------------------------------------


class TestSourceHierarchy:
    def test_source_error_parent(self) -> None:
        raises_as(SourceError, NexusError)

    def test_uia_parent(self) -> None:
        raises_as(UIAError, SourceError)

    def test_dom_parent(self) -> None:
        raises_as(DOMError, SourceError)

    def test_file_adapter_parent(self) -> None:
        raises_as(FileAdapterError, SourceError)

    def test_source_recoverable(self) -> None:
        assert SourceError.recoverable is True

    def test_uia_recoverable(self) -> None:
        assert UIAError.recoverable is True

    def test_dom_recoverable(self) -> None:
        assert DOMError.recoverable is True

    def test_file_adapter_not_recoverable(self) -> None:
        assert FileAdapterError.recoverable is False

    def test_unique_codes(self) -> None:
        codes = {UIAError.code, DOMError.code, FileAdapterError.code}
        assert len(codes) == 3


# ---------------------------------------------------------------------------
# Transport layer hierarchy
# ---------------------------------------------------------------------------


class TestTransportHierarchy:
    def test_transport_error_parent(self) -> None:
        raises_as(TransportError, NexusError)

    def test_native_action_failed_parent(self) -> None:
        raises_as(NativeActionFailedError, TransportError)

    def test_transport_fallback_parent(self) -> None:
        raises_as(TransportFallbackError, TransportError)

    def test_transport_recoverable(self) -> None:
        assert TransportError.recoverable is True

    def test_native_action_failed_recoverable(self) -> None:
        assert NativeActionFailedError.recoverable is True

    def test_transport_fallback_not_recoverable(self) -> None:
        assert TransportFallbackError.recoverable is False

    def test_unique_codes(self) -> None:
        codes = {
            TransportError.code,
            NativeActionFailedError.code,
            TransportFallbackError.code,
        }
        assert len(codes) == 3


# ---------------------------------------------------------------------------
# Capture layer hierarchy
# ---------------------------------------------------------------------------


class TestCaptureHierarchy:
    def test_capture_error_parent(self) -> None:
        raises_as(CaptureError, NexusError)

    def test_stabilization_timeout_parent(self) -> None:
        raises_as(StabilizationTimeoutError, CaptureError)

    def test_frozen_screen_parent(self) -> None:
        raises_as(FrozenScreenError, CaptureError)

    def test_capture_recoverable(self) -> None:
        assert CaptureError.recoverable is True

    def test_stabilization_recoverable(self) -> None:
        assert StabilizationTimeoutError.recoverable is True

    def test_frozen_screen_recoverable(self) -> None:
        assert FrozenScreenError.recoverable is True


# ---------------------------------------------------------------------------
# Perception layer hierarchy
# ---------------------------------------------------------------------------


class TestPerceptionHierarchy:
    def test_perception_error_parent(self) -> None:
        raises_as(PerceptionError, NexusError)

    def test_ocr_parent(self) -> None:
        raises_as(OCRError, PerceptionError)

    def test_locator_parent(self) -> None:
        raises_as(LocatorError, PerceptionError)

    def test_matcher_parent(self) -> None:
        raises_as(MatcherError, PerceptionError)

    def test_arbitration_parent(self) -> None:
        raises_as(ArbitrationError, PerceptionError)

    def test_perception_recoverable(self) -> None:
        assert PerceptionError.recoverable is True

    def test_ocr_recoverable(self) -> None:
        assert OCRError.recoverable is True

    def test_locator_recoverable(self) -> None:
        assert LocatorError.recoverable is True

    def test_matcher_recoverable(self) -> None:
        assert MatcherError.recoverable is True

    def test_arbitration_not_recoverable(self) -> None:
        assert ArbitrationError.recoverable is False


# ---------------------------------------------------------------------------
# Decision layer hierarchy
# ---------------------------------------------------------------------------


class TestDecisionHierarchy:
    def test_decision_error_parent(self) -> None:
        raises_as(DecisionError, NexusError)

    def test_ambiguity_parent(self) -> None:
        raises_as(AmbiguityTooHighError, DecisionError)

    def test_stuck_parent(self) -> None:
        raises_as(StuckDetectedError, DecisionError)

    def test_decision_not_recoverable(self) -> None:
        assert DecisionError.recoverable is False

    def test_ambiguity_recoverable(self) -> None:
        assert AmbiguityTooHighError.recoverable is True

    def test_stuck_not_recoverable(self) -> None:
        assert StuckDetectedError.recoverable is False


# ---------------------------------------------------------------------------
# Action layer hierarchy
# ---------------------------------------------------------------------------


class TestActionHierarchy:
    def test_action_error_parent(self) -> None:
        raises_as(ActionError, NexusError)

    def test_preflight_parent(self) -> None:
        raises_as(PreflightFailedError, ActionError)

    def test_partial_action_parent(self) -> None:
        raises_as(PartialActionError, ActionError)

    def test_action_recoverable(self) -> None:
        assert ActionError.recoverable is True

    def test_preflight_recoverable(self) -> None:
        assert PreflightFailedError.recoverable is True

    def test_partial_action_not_recoverable(self) -> None:
        assert PartialActionError.recoverable is False

    def test_partial_action_attributes(self) -> None:
        e = PartialActionError("failed mid-way", completed=3, total=7)
        assert e.completed == 3
        assert e.total == 7

    def test_partial_action_context(self) -> None:
        e = PartialActionError("", completed=1, total=5, context={"step": "click"})
        assert e.context["step"] == "click"

    def test_partial_action_is_nexus_error(self) -> None:
        raises_as(PartialActionError, NexusError)


# ---------------------------------------------------------------------------
# Verification layer hierarchy
# ---------------------------------------------------------------------------


class TestVerificationHierarchy:
    def test_verification_error_parent(self) -> None:
        raises_as(VerificationError, NexusError)

    def test_false_positive_parent(self) -> None:
        raises_as(FalsePositiveError, VerificationError)

    def test_policy_error_parent(self) -> None:
        raises_as(VerificationPolicyError, VerificationError)

    def test_verification_not_recoverable(self) -> None:
        assert VerificationError.recoverable is False

    def test_false_positive_not_recoverable(self) -> None:
        assert FalsePositiveError.recoverable is False

    def test_policy_error_not_recoverable(self) -> None:
        assert VerificationPolicyError.recoverable is False


# ---------------------------------------------------------------------------
# Cloud / LLM layer hierarchy
# ---------------------------------------------------------------------------


class TestCloudHierarchy:
    def test_cloud_error_parent(self) -> None:
        raises_as(CloudError, NexusError)

    def test_unavailable_parent(self) -> None:
        raises_as(CloudUnavailableError, CloudError)

    def test_quota_parent(self) -> None:
        raises_as(CloudQuotaExceededError, CloudError)

    def test_timeout_parent(self) -> None:
        raises_as(CloudTimeoutError, CloudError)

    def test_invalid_key_parent(self) -> None:
        raises_as(InvalidAPIKeyError, CloudError)

    def test_cloud_recoverable(self) -> None:
        assert CloudError.recoverable is True

    def test_unavailable_recoverable(self) -> None:
        assert CloudUnavailableError.recoverable is True

    def test_quota_recoverable(self) -> None:
        assert CloudQuotaExceededError.recoverable is True

    def test_timeout_recoverable(self) -> None:
        assert CloudTimeoutError.recoverable is True

    def test_invalid_key_not_recoverable(self) -> None:
        assert InvalidAPIKeyError.recoverable is False


# ---------------------------------------------------------------------------
# BudgetExceededError
# ---------------------------------------------------------------------------


class TestBudgetExceededError:
    def test_parent(self) -> None:
        raises_as(BudgetExceededError, NexusError)

    def test_not_recoverable(self) -> None:
        assert BudgetExceededError.recoverable is False

    def test_attributes(self) -> None:
        e = BudgetExceededError("over limit", limit_type="daily", current=1.5, limit=1.0)
        assert e.limit_type == "daily"
        assert e.current == pytest.approx(1.5)
        assert e.limit == pytest.approx(1.0)

    def test_context_stored(self) -> None:
        e = BudgetExceededError(
            limit_type="task", current=0.5, limit=0.4, context={"task_id": "t1"}
        )
        assert e.context["task_id"] == "t1"

    def test_code(self) -> None:
        assert BudgetExceededError.code == "budget_exceeded"


# ---------------------------------------------------------------------------
# StorageError
# ---------------------------------------------------------------------------


class TestStorageError:
    def test_parent(self) -> None:
        raises_as(StorageError, NexusError)

    def test_not_recoverable(self) -> None:
        assert StorageError.recoverable is False

    def test_code(self) -> None:
        assert StorageError.code == "storage_error"


# ---------------------------------------------------------------------------
# PolicyBlockedError
# ---------------------------------------------------------------------------


class TestPolicyBlockedError:
    def test_parent(self) -> None:
        raises_as(PolicyBlockedError, NexusError)

    def test_not_recoverable(self) -> None:
        assert PolicyBlockedError.recoverable is False

    def test_attributes(self) -> None:
        e = PolicyBlockedError("blocked", rule="no_pii", severity="block")
        assert e.rule == "no_pii"
        assert e.severity == "block"

    def test_context_stored(self) -> None:
        e = PolicyBlockedError(rule="r", severity="warn", context={"target": "field_x"})
        assert e.context["target"] == "field_x"

    def test_code(self) -> None:
        assert PolicyBlockedError.code == "policy_blocked"


# ---------------------------------------------------------------------------
# Cross-cutting: all NexusError subclasses have unique codes
# ---------------------------------------------------------------------------


class TestUniqueCodes:
    ALL_CLASSES = [
        NexusError,
        SourceError, UIAError, DOMError, FileAdapterError,
        TransportError, NativeActionFailedError, TransportFallbackError,
        CaptureError, StabilizationTimeoutError, FrozenScreenError,
        PerceptionError, OCRError, LocatorError, MatcherError, ArbitrationError,
        DecisionError, AmbiguityTooHighError, StuckDetectedError,
        ActionError, PreflightFailedError, PartialActionError,
        VerificationError, FalsePositiveError, VerificationPolicyError,
        CloudError, CloudUnavailableError, CloudQuotaExceededError,
        CloudTimeoutError, InvalidAPIKeyError,
        BudgetExceededError,
        StorageError,
        PolicyBlockedError,
    ]

    def test_all_are_nexus_errors(self) -> None:
        for cls in self.ALL_CLASSES:
            assert issubclass(cls, NexusError), f"{cls.__name__} not a NexusError"

    def test_all_codes_unique(self) -> None:
        codes = [cls.code for cls in self.ALL_CLASSES]
        assert len(codes) == len(set(codes)), (
            f"Duplicate codes: {[c for c in codes if codes.count(c) > 1]}"
        )

    def test_all_have_code_attribute(self) -> None:
        for cls in self.ALL_CLASSES:
            assert isinstance(cls.code, str) and cls.code, (
                f"{cls.__name__}.code is missing or empty"
            )

    def test_all_have_recoverable_attribute(self) -> None:
        for cls in self.ALL_CLASSES:
            assert isinstance(cls.recoverable, bool), (
                f"{cls.__name__}.recoverable must be bool"
            )


# ---------------------------------------------------------------------------
# Raise-and-catch smoke tests
# ---------------------------------------------------------------------------


class TestRaiseAndCatch:
    def test_catch_as_nexus_error(self) -> None:
        with pytest.raises(NexusError):
            raise UIAError("uia failed")

    def test_catch_as_source_error(self) -> None:
        with pytest.raises(SourceError):
            raise DOMError("dom failed")

    def test_catch_as_transport_error(self) -> None:
        with pytest.raises(TransportError):
            raise NativeActionFailedError("native failed")

    def test_catch_specific(self) -> None:
        with pytest.raises(PartialActionError) as exc_info:
            raise PartialActionError("mid-fail", completed=2, total=5)
        assert exc_info.value.completed == 2

    def test_catch_budget_exceeded(self) -> None:
        with pytest.raises(BudgetExceededError) as exc_info:
            raise BudgetExceededError(limit_type="task", current=2.0, limit=1.0)
        assert exc_info.value.limit_type == "task"

    def test_catch_policy_blocked(self) -> None:
        with pytest.raises(PolicyBlockedError) as exc_info:
            raise PolicyBlockedError(rule="gdpr", severity="abort")
        assert exc_info.value.severity == "abort"
