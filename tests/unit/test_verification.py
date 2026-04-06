"""
tests/unit/test_verification.py
Unit tests for nexus/verification — policy, result, visual, source.

Coverage
--------
  VerificationPolicy   — defaults, convenience constructors (skip/visual/source)
  VerificationResult   — fields, skipped() factory
  VisualVerifier       — identical frames (no change), changed frames,
                         expected_text hit, expected_text miss,
                         OCR failure (graceful), require_change flag,
                         shape mismatch (conservative 0.0 ratio)
  SourceVerifier       — exact match, contains match, no match,
                         probe returns None (unavailable),
                         retry on failure (success on 2nd attempt),
                         timeout respected, probe exception handled
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from nexus.capture.frame import Frame
from nexus.verification import (
    SourceVerifier,
    VerificationMode,
    VerificationPolicy,
    VerificationResult,
    VisualVerifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame(pixel_value: int = 128, width: int = 4, height: int = 4) -> Frame:
    """Return a minimal Frame filled with a uniform pixel value."""
    data = np.full((height, width, 3), pixel_value, dtype=np.uint8)
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=time.monotonic(),
        captured_at_utc="2026-04-07T00:00:00+00:00",
        sequence_number=1,
    )


def _diff_frame(base: Frame, delta: int = 50) -> Frame:
    """Return a Frame where every pixel differs from *base* by *delta*."""
    new_data = (base.data.astype(np.int32) + delta).clip(0, 255).astype(np.uint8)
    return Frame(
        data=new_data,
        width=base.width,
        height=base.height,
        captured_at_monotonic=time.monotonic(),
        captured_at_utc="2026-04-07T00:00:01+00:00",
        sequence_number=2,
    )


# ---------------------------------------------------------------------------
# VerificationPolicy
# ---------------------------------------------------------------------------


class TestVerificationPolicy:
    def test_defaults(self):
        p = VerificationPolicy()
        assert p.mode == VerificationMode.AUTO
        assert p.timeout_s == 2.0
        assert p.confidence_threshold == 0.80
        assert p.max_retries == 1
        assert p.require_change is False

    def test_skip_factory(self):
        p = VerificationPolicy.skip()
        assert p.mode == VerificationMode.SKIP

    def test_visual_factory(self):
        p = VerificationPolicy.visual(require_change=True)
        assert p.mode == VerificationMode.VISUAL
        assert p.require_change is True

    def test_source_factory(self):
        p = VerificationPolicy.source(confidence_threshold=0.95)
        assert p.mode == VerificationMode.SOURCE
        assert p.confidence_threshold == 0.95

    def test_immutable(self):
        p = VerificationPolicy()
        with pytest.raises((AttributeError, TypeError)):
            p.mode = VerificationMode.SKIP  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------


class TestVerificationResult:
    def test_skipped_factory(self):
        r = VerificationResult.skipped()
        assert r.success is True
        assert r.mode_used == VerificationMode.SKIP
        assert r.confidence == 1.0

    def test_fields(self):
        r = VerificationResult(
            success=False,
            mode_used=VerificationMode.VISUAL,
            confidence=0.3,
            duration_ms=12.5,
            detail="test",
            expected_value="hello",
            observed_value=None,
        )
        assert r.success is False
        assert r.duration_ms == 12.5
        assert r.expected_value == "hello"
        assert r.observed_value is None


# ---------------------------------------------------------------------------
# VisualVerifier
# ---------------------------------------------------------------------------


class TestVisualVerifier:
    def _policy(self, **kw) -> VerificationPolicy:
        return VerificationPolicy(mode=VerificationMode.VISUAL, **kw)

    def test_identical_frames_low_confidence(self):
        verifier = VisualVerifier()
        before = _frame(100)
        after = _frame(100)  # identical
        result = verifier.verify(before, after, self._policy(confidence_threshold=0.5))
        assert result.mode_used == VerificationMode.VISUAL
        assert result.confidence < 0.01  # no change → near-zero confidence

    def test_changed_frames_high_confidence(self):
        verifier = VisualVerifier()
        before = _frame(0)
        after = _diff_frame(before, delta=200)  # large diff
        result = verifier.verify(before, after, self._policy())
        assert result.confidence >= 0.80
        assert result.success is True

    def test_expected_text_hit(self):
        def ocr_fn(img: np.ndarray) -> str:  # noqa: ARG001
            return "Submit button clicked"

        verifier = VisualVerifier(ocr_fn=ocr_fn)
        before = _frame(0)
        after = _diff_frame(before, delta=30)
        result = verifier.verify(
            before, after, self._policy(), expected_text="Submit"
        )
        assert result.confidence >= 0.9
        assert result.success is True
        assert result.observed_value is not None
        assert "Submit" in result.observed_value

    def test_expected_text_miss(self):
        def ocr_fn(img: np.ndarray) -> str:  # noqa: ARG001
            return "Random text without target"

        verifier = VisualVerifier(ocr_fn=ocr_fn)
        before = _frame(0)
        after = _diff_frame(before, delta=10)
        result = verifier.verify(
            before, after, self._policy(confidence_threshold=0.7), expected_text="Submit"
        )
        # OCR miss → low confidence
        assert result.confidence < 0.7
        assert result.success is False

    def test_ocr_exception_handled_gracefully(self):
        def ocr_fn(img: np.ndarray) -> str:
            raise RuntimeError("tesseract unavailable")

        verifier = VisualVerifier(ocr_fn=ocr_fn)
        before = _frame(0)
        after = _diff_frame(before, delta=200)
        # Should not raise; falls back to pixel-diff confidence
        result = verifier.verify(
            before, after, self._policy(), expected_text="anything"
        )
        assert result.mode_used == VerificationMode.VISUAL
        assert result.confidence >= 0.0  # no crash

    def test_require_change_fails_on_identical(self):
        verifier = VisualVerifier()
        before = _frame(50)
        after = _frame(50)  # identical
        policy = VerificationPolicy(
            mode=VerificationMode.VISUAL,
            require_change=True,
            confidence_threshold=0.5,
        )
        result = verifier.verify(before, after, policy)
        assert result.confidence == 0.0
        assert result.success is False

    def test_shape_mismatch_returns_zero_change(self):
        verifier = VisualVerifier()
        before = _frame(0, width=4, height=4)
        # Different size after frame
        after_data = np.full((8, 8, 3), 255, dtype=np.uint8)
        after = Frame(
            data=after_data,
            width=8,
            height=8,
            captured_at_monotonic=time.monotonic(),
            captured_at_utc="2026-04-07T00:00:01+00:00",
            sequence_number=2,
        )
        result = verifier.verify(before, after, self._policy(confidence_threshold=0.5))
        assert result.confidence < 0.01  # shape mismatch → 0 ratio

    def test_no_ocr_fn_with_expected_text(self):
        verifier = VisualVerifier(ocr_fn=None)
        before = _frame(0)
        after = _diff_frame(before, delta=200)
        # Falls back to pixel-diff path
        result = verifier.verify(
            before, after, self._policy(), expected_text="hello"
        )
        assert result.mode_used == VerificationMode.VISUAL
        assert result.confidence >= 0.0


# ---------------------------------------------------------------------------
# SourceVerifier
# ---------------------------------------------------------------------------


class TestSourceVerifier:
    def _policy(self, **kw) -> VerificationPolicy:
        defaults: dict[str, Any] = {
            "mode": VerificationMode.SOURCE,
            "timeout_s": 1.0,
            "confidence_threshold": 0.80,
            "max_retries": 0,
        }
        defaults.update(kw)
        return VerificationPolicy(**defaults)

    def test_exact_match(self):
        probe = lambda ctx: "hello world"  # noqa: E731
        verifier = SourceVerifier(source_probe=probe)
        result = verifier.verify("hello world", self._policy())
        assert result.success is True
        assert result.confidence == 1.0
        assert result.mode_used == VerificationMode.SOURCE
        assert result.observed_value == "hello world"

    def test_contains_match(self):
        probe = lambda ctx: "The value is hello world now"  # noqa: E731
        verifier = SourceVerifier(source_probe=probe)
        result = verifier.verify("hello world", self._policy(confidence_threshold=0.8))
        assert result.success is True
        assert result.confidence == 0.85

    def test_no_match(self):
        probe = lambda ctx: "completely different"  # noqa: E731
        verifier = SourceVerifier(source_probe=probe)
        result = verifier.verify("hello world", self._policy())
        assert result.success is False
        assert result.confidence == 0.0

    def test_probe_returns_none(self):
        probe = lambda ctx: None  # noqa: E731
        verifier = SourceVerifier(source_probe=probe)
        result = verifier.verify("hello", self._policy())
        assert result.success is False
        assert result.confidence == 0.0
        assert "unavailable" in result.detail

    def test_retry_succeeds_on_second_attempt(self):
        call_count = 0

        def probe(ctx: dict[str, Any]) -> str | None:
            nonlocal call_count
            call_count += 1
            return "hello" if call_count >= 2 else "wrong"

        verifier = SourceVerifier(source_probe=probe)
        policy = self._policy(max_retries=2, confidence_threshold=0.9)
        result = verifier.verify("hello", policy)
        assert result.success is True
        assert call_count == 2
        assert result.retries == 1

    def test_probe_exception_handled(self):
        def probe(ctx: dict[str, Any]) -> str | None:
            raise OSError("adapter unavailable")

        verifier = SourceVerifier(source_probe=probe)
        result = verifier.verify("hello", self._policy())
        assert result.success is False
        assert result.confidence == 0.0

    def test_context_forwarded_to_probe(self):
        received: dict[str, Any] = {}

        def probe(ctx: dict[str, Any]) -> str:
            received.update(ctx)
            return "ok"

        verifier = SourceVerifier(source_probe=probe)
        verifier.verify(
            "ok",
            self._policy(),
            context={"element_id": "el-42", "window": "main"},
        )
        assert received["element_id"] == "el-42"
        assert received["window"] == "main"

    def test_case_insensitive_match(self):
        probe = lambda ctx: "HELLO WORLD"  # noqa: E731
        verifier = SourceVerifier(source_probe=probe)
        result = verifier.verify("hello world", self._policy())
        assert result.success is True
        assert result.confidence == 1.0

    def test_whitespace_stripped_in_match(self):
        probe = lambda ctx: "  hello world  "  # noqa: E731
        verifier = SourceVerifier(source_probe=probe)
        result = verifier.verify("hello world", self._policy())
        assert result.success is True

    def test_timeout_exceeded(self):
        """Probe always fails; timeout < retry sleep → exits early."""
        call_count = 0

        def slow_probe(ctx: dict[str, Any]) -> str | None:
            nonlocal call_count
            call_count += 1
            time.sleep(0.15)
            return None

        verifier = SourceVerifier(source_probe=slow_probe)
        # Timeout of 0.1 s means only the first attempt runs.
        policy = VerificationPolicy(
            mode=VerificationMode.SOURCE,
            timeout_s=0.1,
            confidence_threshold=0.8,
            max_retries=5,
        )
        result = verifier.verify("hello", policy)
        assert result.success is False
        # At most 1 probe call before timeout kicks in
        assert call_count <= 2
