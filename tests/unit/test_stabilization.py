"""
tests/unit/test_stabilization.py
Unit tests for nexus/capture/stabilization.py — Faz 19.

Sections:
  1. StabilizationResult value object
  2. _compare_frames — pixel diff metric
  3. _detect_spinner — spinner heuristic
  4. _detect_loading_text — OCR pattern matching
  5. wait_for_stable — integration scenarios
  6. Hypothesis — change_ratio always in [0.0, 1.0]

All tests inject _ocr_fn and _sleep_fn so they run without Tesseract
or actual timing delays.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from itertools import cycle

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nexus.capture.frame import Frame
from nexus.capture.stabilization import (
    PIXEL_CHANGE_THRESHOLD,
    StabilizationGate,
    StabilizationResult,
    _SPINNER_MAX_RATIO,
    _SPINNER_MIN_RATIO,
    _SPINNER_WINDOW,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(
    width: int = 10,
    height: int = 10,
    color: tuple[int, int, int] = (100, 100, 100),
    seq: int = 1,
) -> Frame:
    data = np.full((height, width, 3), color, dtype=np.uint8)
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=time.monotonic(),
        captured_at_utc="2026-04-03T00:00:00+00:00",
        sequence_number=seq,
    )


def _frame_from_array(arr: np.ndarray, seq: int = 1) -> Frame:
    h, w = arr.shape[:2]
    return Frame(
        data=arr.copy(),
        width=w,
        height=h,
        captured_at_monotonic=0.0,
        captured_at_utc="",
        sequence_number=seq,
    )


def _make_gate(
    frames: list[Frame | None],
    *,
    ocr_text: str = "",
) -> StabilizationGate:
    """
    Build a gate backed by a list of frames.

    Frames are yielded by cycling through the list indefinitely, so the
    content pattern (static, changing, spinner) is preserved after the
    list is exhausted rather than collapsing to a constant last frame.
    """
    cyc = cycle(frames)

    def get_frame() -> Frame | None:
        return next(cyc)

    return StabilizationGate(
        _get_frame_fn=get_frame,
        _ocr_fn=lambda f: ocr_text,
        _sleep_fn=lambda s: None,  # no-op sleep → instant test
    )


def _spinner_frames(
    width: int = 10,
    height: int = 10,
    n: int = 8,
) -> list[Frame]:
    """
    Generate *n* frames with a spinner-like change pattern.

    Each frame has a small rotating "dot" in one corner that shifts
    one pixel per frame.  The inter-frame change ratio is tiny (well
    within the spinner band) and consistent.
    """
    frames = []
    for i in range(n):
        data = np.full((height, width, 3), 50, dtype=np.uint8)
        # Move a 1x1 "pixel" in a small 3x3 square — spinner motion
        row = i % 3
        col = i % 3
        data[row, col] = [255, 0, 0]
        frames.append(_frame_from_array(data, seq=i + 1))
    return frames


# ---------------------------------------------------------------------------
# 1. StabilizationResult value object
# ---------------------------------------------------------------------------


class TestStabilizationResult:
    def test_frozen(self) -> None:
        r = StabilizationResult(
            stable=True, waited_ms=50.0, reason="stable", change_ratio_final=0.0
        )
        with pytest.raises((AttributeError, TypeError)):
            r.stable = False  # type: ignore[misc]

    def test_fields(self) -> None:
        r = StabilizationResult(
            stable=False, waited_ms=3000.0, reason="timeout", change_ratio_final=0.25
        )
        assert r.stable is False
        assert r.waited_ms == pytest.approx(3000.0)
        assert r.reason == "timeout"
        assert r.change_ratio_final == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 2. _compare_frames
# ---------------------------------------------------------------------------


class TestCompareFrames:
    def _gate(self) -> StabilizationGate:
        return StabilizationGate(
            _get_frame_fn=lambda: None,
            _ocr_fn=lambda f: "",
            _sleep_fn=lambda s: None,
        )

    def test_identical_frames_return_zero(self) -> None:
        f = _make_frame(color=(128, 64, 32))
        assert self._gate()._compare_frames(f, f) == pytest.approx(0.0)

    def test_same_color_return_zero(self) -> None:
        f1 = _make_frame(color=(200, 200, 200))
        f2 = _make_frame(color=(200, 200, 200))
        assert self._gate()._compare_frames(f1, f2) == pytest.approx(0.0)

    def test_completely_different_colors(self) -> None:
        f1 = _make_frame(color=(0, 0, 0))
        f2 = _make_frame(color=(255, 255, 255))
        ratio = self._gate()._compare_frames(f1, f2)
        assert ratio == pytest.approx(1.0)

    def test_half_pixels_changed(self) -> None:
        """Fill left half with black, right half with white; flip for second frame."""
        w, h = 20, 10
        d1 = np.zeros((h, w, 3), dtype=np.uint8)
        d1[:, w // 2 :] = 255
        d2 = np.zeros((h, w, 3), dtype=np.uint8)
        d2[:, : w // 2] = 255
        f1 = _frame_from_array(d1)
        f2 = _frame_from_array(d2)
        ratio = self._gate()._compare_frames(f1, f2)
        assert ratio == pytest.approx(1.0)  # all pixels changed

    def test_small_noise_below_threshold(self) -> None:
        """Changes within PIXEL_CHANGE_THRESHOLD should not be counted."""
        d1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        d2 = d1.copy()
        # Add noise less than threshold
        d2[0, 0] = [100 + PIXEL_CHANGE_THRESHOLD - 1] * 3
        f1 = _frame_from_array(d1)
        f2 = _frame_from_array(d2)
        ratio = self._gate()._compare_frames(f1, f2)
        assert ratio == pytest.approx(0.0)

    def test_one_pixel_above_threshold(self) -> None:
        d1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        d2 = d1.copy()
        d2[0, 0, 0] = 100 + PIXEL_CHANGE_THRESHOLD + 1
        f1 = _frame_from_array(d1)
        f2 = _frame_from_array(d2)
        ratio = self._gate()._compare_frames(f1, f2)
        # Exactly 1 pixel out of 100 changed
        assert ratio == pytest.approx(1 / 100)

    def test_different_size_returns_one(self) -> None:
        f1 = _make_frame(width=10, height=10)
        f2 = _make_frame(width=20, height=10)
        assert self._gate()._compare_frames(f1, f2) == pytest.approx(1.0)

    def test_result_is_in_0_1_range(self) -> None:
        f1 = _make_frame(color=(0, 0, 0))
        f2 = _make_frame(color=(255, 255, 255))
        r = self._gate()._compare_frames(f1, f2)
        assert 0.0 <= r <= 1.0

    def test_symmetric(self) -> None:
        f1 = _make_frame(color=(10, 20, 30))
        f2 = _make_frame(color=(200, 150, 100))
        gate = self._gate()
        assert gate._compare_frames(f1, f2) == pytest.approx(
            gate._compare_frames(f2, f1)
        )


# ---------------------------------------------------------------------------
# 3. _detect_spinner
# ---------------------------------------------------------------------------


class TestDetectSpinner:
    def _gate(self) -> StabilizationGate:
        return StabilizationGate(
            _get_frame_fn=lambda: None,
            _ocr_fn=lambda f: "",
            _sleep_fn=lambda s: None,
        )

    def test_spinner_detected_in_spinner_frames(self) -> None:
        frames = _spinner_frames(n=_SPINNER_WINDOW + 2)
        gate = self._gate()
        # The spinner frames have consistent small inter-frame changes
        assert gate._detect_spinner(frames) is True

    def test_not_spinner_with_static_frames(self) -> None:
        f = _make_frame(color=(128, 128, 128))
        frames = [f] * (_SPINNER_WINDOW + 2)
        assert self._gate()._detect_spinner(frames) is False

    def test_not_spinner_with_large_changes(self) -> None:
        """Large changes (new page load) must NOT be misidentified as spinner."""
        frames = []
        for i in range(_SPINNER_WINDOW + 2):
            color = (0, 0, 0) if i % 2 == 0 else (255, 255, 255)
            frames.append(_make_frame(color=color, seq=i))
        assert self._gate()._detect_spinner(frames) is False

    def test_requires_minimum_window_frames(self) -> None:
        """Fewer than _SPINNER_WINDOW frames → never detect spinner."""
        frames = _spinner_frames(n=_SPINNER_WINDOW - 1)
        assert self._gate()._detect_spinner(frames) is False

    def test_empty_frames_returns_false(self) -> None:
        assert self._gate()._detect_spinner([]) is False

    def test_single_frame_returns_false(self) -> None:
        assert self._gate()._detect_spinner([_make_frame()]) is False

    def test_spinner_constants_make_sense(self) -> None:
        assert _SPINNER_MIN_RATIO < _SPINNER_MAX_RATIO
        assert _SPINNER_MIN_RATIO > 0
        assert _SPINNER_MAX_RATIO < 1
        assert _SPINNER_WINDOW >= 2


# ---------------------------------------------------------------------------
# 4. _detect_loading_text
# ---------------------------------------------------------------------------


class TestDetectLoadingText:
    def _gate_with_text(self, text: str) -> StabilizationGate:
        return StabilizationGate(
            _get_frame_fn=lambda: None,
            _ocr_fn=lambda f: text,
            _sleep_fn=lambda s: None,
        )

    def test_yükleniyor_detected(self) -> None:
        gate = self._gate_with_text("Lütfen bekleyin — Yükleniyor")
        assert gate._detect_loading_text(_make_frame()) is True

    def test_loading_detected(self) -> None:
        gate = self._gate_with_text("Loading, please wait...")
        assert gate._detect_loading_text(_make_frame()) is True

    def test_please_wait_detected(self) -> None:
        gate = self._gate_with_text("Please Wait")
        assert gate._detect_loading_text(_make_frame()) is True

    def test_lütfen_bekleyin_detected(self) -> None:
        gate = self._gate_with_text("lütfen bekleyin")
        assert gate._detect_loading_text(_make_frame()) is True

    def test_bekleyiniz_detected(self) -> None:
        gate = self._gate_with_text("Bekleyiniz...")
        assert gate._detect_loading_text(_make_frame()) is True

    def test_case_insensitive_loading(self) -> None:
        gate = self._gate_with_text("LOADING")
        assert gate._detect_loading_text(_make_frame()) is True

    def test_case_insensitive_yükleniyor(self) -> None:
        gate = self._gate_with_text("YÜKLENİYOR")
        assert gate._detect_loading_text(_make_frame()) is True

    def test_normal_text_not_detected(self) -> None:
        gate = self._gate_with_text("Welcome to the dashboard")
        assert gate._detect_loading_text(_make_frame()) is False

    def test_empty_text_not_detected(self) -> None:
        gate = self._gate_with_text("")
        assert gate._detect_loading_text(_make_frame()) is False

    def test_partial_match_is_detected(self) -> None:
        """Loading pattern embedded in longer text must still match."""
        gate = self._gate_with_text("System is loading resources...")
        assert gate._detect_loading_text(_make_frame()) is True


# ---------------------------------------------------------------------------
# 5. wait_for_stable — integration scenarios
# ---------------------------------------------------------------------------


class TestWaitForStable:
    def test_identical_frames_stable_true(self) -> None:
        """
        Frames that never change → change_ratio = 0 → stable=True
        immediately after the second frame.
        """
        frame = _make_frame(color=(100, 100, 100))
        gate = _make_gate([frame, frame, frame, frame])
        result = gate.wait_for_stable(timeout_ms=500, poll_ms=1)
        assert result.stable is True
        assert result.reason == "stable"
        assert result.change_ratio_final == pytest.approx(0.0)

    def test_stable_change_ratio_le_threshold(self) -> None:
        """change_ratio must be ≤ change_threshold in stable result."""
        f = _make_frame()
        gate = _make_gate([f, f, f])
        result = gate.wait_for_stable(timeout_ms=500, poll_ms=1, change_threshold=0.02)
        assert result.stable is True
        assert result.change_ratio_final <= 0.02

    def test_continuously_changing_frames_timeout(self) -> None:
        """
        Frames with large random changes never stabilise → timeout.
        """
        rng = np.random.default_rng(42)
        frames = [
            _frame_from_array(rng.integers(0, 256, (10, 10, 3), dtype=np.uint8), seq=i)
            for i in range(20)
        ]
        gate = _make_gate(frames)
        result = gate.wait_for_stable(timeout_ms=5, poll_ms=1)
        assert result.stable is False
        assert result.reason == "timeout"

    def test_timeout_result_has_waited_ms_positive(self) -> None:
        frames = [_make_frame(color=(i * 10 % 255, 0, 0)) for i in range(20)]
        gate = _make_gate(frames)
        result = gate.wait_for_stable(timeout_ms=5, poll_ms=1)
        assert result.waited_ms >= 0.0

    def test_spinner_returns_stable_false(self) -> None:
        """
        Spinner-pattern frames → stable=False with reason="spinner_detected".
        """
        frames = _spinner_frames(n=_SPINNER_WINDOW * 3)
        gate = _make_gate(frames)
        result = gate.wait_for_stable(
            timeout_ms=5000, poll_ms=1, change_threshold=0.5
        )
        assert result.stable is False
        assert result.reason == "spinner_detected"

    def test_loading_text_prevents_stable(self) -> None:
        """
        When OCR finds loading text, the frame is not considered stable
        even if change_ratio is below threshold.
        Outcome is stable=False (timeout).
        """
        frame = _make_frame(color=(200, 200, 200))
        gate = StabilizationGate(
            _get_frame_fn=lambda: frame,
            _ocr_fn=lambda f: "Loading...",
            _sleep_fn=lambda s: None,
        )
        result = gate.wait_for_stable(timeout_ms=5, poll_ms=1)
        assert result.stable is False

    def test_loading_text_reason_on_timeout(self) -> None:
        """
        When timeout is reached while loading text is present, reason
        must be 'timeout_loading_text'.
        """
        frame = _make_frame()
        gate = StabilizationGate(
            _get_frame_fn=lambda: frame,
            _ocr_fn=lambda f: "Please wait",
            _sleep_fn=lambda s: None,
        )
        result = gate.wait_for_stable(timeout_ms=5, poll_ms=1)
        assert result.reason == "timeout_loading_text"

    def test_stable_waited_ms_is_nonnegative(self) -> None:
        f = _make_frame()
        gate = _make_gate([f, f, f])
        result = gate.wait_for_stable(timeout_ms=500, poll_ms=1)
        assert result.waited_ms >= 0.0

    def test_none_frames_are_skipped(self) -> None:
        """
        None frames (no frame available yet) must be skipped.
        After a None a real frame causes stabilisation.
        """
        f = _make_frame()
        gate = _make_gate([None, None, f, f, f])  # type: ignore[list-item]
        result = gate.wait_for_stable(timeout_ms=500, poll_ms=1)
        assert result.stable is True

    def test_change_threshold_respected(self) -> None:
        """
        With change_threshold=1.0 even frames that are totally different
        satisfy stability.
        """
        f1 = _make_frame(color=(0, 0, 0))
        f2 = _make_frame(color=(255, 255, 255))
        gate = _make_gate([f1, f2, f1, f2])
        result = gate.wait_for_stable(
            timeout_ms=500, poll_ms=1, change_threshold=1.0
        )
        assert result.stable is True

    def test_wait_for_stable_returns_stabilization_result(self) -> None:
        f = _make_frame()
        gate = _make_gate([f, f])
        result = gate.wait_for_stable(timeout_ms=100, poll_ms=1)
        assert isinstance(result, StabilizationResult)

    def test_first_frame_does_not_immediately_stabilize(self) -> None:
        """
        The gate needs at least two frames to make a comparison.
        A single frame followed by timeout must give stable=False.
        """
        f = _make_frame()
        # Only one unique frame — iterator exhausted after first yield
        call_count = 0

        def get_once() -> Frame | None:
            nonlocal call_count
            call_count += 1
            return f if call_count == 1 else None  # second call → None

        gate = StabilizationGate(
            _get_frame_fn=get_once,
            _ocr_fn=lambda f: "",
            _sleep_fn=lambda s: None,
        )
        result = gate.wait_for_stable(timeout_ms=2, poll_ms=1)
        # Can't compare → timeout
        assert result.stable is False


# ---------------------------------------------------------------------------
# 6. Hypothesis — change_ratio always in [0.0, 1.0]
# ---------------------------------------------------------------------------


class TestHypothesisChangeRatio:
    """
    Property-based tests that verify _compare_frames returns a value in
    [0.0, 1.0] for any pair of same-shaped uint8 pixel arrays.
    """

    _gate = StabilizationGate(
        _get_frame_fn=lambda: None,
        _ocr_fn=lambda f: "",
        _sleep_fn=lambda s: None,
    )

    @given(
        arrays(dtype=np.dtype("uint8"), shape=(8, 8, 3)),
        arrays(dtype=np.dtype("uint8"), shape=(8, 8, 3)),
    )
    @settings(max_examples=200)
    def test_change_ratio_between_0_and_1(
        self, arr1: np.ndarray, arr2: np.ndarray
    ) -> None:
        """For any two same-shape uint8 arrays, result must be in [0.0, 1.0]."""
        f1 = _frame_from_array(arr1)
        f2 = _frame_from_array(arr2)
        ratio = self._gate._compare_frames(f1, f2)
        assert 0.0 <= ratio <= 1.0, f"Got {ratio!r} outside [0, 1]"

    @given(
        arrays(dtype=np.dtype("uint8"), shape=st.tuples(
            st.integers(1, 16), st.integers(1, 16), st.just(3)
        )),
    )
    @settings(max_examples=100)
    def test_compare_with_self_is_zero(self, arr: np.ndarray) -> None:
        """A frame compared with itself must always return 0.0."""
        f = _frame_from_array(arr)
        ratio = self._gate._compare_frames(f, f)
        assert ratio == pytest.approx(0.0), f"Self-compare gave {ratio!r}"

    @given(
        st.integers(1, 20),
        st.integers(1, 20),
    )
    @settings(max_examples=50)
    def test_different_sizes_return_one(self, w1: int, w2: int) -> None:
        """Frames with different widths must always return 1.0."""
        if w1 == w2:
            w2 = w2 + 1  # force different width
        f1 = _make_frame(width=w1, height=5)
        f2 = _make_frame(width=w2, height=5)
        ratio = self._gate._compare_frames(f1, f2)
        assert ratio == pytest.approx(1.0)
