"""
tests/unit/test_temporal_expert.py
Unit tests for nexus/perception/temporal/temporal_expert.py — Faz 27.

Sections:
  1.  ScreenState value object
  2.  StateType enum completeness
  3.  _change_ratio helper
  4.  _elapsed_s helper
  5.  _has_loading_text helper
  6.  _spinner_detected helper
  7.  TemporalExpert.analyze — insufficient frames → UNKNOWN
  8.  TemporalExpert.analyze — STABLE detection
  9.  TemporalExpert.analyze — FROZEN detection
  10. TemporalExpert.analyze — LOADING via text
  11. TemporalExpert.analyze — LOADING via spinner
  12. TemporalExpert.analyze — ANIMATING detection
  13. TemporalExpert.analyze — TRANSITIONING detection
  14. TemporalExpert.analyze — blocking decisions
  15. TemporalExpert.analyze — retry_after_ms values
  16. TemporalExpert.analyze — OCR failure non-fatal
  17. Hypothesis — arbitrary frame history never raises
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nexus.capture.frame import Frame
from nexus.perception.temporal.temporal_expert import (
    ScreenState,
    StateType,
    TemporalExpert,
    _change_ratio,
    _elapsed_s,
    _has_loading_text,
    _spinner_detected,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = "2026-04-05T00:00:00+00:00"


def _frame(
    color: tuple[int, int, int] = (128, 128, 128),
    t: float = 0.0,
    seq: int = 1,
    w: int = 64,
    h: int = 48,
) -> Frame:
    data = np.full((h, w, 3), color, dtype=np.uint8)
    return Frame(
        data=data,
        width=w,
        height=h,
        captured_at_monotonic=t,
        captured_at_utc=_UTC,
        sequence_number=seq,
    )


def _stable_history(n: int = 5, w: int = 64, h: int = 48) -> list[Frame]:
    """N identical frames spaced 0.1 s apart — should classify as STABLE."""
    return [_frame(color=(100, 100, 100), t=i * 0.1, seq=i + 1, w=w, h=h) for i in range(n)]


def _animating_history(n: int = 5) -> list[Frame]:
    """Frames with high change ratio — should classify as ANIMATING or TRANSITIONING."""
    frames = []
    for i in range(n):
        # Alternate between two very different colors to create high change
        color = (200, 200, 200) if i % 2 == 0 else (10, 10, 10)
        frames.append(_frame(color=color, t=i * 0.1, seq=i + 1))
    return frames


def _moderate_change_history(n: int = 5, change_frac: float = 0.15) -> list[Frame]:
    """
    Frames where each consecutive pair has ~change_frac of pixels altered.
    Produces change ratios in the ANIMATING band (> anim_threshold, << trans_threshold).
    """
    w, h = 64, 48
    n_changed = max(1, int(change_frac * w * h))
    frames = []
    for i in range(n):
        data = np.full((h, w, 3), 100, dtype=np.uint8)
        # Cycle which columns are different to create consistent moderate change
        start_col = (i * (w // n)) % w
        end_col = min(w, start_col + n_changed // h + 1)
        data[:, start_col:end_col, 0] = 200 if i % 2 == 0 else 50
        frames.append(Frame(
            data=data, width=w, height=h,
            captured_at_monotonic=i * 0.1, captured_at_utc=_UTC, sequence_number=i + 1,
        ))
    return frames


def _spinner_history() -> list[Frame]:
    """
    Four inter-frame changes all in [SPINNER_MIN, SPINNER_MAX] →
    5 frames needed (4 gaps).
    """
    from nexus.perception.temporal.temporal_expert import (
        _SPINNER_MAX_RATIO,
        _SPINNER_MIN_RATIO,
        _SPINNER_WINDOW,
    )
    n = _SPINNER_WINDOW + 1  # 5 frames
    frames = []
    w, h = 64, 48
    total = w * h
    # Each consecutive pair has exactly ~0.01 of pixels changed (spinner-band)
    target_ratio = (_SPINNER_MIN_RATIO + _SPINNER_MAX_RATIO) / 2  # ≈ 0.025
    n_changed = max(1, int(target_ratio * total))
    base = np.full((h, w, 3), 128, dtype=np.uint8)

    for i in range(n):
        data = base.copy()
        # Cycle: change a different row to create small localized difference
        row = i % h
        data[row, :n_changed, 0] = 200 if i % 2 == 0 else 50  # > threshold=10
        frames.append(Frame(
            data=data,
            width=w,
            height=h,
            captured_at_monotonic=float(i) * 0.1,
            captured_at_utc=_UTC,
            sequence_number=i + 1,
        ))
    return frames


def _loading_history() -> list[Frame]:
    """Identical frames; OCR returns loading text."""
    return _stable_history(4)


def _frozen_history(elapsed_s: float = 6.0) -> list[Frame]:
    """Identical frames spanning elapsed_s seconds — should be FROZEN."""
    n = 5
    dt = elapsed_s / (n - 1)
    return [_frame(color=(80, 80, 80), t=i * dt, seq=i + 1) for i in range(n)]


def _no_ocr(frame: Frame) -> str:
    return ""


# ---------------------------------------------------------------------------
# Section 1 — ScreenState value object
# ---------------------------------------------------------------------------


class TestScreenState:
    def test_frozen(self):
        ss = ScreenState(
            state_type=StateType.STABLE,
            confidence=0.9,
            blocks_perception=False,
            reason="stable",
            retry_after_ms=0,
        )
        with pytest.raises((AttributeError, TypeError)):
            ss.confidence = 0.5  # type: ignore[misc]

    def test_fields_accessible(self):
        ss = ScreenState(
            state_type=StateType.LOADING,
            confidence=0.85,
            blocks_perception=True,
            reason="loading_text",
            retry_after_ms=500,
        )
        assert ss.state_type is StateType.LOADING
        assert ss.blocks_perception is True
        assert ss.retry_after_ms == 500


# ---------------------------------------------------------------------------
# Section 2 — StateType enum completeness
# ---------------------------------------------------------------------------


class TestStateTypeEnum:
    _REQUIRED = {
        "STABLE", "LOADING", "ANIMATING", "TRANSITIONING", "FROZEN", "UNKNOWN",
    }

    def test_all_required_members(self):
        assert {m.name for m in StateType} >= self._REQUIRED


# ---------------------------------------------------------------------------
# Section 3 — _change_ratio helper
# ---------------------------------------------------------------------------


class TestChangeRatio:
    def test_identical_frames_zero(self):
        f = _frame((100, 100, 100))
        assert _change_ratio(f, f) == pytest.approx(0.0)

    def test_fully_different_frames(self):
        a = _frame((0, 0, 0))
        b = _frame((255, 255, 255))
        assert _change_ratio(a, b) == pytest.approx(1.0)

    def test_half_changed(self):
        w, h = 64, 48
        a_data = np.full((h, w, 3), 0, dtype=np.uint8)
        b_data = np.full((h, w, 3), 0, dtype=np.uint8)
        # Change left half significantly
        b_data[:, : w // 2, :] = 200
        a = Frame(data=a_data, width=w, height=h, captured_at_monotonic=0.0,
                  captured_at_utc=_UTC, sequence_number=1)
        b = Frame(data=b_data, width=w, height=h, captured_at_monotonic=0.1,
                  captured_at_utc=_UTC, sequence_number=2)
        ratio = _change_ratio(a, b)
        assert ratio == pytest.approx(0.5, abs=0.01)

    def test_different_shape_returns_one(self):
        a = _frame(w=10, h=10)
        b = _frame(w=20, h=20)
        assert _change_ratio(a, b) == pytest.approx(1.0)

    def test_result_in_range(self):
        a = _frame((10, 20, 30))
        b = _frame((50, 60, 70))
        r = _change_ratio(a, b)
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# Section 4 — _elapsed_s helper
# ---------------------------------------------------------------------------


class TestElapsedS:
    def test_two_frames(self):
        frames = [_frame(t=0.0), _frame(t=3.5)]
        assert _elapsed_s(frames) == pytest.approx(3.5)

    def test_single_frame_zero(self):
        assert _elapsed_s([_frame(t=1.0)]) == pytest.approx(0.0)

    def test_empty_zero(self):
        assert _elapsed_s([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Section 5 — _has_loading_text helper
# ---------------------------------------------------------------------------


class TestHasLoadingText:
    @pytest.mark.parametrize("text", [
        "yükleniyor",
        "loading",
        "please wait",
        "lütfen bekleyin",
        "bekleyiniz",
        "app is loading now",
        "yükleniyor lütfen bekleyin",
    ])
    def test_loading_patterns_detected(self, text: str):
        assert _has_loading_text(text) is True

    @pytest.mark.parametrize("text", [
        "kaydet",
        "tamam",
        "submit",
        "",
        "dosya açıldı",
    ])
    def test_non_loading_text(self, text: str):
        assert _has_loading_text(text) is False


# ---------------------------------------------------------------------------
# Section 6 — _spinner_detected helper
# ---------------------------------------------------------------------------


class TestSpinnerDetected:
    def test_spinner_band_all_in_range(self):
        from nexus.perception.temporal.temporal_expert import (
            _SPINNER_MAX_RATIO,
            _SPINNER_MIN_RATIO,
            _SPINNER_WINDOW,
        )
        mid = (_SPINNER_MIN_RATIO + _SPINNER_MAX_RATIO) / 2
        ratios = [mid] * _SPINNER_WINDOW
        assert _spinner_detected(ratios) is True

    def test_too_few_ratios_false(self):
        from nexus.perception.temporal.temporal_expert import _SPINNER_WINDOW
        ratios = [0.01] * (_SPINNER_WINDOW - 1)
        assert _spinner_detected(ratios) is False

    def test_one_ratio_out_of_band_false(self):
        from nexus.perception.temporal.temporal_expert import (
            _SPINNER_MAX_RATIO,
            _SPINNER_MIN_RATIO,
            _SPINNER_WINDOW,
        )
        mid = (_SPINNER_MIN_RATIO + _SPINNER_MAX_RATIO) / 2
        ratios = [mid] * (_SPINNER_WINDOW - 1) + [1.0]  # last one too high
        assert _spinner_detected(ratios) is False

    def test_zero_ratios_false(self):
        ratios = [0.0] * 10
        assert _spinner_detected(ratios) is False


# ---------------------------------------------------------------------------
# Section 7 — insufficient frames → UNKNOWN
# ---------------------------------------------------------------------------


class TestInsufficientFrames:
    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_fewer_than_3_frames_unknown(self, n: int):
        expert = TemporalExpert(_ocr_fn=_no_ocr)
        frames = _stable_history(n)
        result = expert.analyze(frames)
        assert result.state_type is StateType.UNKNOWN

    def test_empty_list_unknown(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr)
        result = expert.analyze([])
        assert result.state_type is StateType.UNKNOWN


# ---------------------------------------------------------------------------
# Section 8 — STABLE detection
# ---------------------------------------------------------------------------


class TestStableDetection:
    def test_identical_frames_stable(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr)
        frames = _stable_history(5)
        result = expert.analyze(frames)
        assert result.state_type is StateType.STABLE

    def test_stable_does_not_block(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr)
        result = expert.analyze(_stable_history(5))
        assert result.blocks_perception is False

    def test_stable_retry_zero(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr)
        result = expert.analyze(_stable_history(5))
        assert result.retry_after_ms == 0

    def test_stable_confidence_high(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr)
        result = expert.analyze(_stable_history(5))
        assert result.confidence >= 0.80


# ---------------------------------------------------------------------------
# Section 9 — FROZEN detection
# ---------------------------------------------------------------------------


class TestFrozenDetection:
    def test_long_identical_history_frozen(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr, frozen_threshold_s=5.0)
        frames = _frozen_history(elapsed_s=6.0)
        result = expert.analyze(frames)
        assert result.state_type is StateType.FROZEN

    def test_frozen_blocks_perception(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr, frozen_threshold_s=5.0)
        result = expert.analyze(_frozen_history(6.0))
        assert result.blocks_perception is True

    def test_frozen_retry_nonzero(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr, frozen_threshold_s=5.0)
        result = expert.analyze(_frozen_history(6.0))
        assert result.retry_after_ms > 0

    def test_short_identical_not_frozen(self):
        """Same pixels but only 1s elapsed — not FROZEN."""
        expert = TemporalExpert(_ocr_fn=_no_ocr, frozen_threshold_s=5.0)
        frames = _frozen_history(elapsed_s=1.0)
        result = expert.analyze(frames)
        assert result.state_type is not StateType.FROZEN

    def test_custom_frozen_threshold(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr, frozen_threshold_s=2.0)
        frames = _frozen_history(elapsed_s=3.0)
        result = expert.analyze(frames)
        assert result.state_type is StateType.FROZEN


# ---------------------------------------------------------------------------
# Section 10 — LOADING via text
# ---------------------------------------------------------------------------


class TestLoadingViaText:
    @pytest.mark.parametrize("loading_text", [
        "yükleniyor",
        "loading",
        "please wait",
        "lütfen bekleyin",
        "bekleyiniz",
    ])
    def test_loading_text_detected(self, loading_text: str):
        expert = TemporalExpert(_ocr_fn=lambda f: loading_text)
        frames = _stable_history(4)
        result = expert.analyze(frames)
        assert result.state_type is StateType.LOADING

    def test_loading_blocks_perception(self):
        expert = TemporalExpert(_ocr_fn=lambda f: "loading")
        result = expert.analyze(_stable_history(4))
        assert result.blocks_perception is True

    def test_loading_retry_500ms(self):
        expert = TemporalExpert(_ocr_fn=lambda f: "loading")
        result = expert.analyze(_stable_history(4))
        assert result.retry_after_ms == 500

    def test_loading_reason_loading_text(self):
        expert = TemporalExpert(_ocr_fn=lambda f: "yükleniyor")
        result = expert.analyze(_stable_history(4))
        assert result.reason == "loading_text"


# ---------------------------------------------------------------------------
# Section 11 — LOADING via spinner
# ---------------------------------------------------------------------------


class TestLoadingViaSpinner:
    def test_spinner_pattern_loading(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr)
        frames = _spinner_history()
        result = expert.analyze(frames)
        assert result.state_type is StateType.LOADING

    def test_spinner_reason(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr)
        result = expert.analyze(_spinner_history())
        assert result.reason == "spinner"

    def test_spinner_blocks_perception(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr)
        result = expert.analyze(_spinner_history())
        assert result.blocks_perception is True


# ---------------------------------------------------------------------------
# Section 12 — ANIMATING detection
# ---------------------------------------------------------------------------


class TestAnimatingDetection:
    def test_high_change_animating_or_transitioning(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr)
        frames = _animating_history(5)
        result = expert.analyze(frames)
        assert result.state_type in (StateType.ANIMATING, StateType.TRANSITIONING)

    def test_animating_blocks(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr)
        # Force ANIMATING by using moderate change (above anim_threshold but below trans)
        expert2 = TemporalExpert(
            _ocr_fn=_no_ocr,
            anim_threshold=0.01,
            trans_threshold=0.90,  # very high trans threshold → won't trigger TRANS
        )
        frames = _animating_history(5)
        result = expert2.analyze(frames)
        assert result.blocks_perception is True

    def test_animating_retry_200ms(self):
        expert = TemporalExpert(
            _ocr_fn=_no_ocr,
            anim_threshold=0.01,
            trans_threshold=0.90,
        )
        frames = _animating_history(5)
        result = expert.analyze(frames)
        # ANIMATING has retry=200, TRANSITIONING has retry=300
        assert result.retry_after_ms in (200, 300)


# ---------------------------------------------------------------------------
# Section 13 — TRANSITIONING detection
# ---------------------------------------------------------------------------


class TestTransitioningDetection:
    def test_full_screen_change_transitioning(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr, trans_threshold=0.10)
        # Use black/white alternation which gives ~100% change
        frames = _animating_history(5)
        result = expert.analyze(frames)
        assert result.state_type is StateType.TRANSITIONING

    def test_transitioning_blocks(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr, trans_threshold=0.10)
        result = expert.analyze(_animating_history(5))
        assert result.blocks_perception is True

    def test_transitioning_retry_300ms(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr, trans_threshold=0.10)
        result = expert.analyze(_animating_history(5))
        assert result.retry_after_ms == 300


# ---------------------------------------------------------------------------
# Section 14 — blocking decisions correctness
# ---------------------------------------------------------------------------


class TestBlockingDecisions:
    @pytest.mark.parametrize("state,should_block", [
        (StateType.STABLE,        False),
        (StateType.LOADING,       True),
        (StateType.ANIMATING,     True),
        (StateType.TRANSITIONING, True),
        (StateType.FROZEN,        True),
        (StateType.UNKNOWN,       True),
    ])
    def test_blocking_per_state(self, state: StateType, should_block: bool):
        # Build scenarios that produce each state
        expert = TemporalExpert(_ocr_fn=_no_ocr)
        if state is StateType.STABLE:
            result = expert.analyze(_stable_history(5))
        elif state is StateType.LOADING:
            result = expert.analyze(
                _stable_history(4),
                # Override via loading_text
            )
            expert2 = TemporalExpert(_ocr_fn=lambda f: "loading")
            result = expert2.analyze(_stable_history(4))
        elif state is StateType.FROZEN:
            result = TemporalExpert(
                _ocr_fn=_no_ocr, frozen_threshold_s=5.0
            ).analyze(_frozen_history(6.0))
        elif state in (StateType.ANIMATING, StateType.TRANSITIONING):
            result = TemporalExpert(
                _ocr_fn=_no_ocr, trans_threshold=0.10
            ).analyze(_animating_history(5))
        else:  # UNKNOWN
            result = expert.analyze([])

        assert result.blocks_perception is should_block


# ---------------------------------------------------------------------------
# Section 15 — retry_after_ms values
# ---------------------------------------------------------------------------


class TestRetryAfterMs:
    def test_stable_retry_zero(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr)
        result = expert.analyze(_stable_history(5))
        assert result.retry_after_ms == 0

    def test_loading_retry_500(self):
        expert = TemporalExpert(_ocr_fn=lambda f: "loading")
        result = expert.analyze(_stable_history(4))
        assert result.retry_after_ms == 500

    def test_animating_retry_200(self):
        # Use moderate-change frames (15%) with a high trans_threshold so
        # we land in ANIMATING, not TRANSITIONING
        expert = TemporalExpert(
            _ocr_fn=_no_ocr,
            anim_threshold=0.01,
            trans_threshold=0.90,
        )
        result = expert.analyze(_moderate_change_history(5, change_frac=0.15))
        assert result.state_type is StateType.ANIMATING
        assert result.retry_after_ms == 200

    def test_transitioning_retry_300(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr, trans_threshold=0.10)
        result = expert.analyze(_animating_history(5))
        assert result.retry_after_ms == 300

    def test_frozen_retry_1000(self):
        expert = TemporalExpert(_ocr_fn=_no_ocr, frozen_threshold_s=5.0)
        result = expert.analyze(_frozen_history(6.0))
        assert result.retry_after_ms == 1000


# ---------------------------------------------------------------------------
# Section 16 — OCR failure non-fatal
# ---------------------------------------------------------------------------


class TestOCRFailureNonFatal:
    def test_raising_ocr_does_not_propagate(self):
        def _bad_ocr(f: Frame) -> str:
            raise RuntimeError("OCR boom")

        expert = TemporalExpert(_ocr_fn=_bad_ocr)
        result = expert.analyze(_stable_history(5))
        # Must not raise; state still determined
        assert isinstance(result, ScreenState)
        assert isinstance(result.state_type, StateType)

    def test_raising_ocr_can_still_be_stable(self):
        def _bad_ocr(f: Frame) -> str:
            raise ValueError("no tesseract")

        expert = TemporalExpert(_ocr_fn=_bad_ocr)
        result = expert.analyze(_stable_history(5))
        # With no loading text, identical frames → STABLE
        assert result.state_type is StateType.STABLE


# ---------------------------------------------------------------------------
# Section 17 — Hypothesis: arbitrary frame history never raises
# ---------------------------------------------------------------------------

_frame_strategy = st.builds(
    lambda data, t, seq: Frame(
        data=data,
        width=data.shape[1],
        height=data.shape[0],
        captured_at_monotonic=t,
        captured_at_utc=_UTC,
        sequence_number=seq,
    ),
    data=arrays(
        dtype=np.uint8,
        shape=st.tuples(
            st.integers(1, 32),   # height
            st.integers(1, 32),   # width
            st.just(3),           # channels
        ),
    ),
    t=st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
    seq=st.integers(1, 10_000),
)


@given(st.lists(_frame_strategy, min_size=0, max_size=10))
@settings(max_examples=200, deadline=None)
def test_hypothesis_no_exception(frames: list[Frame]) -> None:
    """
    TemporalExpert.analyze() must never raise regardless of frame content,
    dimensions, timing, or history length.
    """
    expert = TemporalExpert(_ocr_fn=_no_ocr)
    result = expert.analyze(frames)
    assert isinstance(result, ScreenState)
    assert isinstance(result.state_type, StateType)
    assert 0.0 <= result.confidence <= 1.0
    assert result.retry_after_ms >= 0
