"""
tests/unit/test_arbitrator.py
Unit tests for nexus/perception/arbitration/arbitrator.py — Faz 29.

Sections:
  1.  ArbitrationResult value object
  2.  Temporal veto (blocks_perception=True → ArbitrationError)
  3.  No conflicts (clean pass-through)
  4.  Type conflict: Locator vs Matcher → Matcher wins
  5.  Correction memory override → memory wins
  6.  Correction memory vs type conflict → memory still wins
  7.  Confidence weighting → low combined → UNKNOWN
  8.  Occlusion penalty
  9.  Active-window penalty (outside → confidence × 0.5)
  10. Active-window (inside → no penalty)
  11. Overall confidence computation
  12. conflict counts: detected and resolved
  13. Empty element list
  14. Mismatched length raises ValueError
"""
from __future__ import annotations

import uuid

import pytest

from nexus.core.errors import ArbitrationError
from nexus.core.types import ElementId, Rect
from nexus.perception.arbitration.arbitrator import (
    ArbitrationResult,
    PerceptionArbitrator,
    _replace_label,
)
from nexus.perception.locator.locator import ElementType, UIElement
from nexus.perception.matcher.matcher import Affordance, SemanticLabel
from nexus.perception.reader.reader import ReaderOutput
from nexus.perception.temporal.temporal_expert import ScreenState, StateType

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _el(
    x: int = 0,
    y: int = 0,
    w: int = 80,
    h: int = 30,
    el_type: ElementType = ElementType.BUTTON,
    confidence: float = 0.9,
    is_occluded: bool = False,
    occlusion_ratio: float = 0.0,
    z_order: int = 0,
) -> UIElement:
    return UIElement(
        id=ElementId(str(uuid.uuid4())),
        element_type=el_type,
        bounding_box=Rect(x, y, w, h),
        confidence=confidence,
        is_visible=True,
        is_occluded=is_occluded,
        occlusion_ratio=occlusion_ratio,
        z_order_estimate=z_order,
    )


def _sem(
    el: UIElement,
    label: str = "Button",
    affordance: Affordance = Affordance.CLICKABLE,
    confidence: float = 0.9,
    is_destructive: bool = False,
) -> SemanticLabel:
    return SemanticLabel(
        element_id=el.id,
        primary_label=label,
        secondary_labels=(),
        confidence=confidence,
        affordance=affordance,
        is_destructive=is_destructive,
    )


def _stable_state() -> ScreenState:
    return ScreenState(
        state_type=StateType.STABLE,
        confidence=0.95,
        blocks_perception=False,
        reason="stable",
        retry_after_ms=0,
    )


def _blocking_state(reason: str = "loading_text") -> ScreenState:
    return ScreenState(
        state_type=StateType.LOADING,
        confidence=0.85,
        blocks_perception=True,
        reason=reason,
        retry_after_ms=500,
    )


def _empty_reader() -> ReaderOutput:
    return ReaderOutput(
        element_texts={},
        text_blocks=[],
        layout_regions=[],
        reading_order=[],
        table_data=None,
    )


def _arbitrate(
    elements: list[UIElement],
    labels: list[SemanticLabel],
    state: ScreenState | None = None,
    memory: dict[ElementId, SemanticLabel] | None = None,
    active_window: Rect | None = None,
) -> ArbitrationResult:
    arb = PerceptionArbitrator()
    return arb.arbitrate(
        locator_elements=elements,
        reader_output=_empty_reader(),
        semantic_labels=labels,
        screen_state=state or _stable_state(),
        correction_memory=memory,
        active_window=active_window,
    )


# ---------------------------------------------------------------------------
# Section 1 — ArbitrationResult value object
# ---------------------------------------------------------------------------


class TestArbitrationResult:
    def test_frozen(self):
        el = _el()
        result = _arbitrate([el], [_sem(el)])
        with pytest.raises((AttributeError, TypeError)):
            result.overall_confidence = 0.5  # type: ignore[misc]

    def test_tuple_fields(self):
        el = _el()
        result = _arbitrate([el], [_sem(el)])
        assert isinstance(result.resolved_elements, tuple)
        assert isinstance(result.resolved_labels, tuple)

    def test_temporal_blocked_false_in_success(self):
        el = _el()
        result = _arbitrate([el], [_sem(el)])
        assert result.temporal_blocked is False


# ---------------------------------------------------------------------------
# Section 2 — Temporal veto
# ---------------------------------------------------------------------------


class TestTemporalVeto:
    @pytest.mark.parametrize("reason", [
        "loading_text", "spinner", "frozen_no_change", "high_change",
    ])
    def test_blocking_state_raises(self, reason: str):
        el = _el()
        state = ScreenState(
            state_type=StateType.LOADING,
            confidence=0.9,
            blocks_perception=True,
            reason=reason,
            retry_after_ms=500,
        )
        with pytest.raises(ArbitrationError):
            _arbitrate([el], [_sem(el)], state=state)

    def test_stable_state_does_not_raise(self):
        el = _el()
        result = _arbitrate([el], [_sem(el)], state=_stable_state())
        assert isinstance(result, ArbitrationResult)

    def test_error_contains_state_name(self):
        el = _el()
        try:
            _arbitrate([el], [_sem(el)], state=_blocking_state("spinner"))
        except ArbitrationError as exc:
            assert "LOADING" in str(exc) or "spinner" in str(exc)
        else:
            pytest.fail("ArbitrationError not raised")

    def test_error_is_recoverable_false(self):
        el = _el()
        with pytest.raises(ArbitrationError) as exc_info:
            _arbitrate([el], [_sem(el)], state=_blocking_state())
        assert exc_info.value.recoverable is False

    @pytest.mark.parametrize("state_type,blocks", [
        (StateType.FROZEN,        True),
        (StateType.ANIMATING,     True),
        (StateType.TRANSITIONING, True),
        (StateType.UNKNOWN,       True),
        (StateType.STABLE,        False),
    ])
    def test_various_blocking_states(self, state_type: StateType, blocks: bool):
        el = _el()
        state = ScreenState(
            state_type=state_type,
            confidence=0.8,
            blocks_perception=blocks,
            reason="test",
            retry_after_ms=0,
        )
        if blocks:
            with pytest.raises(ArbitrationError):
                _arbitrate([el], [_sem(el)], state=state)
        else:
            result = _arbitrate([el], [_sem(el)], state=state)
            assert isinstance(result, ArbitrationResult)


# ---------------------------------------------------------------------------
# Section 3 — Clean pass-through (no conflicts)
# ---------------------------------------------------------------------------


class TestCleanPassThrough:
    def test_elements_preserved(self):
        el = _el(el_type=ElementType.BUTTON)
        sem = _sem(el, affordance=Affordance.CLICKABLE)
        result = _arbitrate([el], [sem])
        assert len(result.resolved_elements) == 1
        assert result.resolved_elements[0].id == el.id

    def test_label_preserved_when_no_conflict(self):
        el = _el(el_type=ElementType.BUTTON)
        sem = _sem(el, label="Kaydet", affordance=Affordance.CLICKABLE)
        result = _arbitrate([el], [sem])
        assert result.resolved_labels[0].primary_label == "Kaydet"

    def test_zero_conflicts_when_compatible(self):
        el = _el(el_type=ElementType.BUTTON)
        sem = _sem(el, affordance=Affordance.CLICKABLE)
        result = _arbitrate([el], [sem])
        assert result.conflicts_detected == 0
        assert result.conflicts_resolved == 0


# ---------------------------------------------------------------------------
# Section 4 — Type conflict: Matcher wins
# ---------------------------------------------------------------------------


class TestTypeConflictMatcherWins:
    def test_conflict_detected(self):
        # BUTTON type but TYPEABLE affordance → conflict
        el = _el(el_type=ElementType.BUTTON)
        sem = _sem(el, affordance=Affordance.TYPEABLE)
        result = _arbitrate([el], [sem])
        assert result.conflicts_detected >= 1

    def test_matcher_affordance_kept(self):
        el = _el(el_type=ElementType.BUTTON)
        sem = _sem(el, affordance=Affordance.TYPEABLE)
        result = _arbitrate([el], [sem])
        # Matcher wins → affordance stays TYPEABLE
        assert result.resolved_labels[0].affordance is Affordance.TYPEABLE

    def test_conflict_resolved(self):
        el = _el(el_type=ElementType.BUTTON)
        sem = _sem(el, affordance=Affordance.TYPEABLE)
        result = _arbitrate([el], [sem])
        assert result.conflicts_resolved == result.conflicts_detected

    def test_label_affordance_kept_from_matcher(self):
        el = _el(el_type=ElementType.INPUT)
        # INPUT expected to be TYPEABLE; Matcher says CLICKABLE → conflict
        sem = _sem(el, label="Ara", affordance=Affordance.CLICKABLE)
        result = _arbitrate([el], [sem])
        # Matcher wins → CLICKABLE kept
        assert result.resolved_labels[0].affordance is Affordance.CLICKABLE

    @pytest.mark.parametrize("el_type,wrong_affordance", [
        (ElementType.BUTTON,    Affordance.TYPEABLE),
        (ElementType.INPUT,     Affordance.CLICKABLE),
        (ElementType.LABEL,     Affordance.CLICKABLE),
        (ElementType.SCROLLBAR, Affordance.TYPEABLE),
        (ElementType.CHECKBOX,  Affordance.CLICKABLE),
    ])
    def test_various_type_conflicts(
        self, el_type: ElementType, wrong_affordance: Affordance
    ):
        el = _el(el_type=el_type)
        sem = _sem(el, affordance=wrong_affordance)
        result = _arbitrate([el], [sem])
        assert result.conflicts_detected >= 1
        assert result.resolved_labels[0].affordance is wrong_affordance  # Matcher wins


# ---------------------------------------------------------------------------
# Section 5 — Correction memory override
# ---------------------------------------------------------------------------


class TestCorrectionMemoryOverride:
    def test_memory_wins_over_matcher(self):
        el = _el(el_type=ElementType.BUTTON)
        matcher_sem = _sem(el, label="Button", affordance=Affordance.CLICKABLE)
        memory_sem = SemanticLabel(
            element_id=el.id,
            primary_label="Kaydet",
            secondary_labels=(),
            confidence=0.95,
            affordance=Affordance.CLICKABLE,
            is_destructive=False,
        )
        result = _arbitrate([el], [matcher_sem], memory={el.id: memory_sem})
        assert result.resolved_labels[0].primary_label == "Kaydet"

    def test_memory_override_counts_as_conflict(self):
        el = _el()
        matcher_sem = _sem(el, label="X")
        memory_sem = SemanticLabel(
            element_id=el.id,
            primary_label="Y",
            secondary_labels=(),
            confidence=0.9,
            affordance=Affordance.CLICKABLE,
            is_destructive=False,
        )
        result = _arbitrate([el], [matcher_sem], memory={el.id: memory_sem})
        assert result.conflicts_detected >= 1
        assert result.conflicts_resolved >= 1

    def test_no_memory_entry_not_overridden(self):
        el = _el()
        matcher_sem = _sem(el, label="Matcher")
        other_el = _el()
        memory_sem = SemanticLabel(
            element_id=other_el.id,
            primary_label="Memory",
            secondary_labels=(),
            confidence=0.9,
            affordance=Affordance.CLICKABLE,
            is_destructive=False,
        )
        result = _arbitrate([el], [matcher_sem], memory={other_el.id: memory_sem})
        # Only other_el in memory, not el → el not overridden
        assert result.resolved_labels[0].primary_label == "Matcher"


# ---------------------------------------------------------------------------
# Section 6 — Memory wins over type conflict
# ---------------------------------------------------------------------------


class TestMemoryVsTypeConflict:
    def test_memory_beats_both_locator_and_matcher(self):
        # Locator: INPUT, Matcher: CLICKABLE (conflict), Memory: TYPEABLE
        el = _el(el_type=ElementType.INPUT)
        matcher_sem = _sem(el, affordance=Affordance.CLICKABLE)  # conflict with INPUT
        memory_sem = SemanticLabel(
            element_id=el.id,
            primary_label="E-posta",
            secondary_labels=(),
            confidence=0.95,
            affordance=Affordance.TYPEABLE,
            is_destructive=False,
        )
        result = _arbitrate([el], [matcher_sem], memory={el.id: memory_sem})
        assert result.resolved_labels[0].primary_label == "E-posta"
        assert result.resolved_labels[0].affordance is Affordance.TYPEABLE
        assert result.conflicts_detected >= 1
        assert result.conflicts_resolved >= 1


# ---------------------------------------------------------------------------
# Section 7 — Confidence weighting → low combined → UNKNOWN
# ---------------------------------------------------------------------------


class TestLowConfidenceDowngrade:
    def test_low_combined_downgrades_to_unknown(self):
        # element_confidence=0.3, label_confidence=0.4 → combined=0.12 < 0.25
        el = _el(confidence=0.3)
        sem = _sem(el, affordance=Affordance.CLICKABLE, confidence=0.4)
        result = _arbitrate([el], [sem])
        assert result.resolved_labels[0].affordance is Affordance.UNKNOWN

    def test_low_combined_confidence_value_preserved(self):
        el = _el(confidence=0.3)
        sem = _sem(el, confidence=0.4)
        result = _arbitrate([el], [sem])
        # combined = 0.12; confidence should reflect the combined value
        assert result.resolved_labels[0].confidence == pytest.approx(0.12, abs=0.01)

    def test_sufficient_combined_not_downgraded(self):
        el = _el(confidence=0.9)
        sem = _sem(el, affordance=Affordance.CLICKABLE, confidence=0.9)
        result = _arbitrate([el], [sem])
        assert result.resolved_labels[0].affordance is Affordance.CLICKABLE

    def test_exact_threshold_not_downgraded(self):
        # combined = 0.25 is NOT below threshold (threshold is strictly <)
        el = _el(confidence=0.5)
        sem = _sem(el, confidence=0.5)  # combined = 0.25
        result = _arbitrate([el], [sem])
        assert result.resolved_labels[0].affordance is not Affordance.UNKNOWN


# ---------------------------------------------------------------------------
# Section 8 — Occlusion penalty
# ---------------------------------------------------------------------------


class TestOcclusionPenalty:
    def test_occluded_element_confidence_reduced(self):
        el = _el(is_occluded=True, occlusion_ratio=0.6)
        sem = _sem(el, confidence=0.9)
        result = _arbitrate([el], [sem])
        # penalty = 1 - 0.6 * 0.5 = 0.70; 0.9 * 0.70 = 0.63
        assert result.resolved_labels[0].confidence < 0.9

    def test_not_occluded_no_penalty(self):
        el = _el(is_occluded=False, occlusion_ratio=0.0)
        sem = _sem(el, confidence=0.9)
        result = _arbitrate([el], [sem])
        # No occlusion penalty; confidence should still be 0.9 (or very close)
        assert result.resolved_labels[0].confidence == pytest.approx(0.9, abs=0.05)

    def test_full_occlusion_penalty(self):
        el = _el(is_occluded=True, occlusion_ratio=1.0)
        sem = _sem(el, confidence=1.0)
        result = _arbitrate([el], [sem])
        # factor = 1 - 1.0 * 0.5 = 0.5
        assert result.resolved_labels[0].confidence == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Section 9 — Active-window penalty (outside)
# ---------------------------------------------------------------------------


class TestActiveWindowPenalty:
    def test_element_outside_window_halved(self):
        el = _el(x=500, y=500, w=80, h=30)
        sem = _sem(el, confidence=0.8)
        window = Rect(0, 0, 200, 200)  # element is outside
        result = _arbitrate([el], [sem], active_window=window)
        assert result.resolved_labels[0].confidence == pytest.approx(0.4, abs=0.05)

    def test_outside_window_penalty_ratio(self):
        el = _el(x=1000, y=1000, w=30, h=20)
        sem = _sem(el, confidence=1.0)
        window = Rect(0, 0, 100, 100)
        result = _arbitrate([el], [sem], active_window=window)
        assert result.resolved_labels[0].confidence == pytest.approx(0.5, abs=0.01)

    def test_no_window_no_penalty(self):
        el = _el(x=500, y=500)
        sem = _sem(el, confidence=0.8)
        result = _arbitrate([el], [sem], active_window=None)
        # No penalty (confidence may be adjusted by other steps but not window)
        assert result.resolved_labels[0].confidence == pytest.approx(0.8, abs=0.01)


# ---------------------------------------------------------------------------
# Section 10 — Active-window (inside → no penalty)
# ---------------------------------------------------------------------------


class TestActiveWindowInsideNoPenalty:
    def test_inside_window_no_penalty(self):
        el = _el(x=10, y=10, w=80, h=30)
        sem = _sem(el, confidence=0.85)
        window = Rect(0, 0, 400, 300)
        result = _arbitrate([el], [sem], active_window=window)
        assert result.resolved_labels[0].confidence == pytest.approx(0.85, abs=0.01)

    def test_overlapping_window_no_penalty(self):
        # Partially inside window — overlaps → no penalty
        el = _el(x=150, y=0, w=100, h=50)
        sem = _sem(el, confidence=0.75)
        window = Rect(0, 0, 200, 100)  # el overlaps (x=150..250 vs window x=0..200)
        result = _arbitrate([el], [sem], active_window=window)
        assert result.resolved_labels[0].confidence == pytest.approx(0.75, abs=0.01)


# ---------------------------------------------------------------------------
# Section 11 — Overall confidence
# ---------------------------------------------------------------------------


class TestOverallConfidence:
    def test_single_element_overall(self):
        el = _el(confidence=1.0)
        sem = _sem(el, confidence=0.8)
        result = _arbitrate([el], [sem])
        assert result.overall_confidence == pytest.approx(0.8, abs=0.05)

    def test_multiple_elements_average(self):
        el1 = _el(x=0, confidence=1.0)
        el2 = _el(x=100, confidence=1.0)
        sem1 = _sem(el1, confidence=0.6)
        sem2 = _sem(el2, confidence=0.8)
        result = _arbitrate([el1, el2], [sem1, sem2])
        # (0.6 + 0.8) / 2 = 0.7
        assert result.overall_confidence == pytest.approx(0.7, abs=0.05)

    def test_empty_elements_overall_zero(self):
        result = _arbitrate([], [])
        assert result.overall_confidence == pytest.approx(0.0)

    def test_overall_in_valid_range(self):
        els = [_el(x=i * 100) for i in range(5)]
        sems = [_sem(e, confidence=0.7) for e in els]
        result = _arbitrate(els, sems)
        assert 0.0 <= result.overall_confidence <= 1.0


# ---------------------------------------------------------------------------
# Section 12 — Conflict counts
# ---------------------------------------------------------------------------


class TestConflictCounts:
    def test_multiple_conflicts_counted(self):
        els = [
            _el(el_type=ElementType.BUTTON),  # conflict: TYPEABLE
            _el(el_type=ElementType.INPUT),   # conflict: CLICKABLE
            _el(el_type=ElementType.BUTTON),  # no conflict: CLICKABLE
        ]
        sems = [
            _sem(els[0], affordance=Affordance.TYPEABLE),   # conflict
            _sem(els[1], affordance=Affordance.CLICKABLE),  # conflict
            _sem(els[2], affordance=Affordance.CLICKABLE),  # OK
        ]
        result = _arbitrate(els, sems)
        assert result.conflicts_detected == 2
        assert result.conflicts_resolved == 2

    def test_resolved_leq_detected(self):
        els = [_el() for _ in range(5)]
        sems = [_sem(e, affordance=Affordance.TYPEABLE) for e in els]
        result = _arbitrate(els, sems)
        assert result.conflicts_resolved <= result.conflicts_detected

    def test_memory_and_type_conflict_both_counted(self):
        el1 = _el(el_type=ElementType.BUTTON)
        el2 = _el(el_type=ElementType.BUTTON)
        # el1 has type conflict AND is in memory
        # el2 has type conflict only
        mem_label = SemanticLabel(
            element_id=el1.id,
            primary_label="Memory",
            secondary_labels=(),
            confidence=0.9,
            affordance=Affordance.CLICKABLE,
            is_destructive=False,
        )
        sems = [
            _sem(el1, affordance=Affordance.TYPEABLE),  # would be conflict, but memory wins
            _sem(el2, affordance=Affordance.TYPEABLE),  # type conflict
        ]
        result = _arbitrate(
            [el1, el2], sems, memory={el1.id: mem_label}
        )
        # el1: memory override → 1 conflict; el2: type conflict → 1 conflict
        assert result.conflicts_detected == 2
        assert result.conflicts_resolved == 2


# ---------------------------------------------------------------------------
# Section 13 — Empty element list
# ---------------------------------------------------------------------------


class TestEmptyElements:
    def test_empty_returns_empty_result(self):
        result = _arbitrate([], [])
        assert result.resolved_elements == ()
        assert result.resolved_labels == ()
        assert result.conflicts_detected == 0
        assert result.overall_confidence == 0.0
        assert result.temporal_blocked is False


# ---------------------------------------------------------------------------
# Section 14 — Mismatched length raises
# ---------------------------------------------------------------------------


class TestMismatchedLengths:
    def test_more_elements_than_labels(self):
        el1 = _el()
        el2 = _el()
        with pytest.raises(ValueError, match="equal length"):
            _arbitrate([el1, el2], [_sem(el1)])

    def test_more_labels_than_elements(self):
        el = _el()
        with pytest.raises(ValueError):
            _arbitrate([el], [_sem(el), _sem(_el())])


# ---------------------------------------------------------------------------
# Section — _replace_label helper
# ---------------------------------------------------------------------------


class TestReplaceLabelHelper:
    def test_replace_affordance(self):
        el = _el()
        sem = _sem(el, affordance=Affordance.CLICKABLE)
        new = _replace_label(sem, affordance=Affordance.UNKNOWN)
        assert new.affordance is Affordance.UNKNOWN
        assert new.primary_label == sem.primary_label

    def test_replace_confidence(self):
        el = _el()
        sem = _sem(el, confidence=0.9)
        new = _replace_label(sem, confidence=0.5)
        assert new.confidence == pytest.approx(0.5)
        assert new.affordance is sem.affordance

    def test_replace_preserves_other_fields(self):
        el = _el()
        sem = _sem(el, label="Sil", is_destructive=True)
        new = _replace_label(sem, confidence=0.7)
        assert new.primary_label == "Sil"
        assert new.is_destructive is True
        assert new.element_id == el.id
