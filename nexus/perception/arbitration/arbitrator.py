"""
nexus/perception/arbitration/arbitrator.py
Perception Arbitrator — resolves conflicts between Locator, Matcher, and memory.

ArbitrationResult
-----------------
  resolved_elements   : list[UIElement]      final set of elements
  resolved_labels     : list[SemanticLabel]  final set of labels (same order)
  conflicts_detected  : int                  type / confidence mismatches found
  conflicts_resolved  : int                  conflicts successfully resolved
  temporal_blocked    : bool                 True when arbitration was vetoed
  overall_confidence  : float                mean confidence across all labels

PerceptionArbitrator.arbitrate()
---------------------------------
STEP 1 — Temporal veto
  If screen_state.blocks_perception → raise ArbitrationError immediately.
  temporal_blocked is set True in the result when this has happened
  (only reachable by callers who catch the error and inspect the partial result).

STEP 2 — Type-conflict resolution (Locator vs Matcher)
  For each (UIElement, SemanticLabel) pair:
    a. correction_memory[element_id] exists → memory wins; log conflict.
    b. element_type's expected affordance ≠ semantic.affordance → conflict;
       Matcher wins (original SemanticLabel kept); log conflict.
  conflicts_detected / conflicts_resolved updated accordingly.

STEP 3 — Confidence weighting
  combined = element.confidence × label.confidence
  combined < LOW_COMBINED_THRESHOLD → downgrade label to Affordance.UNKNOWN
  with confidence = combined (keeps numeric value for ranking).

STEP 4 — Occlusion adjustment
  element.is_occluded → multiply label confidence by
  (1 – occlusion_ratio × OCCL_PENALTY)

STEP 5 — Active-window penalty
  element bbox does not overlap active_window → multiply confidence by 0.50

STEP 6 — Overall confidence
  arithmetic mean of resolved label confidences; 0.0 when no elements.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nexus.core.errors import ArbitrationError
from nexus.core.types import ElementId, Rect
from nexus.infra.logger import get_logger
from nexus.perception.locator.locator import ElementType, UIElement
from nexus.perception.matcher.matcher import Affordance, SemanticLabel
from nexus.perception.reader.reader import ReaderOutput
from nexus.perception.temporal.temporal_expert import ScreenState

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

_LOW_COMBINED_THRESHOLD: float = 0.25   # below → downgrade to UNKNOWN
_OCCL_PENALTY: float = 0.5              # occluded element loses up to 50%
_ACTIVE_WINDOW_PENALTY: float = 0.50    # outside active window → ×0.5

# Which affordances are "expected" for each element type.
# UNKNOWN type accepts any affordance (no conflict possible).
_TYPE_EXPECTED_AFFORDANCES: dict[ElementType, frozenset[Affordance]] = {
    ElementType.BUTTON:    frozenset({Affordance.CLICKABLE}),
    ElementType.INPUT:     frozenset({Affordance.TYPEABLE}),
    ElementType.LABEL:     frozenset({Affordance.READ_ONLY}),
    ElementType.IMAGE:     frozenset({Affordance.READ_ONLY}),
    ElementType.ICON:      frozenset({Affordance.CLICKABLE, Affordance.UNKNOWN}),
    ElementType.CONTAINER: frozenset({Affordance.UNKNOWN}),
    ElementType.PANEL:     frozenset({Affordance.UNKNOWN}),
    ElementType.MENU:      frozenset({Affordance.CLICKABLE}),
    ElementType.DROPDOWN:  frozenset({Affordance.SELECTABLE, Affordance.CLICKABLE}),
    ElementType.CHECKBOX:  frozenset({Affordance.SELECTABLE}),
    ElementType.RADIO:     frozenset({Affordance.SELECTABLE}),
    ElementType.LINK:      frozenset({Affordance.CLICKABLE}),
    ElementType.SCROLLBAR: frozenset({Affordance.SCROLLABLE}),
    ElementType.TAB:       frozenset({Affordance.CLICKABLE}),
    ElementType.DIALOG:    frozenset({Affordance.UNKNOWN}),
    ElementType.TOOLTIP:   frozenset({Affordance.READ_ONLY, Affordance.UNKNOWN}),
    ElementType.UNKNOWN:   frozenset(Affordance),     # any affordance OK
}


# ---------------------------------------------------------------------------
# ArbitrationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArbitrationResult:
    """
    Output of a single PerceptionArbitrator.arbitrate() call.

    Attributes
    ----------
    resolved_elements:
        UIElements that survived arbitration (same order as resolved_labels).
    resolved_labels:
        SemanticLabels after conflict resolution, confidence adjustments,
        and occlusion / active-window penalties.  Same length and order as
        resolved_elements.
    conflicts_detected:
        Number of type or confidence mismatches found.
    conflicts_resolved:
        Number of those conflicts that were successfully resolved (≤ detected).
    temporal_blocked:
        True when the temporal veto was active (should equal False in any
        result that is actually returned, because the veto raises instead).
    overall_confidence:
        Arithmetic mean of resolved label confidences; 0.0 for empty results.
    """

    resolved_elements: tuple[UIElement, ...]
    resolved_labels: tuple[SemanticLabel, ...]
    conflicts_detected: int
    conflicts_resolved: int
    temporal_blocked: bool
    overall_confidence: float


# ---------------------------------------------------------------------------
# PerceptionArbitrator
# ---------------------------------------------------------------------------


class PerceptionArbitrator:
    """
    Merges and validates the outputs of Locator, Reader, and Matcher.

    This class is stateless; all state is passed explicitly on each call.
    """

    def arbitrate(
        self,
        locator_elements: Sequence[UIElement],
        reader_output: ReaderOutput,
        semantic_labels: Sequence[SemanticLabel],
        screen_state: ScreenState,
        correction_memory: dict[ElementId, SemanticLabel] | None = None,
        active_window: Rect | None = None,
    ) -> ArbitrationResult:
        """
        Run the full arbitration pipeline and return a resolved result.

        Parameters
        ----------
        locator_elements:
            UIElements detected by Locator (same length as semantic_labels).
        reader_output:
            Output of Reader.read() for this frame.
        semantic_labels:
            SemanticLabels from Matcher (same order as locator_elements).
        screen_state:
            Current screen state from TemporalExpert.
        correction_memory:
            User / agent corrections indexed by element ID.  When an entry
            exists its SemanticLabel completely overrides the Matcher result.
        active_window:
            Optional bounding rect of the active application window.
            Elements outside this rect receive a confidence penalty.

        Returns
        -------
        ArbitrationResult

        Raises
        ------
        ArbitrationError
            When screen_state.blocks_perception is True (temporal veto).
        """
        if len(locator_elements) != len(semantic_labels):
            raise ValueError(
                f"locator_elements ({len(locator_elements)}) and "
                f"semantic_labels ({len(semantic_labels)}) must have equal length"
            )

        # ── STEP 1: Temporal veto ──────────────────────────────────────────
        if screen_state.blocks_perception:
            _log.warning(
                "arbitration_temporal_veto",
                state=screen_state.state_type.name,
                reason=screen_state.reason,
            )
            raise ArbitrationError(
                f"Temporal veto: screen state is "
                f"{screen_state.state_type.name} ({screen_state.reason})",
                context={
                    "state": screen_state.state_type.name,
                    "reason": screen_state.reason,
                    "retry_after_ms": screen_state.retry_after_ms,
                },
            )

        memory = correction_memory or {}
        conflicts_detected = 0
        conflicts_resolved = 0

        resolved_elements: list[UIElement] = []
        resolved_labels: list[SemanticLabel] = []

        pairs = list(zip(locator_elements, semantic_labels, strict=True))

        for el, sem in pairs:
            label = sem  # start with Matcher's output

            # ── STEP 2a: Correction memory override ───────────────────────
            if el.id in memory:
                _log.debug(
                    "arbitration_memory_override",
                    element_id=str(el.id)[:8],
                    old_label=sem.primary_label,
                    new_label=memory[el.id].primary_label,
                )
                conflicts_detected += 1
                conflicts_resolved += 1
                label = memory[el.id]

            else:
                # ── STEP 2b: Type conflict (Locator vs Matcher) ───────────
                expected = _TYPE_EXPECTED_AFFORDANCES.get(
                    el.element_type, frozenset(Affordance)
                )
                if sem.affordance not in expected:
                    _log.debug(
                        "arbitration_type_conflict",
                        element_id=str(el.id)[:8],
                        element_type=el.element_type.name,
                        matcher_affordance=sem.affordance.name,
                        expected=[a.name for a in expected],
                    )
                    conflicts_detected += 1
                    conflicts_resolved += 1
                    # Matcher wins: keep `label` as-is (already = sem)

            # ── STEP 3: Confidence weighting ──────────────────────────────
            combined = el.confidence * label.confidence
            if combined < _LOW_COMBINED_THRESHOLD:
                _log.debug(
                    "arbitration_low_combined_confidence",
                    element_id=str(el.id)[:8],
                    combined=round(combined, 3),
                )
                label = _replace_label(
                    label,
                    affordance=Affordance.UNKNOWN,
                    confidence=combined,
                )

            # ── STEP 4: Occlusion adjustment ──────────────────────────────
            if el.is_occluded and el.occlusion_ratio > 0:
                factor = 1.0 - el.occlusion_ratio * _OCCL_PENALTY
                label = _replace_label(
                    label, confidence=label.confidence * factor
                )

            # ── STEP 5: Active-window penalty ─────────────────────────────
            if (
                active_window is not None
                and not el.bounding_box.overlaps(active_window)
            ):
                label = _replace_label(
                    label,
                    confidence=label.confidence * _ACTIVE_WINDOW_PENALTY,
                )

            resolved_elements.append(el)
            resolved_labels.append(label)

        # ── STEP 6: Overall confidence ────────────────────────────────────
        if resolved_labels:
            total_conf = sum(lb.confidence for lb in resolved_labels)
            overall = total_conf / len(resolved_labels)
        else:
            overall = 0.0

        _log.debug(
            "arbitration_done",
            elements=len(resolved_elements),
            conflicts_detected=conflicts_detected,
            conflicts_resolved=conflicts_resolved,
            overall_confidence=round(overall, 3),
        )

        return ArbitrationResult(
            resolved_elements=tuple(resolved_elements),
            resolved_labels=tuple(resolved_labels),
            conflicts_detected=conflicts_detected,
            conflicts_resolved=conflicts_resolved,
            temporal_blocked=False,
            overall_confidence=round(overall, 4),
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _replace_label(
    label: SemanticLabel,
    *,
    affordance: Affordance | None = None,
    confidence: float | None = None,
) -> SemanticLabel:
    """Return a new SemanticLabel with the given fields replaced."""
    return SemanticLabel(
        element_id=label.element_id,
        primary_label=label.primary_label,
        secondary_labels=label.secondary_labels,
        confidence=round(confidence if confidence is not None else label.confidence, 4),
        affordance=affordance if affordance is not None else label.affordance,
        is_destructive=label.is_destructive,
    )
