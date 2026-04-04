"""
nexus/perception/orchestrator.py
Perception Orchestrator — coordinates all perception subsystems.

PerceptionResult
----------------
  spatial_graph   : SpatialGraph          queryable element graph
  screen_state    : ScreenState           temporal classification
  arbitration     : ArbitrationResult     conflict-resolved labels
  source_result   : SourceResult          winning source info
  perception_ms   : float                 total wall-clock time
  frame_sequence  : int                   from stable_frame.sequence_number
  timestamp       : str                   from stable_frame.captured_at_utc

PerceptionOrchestrator.perceive()
---------------------------------
SOURCE uia/dom/file (structured):
  - Skip Locator and OCREngine (pipeline bypass).
  - Build a STABLE ScreenState and an empty ArbitrationResult.
  - Return a PerceptionResult with an empty SpatialGraph.

SOURCE visual:
  1. asyncio.gather in parallel:
       a. TemporalExpert.analyze(frame_history)   → ScreenState
       b. Locator.locate(stable_frame, ...)        → list[UIElement]
       c. OCREngine.extract(stable_frame.data)     → list[OCRResult]
  2. Correlate OCR results with element bounding boxes → element_texts.
  3. Matcher.match(elements, element_texts)         → list[SemanticLabel]
  4. PerceptionArbitrator.arbitrate(...)            → ArbitrationResult
     Raises ArbitrationError when temporal veto is active.
  5. SpatialGraph(resolved_elements, resolved_labels, element_texts).

Cache
-----
  Key   : frame_sequence (int)
  TTL   : 200 ms (configurable via cache_ttl_s)
  Policy: hit within TTL → return cached PerceptionResult immediately.

frame_history
-------------
  Required by TemporalExpert (minimum 3 frames).  When not provided the
  orchestrator synthesises a 3-frame history of identical copies of
  stable_frame; change ratios will be ≈ 0 and the state STABLE.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from nexus.capture.frame import Frame
from nexus.core.types import ElementId, Rect
from nexus.infra.logger import get_logger
from nexus.perception.arbitration.arbitrator import (
    ArbitrationResult,
    PerceptionArbitrator,
)
from nexus.perception.locator.locator import Locator, UIElement
from nexus.perception.matcher.matcher import Matcher, SemanticLabel
from nexus.perception.reader.ocr_engine import OCREngine, OCRResult
from nexus.perception.reader.reader import ReaderOutput
from nexus.perception.spatial_graph import SpatialGraph
from nexus.perception.temporal.temporal_expert import (
    ScreenState,
    StateType,
    TemporalExpert,
)
from nexus.source.resolver import SourceResult

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_TTL_S: float = 0.200   # 200 ms
_STRUCTURED_SOURCES: frozenset[str] = frozenset({"uia", "dom", "file"})

# Stable state used for structured sources (they already resolved element info).
_STRUCTURED_STABLE_STATE = ScreenState(
    state_type=StateType.STABLE,
    confidence=1.0,
    blocks_perception=False,
    reason="structured_source",
    retry_after_ms=0,
)

_EMPTY_ARBITRATION = ArbitrationResult(
    resolved_elements=(),
    resolved_labels=(),
    conflicts_detected=0,
    conflicts_resolved=0,
    temporal_blocked=False,
    overall_confidence=0.0,
)


# ---------------------------------------------------------------------------
# PerceptionResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerceptionResult:
    """
    Output of a single PerceptionOrchestrator.perceive() call.

    Attributes
    ----------
    spatial_graph:
        Queryable graph of resolved UI elements and their spatial relations.
    screen_state:
        Temporal classification of the screen at capture time.
    arbitration:
        Conflict-resolved labels and confidence scores.
    source_result:
        The winning source result that triggered this perception run.
    perception_ms:
        Total wall-clock time for the perception pipeline (milliseconds).
    frame_sequence:
        Sequence number from the input stable_frame.
    timestamp:
        ISO-8601 UTC timestamp from the input stable_frame.
    """

    spatial_graph: SpatialGraph
    screen_state: ScreenState
    arbitration: ArbitrationResult
    source_result: SourceResult
    perception_ms: float
    frame_sequence: int
    timestamp: str


# ---------------------------------------------------------------------------
# Cache entry (internal)
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    result: PerceptionResult
    stored_at: float


# ---------------------------------------------------------------------------
# PerceptionOrchestrator
# ---------------------------------------------------------------------------


class PerceptionOrchestrator:
    """
    Coordinates Locator, OCREngine, TemporalExpert, Matcher, Arbitrator,
    and SpatialGraph into a single async perception pipeline.

    Parameters
    ----------
    temporal_expert:
        Screen-state classifier.
    locator:
        CV-based UI element detector.
    matcher:
        Rule-based semantic labeller.
    arbitrator:
        Conflict resolver between Locator and Matcher outputs.
    ocr_engine:
        OCR engine for full-frame text extraction (visual path only).
    cache_ttl_s:
        How long (seconds) a result for a given frame_sequence is valid.
        Defaults to 200 ms.
    _time_fn:
        Injectable clock; defaults to ``time.monotonic``.
    """

    def __init__(
        self,
        temporal_expert: TemporalExpert,
        locator: Locator,
        matcher: Matcher,
        arbitrator: PerceptionArbitrator,
        ocr_engine: OCREngine,
        cache_ttl_s: float = _DEFAULT_CACHE_TTL_S,
        *,
        _time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._temporal_expert = temporal_expert
        self._locator = locator
        self._matcher = matcher
        self._arbitrator = arbitrator
        self._ocr_engine = ocr_engine
        self._cache_ttl = cache_ttl_s
        self._time: Callable[[], float] = _time_fn or time.monotonic
        self._cache: dict[int, _CacheEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def perceive(
        self,
        stable_frame: Frame,
        source_result: SourceResult,
        correction_memory: dict[ElementId, SemanticLabel] | None = None,
        frame_history: Sequence[Frame] | None = None,
        active_window: Rect | None = None,
        dirty_regions: Sequence[Rect] | None = None,
    ) -> PerceptionResult:
        """
        Run the perception pipeline for *stable_frame*.

        Parameters
        ----------
        stable_frame:
            The captured, stabilised screen frame to analyse.
        source_result:
            Result from SourcePriorityResolver indicating which source is
            active.  Determines which pipeline branch is taken.
        correction_memory:
            User / agent label corrections keyed by element ID.
            Passed through to the arbitrator.
        frame_history:
            Sequence of recent frames (oldest first) for temporal analysis.
            When None, a synthetic history of 3 identical copies of
            ``stable_frame`` is used (guarantees ≥ 3 frames).
        active_window:
            Optional bounding rect of the active application window.
            Passed to Locator and Arbitrator for penalty/filtering.
        dirty_regions:
            Optional dirty rectangles forwarded to Locator.

        Returns
        -------
        PerceptionResult

        Raises
        ------
        ArbitrationError
            When the visual pipeline is used and the temporal expert
            classifies the screen as blocking perception.
        """
        t0 = self._time()

        # ── Cache lookup ──────────────────────────────────────────────
        seq = stable_frame.sequence_number
        cached = self._cache.get(seq)
        if cached is not None and (self._time() - cached.stored_at) < self._cache_ttl:
            _log.debug(
                "perception_cache_hit",
                frame_sequence=seq,
            )
            return cached.result

        # ── Source branch ─────────────────────────────────────────────
        if source_result.source_type in _STRUCTURED_SOURCES:
            result = self._run_structured(
                stable_frame=stable_frame,
                source_result=source_result,
                t0=t0,
            )
        else:
            result = await self._run_visual(
                stable_frame=stable_frame,
                source_result=source_result,
                correction_memory=correction_memory,
                frame_history=frame_history,
                active_window=active_window,
                dirty_regions=dirty_regions,
                t0=t0,
            )

        # ── Cache store ───────────────────────────────────────────────
        self._cache[seq] = _CacheEntry(result=result, stored_at=self._time())
        return result

    # ------------------------------------------------------------------
    # Structured source path (uia / dom / file)
    # ------------------------------------------------------------------

    def _run_structured(
        self,
        *,
        stable_frame: Frame,
        source_result: SourceResult,
        t0: float,
    ) -> PerceptionResult:
        """
        Build a PerceptionResult without running Locator or OCR.

        Structured sources (UIA, DOM, File) provide their own element tree;
        vision-based detection is unnecessary and expensive.  We return an
        empty SpatialGraph alongside a guaranteed STABLE ScreenState.
        """
        _log.debug(
            "perception_structured_bypass",
            source=source_result.source_type,
            frame_sequence=stable_frame.sequence_number,
        )
        perception_ms = (self._time() - t0) * 1000
        graph = SpatialGraph([], [], {})
        return PerceptionResult(
            spatial_graph=graph,
            screen_state=_STRUCTURED_STABLE_STATE,
            arbitration=_EMPTY_ARBITRATION,
            source_result=source_result,
            perception_ms=round(perception_ms, 3),
            frame_sequence=stable_frame.sequence_number,
            timestamp=stable_frame.captured_at_utc,
        )

    # ------------------------------------------------------------------
    # Visual source path
    # ------------------------------------------------------------------

    async def _run_visual(
        self,
        *,
        stable_frame: Frame,
        source_result: SourceResult,
        correction_memory: dict[ElementId, SemanticLabel] | None,
        frame_history: Sequence[Frame] | None,
        active_window: Rect | None,
        dirty_regions: Sequence[Rect] | None,
        t0: float,
    ) -> PerceptionResult:
        """
        Full visual perception pipeline with parallel analysis.

        Three subsystems are dispatched concurrently:
          1. TemporalExpert.analyze()  — screen state
          2. Locator.locate()          — element bounding boxes
          3. OCREngine.extract()       — full-frame text
        """
        # Build a 3-frame history when none is supplied.
        history: Sequence[Frame] = (
            frame_history
            if frame_history is not None
            else [stable_frame, stable_frame, stable_frame]
        )

        # ── STEP 1: Parallel execution ────────────────────────────────
        screen_state, elements, ocr_results = await asyncio.gather(
            asyncio.to_thread(self._temporal_expert.analyze, history),
            asyncio.to_thread(
                self._locator.locate,
                stable_frame,
                dirty_regions,
                active_window,
            ),
            asyncio.to_thread(self._ocr_engine.extract, stable_frame.data),
        )

        _log.debug(
            "perception_visual_parallel_done",
            frame_sequence=stable_frame.sequence_number,
            state=screen_state.state_type.name,
            elements=len(elements),
            ocr_words=len(ocr_results),
        )

        # ── STEP 2: Correlate OCR words with element bounding boxes ───
        element_texts = _build_element_texts(elements, ocr_results)

        # ── STEP 3: Semantic labelling ────────────────────────────────
        labels = self._matcher.match(elements, element_texts)

        # ── STEP 4: Arbitration (may raise ArbitrationError) ─────────
        reader_output = _make_reader_output(element_texts, ocr_results)
        arbitration = self._arbitrator.arbitrate(
            locator_elements=elements,
            reader_output=reader_output,
            semantic_labels=labels,
            screen_state=screen_state,
            correction_memory=correction_memory,
            active_window=active_window,
        )

        # ── STEP 5: Spatial graph ─────────────────────────────────────
        graph = SpatialGraph(
            list(arbitration.resolved_elements),
            list(arbitration.resolved_labels),
            element_texts,
        )

        perception_ms = (self._time() - t0) * 1000
        _log.debug(
            "perception_done",
            frame_sequence=stable_frame.sequence_number,
            perception_ms=round(perception_ms, 3),
            overall_confidence=arbitration.overall_confidence,
        )

        return PerceptionResult(
            spatial_graph=graph,
            screen_state=screen_state,
            arbitration=arbitration,
            source_result=source_result,
            perception_ms=round(perception_ms, 3),
            frame_sequence=stable_frame.sequence_number,
            timestamp=stable_frame.captured_at_utc,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_element_texts(
    elements: list[UIElement],
    ocr_results: list[OCRResult],
) -> dict[ElementId, str]:
    """
    Map each element to the OCR text whose centre point lies within its bbox.

    OCR results are assigned to the first matching element (no double-assignment).
    Words are joined with a single space in left-to-right, top-to-bottom order.
    """
    texts: dict[ElementId, list[str]] = {el.id: [] for el in elements}

    for result in ocr_results:
        rb = result.bounding_box
        cx = rb.x + rb.width // 2
        cy = rb.y + rb.height // 2
        for el in elements:
            bb = el.bounding_box
            if bb.x <= cx <= bb.x + bb.width and bb.y <= cy <= bb.y + bb.height:
                texts[el.id].append(result.text)
                break   # assign word to first (outermost) matching element

    return {eid: " ".join(words).strip() for eid, words in texts.items()}


def _make_reader_output(
    element_texts: dict[ElementId, str],
    ocr_results: list[OCRResult],
) -> ReaderOutput:
    """Build a minimal ReaderOutput from orchestrator-derived data."""
    return ReaderOutput(
        element_texts=element_texts,
        text_blocks=list(ocr_results),
        layout_regions=[],
        reading_order=list(element_texts.keys()),
        table_data=None,
    )


def _noop(*_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
    """No-operation placeholder (unused in current implementation)."""
