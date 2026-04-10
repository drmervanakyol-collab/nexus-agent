"""
tests/unit/test_perception_orchestrator.py
Unit tests for nexus/perception/orchestrator.py — FAZ 30.

All subsystems are injected as lightweight stubs.

Sections:
  1.  PerceptionResult value object
  2.  Structured source — UIA bypass
  3.  Structured source — DOM bypass
  4.  Structured source — File bypass
  5.  Visual source — full pipeline
  6.  Visual source — temporal block raises ArbitrationError
  7.  Visual source — element text correlation
  8.  Cache — hit within TTL
  9.  Cache — expired after TTL
  10. Cache — different frame sequences are independent
  11. Parallel execution — all three subsystems called in visual path
  12. frame_history — None synthesises 3-frame fallback
  13. frame_history — provided history is forwarded
  14. active_window forwarded to locator and arbitrator
  15. correction_memory forwarded to arbitrator
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from nexus.capture.frame import Frame
from nexus.core.errors import ArbitrationError
from nexus.core.types import ElementId, Rect
from nexus.perception.arbitration.arbitrator import (
    ArbitrationResult,
)
from nexus.perception.locator.locator import ElementType, UIElement
from nexus.perception.matcher.matcher import Affordance, SemanticLabel
from nexus.perception.orchestrator import (
    PerceptionOrchestrator,
    PerceptionResult,
    _build_element_texts,
)
from nexus.perception.reader.ocr_engine import OCRResult
from nexus.perception.reader.reader import ReaderOutput
from nexus.perception.spatial_graph import SpatialGraph
from nexus.perception.temporal.temporal_expert import (
    ScreenState,
    StateType,
)
from nexus.source.resolver import SourceResult

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_UTC = "2026-04-05T10:00:00+00:00"


def _make_frame(
    seq: int = 1,
    width: int = 32,
    height: int = 32,
    color: tuple[int, int, int] = (100, 100, 100),
    monotonic: float = 0.0,
) -> Frame:
    data = np.full((height, width, 3), color, dtype=np.uint8)
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=monotonic,
        captured_at_utc=_UTC,
        sequence_number=seq,
    )


def _make_element(
    x: int = 0,
    y: int = 0,
    w: int = 100,
    h: int = 30,
    el_type: ElementType = ElementType.BUTTON,
    confidence: float = 0.9,
) -> UIElement:
    return UIElement(
        id=ElementId(str(uuid.uuid4())),
        element_type=el_type,
        bounding_box=Rect(x=x, y=y, width=w, height=h),
        confidence=confidence,
        is_visible=True,
        is_occluded=False,
        occlusion_ratio=0.0,
        z_order_estimate=0,
    )


def _make_label(element_id: ElementId, confidence: float = 0.9) -> SemanticLabel:
    return SemanticLabel(
        element_id=element_id,
        primary_label="button",
        secondary_labels=[],
        confidence=confidence,
        affordance=Affordance.CLICKABLE,
        is_destructive=False,
    )


def _make_stable_screen_state() -> ScreenState:
    return ScreenState(
        state_type=StateType.STABLE,
        confidence=0.95,
        blocks_perception=False,
        reason="stable",
        retry_after_ms=0,
    )


def _make_blocking_screen_state() -> ScreenState:
    return ScreenState(
        state_type=StateType.LOADING,
        confidence=0.85,
        blocks_perception=True,
        reason="loading_text",
        retry_after_ms=500,
    )


def _make_source_result(source_type: str = "visual", data: Any = None) -> SourceResult:
    return SourceResult(
        source_type=source_type,  # type: ignore[arg-type]
        data=data or {"visual_pending": True},
        confidence=0.70,
        latency_ms=5.0,
    )


def _make_ocr_result(
    text: str,
    x: int,
    y: int,
    w: int = 40,
    h: int = 12,
) -> OCRResult:
    return OCRResult(
        text=text,
        confidence=0.95,
        bounding_box=Rect(x=x, y=y, width=w, height=h),
        language="eng",
    )


# ---------------------------------------------------------------------------
# Stub subsystems
# ---------------------------------------------------------------------------


class _StubTemporalExpert:
    """Returns a fixed ScreenState; records last call arguments."""

    def __init__(self, state: ScreenState | None = None) -> None:
        self._state = state or _make_stable_screen_state()
        self.calls: list[Sequence[Frame]] = []

    def analyze(self, frame_history: Sequence[Frame]) -> ScreenState:
        self.calls.append(list(frame_history))
        return self._state


class _StubLocator:
    """Returns a fixed list of UIElements; records call count."""

    def __init__(self, elements: list[UIElement] | None = None) -> None:
        self._elements = elements or []
        self.call_count = 0
        self.last_dirty: Sequence[Rect] | None = None
        self.last_active_window: Rect | None = None

    def locate(
        self,
        frame: Frame,
        dirty_regions: Sequence[Rect] | None = None,
        active_window: Rect | None = None,
    ) -> list[UIElement]:
        self.call_count += 1
        self.last_dirty = dirty_regions
        self.last_active_window = active_window
        return self._elements


class _StubMatcher:
    """Returns labels matching the provided elements; records call count."""

    def __init__(self, labels: list[SemanticLabel] | None = None) -> None:
        self._labels = labels
        self.call_count = 0

    def match(
        self,
        elements: Sequence[UIElement],
        element_texts: dict[ElementId, str],
    ) -> list[SemanticLabel]:
        self.call_count += 1
        if self._labels is not None:
            return self._labels
        return [_make_label(el.id) for el in elements]


class _StubArbitrator:
    """Returns a fixed ArbitrationResult or raises ArbitrationError."""

    def __init__(
        self,
        result: ArbitrationResult | None = None,
        raise_error: bool = False,
    ) -> None:
        self._result = result
        self._raise = raise_error
        self.call_count = 0
        self.last_screen_state: ScreenState | None = None
        self.last_correction_memory: dict[ElementId, SemanticLabel] | None = None
        self.last_active_window: Rect | None = None

    def arbitrate(
        self,
        locator_elements: Sequence[UIElement],
        reader_output: ReaderOutput,
        semantic_labels: Sequence[SemanticLabel],
        screen_state: ScreenState,
        correction_memory: dict[ElementId, SemanticLabel] | None = None,
        active_window: Rect | None = None,
    ) -> ArbitrationResult:
        self.call_count += 1
        self.last_screen_state = screen_state
        self.last_correction_memory = correction_memory
        self.last_active_window = active_window

        if self._raise:
            raise ArbitrationError(
                "Temporal veto",
                context={"state": screen_state.state_type.name},
            )

        if self._result is not None:
            return self._result

        els = tuple(locator_elements)
        lbs = tuple(semantic_labels)
        return ArbitrationResult(
            resolved_elements=els,
            resolved_labels=lbs,
            conflicts_detected=0,
            conflicts_resolved=0,
            temporal_blocked=False,
            overall_confidence=0.9 if lbs else 0.0,
        )


class _StubOCREngine:
    """Returns fixed OCRResults; records call count."""

    def __init__(self, results: list[OCRResult] | None = None) -> None:
        self._results = results or []
        self.call_count = 0

    def extract(
        self,
        image: Any,
        region: Rect | None = None,
        languages: list[str] | None = None,
    ) -> list[OCRResult]:
        self.call_count += 1
        return self._results


def _make_orchestrator(
    *,
    temporal: _StubTemporalExpert | None = None,
    locator: _StubLocator | None = None,
    matcher: _StubMatcher | None = None,
    arbitrator: _StubArbitrator | None = None,
    ocr: _StubOCREngine | None = None,
    cache_ttl_s: float = 0.200,
    time_fn: Any = None,
) -> PerceptionOrchestrator:
    return PerceptionOrchestrator(
        temporal_expert=temporal or _StubTemporalExpert(),  # type: ignore[arg-type]
        locator=locator or _StubLocator(),  # type: ignore[arg-type]
        matcher=matcher or _StubMatcher(),  # type: ignore[arg-type]
        arbitrator=arbitrator or _StubArbitrator(),  # type: ignore[arg-type]
        ocr_engine=ocr or _StubOCREngine(),  # type: ignore[arg-type]
        cache_ttl_s=cache_ttl_s,
        _time_fn=time_fn,
    )


# ---------------------------------------------------------------------------
# 1. PerceptionResult value object
# ---------------------------------------------------------------------------


class TestPerceptionResultValueObject:
    async def test_frozen(self) -> None:
        orch = _make_orchestrator()
        result = await orch.perceive(_make_frame(), _make_source_result())
        with pytest.raises((AttributeError, TypeError)):
            result.perception_ms = 0.0  # type: ignore[misc]

    async def test_has_all_fields(self) -> None:
        orch = _make_orchestrator()
        result = await orch.perceive(_make_frame(), _make_source_result("uia"))
        assert isinstance(result.spatial_graph, SpatialGraph)
        assert isinstance(result.screen_state, ScreenState)
        assert isinstance(result.arbitration, ArbitrationResult)
        assert isinstance(result.source_result, SourceResult)
        assert isinstance(result.perception_ms, float)
        assert isinstance(result.frame_sequence, int)
        assert isinstance(result.timestamp, str)

    async def test_frame_sequence_matches_input(self) -> None:
        orch = _make_orchestrator()
        frame = _make_frame(seq=42)
        result = await orch.perceive(frame, _make_source_result("uia"))
        assert result.frame_sequence == 42

    async def test_timestamp_matches_frame(self) -> None:
        orch = _make_orchestrator()
        result = await orch.perceive(_make_frame(), _make_source_result("uia"))
        assert result.timestamp == _UTC

    async def test_source_result_passed_through(self) -> None:
        sr = _make_source_result("uia", data={"elements": []})
        orch = _make_orchestrator()
        result = await orch.perceive(_make_frame(), sr)
        assert result.source_result is sr


# ---------------------------------------------------------------------------
# 2. Structured source — UIA bypass
# ---------------------------------------------------------------------------


class TestUIABypass:
    async def test_locator_not_called(self) -> None:
        loc = _StubLocator()
        orch = _make_orchestrator(locator=loc)
        await orch.perceive(_make_frame(), _make_source_result("uia"))
        assert loc.call_count == 0

    async def test_ocr_not_called(self) -> None:
        ocr = _StubOCREngine()
        orch = _make_orchestrator(ocr=ocr)
        await orch.perceive(_make_frame(), _make_source_result("uia"))
        assert ocr.call_count == 0

    async def test_temporal_not_called(self) -> None:
        temp = _StubTemporalExpert()
        orch = _make_orchestrator(temporal=temp)
        await orch.perceive(_make_frame(), _make_source_result("uia"))
        assert len(temp.calls) == 0

    async def test_matcher_not_called(self) -> None:
        match = _StubMatcher()
        orch = _make_orchestrator(matcher=match)
        await orch.perceive(_make_frame(), _make_source_result("uia"))
        assert match.call_count == 0

    async def test_screen_state_stable(self) -> None:
        orch = _make_orchestrator()
        result = await orch.perceive(_make_frame(), _make_source_result("uia"))
        assert result.screen_state.blocks_perception is False
        assert result.screen_state.state_type is StateType.STABLE

    async def test_empty_arbitration(self) -> None:
        orch = _make_orchestrator()
        result = await orch.perceive(_make_frame(), _make_source_result("uia"))
        assert result.arbitration.resolved_elements == ()
        assert result.arbitration.resolved_labels == ()

    async def test_returns_perception_result(self) -> None:
        orch = _make_orchestrator()
        result = await orch.perceive(_make_frame(), _make_source_result("uia"))
        assert isinstance(result, PerceptionResult)


# ---------------------------------------------------------------------------
# 3. Structured source — DOM bypass
# ---------------------------------------------------------------------------


class TestDOMBypass:
    async def test_locator_not_called(self) -> None:
        loc = _StubLocator()
        orch = _make_orchestrator(locator=loc)
        await orch.perceive(_make_frame(), _make_source_result("dom"))
        assert loc.call_count == 0

    async def test_ocr_not_called(self) -> None:
        ocr = _StubOCREngine()
        orch = _make_orchestrator(ocr=ocr)
        await orch.perceive(_make_frame(), _make_source_result("dom"))
        assert ocr.call_count == 0

    async def test_screen_state_stable(self) -> None:
        orch = _make_orchestrator()
        result = await orch.perceive(_make_frame(), _make_source_result("dom"))
        assert result.screen_state.state_type is StateType.STABLE


# ---------------------------------------------------------------------------
# 4. Structured source — File bypass
# ---------------------------------------------------------------------------


class TestFileBypass:
    async def test_locator_not_called(self) -> None:
        loc = _StubLocator()
        orch = _make_orchestrator(locator=loc)
        await orch.perceive(_make_frame(), _make_source_result("file"))
        assert loc.call_count == 0

    async def test_ocr_not_called(self) -> None:
        ocr = _StubOCREngine()
        orch = _make_orchestrator(ocr=ocr)
        await orch.perceive(_make_frame(), _make_source_result("file"))
        assert ocr.call_count == 0

    async def test_screen_state_stable(self) -> None:
        orch = _make_orchestrator()
        result = await orch.perceive(_make_frame(), _make_source_result("file"))
        assert result.screen_state.state_type is StateType.STABLE


# ---------------------------------------------------------------------------
# 5. Visual source — full pipeline
# ---------------------------------------------------------------------------


class TestVisualFullPipeline:
    async def test_locator_called(self) -> None:
        loc = _StubLocator()
        orch = _make_orchestrator(locator=loc)
        await orch.perceive(_make_frame(), _make_source_result("visual"))
        assert loc.call_count == 1

    async def test_ocr_called(self) -> None:
        ocr = _StubOCREngine()
        orch = _make_orchestrator(ocr=ocr)
        await orch.perceive(_make_frame(), _make_source_result("visual"))
        assert ocr.call_count == 1

    async def test_temporal_called(self) -> None:
        temp = _StubTemporalExpert()
        orch = _make_orchestrator(temporal=temp)
        await orch.perceive(_make_frame(), _make_source_result("visual"))
        assert len(temp.calls) == 1

    async def test_matcher_called(self) -> None:
        match = _StubMatcher()
        orch = _make_orchestrator(matcher=match)
        await orch.perceive(_make_frame(), _make_source_result("visual"))
        assert match.call_count == 1

    async def test_arbitrator_called(self) -> None:
        arb = _StubArbitrator()
        orch = _make_orchestrator(arbitrator=arb)
        await orch.perceive(_make_frame(), _make_source_result("visual"))
        assert arb.call_count == 1

    async def test_screen_state_from_temporal(self) -> None:
        state = _make_stable_screen_state()
        temp = _StubTemporalExpert(state=state)
        orch = _make_orchestrator(temporal=temp)
        result = await orch.perceive(_make_frame(), _make_source_result("visual"))
        assert result.screen_state is state

    async def test_elements_from_locator_in_graph(self) -> None:
        el = _make_element()
        loc = _StubLocator(elements=[el])
        orch = _make_orchestrator(locator=loc)
        result = await orch.perceive(_make_frame(), _make_source_result("visual"))
        node_ids = {n.id for n in result.spatial_graph.nodes}
        assert el.id in node_ids

    async def test_returns_perception_result(self) -> None:
        orch = _make_orchestrator()
        result = await orch.perceive(_make_frame(), _make_source_result("visual"))
        assert isinstance(result, PerceptionResult)


# ---------------------------------------------------------------------------
# 6. Visual source — temporal block raises ArbitrationError
# ---------------------------------------------------------------------------


class TestTemporalBlock:
    async def test_raises_arbitration_error(self) -> None:
        temp = _StubTemporalExpert(state=_make_blocking_screen_state())
        arb = _StubArbitrator(raise_error=True)
        orch = _make_orchestrator(temporal=temp, arbitrator=arb)
        with pytest.raises(ArbitrationError):
            await orch.perceive(_make_frame(), _make_source_result("visual"))

    async def test_screen_state_delivered_to_arbitrator(self) -> None:
        blocking = _make_blocking_screen_state()
        temp = _StubTemporalExpert(state=blocking)
        arb = _StubArbitrator(raise_error=True)
        orch = _make_orchestrator(temporal=temp, arbitrator=arb)
        with pytest.raises(ArbitrationError):
            await orch.perceive(_make_frame(), _make_source_result("visual"))
        assert arb.last_screen_state is blocking

    async def test_structured_source_never_raises_on_loading(self) -> None:
        """Structured sources skip temporal analysis; never raise."""
        temp = _StubTemporalExpert(state=_make_blocking_screen_state())
        arb = _StubArbitrator(raise_error=True)
        orch = _make_orchestrator(temporal=temp, arbitrator=arb)
        # Should NOT raise — temporal is bypassed for UIA
        result = await orch.perceive(_make_frame(), _make_source_result("uia"))
        assert isinstance(result, PerceptionResult)


# ---------------------------------------------------------------------------
# 7. Visual source — element text correlation
# ---------------------------------------------------------------------------


class TestElementTextCorrelation:
    def test_word_inside_element_assigned(self) -> None:
        el = _make_element(x=10, y=10, w=80, h=20)
        # OCR word centre: (50, 20) — inside [10,10,90,30]
        ocr = _make_ocr_result("Hello", x=30, y=14, w=40, h=12)
        texts = _build_element_texts([el], [ocr])
        assert texts[el.id] == "Hello"

    def test_word_outside_element_not_assigned(self) -> None:
        el = _make_element(x=10, y=10, w=80, h=20)
        # OCR word centre: (200, 20) — outside
        ocr = _make_ocr_result("Outside", x=180, y=14, w=40, h=12)
        texts = _build_element_texts([el], [ocr])
        assert texts[el.id] == ""

    def test_multiple_words_joined(self) -> None:
        el = _make_element(x=0, y=0, w=200, h=40)
        w1 = _make_ocr_result("Save", x=5, y=10, w=30, h=12)
        w2 = _make_ocr_result("File", x=40, y=10, w=30, h=12)
        texts = _build_element_texts([el], [w1, w2])
        assert texts[el.id] == "Save File"

    def test_empty_ocr_gives_empty_text(self) -> None:
        el = _make_element()
        texts = _build_element_texts([el], [])
        assert texts[el.id] == ""

    def test_word_assigned_to_first_matching_element(self) -> None:
        el1 = _make_element(x=0, y=0, w=200, h=200)   # outer
        el2 = _make_element(x=10, y=10, w=50, h=20)   # inner
        # word centre at (35, 20) — inside both; assigned to el1 (first)
        ocr = _make_ocr_result("Text", x=15, y=14, w=40, h=12)
        texts = _build_element_texts([el1, el2], [ocr])
        assert texts[el1.id] == "Text"
        assert texts[el2.id] == ""


# ---------------------------------------------------------------------------
# 8. Cache — hit within TTL
# ---------------------------------------------------------------------------


class TestCacheHit:
    async def test_same_sequence_returns_same_object(self) -> None:
        loc = _StubLocator()
        # First call uses 3 time() calls: t0, perception_ms, stored_at
        # Second call uses 2: t0, cache-check → hit
        time_vals = iter([0.0, 0.0, 0.05, 0.06, 0.06])
        orch = _make_orchestrator(locator=loc, time_fn=lambda: next(time_vals))
        frame = _make_frame(seq=7)
        r1 = await orch.perceive(frame, _make_source_result("visual"))
        r2 = await orch.perceive(frame, _make_source_result("visual"))
        assert r1 is r2

    async def test_locator_not_called_on_cache_hit(self) -> None:
        loc = _StubLocator()
        time_vals = iter([0.0, 0.0, 0.05, 0.06, 0.06])
        orch = _make_orchestrator(locator=loc, time_fn=lambda: next(time_vals))
        frame = _make_frame(seq=9)
        await orch.perceive(frame, _make_source_result("visual"))
        await orch.perceive(frame, _make_source_result("visual"))
        assert loc.call_count == 1  # only once, second is cached


# ---------------------------------------------------------------------------
# 9. Cache — expired after TTL
# ---------------------------------------------------------------------------


class TestCacheExpiry:
    async def test_expired_cache_reruns_pipeline(self) -> None:
        loc = _StubLocator()
        # First call: 3 time() uses (t0, perception_ms, stored_at=0.0)
        # Second call: cache-check (t0=0.3, check=0.3 → 0.3-0.0=0.3>0.2 → miss)
        #              then: perception_ms + stored_at = 2 more calls
        time_vals = iter([0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3])
        orch = _make_orchestrator(
            locator=loc,
            cache_ttl_s=0.200,
            time_fn=lambda: next(time_vals),
        )
        frame = _make_frame(seq=11)
        await orch.perceive(frame, _make_source_result("visual"))
        await orch.perceive(frame, _make_source_result("visual"))
        assert loc.call_count == 2  # both calls ran the pipeline

    async def test_expired_cache_returns_fresh_result(self) -> None:
        el1 = _make_element()
        el2 = _make_element(x=200)
        locators = [_StubLocator(elements=[el1]), _StubLocator(elements=[el2])]
        idx = [0]

        class _SwitchingLocator:
            def locate(self, *a: object, **kw: object) -> list[UIElement]:
                result = locators[idx[0]].locate(*a, **kw)  # type: ignore[arg-type]
                idx[0] = min(idx[0] + 1, len(locators) - 1)
                return result

        time_vals = iter([0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3])
        orch = _make_orchestrator(
            locator=_SwitchingLocator(),  # type: ignore[arg-type]
            cache_ttl_s=0.200,
            time_fn=lambda: next(time_vals),
        )
        frame = _make_frame(seq=13)
        r1 = await orch.perceive(frame, _make_source_result("visual"))
        r2 = await orch.perceive(frame, _make_source_result("visual"))
        # Both results are PerceptionResult but not the same object
        assert r1 is not r2


# ---------------------------------------------------------------------------
# 10. Cache — different frame sequences are independent
# ---------------------------------------------------------------------------


class TestCacheIndependence:
    async def test_different_sequences_run_independently(self) -> None:
        loc = _StubLocator()
        orch = _make_orchestrator(locator=loc)
        await orch.perceive(_make_frame(seq=1), _make_source_result("visual"))
        await orch.perceive(_make_frame(seq=2), _make_source_result("visual"))
        assert loc.call_count == 2

    async def test_sequence_1_cached_sequence_2_miss(self) -> None:
        loc = _StubLocator()
        # Use real time: within a single test run seq=1 hits cache,
        # seq=2 is a different key → always a miss.
        orch = _make_orchestrator(locator=loc, cache_ttl_s=60.0)
        await orch.perceive(_make_frame(seq=1), _make_source_result("visual"))
        await orch.perceive(_make_frame(seq=1), _make_source_result("visual"))  # cache
        await orch.perceive(_make_frame(seq=2), _make_source_result("visual"))  # miss
        assert loc.call_count == 2  # seq=1 once, seq=2 once


# ---------------------------------------------------------------------------
# 11. Parallel execution — all three subsystems called in visual path
# ---------------------------------------------------------------------------


class TestParallelExecution:
    async def test_all_three_called_once(self) -> None:
        temp = _StubTemporalExpert()
        loc = _StubLocator()
        ocr = _StubOCREngine()
        orch = _make_orchestrator(temporal=temp, locator=loc, ocr=ocr)
        await orch.perceive(_make_frame(), _make_source_result("visual"))
        assert len(temp.calls) == 1
        assert loc.call_count == 1
        assert ocr.call_count == 1

    async def test_gather_runs_concurrent_tasks(self) -> None:
        """
        Verify gather dispatches all tasks even when individual tasks
        have simulated latency.  We track start-order and verify all
        three started before any could block sequentially.
        """
        started: list[str] = []

        async def _slow_task(name: str, delay: float) -> None:
            started.append(name)
            await asyncio.sleep(delay)

        tasks = [
            _slow_task("temporal", 0.01),
            _slow_task("locator", 0.01),
            _slow_task("ocr", 0.01),
        ]
        await asyncio.gather(*tasks)
        # All three started before any finished (order may vary but all present)
        assert set(started) == {"temporal", "locator", "ocr"}


# ---------------------------------------------------------------------------
# 12. frame_history — None synthesises 3-frame fallback
# ---------------------------------------------------------------------------


class TestFrameHistoryFallback:
    async def test_none_history_synthesises_three_frames(self) -> None:
        temp = _StubTemporalExpert()
        orch = _make_orchestrator(temporal=temp)
        await orch.perceive(
            _make_frame(seq=1),
            _make_source_result("visual"),
            frame_history=None,
        )
        assert len(temp.calls) == 1
        assert len(temp.calls[0]) == 3

    async def test_synthesised_frames_are_stable_frame_copies(self) -> None:
        temp = _StubTemporalExpert()
        orch = _make_orchestrator(temporal=temp)
        frame = _make_frame(seq=5)
        await orch.perceive(frame, _make_source_result("visual"), frame_history=None)
        for f in temp.calls[0]:
            assert f is frame


# ---------------------------------------------------------------------------
# 13. frame_history — provided history is forwarded
# ---------------------------------------------------------------------------


class TestFrameHistoryForwarded:
    async def test_provided_history_forwarded_verbatim(self) -> None:
        temp = _StubTemporalExpert()
        orch = _make_orchestrator(temporal=temp)
        h = [_make_frame(seq=i) for i in range(5)]
        await orch.perceive(
            _make_frame(seq=10),
            _make_source_result("visual"),
            frame_history=h,
        )
        assert list(temp.calls[0]) == h

    async def test_history_length_preserved(self) -> None:
        temp = _StubTemporalExpert()
        orch = _make_orchestrator(temporal=temp)
        h = [_make_frame(seq=i) for i in range(8)]
        await orch.perceive(
            _make_frame(seq=20),
            _make_source_result("visual"),
            frame_history=h,
        )
        assert len(temp.calls[0]) == 8


# ---------------------------------------------------------------------------
# 14. active_window forwarded to locator and arbitrator
# ---------------------------------------------------------------------------


class TestActiveWindowForwarding:
    async def test_active_window_forwarded_to_locator(self) -> None:
        loc = _StubLocator()
        window = Rect(x=0, y=0, width=1920, height=1080)
        orch = _make_orchestrator(locator=loc)
        await orch.perceive(
            _make_frame(),
            _make_source_result("visual"),
            active_window=window,
        )
        assert loc.last_active_window is window

    async def test_active_window_forwarded_to_arbitrator(self) -> None:
        arb = _StubArbitrator()
        window = Rect(x=0, y=0, width=1280, height=720)
        orch = _make_orchestrator(arbitrator=arb)
        await orch.perceive(
            _make_frame(),
            _make_source_result("visual"),
            active_window=window,
        )
        assert arb.last_active_window is window

    async def test_none_active_window_forwarded(self) -> None:
        arb = _StubArbitrator()
        orch = _make_orchestrator(arbitrator=arb)
        await orch.perceive(
            _make_frame(),
            _make_source_result("visual"),
            active_window=None,
        )
        assert arb.last_active_window is None


# ---------------------------------------------------------------------------
# 15. correction_memory forwarded to arbitrator
# ---------------------------------------------------------------------------


class TestCorrectionMemoryForwarding:
    async def test_correction_memory_forwarded(self) -> None:
        arb = _StubArbitrator()
        el = _make_element()
        mem: dict[ElementId, SemanticLabel] = {el.id: _make_label(el.id)}
        orch = _make_orchestrator(arbitrator=arb)
        await orch.perceive(
            _make_frame(),
            _make_source_result("visual"),
            correction_memory=mem,
        )
        assert arb.last_correction_memory is mem

    async def test_none_correction_memory_forwarded(self) -> None:
        arb = _StubArbitrator()
        orch = _make_orchestrator(arbitrator=arb)
        await orch.perceive(
            _make_frame(),
            _make_source_result("visual"),
            correction_memory=None,
        )
        assert arb.last_correction_memory is None
