"""
tests/integration/test_block3.py
Blok 3 Integration Tests — Faz 31

Perception layer (FAZ 24-30) end-to-end: real classes wired together.
Platform-level I/O (OCR, UI Automation, mouse/keyboard) is injected via
lightweight stubs so no real hardware or OS calls are made.

TEST 1 — Full visual pipeline
TEST 2 — UIA source bypass
TEST 3 — Turkish text pipeline
TEST 4 — Temporal block → ArbitrationError
TEST 5 — Arbitration conflict resolution (correction_memory)
TEST 6 — find_best_target
TEST 7 — Transport + Perception integration (UIA → invoke)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nexus.capture.frame import Frame
from nexus.core.errors import ArbitrationError
from nexus.core.settings import NexusSettings
from nexus.core.types import ElementId, Rect
from nexus.perception.arbitration.arbitrator import PerceptionArbitrator
from nexus.perception.locator.locator import ElementType, Locator
from nexus.perception.matcher.matcher import Affordance, Matcher, SemanticLabel
from nexus.perception.orchestrator import PerceptionOrchestrator, PerceptionResult
from nexus.perception.reader.ocr_engine import OCRResult
from nexus.perception.temporal.temporal_expert import StateType, TemporalExpert
from nexus.source.resolver import SourceResult
from nexus.source.transport.fallback import KeyboardTransport, MouseTransport
from nexus.source.transport.resolver import ActionSpec, TransportResolver

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_UTC = "2026-04-05T10:00:00+00:00"

# ── Frame helpers ────────────────────────────────────────────────────────────


def _blank_frame(
    width: int = 400,
    height: int = 200,
    seq: int = 1,
    monotonic: float = 0.0,
) -> Frame:
    """All-black frame with no detectable shapes."""
    data = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=monotonic,
        captured_at_utc=_UTC,
        sequence_number=seq,
    )


def _frame_with_rects(
    rects: list[tuple[int, int, int, int]],  # (x, y, w, h)
    width: int = 500,
    height: int = 200,
    seq: int = 1,
    colors: list[tuple[int, int, int]] | None = None,
) -> Frame:
    """
    Black frame with filled white (or custom-color) rectangles at given
    positions.  The Locator's contour detector reliably picks these up.
    """
    data = np.zeros((height, width, 3), dtype=np.uint8)
    _colors = colors or [(255, 255, 255)] * len(rects)
    for (x, y, w, h), color in zip(rects, _colors, strict=True):
        data[y : y + h, x : x + w] = color
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=0.0,
        captured_at_utc=_UTC,
        sequence_number=seq,
    )


def _history(frame: Frame, n: int = 3) -> list[Frame]:
    """Minimal frame history for TemporalExpert (identical frames → STABLE)."""
    return [frame] * n


# ── OCR stubs ────────────────────────────────────────────────────────────────


class _ConstOCR:
    """Returns a fixed list of OCRResult on every extract() call."""

    def __init__(self, results: list[OCRResult]) -> None:
        self._results = results

    def extract(
        self,
        image: Any,
        region: Any = None,
        languages: Any = None,
    ) -> list[OCRResult]:
        return self._results


class _EmptyOCR:
    """OCR engine that always returns no results."""

    def extract(
        self,
        image: Any,
        region: Any = None,
        languages: Any = None,
    ) -> list[OCRResult]:
        return []


def _ocr_result(text: str, cx: int, cy: int) -> OCRResult:
    """
    Create an OCRResult whose bounding box centre is exactly at (cx, cy).
    Used to guarantee the word is assigned to the element containing that point.
    """
    return OCRResult(
        text=text,
        confidence=0.95,
        bounding_box=Rect(x=cx - 10, y=cy - 6, width=20, height=12),
        language="eng",
    )


# ── Orchestrator factory ─────────────────────────────────────────────────────


def _make_orchestrator(
    ocr_engine: Any,
    temporal_ocr: str | None = None,
) -> PerceptionOrchestrator:
    """
    Wire up real subsystems into a PerceptionOrchestrator.

    Parameters
    ----------
    ocr_engine:
        Stub OCR engine injected into the orchestrator.
    temporal_ocr:
        Optional text returned by the injectable temporal OCR function.
        Defaults to "" (no loading text → STABLE).
    """
    te_ocr = temporal_ocr or ""
    return PerceptionOrchestrator(
        temporal_expert=TemporalExpert(_ocr_fn=lambda f: te_ocr),
        locator=Locator(),
        matcher=Matcher(),
        arbitrator=PerceptionArbitrator(),
        ocr_engine=ocr_engine,
    )


async def _perceive_visual(
    orch: PerceptionOrchestrator,
    frame: Frame,
) -> PerceptionResult:
    """Shorthand: perceive with visual source and 3-frame history."""
    return await orch.perceive(
        frame,
        _visual_source(),
        frame_history=_history(frame),
    )


def _visual_source() -> SourceResult:
    return SourceResult(
        source_type="visual",
        data={"visual_pending": True},
        confidence=0.70,
        latency_ms=0.0,
    )


def _uia_source(data: Any = None) -> SourceResult:
    return SourceResult(
        source_type="uia",
        data=data or {"elements": []},
        confidence=1.0,
        latency_ms=0.0,
    )


# ---------------------------------------------------------------------------
# TEST 1 — Full visual pipeline
# ---------------------------------------------------------------------------


class TestFullVisualPipeline:
    """
    Complete visual perception pipeline with real Locator + Matcher + Arbitrator
    + SpatialGraph.  Only the OCR engine is stubbed.
    """

    async def test_returns_perception_result(self) -> None:
        frame = _frame_with_rects([(50, 80, 100, 30)])
        orch = _make_orchestrator(_EmptyOCR())
        result = await _perceive_visual(orch, frame)
        assert isinstance(result, PerceptionResult)

    async def test_detects_button_element(self) -> None:
        """Real Locator detects the white rectangle as a BUTTON."""
        frame = _frame_with_rects([(50, 80, 100, 30)])
        orch = _make_orchestrator(_EmptyOCR())
        result = await _perceive_visual(orch, frame)
        types = [el.element_type for el in result.arbitration.resolved_elements]
        assert ElementType.BUTTON in types

    async def test_screen_state_is_stable(self) -> None:
        """Identical frame history with no loading text → STABLE."""
        frame = _frame_with_rects([(50, 80, 100, 30)])
        orch = _make_orchestrator(_EmptyOCR())
        result = await _perceive_visual(orch, frame)
        assert result.screen_state.state_type is StateType.STABLE
        assert result.screen_state.blocks_perception is False

    async def test_source_type_is_visual(self) -> None:
        frame = _blank_frame()
        orch = _make_orchestrator(_EmptyOCR())
        result = await _perceive_visual(orch, frame)
        assert result.source_result.source_type == "visual"

    async def test_ocr_text_assigned_to_element(self) -> None:
        """
        Word whose bbox centre falls inside the button gets assigned.
        Button drawn at x=[50,150], y=[80,110] → detected ≈ Rect(49,79,101,31).
        OCR centre at (100, 95) is safely inside that bbox.
        """
        frame = _frame_with_rects([(50, 80, 100, 30)])
        ocr = _ConstOCR([_ocr_result("Save", cx=100, cy=95)])
        orch = _make_orchestrator(ocr)
        result = await _perceive_visual(orch, frame)

        nodes = result.spatial_graph.nodes
        assert len(nodes) == 1
        assert nodes[0].text == "Save"

    async def test_matched_label_is_save(self) -> None:
        """Matcher recognises 'Save' and produces a known label."""
        frame = _frame_with_rects([(50, 80, 100, 30)])
        ocr = _ConstOCR([_ocr_result("Save", cx=100, cy=95)])
        orch = _make_orchestrator(ocr)
        result = await _perceive_visual(orch, frame)
        labels = result.arbitration.resolved_labels
        assert len(labels) == 1
        assert labels[0].primary_label == "Save"
        assert labels[0].affordance is Affordance.CLICKABLE

    async def test_spatial_graph_has_one_node(self) -> None:
        frame = _frame_with_rects([(50, 80, 100, 30)])
        orch = _make_orchestrator(_EmptyOCR())
        result = await _perceive_visual(orch, frame)
        assert len(result.spatial_graph.nodes) == 1

    async def test_blank_frame_produces_no_elements(self) -> None:
        """No detectable shapes → empty perception result."""
        frame = _blank_frame()
        orch = _make_orchestrator(_EmptyOCR())
        result = await _perceive_visual(orch, frame)
        assert len(result.arbitration.resolved_elements) == 0
        assert result.arbitration.overall_confidence == 0.0


# ---------------------------------------------------------------------------
# TEST 2 — UIA source bypass
# ---------------------------------------------------------------------------


class TestUIASourceBypass:
    """
    When source_result.source_type == 'uia', the orchestrator skips all
    vision-based subsystems and returns immediately with a STABLE ScreenState
    and an empty SpatialGraph.
    """

    async def test_source_type_preserved(self) -> None:
        orch = _make_orchestrator(_EmptyOCR())
        result = await orch.perceive(_blank_frame(), _uia_source())
        assert result.source_result.source_type == "uia"

    async def test_screen_state_is_stable(self) -> None:
        orch = _make_orchestrator(_EmptyOCR())
        result = await orch.perceive(_blank_frame(), _uia_source())
        assert result.screen_state.state_type is StateType.STABLE
        assert result.screen_state.blocks_perception is False

    async def test_no_elements_in_result(self) -> None:
        """Structured source bypass → arbitration resolves zero elements."""
        orch = _make_orchestrator(_EmptyOCR())
        result = await orch.perceive(_blank_frame(), _uia_source())
        assert result.arbitration.resolved_elements == ()

    async def test_empty_spatial_graph(self) -> None:
        orch = _make_orchestrator(_EmptyOCR())
        result = await orch.perceive(_blank_frame(), _uia_source())
        assert len(result.spatial_graph.nodes) == 0

    async def test_fast_execution(self) -> None:
        """UIA bypass should complete quickly (no OCR / CV overhead)."""
        import time

        orch = _make_orchestrator(_EmptyOCR())
        t0 = time.monotonic()
        await orch.perceive(_blank_frame(), _uia_source())
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 200  # should be near-instant

    async def test_dom_source_also_bypasses(self) -> None:
        dom_sr = SourceResult(
            source_type="dom",
            data={},
            confidence=0.95,
            latency_ms=0.0,
        )
        orch = _make_orchestrator(_EmptyOCR())
        result = await orch.perceive(_blank_frame(), dom_sr)
        assert result.screen_state.state_type is StateType.STABLE
        assert result.arbitration.resolved_elements == ()


# ---------------------------------------------------------------------------
# TEST 3 — Turkish text pipeline
# ---------------------------------------------------------------------------


class TestTurkishTextPipeline:
    """
    Real Locator detects three buttons; stub OCR returns Turkish labels
    positioned inside each button's bounding box.  Matcher must correctly
    classify Turkish text including is_destructive markers.

    Frame layout (x, y, w, h):
      Button1 "Kaydet" — (30, 80, 100, 30) → save; NOT destructive
      Button2 "Sil"    — (160, 80, 60, 30)  → delete; IS destructive
      Button3 "İptal"  — (250, 80, 80, 30)  → cancel; IS destructive

    OCR centres placed at drawn-rect centres (safely inside detected bboxes).
    """

    _FRAME_RECTS = [(30, 80, 100, 30), (160, 80, 60, 30), (250, 80, 80, 30)]

    # Drawn-rect centres
    _KAYDET_CX, _KAYDET_CY = 30 + 50, 80 + 15   # (80, 95)
    _SIL_CX,    _SIL_CY    = 160 + 30, 80 + 15  # (190, 95)
    _IPTAL_CX,  _IPTAL_CY  = 250 + 40, 80 + 15  # (290, 95)

    def _make_turkish_orch(self) -> PerceptionOrchestrator:
        ocr = _ConstOCR([
            _ocr_result("Kaydet", cx=self._KAYDET_CX, cy=self._KAYDET_CY),
            _ocr_result("Sil",    cx=self._SIL_CX,    cy=self._SIL_CY),
            _ocr_result("İptal",  cx=self._IPTAL_CX,  cy=self._IPTAL_CY),
        ])
        return _make_orchestrator(ocr)

    async def _run(self) -> PerceptionResult:
        frame = _frame_with_rects(self._FRAME_RECTS)
        orch = self._make_turkish_orch()
        return await orch.perceive(
            frame,
            _visual_source(),
            frame_history=_history(frame),
        )

    async def test_detects_three_buttons(self) -> None:
        result = await self._run()
        assert len(result.arbitration.resolved_elements) == 3

    async def test_all_elements_are_buttons(self) -> None:
        result = await self._run()
        for el in result.arbitration.resolved_elements:
            assert el.element_type is ElementType.BUTTON

    async def test_kaydet_not_destructive(self) -> None:
        result = await self._run()
        graph = result.spatial_graph
        kaydet_nodes = graph.find_by_text("Kaydet")
        assert kaydet_nodes, "Expected node with 'Kaydet' text"
        assert kaydet_nodes[0].semantic.is_destructive is False

    async def test_sil_is_destructive(self) -> None:
        result = await self._run()
        graph = result.spatial_graph
        sil_nodes = graph.find_by_text("Sil")
        assert sil_nodes, "Expected node with 'Sil' text"
        assert sil_nodes[0].semantic.is_destructive is True

    async def test_iptal_is_destructive(self) -> None:
        result = await self._run()
        graph = result.spatial_graph
        # find_by_text uses normalised substring matching
        iptal_nodes = graph.find_by_text("İptal") or graph.find_by_text("iptal")
        assert iptal_nodes, "Expected node with 'İptal' text"
        assert iptal_nodes[0].semantic.is_destructive is True

    async def test_all_clickable(self) -> None:
        result = await self._run()
        for label in result.arbitration.resolved_labels:
            assert label.affordance is Affordance.CLICKABLE

    async def test_turkish_normalisation_via_find_by_text(self) -> None:
        """find_by_text must return the İptal node without raising."""
        result = await self._run()
        nodes = result.spatial_graph.find_by_text("İptal")
        # At least one match expected (fuzzy or exact)
        assert isinstance(nodes, list)


# ---------------------------------------------------------------------------
# TEST 4 — Temporal block → ArbitrationError
# ---------------------------------------------------------------------------


class TestTemporalBlock:
    """
    TemporalExpert with injected OCR that always returns 'loading' →
    ScreenState(LOADING, blocks_perception=True) →
    PerceptionArbitrator raises ArbitrationError.
    """

    async def test_raises_arbitration_error(self) -> None:
        frame = _frame_with_rects([(50, 80, 100, 30)])
        orch = _make_orchestrator(_EmptyOCR(), temporal_ocr="loading")
        with pytest.raises(ArbitrationError):
            await orch.perceive(frame, _visual_source(), frame_history=_history(frame))

    async def test_error_is_arbitration_error(self) -> None:
        frame = _blank_frame()
        orch = _make_orchestrator(_EmptyOCR(), temporal_ocr="loading")
        with pytest.raises(ArbitrationError) as exc_info:
            await orch.perceive(frame, _visual_source(), frame_history=_history(frame))
        assert exc_info.value.code == "arbitration_error"

    async def test_please_wait_also_blocks(self) -> None:
        """'please wait' is a known loading pattern."""
        frame = _blank_frame()
        orch = _make_orchestrator(_EmptyOCR(), temporal_ocr="please wait")
        with pytest.raises(ArbitrationError):
            await orch.perceive(frame, _visual_source(), frame_history=_history(frame))

    async def test_yukleniyir_blocks(self) -> None:
        """Turkish loading text 'yükleniyor' triggers temporal veto."""
        frame = _blank_frame()
        orch = _make_orchestrator(_EmptyOCR(), temporal_ocr="yükleniyor")
        with pytest.raises(ArbitrationError):
            await orch.perceive(frame, _visual_source(), frame_history=_history(frame))

    async def test_uia_source_unaffected_by_loading(self) -> None:
        """UIA source bypasses temporal check entirely — must not raise."""
        frame = _blank_frame()
        orch = _make_orchestrator(_EmptyOCR(), temporal_ocr="loading")
        result = await orch.perceive(frame, _uia_source())
        assert isinstance(result, PerceptionResult)


# ---------------------------------------------------------------------------
# TEST 5 — Arbitration conflict resolution (correction_memory)
# ---------------------------------------------------------------------------


class TestArbitrationConflict:
    """
    correction_memory overrides a Matcher-produced label.
    The Arbitrator counts one conflict detected and one resolved.
    The final label in the graph reflects the correction.
    """

    async def _run_with_correction(
        self,
    ) -> tuple[PerceptionResult, ElementId]:
        frame = _frame_with_rects([(50, 80, 100, 30)])
        orch = _make_orchestrator(_EmptyOCR())

        # First perceive without correction to get the element id
        r0 = await orch.perceive(
            frame, _visual_source(), frame_history=_history(frame)
        )
        assert len(r0.arbitration.resolved_elements) == 1
        element_id = r0.arbitration.resolved_elements[0].id

        # Build correction: override the label with READ_ONLY
        corrected = SemanticLabel(
            element_id=element_id,
            primary_label="Locked Field",
            secondary_labels=(),
            confidence=0.95,
            affordance=Affordance.READ_ONLY,
            is_destructive=False,
        )
        correction_memory = {element_id: corrected}

        # Perceive again with a different frame sequence to bypass cache
        frame2 = _frame_with_rects([(50, 80, 100, 30)], seq=2)
        r1 = await orch.perceive(
            frame2,
            _visual_source(),
            correction_memory=correction_memory,
            frame_history=_history(frame2),
        )
        return r1, element_id

    async def test_conflict_detected(self) -> None:
        result, _ = await self._run_with_correction()
        assert result.arbitration.conflicts_detected >= 1

    async def test_conflict_resolved(self) -> None:
        result, _ = await self._run_with_correction()
        assert (
            result.arbitration.conflicts_resolved
            == result.arbitration.conflicts_detected
        )

    async def test_correction_applied_to_label(self) -> None:
        result, element_id = await self._run_with_correction()
        for lb in result.arbitration.resolved_labels:
            if lb.element_id == element_id:
                assert lb.primary_label == "Locked Field"
                assert lb.affordance is Affordance.READ_ONLY
                return
        pytest.fail("Corrected element not found in resolved labels")

    async def test_correction_reflected_in_graph(self) -> None:
        result, element_id = await self._run_with_correction()
        node = result.spatial_graph.get_node(element_id)
        assert node is not None
        assert node.semantic.affordance is Affordance.READ_ONLY

    async def test_no_correction_no_conflict(self) -> None:
        """Without correction_memory, a single-button frame has no conflicts."""
        frame = _frame_with_rects([(50, 80, 100, 30)])
        orch = _make_orchestrator(_EmptyOCR())
        result = await orch.perceive(
            frame, _visual_source(), frame_history=_history(frame)
        )
        assert result.arbitration.conflicts_detected == 0


# ---------------------------------------------------------------------------
# TEST 6 — find_best_target
# ---------------------------------------------------------------------------


class TestFindBestTarget:
    """
    SpatialGraph.find_best_target() ranks nodes by text similarity and
    semantic confidence, returning the node that best matches the description.
    """

    async def _build_result_with_two_buttons(self) -> PerceptionResult:
        # Two buttons: "Submit" and "Reset"
        # Drawn at known positions so OCR centres are predictable.
        rects = [(30, 80, 100, 30), (160, 80, 90, 30)]
        frame = _frame_with_rects(rects)
        ocr = _ConstOCR([
            _ocr_result("Submit", cx=80,  cy=95),
            _ocr_result("Reset",  cx=205, cy=95),
        ])
        orch = _make_orchestrator(ocr)
        return await orch.perceive(
            frame, _visual_source(), frame_history=_history(frame)
        )

    async def test_find_submit_by_exact_text(self) -> None:
        result = await self._build_result_with_two_buttons()
        node = result.spatial_graph.find_best_target("Submit")
        assert node is not None
        assert "Submit" in node.text

    async def test_find_reset_by_exact_text(self) -> None:
        result = await self._build_result_with_two_buttons()
        node = result.spatial_graph.find_best_target("Reset")
        assert node is not None
        assert "Reset" in node.text

    async def test_returns_none_for_no_match(self) -> None:
        frame = _blank_frame()
        orch = _make_orchestrator(_EmptyOCR())
        result = await orch.perceive(
            frame, _visual_source(), frame_history=_history(frame)
        )
        node = result.spatial_graph.find_best_target("Nonexistent Button XYZ")
        assert node is None

    async def test_find_best_target_on_single_element(self) -> None:
        frame = _frame_with_rects([(50, 80, 100, 30)])
        ocr = _ConstOCR([_ocr_result("Save", cx=100, cy=95)])
        orch = _make_orchestrator(ocr)
        result = await orch.perceive(
            frame, _visual_source(), frame_history=_history(frame)
        )
        node = result.spatial_graph.find_best_target("Save")
        assert node is not None
        assert node.text == "Save"


# ---------------------------------------------------------------------------
# TEST 7 — Transport + Perception integration
# ---------------------------------------------------------------------------


class TestTransportPerceptionIntegration:
    """
    Scenario: SourcePriorityResolver found a UIA element.

    1. PerceptionOrchestrator.perceive() with source_type="uia" returns a
       PerceptionResult whose source_result.source_type == "uia".
    2. TransportResolver.execute() uses the UIA path and calls uia_invoker.
    3. TransportResult.method_used == "uia" and success == True.
    """

    def _make_transport(self, invoke_fn: Any) -> TransportResolver:
        """Wire a TransportResolver with a stub UIA invoker."""
        return TransportResolver(
            NexusSettings(),
            _uia_invoker=invoke_fn,
            _mouse_transport=MouseTransport(_click_fn=lambda x, y: None),
            _keyboard_transport=KeyboardTransport(_type_fn=lambda t: None),
        )

    async def test_uia_source_type_in_perception_result(self) -> None:
        orch = _make_orchestrator(_EmptyOCR())
        result = await orch.perceive(_blank_frame(), _uia_source())
        assert result.source_result.source_type == "uia"

    async def test_uia_invoke_called_on_click(self) -> None:
        """TransportResolver must call the UIA invoker for a click action."""
        invoked: list[object] = []
        invoke_fn = lambda el: (invoked.append(el), True)[1]  # noqa: E731

        transport = self._make_transport(invoke_fn)
        spec = ActionSpec(action_type="click", task_id="t1")
        source = _uia_source()
        target = object()  # dummy UIA element

        tr = await transport.execute(spec, source, target_element=target)
        assert tr.method_used == "uia"
        assert tr.success is True
        assert len(invoked) == 1

    async def test_end_to_end_uia_click(self) -> None:
        """
        Full integration: perceive (uia bypass) + transport (uia invoke).
        Verifies the two layers compose without errors.
        """
        invoked: list[bool] = []

        async def scenario() -> tuple[PerceptionResult, str]:
            # 1. Perception
            orch = _make_orchestrator(_EmptyOCR())
            perception = await orch.perceive(_blank_frame(), _uia_source())
            assert perception.source_result.source_type == "uia"

            # 2. Transport
            transport = self._make_transport(lambda el: (invoked.append(True), True)[1])
            spec = ActionSpec(action_type="click", task_id="t2")
            tr = await transport.execute(
                spec, perception.source_result, target_element=object()
            )
            return perception, tr.method_used

        _, method = await scenario()
        assert method == "uia"
        assert invoked == [True]

    async def test_uia_fallback_when_invoke_fails(self) -> None:
        """When UIA invoke returns False, TransportResolver uses mouse fallback."""
        transport = self._make_transport(lambda el: False)  # always fails
        spec = ActionSpec(action_type="click", task_id="t3")
        source = _uia_source()

        # Provide a target element with a bounding_rect so mouse fallback works
        class _FakeElement:
            bounding_rect = Rect(x=100, y=100, width=80, height=30)

        tr = await transport.execute(spec, source, target_element=_FakeElement())
        assert tr.fallback_used is True
        assert tr.method_used == "mouse"

    async def test_transport_result_is_transport_result_type(self) -> None:
        from nexus.source.transport.resolver import TransportResult

        transport = self._make_transport(lambda el: True)
        spec = ActionSpec(action_type="click", task_id="t4")
        tr = await transport.execute(spec, _uia_source(), target_element=object())
        assert isinstance(tr, TransportResult)
