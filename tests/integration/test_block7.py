"""
tests/integration/test_block7.py
Blok 7 Integration Tests — Faz 51

Browser-DOM-first and UIA-first skill pipelines, plus the full PDF pipeline.
No real browser, UIA COM stack, file I/O, or OS input is exercised —
all external callables are replaced with lightweight stubs.

TEST 1 — Browser cookie DOM-first
  DOMAdapter.get_elements() returns a visible accept button →
  CookieBannerHandler clicks via DOM.
  Assert: DOM click used, MouseTransport NOT called.

TEST 2 — Cookie banner DOM fail → visual fallback
  DOMAdapter.get_elements() returns [] for every selector →
  PerceptionResult SpatialGraph has one "Accept" node →
  CookieBannerHandler falls through to visual path.
  Assert: MouseTransport.click() called, DOM click NOT called.

TEST 3 — Spreadsheet UIA-first cell navigation
  _find_name_box_fn returns a UIAElement stub →
  UIAAdapter.set_value() returns True →
  go_to_cell("B5") succeeds via UIA path.
  Assert: special_key("f5") NOT called (keyboard fallback skipped).

TEST 4 — PDF full pipeline: text field extraction
  FileAdapter stub yields source_type="pdf_text" with an invoice page →
  PDFReader.read(file_path) returns DocumentContent →
  PDFExtractor.extract_field("Invoice Number") returns correct value.

TEST 5 — PDF OCR fallback
  No file_path provided; current_frame supplied →
  _ocr_frame_fn returns OCR text →
  PDFReader.read(current_frame=...) returns source_type="pdf_ocr".
"""
from __future__ import annotations

import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from nexus.capture.frame import Frame
from nexus.core.hitl_manager import HITLManager
from nexus.core.settings import NexusSettings
from nexus.core.types import ElementId, Rect
from nexus.perception.arbitration.arbitrator import ArbitrationResult
from nexus.perception.locator.locator import ElementType, UIElement
from nexus.perception.matcher.matcher import Affordance, SemanticLabel
from nexus.perception.orchestrator import PerceptionResult
from nexus.perception.spatial_graph import SpatialGraph
from nexus.perception.temporal.temporal_expert import ScreenState, StateType
from nexus.skills.browser.cookie_handler import CookieBannerHandler
from nexus.skills.pdf.extractor import PDFExtractor
from nexus.skills.pdf.reader import PDFReader
from nexus.skills.spreadsheet.navigation import SpreadsheetNavigator
from nexus.source.dom.adapter import DOMAdapter, DOMElement
from nexus.source.file.adapter import DocumentContent, FileAdapter
from nexus.source.resolver import SourceResult
from nexus.source.transport.fallback import MouseTransport
from nexus.source.uia.adapter import UIAAdapter, UIAElement

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_UTC = "2026-04-08T00:00:00Z"
_DEFAULT_RECT = Rect(10, 20, 100, 40)


def _settings() -> NexusSettings:
    return NexusSettings()


def _uia_stub() -> UIAAdapter:
    mock_auto = MagicMock()
    return UIAAdapter(_settings(), _automation_factory=lambda: mock_auto)


def _dom_stub() -> DOMAdapter:
    mock_factory = MagicMock()
    return DOMAdapter(_settings(), _session_factory=mock_factory)


def _dom_element(
    *,
    tag: str = "button",
    text: str = "Accept",
    is_visible: bool = True,
    rect: Rect | None = None,
) -> DOMElement:
    return DOMElement(
        tag=tag,
        id="el-accept",
        class_name="accept-btn",
        text=text,
        href=None,
        bounding_rect=rect or _DEFAULT_RECT,
        is_visible=is_visible,
        attributes={"id": "accept-btn"},
        node_id=1,
    )


def _uia_element(
    *,
    name: str = "Box",
    value: str = "A1",
    supports_value: bool = True,
) -> UIAElement:
    return UIAElement(
        automation_id=name,
        name=name,
        control_type=50004,  # Edit control type
        bounding_rect=_DEFAULT_RECT,
        is_enabled=True,
        is_visible=True,
        value=value,
        supports_value=supports_value,
        _raw=MagicMock(),
    )


def _make_perception_with_text(text: str, rect: Rect | None = None) -> PerceptionResult:
    """PerceptionResult whose SpatialGraph contains one node with *text*."""
    eid = ElementId(str(uuid.uuid4()))
    el = UIElement(
        id=eid,
        element_type=ElementType.BUTTON,
        bounding_box=rect or _DEFAULT_RECT,
        confidence=0.9,
        is_visible=True,
        is_occluded=False,
        occlusion_ratio=0.0,
        z_order_estimate=1,
    )
    sem = SemanticLabel(
        element_id=eid,
        primary_label=text,
        secondary_labels=[],
        confidence=0.9,
        affordance=Affordance.CLICKABLE,
        is_destructive=False,
    )
    graph = SpatialGraph([el], [sem], {eid: text})
    stable = ScreenState(
        state_type=StateType.STABLE,
        confidence=1.0,
        blocks_perception=False,
        reason="stable",
        retry_after_ms=0,
    )
    arb = ArbitrationResult(
        resolved_elements=(),
        resolved_labels=(),
        conflicts_detected=0,
        conflicts_resolved=0,
        temporal_blocked=False,
        overall_confidence=1.0,
    )
    return PerceptionResult(
        spatial_graph=graph,
        screen_state=stable,
        arbitration=arb,
        source_result=SourceResult(
            source_type="visual", data={}, confidence=0.8, latency_ms=0.0
        ),
        perception_ms=0.0,
        frame_sequence=1,
        timestamp=_UTC,
    )


def _empty_perception() -> PerceptionResult:
    stable = ScreenState(
        state_type=StateType.STABLE,
        confidence=1.0,
        blocks_perception=False,
        reason="stable",
        retry_after_ms=0,
    )
    arb = ArbitrationResult(
        resolved_elements=(),
        resolved_labels=(),
        conflicts_detected=0,
        conflicts_resolved=0,
        temporal_blocked=False,
        overall_confidence=1.0,
    )
    return PerceptionResult(
        spatial_graph=SpatialGraph([], [], {}),
        screen_state=stable,
        arbitration=arb,
        source_result=SourceResult(
            source_type="visual", data={}, confidence=0.8, latency_ms=0.0
        ),
        perception_ms=0.0,
        frame_sequence=1,
        timestamp=_UTC,
    )


def _frame() -> Frame:
    data = np.zeros((8, 8, 3), dtype=np.uint8)
    return Frame(
        data=data,
        width=8,
        height=8,
        captured_at_monotonic=time.monotonic(),
        captured_at_utc=_UTC,
        sequence_number=1,
    )


# ---------------------------------------------------------------------------
# TEST 1 — Browser cookie DOM-first
# ---------------------------------------------------------------------------


class TestCookieDOMFirst:
    """
    DOMAdapter returns a visible accept button → CookieBannerHandler
    clicks via the DOM path.  MouseTransport must NOT be called.
    """

    @pytest.mark.asyncio
    async def test_dom_click_used_mouse_not_called(self):
        dom = _dom_stub()
        accept_elem = _dom_element(text="Accept All")

        # DOM returns the accept button on the first matching selector
        dom.get_elements = AsyncMock(return_value=[accept_elem])
        dom.click = AsyncMock(return_value=True)

        mouse_clicks: list[tuple[int, int]] = []
        mouse = MouseTransport(_click_fn=lambda x, y: mouse_clicks.append((x, y)))

        handler = CookieBannerHandler(dom, mouse)
        result = await handler.handle(_empty_perception())

        assert result is True
        dom.click.assert_awaited_once_with(accept_elem)
        assert mouse_clicks == [], "MouseTransport must not be called on DOM success"


# ---------------------------------------------------------------------------
# TEST 2 — Cookie banner DOM fail → visual fallback
# ---------------------------------------------------------------------------


class TestCookieDOMFailVisual:
    """
    DOMAdapter returns [] for all selectors → visual fallback:
    MouseTransport.click() called at the node centre.
    """

    @pytest.mark.asyncio
    async def test_visual_fallback_called_when_dom_fails(self):
        dom = _dom_stub()
        dom.get_elements = AsyncMock(return_value=[])  # no DOM elements

        mouse_clicks: list[tuple[int, int]] = []

        def _click(x: int, y: int) -> None:
            mouse_clicks.append((x, y))

        mouse = MouseTransport(_click_fn=_click)

        # Perception has an "Accept" button node
        rect = Rect(50, 60, 100, 40)  # center = (100, 80)
        perception = _make_perception_with_text("Accept", rect=rect)

        handler = CookieBannerHandler(dom, mouse)
        result = await handler.handle(perception)

        assert result is True
        assert len(mouse_clicks) == 1
        x, y = mouse_clicks[0]
        # center of Rect(50, 60, 100, 40) → (50+100//2, 60+40//2) = (100, 80)
        assert x == 100
        assert y == 80


# ---------------------------------------------------------------------------
# TEST 3 — Spreadsheet UIA-first cell navigation
# ---------------------------------------------------------------------------


class TestSpreadsheetUIAFirst:
    """
    _find_name_box_fn returns a UIAElement → UIAAdapter.set_value returns True →
    go_to_cell() succeeds via UIA path.  F5 keyboard fallback must NOT trigger.
    """

    @pytest.mark.asyncio
    async def test_uia_path_no_keyboard_fallback(self):
        uia = _uia_stub()
        name_box = _uia_element(name="Box", value="A1")

        # UIA set_value succeeds
        uia.set_value = MagicMock(return_value=True)

        special_key_calls: list[str] = []

        async def _special_key(key: str) -> bool:
            special_key_calls.append(key)
            return True

        nav = SpreadsheetNavigator(
            uia,
            _find_name_box_fn=lambda: name_box,
            _special_key_fn=_special_key,
        )
        result = await nav.go_to_cell("B5")

        assert result is True
        uia.set_value.assert_called_once_with(name_box, "B5")
        # UIA succeeded → only "enter" should be called (to confirm), not "f5"
        assert "f5" not in special_key_calls, "Keyboard fallback must not trigger"
        assert "enter" in special_key_calls


# ---------------------------------------------------------------------------
# TEST 4 — PDF full pipeline: text field extraction
# ---------------------------------------------------------------------------


class TestPDFFullPipeline:
    """
    FileAdapter stub returns source_type="pdf_text" →
    PDFReader.read(file_path) passes content through →
    PDFExtractor.extract_field() finds the correct value.
    """

    @pytest.mark.asyncio
    async def test_extract_field_from_text_pdf(self, tmp_path):
        invoice_page = (
            "INVOICE\n"
            "Invoice Number: INV-2026-001\n"
            "Date: 2026-04-08\n"
            "Vendor: Acme Corp\n"
            "Total: 1500.00 USD\n"
        )
        content = DocumentContent(
            source_type="pdf_text",
            pages=[invoice_page],
            tables=[],
            metadata={"page_count": 1},
            extraction_confidence=1.0,
        )

        file_adapter = MagicMock(spec=FileAdapter)
        file_adapter.extract = MagicMock(return_value=content)

        hitl = MagicMock(spec=HITLManager)
        hitl.request = AsyncMock()

        pdf_path = tmp_path / "invoice.pdf"
        pdf_path.touch()

        reader = PDFReader(
            file_adapter,
            hitl,
            _is_encrypted_fn=lambda _: False,
        )
        result = await reader.read(file_path=pdf_path)

        assert result is not None
        assert result.source_type == "pdf_text"
        assert result.pages[0] == invoice_page

        extractor = PDFExtractor()
        invoice_no = extractor.extract_field("Invoice Number", result)
        vendor = extractor.extract_field("Vendor", result)

        assert invoice_no == "INV-2026-001"
        assert vendor == "Acme Corp"
        hitl.request.assert_not_awaited()


# ---------------------------------------------------------------------------
# TEST 5 — PDF OCR fallback
# ---------------------------------------------------------------------------


class TestPDFOCRFallback:
    """
    No file_path supplied; current_frame provided →
    _ocr_frame_fn invoked → DocumentContent.source_type == "pdf_ocr".
    """

    @pytest.mark.asyncio
    async def test_frame_ocr_path_returns_pdf_ocr(self):
        ocr_text = "Scanned page content: Item A  10  100.00"

        file_adapter = MagicMock(spec=FileAdapter)
        hitl = MagicMock(spec=HITLManager)
        hitl.request = AsyncMock()

        reader = PDFReader(
            file_adapter,
            hitl,
            _ocr_frame_fn=lambda _frame: ocr_text,
        )

        frame = _frame()
        result = await reader.read(current_frame=frame)

        assert result is not None
        assert result.source_type == "pdf_ocr"
        assert result.pages == [ocr_text]
        assert result.extraction_confidence == 0.85
        # FileAdapter must NOT have been called
        file_adapter.extract.assert_not_called()
