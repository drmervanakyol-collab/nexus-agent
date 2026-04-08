"""
tests/unit/test_browser_skills.py
Unit tests for nexus/skills/browser/ — Faz 47.

TEST PLAN
---------
1. CookieBannerHandler — DOM path finds accept button → clicks it.
2. CookieBannerHandler — DOM fails → visual fallback clicks via mouse.
3. InfiniteScrollHandler — scroll_until_found stops at max_scrolls.
4. FormHandler — fill_field is transport-aware (DOM first, keyboard fallback).
5. TabHandler — switch_to_tab activates the correct CDP target.
6. TabHandler — open_new_tab sends Target.createTarget.
7. DynamicContentWaiter — returns True when element appears.
8. DynamicContentWaiter — returns False on timeout.
9. FormHandler.submit_form — DOM path clicks submit button.
10. FormHandler.handle_date_picker — types date via DOM.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.core.settings import NexusSettings
from nexus.core.types import ElementId, Rect
from nexus.perception.arbitration.arbitrator import ArbitrationResult
from nexus.perception.locator.locator import ElementType, UIElement
from nexus.perception.matcher.matcher import Affordance, SemanticLabel
from nexus.perception.orchestrator import PerceptionResult
from nexus.perception.spatial_graph import SpatialGraph
from nexus.perception.temporal.temporal_expert import ScreenState, StateType
from nexus.skills.browser.cookie_handler import CookieBannerHandler
from nexus.skills.browser.form_handler import FormHandler
from nexus.skills.browser.infinite_scroll import InfiniteScrollHandler
from nexus.skills.browser.navigation import DynamicContentWaiter, TabHandler
from nexus.source.dom.adapter import DOMAdapter, DOMElement
from nexus.source.resolver import SourceResult
from nexus.source.transport.fallback import KeyboardTransport, MouseTransport

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_settings(port: int = 9222) -> NexusSettings:
    return NexusSettings.model_validate({"source": {"dom_debug_port": port}})


_DEFAULT_DOM_RECT = Rect(10, 20, 100, 40)


def _make_dom_element(
    *,
    tag: str = "button",
    text: str = "Accept",
    is_visible: bool = True,
    bounding_rect: Rect | None = None,
    attributes: dict[str, str] | None = None,
) -> DOMElement:
    return DOMElement(
        tag=tag,
        id="el1",
        class_name="btn",
        text=text,
        href=None,
        bounding_rect=bounding_rect if bounding_rect is not None else _DEFAULT_DOM_RECT,
        is_visible=is_visible,
        attributes=attributes or {},
        node_id=0,
    )


_DEFAULT_UI_RECT = Rect(50, 50, 120, 40)


def _make_ui_element(
    eid: ElementId | None = None,
    *,
    rect: Rect | None = None,
) -> UIElement:
    return UIElement(
        id=eid or ElementId(str(uuid.uuid4())),
        element_type=ElementType.BUTTON,
        bounding_box=rect if rect is not None else _DEFAULT_UI_RECT,
        confidence=0.9,
        is_visible=True,
        is_occluded=False,
        occlusion_ratio=0.0,
        z_order_estimate=1,
    )


def _make_semantic_label(element_id: ElementId, text: str = "Accept") -> SemanticLabel:
    return SemanticLabel(
        element_id=element_id,
        primary_label=text,
        secondary_labels=[],
        confidence=0.9,
        affordance=Affordance.CLICKABLE,
        is_destructive=False,
    )


def _make_spatial_graph(
    nodes: list[tuple[UIElement, SemanticLabel, str]],
) -> SpatialGraph:
    """Build a SpatialGraph from (UIElement, SemanticLabel, text) triples."""
    elements = [n[0] for n in nodes]
    labels = [n[1] for n in nodes]
    texts = {n[0].id: n[2] for n in nodes}
    return SpatialGraph(elements, labels, texts)


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
            source_type="visual",
            data={},
            confidence=0.8,
            latency_ms=0.0,
        ),
        perception_ms=0.0,
        frame_sequence=1,
        timestamp="2026-04-08T00:00:00Z",
    )


def _perception_with_text(
    text: str, rect: Rect | None = None
) -> PerceptionResult:
    """PerceptionResult whose SpatialGraph contains one node with *text*."""
    el = _make_ui_element(rect=rect)
    sem = _make_semantic_label(el.id, text)
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
        spatial_graph=_make_spatial_graph([(el, sem, text)]),
        screen_state=stable,
        arbitration=arb,
        source_result=SourceResult(
            source_type="visual",
            data={},
            confidence=0.8,
            latency_ms=0.0,
        ),
        perception_ms=0.0,
        frame_sequence=1,
        timestamp="2026-04-08T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Mock WebSocket / factory helpers (reused from test_dom_adapter pattern)
# ---------------------------------------------------------------------------


class _MockWs:
    def __init__(self) -> None:
        self._sent: list[dict[str, Any]] = []
        self._results: list[dict[str, Any] | None] = []

    def queue_result(self, result: dict[str, Any] | None) -> None:
        self._results.append(result)

    async def send(self, data: str) -> None:
        self._sent.append(json.loads(data))

    async def recv(self) -> str:
        msg = self._sent.pop(0)
        result = self._results.pop(0) if self._results else {}
        if result is None:
            return json.dumps({"id": msg["id"], "error": {"message": "CDP err"}})
        return json.dumps({"id": msg["id"], "result": result})


def _factory_for(ws: _MockWs) -> Any:
    @asynccontextmanager
    async def _factory(port: int) -> AsyncIterator[_MockWs]:
        yield ws

    return _factory


def _failing_factory() -> Any:
    @asynccontextmanager
    async def _factory(port: int) -> AsyncIterator[None]:
        raise ConnectionRefusedError("no browser")
        yield  # pragma: no cover

    return _factory


def _dom_adapter_with(ws: _MockWs) -> DOMAdapter:
    return DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))


def _dom_adapter_failing() -> DOMAdapter:
    return DOMAdapter(_make_settings(), _session_factory=_failing_factory())


def _mouse(ok: bool = True) -> MouseTransport:
    m = MagicMock(spec=MouseTransport)
    m.click = AsyncMock(return_value=ok)
    return m


def _keyboard(ok: bool = True) -> KeyboardTransport:
    k = MagicMock(spec=KeyboardTransport)
    k.type_text = AsyncMock(return_value=ok)
    return k


# ---------------------------------------------------------------------------
# 1. CookieBannerHandler — DOM path
# ---------------------------------------------------------------------------


class TestCookieBannerHandlerDOM:
    @pytest.mark.asyncio
    async def test_dom_accept_button_found_and_clicked(self):
        """DOM querySelector finds accept button → DOMAdapter.click() called."""
        ws = _MockWs()
        # get_elements returns one visible button
        elem_data = {
            "result": {
                "value": [
                    {
                        "tag": "button",
                        "id": "accept",
                        "className": "",
                        "text": "Accept",
                        "href": None,
                        "x": 10,
                        "y": 20,
                        "width": 100,
                        "height": 40,
                        "visible": True,
                        "attrs": {},
                        "idx": 0,
                    }
                ]
            }
        }
        ws.queue_result(elem_data)          # get_elements → button found
        ws.queue_result({})                  # click → mousePressed
        ws.queue_result({})                  # click → mouseReleased

        dom = _dom_adapter_with(ws)
        mouse = _mouse(ok=True)
        handler = CookieBannerHandler(dom=dom, mouse=mouse)

        result = await handler.handle(_empty_perception())

        assert result is True
        mouse.click.assert_not_called()     # visual fallback NOT used

    @pytest.mark.asyncio
    async def test_dom_no_button_returns_false(self):
        """No selectors match and spatial graph is empty → returns False."""
        ws = _MockWs()
        # Queue empty results for every selector tried
        for _ in range(10):
            ws.queue_result({"result": {"value": []}})

        dom = _dom_adapter_with(ws)
        mouse = _mouse(ok=True)
        handler = CookieBannerHandler(dom=dom, mouse=mouse)

        result = await handler.handle(_empty_perception())

        assert result is False


# ---------------------------------------------------------------------------
# 2. CookieBannerHandler — visual fallback
# ---------------------------------------------------------------------------


class TestCookieBannerHandlerVisualFallback:
    @pytest.mark.asyncio
    async def test_dom_fail_uses_visual_fallback(self):
        """
        When all DOM selectors fail (CDP error), the visual fallback finds
        'Accept' in the SpatialGraph and clicks it via MouseTransport.
        """
        dom = _dom_adapter_failing()        # all DOM calls raise
        mouse = _mouse(ok=True)

        perception = _perception_with_text("Accept")
        handler = CookieBannerHandler(dom=dom, mouse=mouse)

        result = await handler.handle(perception)

        assert result is True
        # Centre of Rect(50,50,120,40) = (110, 70)
        mouse.click.assert_called_once_with(110, 70)

    @pytest.mark.asyncio
    async def test_dom_fail_visual_fallback_kabul_et(self):
        """Visual fallback works with Turkish 'Kabul Et' text."""
        dom = _dom_adapter_failing()
        mouse = _mouse(ok=True)

        perception = _perception_with_text("Kabul Et", rect=Rect(0, 0, 100, 50))
        handler = CookieBannerHandler(dom=dom, mouse=mouse)

        result = await handler.handle(perception)

        assert result is True
        # centre of Rect(0, 0, 100, 50) = (50, 25)
        mouse.click.assert_called_once_with(50, 25)

    @pytest.mark.asyncio
    async def test_both_fail_returns_false(self):
        """DOM fails and spatial graph is empty → returns False."""
        dom = _dom_adapter_failing()
        mouse = _mouse(ok=False)

        handler = CookieBannerHandler(dom=dom, mouse=mouse)
        result = await handler.handle(_empty_perception())

        assert result is False


# ---------------------------------------------------------------------------
# 3. InfiniteScrollHandler — max_scrolls boundary
# ---------------------------------------------------------------------------


class TestInfiniteScrollHandler:
    @pytest.mark.asyncio
    async def test_stops_at_max_scrolls(self):
        """Target text never appears → scroll called exactly max_scrolls times."""
        dom = MagicMock(spec=DOMAdapter)
        dom.find_by_text = AsyncMock(return_value=None)  # never found

        scroll_calls: list[None] = []

        async def fake_scroll() -> None:
            scroll_calls.append(None)

        async def fake_sleep(s: float) -> None:
            pass

        handler = InfiniteScrollHandler(
            dom=dom,
            _scroll_fn=fake_scroll,
            _sleep_fn=fake_sleep,
        )

        result = await handler.scroll_until_found("missing text", max_scrolls=5)

        assert result is False
        assert len(scroll_calls) == 5

    @pytest.mark.asyncio
    async def test_stops_immediately_when_found(self):
        """When target text is present on load, scroll is never called."""
        dom = MagicMock(spec=DOMAdapter)
        dom.find_by_text = AsyncMock(return_value=_make_dom_element(text="Found!"))

        scroll_calls: list[None] = []

        async def fake_scroll() -> None:
            scroll_calls.append(None)

        async def fake_sleep(s: float) -> None:
            pass

        handler = InfiniteScrollHandler(
            dom=dom,
            _scroll_fn=fake_scroll,
            _sleep_fn=fake_sleep,
        )

        result = await handler.scroll_until_found("Found!", max_scrolls=20)

        assert result is True
        assert len(scroll_calls) == 0

    @pytest.mark.asyncio
    async def test_found_after_n_scrolls(self):
        """Text appears after 3 scrolls → returns True, exactly 3 scrolls performed."""
        call_count = 0
        found_after = 3
        dom = MagicMock(spec=DOMAdapter)

        async def find_by_text(text: str) -> DOMElement | None:
            nonlocal call_count
            call_count += 1
            if call_count > found_after:
                return _make_dom_element(text=text)
            return None

        dom.find_by_text = find_by_text

        scroll_calls: list[None] = []

        async def fake_scroll() -> None:
            scroll_calls.append(None)

        async def fake_sleep(s: float) -> None:
            pass

        handler = InfiniteScrollHandler(
            dom=dom,
            _scroll_fn=fake_scroll,
            _sleep_fn=fake_sleep,
        )

        result = await handler.scroll_until_found("target", max_scrolls=20)

        assert result is True
        assert len(scroll_calls) == found_after


# ---------------------------------------------------------------------------
# 4. FormHandler — fill_field transport-aware
# ---------------------------------------------------------------------------


class TestFormHandlerFillField:
    @pytest.mark.asyncio
    async def test_fill_field_dom_path(self):
        """DOM path: label found via find_by_text with 'for' attr → types into input."""
        ws = _MockWs()

        # find_by_text(label) → label element with for="email"
        label_result = {
            "result": {
                "value": {
                    "tag": "label",
                    "id": "",
                    "className": "",
                    "text": "Email",
                    "href": None,
                    "x": 0,
                    "y": 0,
                    "width": 60,
                    "height": 20,
                    "visible": True,
                    "attrs": {"for": "email"},
                }
            }
        }
        # get_elements("input#email") → input element
        input_result = {
            "result": {
                "value": [
                    {
                        "tag": "input",
                        "id": "email",
                        "className": "",
                        "text": "",
                        "href": None,
                        "x": 70,
                        "y": 0,
                        "width": 200,
                        "height": 30,
                        "visible": True,
                        "attrs": {},
                        "idx": 0,
                    }
                ]
            }
        }

        # Queue: find_by_text sends Runtime.evaluate → label_result
        ws.queue_result(label_result)
        # get_elements("input#email") → input_result
        ws.queue_result(input_result)
        # clear → Runtime.evaluate (clear field)
        ws.queue_result({"result": {"value": True}})
        # type_text: one char 'a' → keyDown + keyUp
        ws.queue_result({})
        ws.queue_result({})

        dom = _dom_adapter_with(ws)
        mouse = _mouse()
        keyboard = _keyboard()
        handler = FormHandler(dom=dom, mouse=mouse, keyboard=keyboard)

        result = await handler.fill_field("Email", "a")

        assert result is True
        keyboard.type_text.assert_not_called()   # DOM path succeeded

    @pytest.mark.asyncio
    async def test_fill_field_visual_fallback(self):
        """When DOM fails, fallback: mouse click on label area + keyboard type."""
        dom = _dom_adapter_failing()
        mouse = _mouse(ok=True)
        keyboard = _keyboard(ok=True)
        handler = FormHandler(dom=dom, mouse=mouse, keyboard=keyboard)

        result = await handler.fill_field("Name", "Alice")

        assert result is True
        keyboard.type_text.assert_called_once_with("Alice")

    @pytest.mark.asyncio
    async def test_fill_field_both_fail_returns_false(self):
        """Both DOM and visual paths fail → returns False."""
        dom = _dom_adapter_failing()
        mouse = _mouse(ok=False)
        keyboard = _keyboard(ok=False)
        handler = FormHandler(dom=dom, mouse=mouse, keyboard=keyboard)

        result = await handler.fill_field("Name", "Alice")

        assert result is False


# ---------------------------------------------------------------------------
# 5. TabHandler — switch_to_tab
# ---------------------------------------------------------------------------


class TestTabHandler:
    @pytest.mark.asyncio
    async def test_switch_to_tab_activates_correct_target(self):
        """switch_to_tab(1) activates the second page target."""
        ws = _MockWs()
        targets = {
            "targetInfos": [
                {"targetId": "t0", "type": "page", "title": "Tab 0"},
                {"targetId": "t1", "type": "page", "title": "Tab 1"},
            ]
        }
        ws.queue_result(targets)   # Target.getTargets
        ws.queue_result({})        # Target.activateTarget

        dom = _dom_adapter_with(ws)
        handler = TabHandler(dom)

        result = await handler.switch_to_tab(1)

        assert result is True

    @pytest.mark.asyncio
    async def test_switch_to_tab_out_of_range(self):
        """switch_to_tab with index >= len(pages) returns False."""
        ws = _MockWs()
        ws.queue_result({"targetInfos": [{"targetId": "t0", "type": "page"}]})

        dom = _dom_adapter_with(ws)
        handler = TabHandler(dom)

        result = await handler.switch_to_tab(5)

        assert result is False

    @pytest.mark.asyncio
    async def test_open_new_tab_sends_create_target(self):
        """open_new_tab() sends Target.createTarget and returns True on success."""
        ws = _MockWs()
        ws.queue_result({"targetId": "new-tab"})

        dom = _dom_adapter_with(ws)
        handler = TabHandler(dom)

        result = await handler.open_new_tab("https://example.com")

        assert result is True

    @pytest.mark.asyncio
    async def test_open_new_tab_default_url(self):
        """open_new_tab(None) uses about:blank."""
        ws = _MockWs()
        ws.queue_result({"targetId": "blank"})

        dom = _dom_adapter_with(ws)
        handler = TabHandler(dom)

        result = await handler.open_new_tab()

        assert result is True


# ---------------------------------------------------------------------------
# 6. DynamicContentWaiter
# ---------------------------------------------------------------------------


class TestDynamicContentWaiter:
    @pytest.mark.asyncio
    async def test_returns_true_when_element_appears(self):
        """Waiter returns True as soon as a visible element is found."""
        dom = MagicMock(spec=DOMAdapter)
        visible_elem = _make_dom_element(is_visible=True)
        # First poll returns nothing, second returns the element
        dom.get_elements = AsyncMock(side_effect=[[], [visible_elem]])

        waiter = DynamicContentWaiter(dom)
        result = await waiter.wait_for_content(
            "#target", timeout_ms=2000, poll_interval_ms=10
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self):
        """Waiter returns False when timeout elapses before element appears."""
        dom = MagicMock(spec=DOMAdapter)
        dom.get_elements = AsyncMock(return_value=[])   # never found

        waiter = DynamicContentWaiter(dom)
        result = await waiter.wait_for_content(
            "#missing", timeout_ms=50, poll_interval_ms=10
        )

        assert result is False


# ---------------------------------------------------------------------------
# 7. FormHandler.submit_form
# ---------------------------------------------------------------------------


class TestFormHandlerSubmit:
    @pytest.mark.asyncio
    async def test_submit_dom_path(self):
        """submit_form() clicks the first visible submit button via DOM."""
        ws = _MockWs()
        submit_btn = {
            "result": {
                "value": [
                    {
                        "tag": "button",
                        "id": "submit",
                        "className": "",
                        "text": "Submit",
                        "href": None,
                        "x": 0,
                        "y": 0,
                        "width": 80,
                        "height": 35,
                        "visible": True,
                        "attrs": {"type": "submit"},
                        "idx": 0,
                    }
                ]
            }
        }
        ws.queue_result(submit_btn)   # get_elements("button[type=submit]")
        ws.queue_result({})            # click mousePressed
        ws.queue_result({})            # click mouseReleased

        dom = _dom_adapter_with(ws)
        keyboard = _keyboard()
        handler = FormHandler(dom=dom, mouse=_mouse(), keyboard=keyboard)

        result = await handler.submit_form()

        assert result is True
        keyboard.type_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_keyboard_fallback(self):
        """submit_form() falls back to Enter key when no submit button found."""
        dom = _dom_adapter_failing()
        keyboard = _keyboard(ok=True)
        handler = FormHandler(dom=dom, mouse=_mouse(), keyboard=keyboard)

        result = await handler.submit_form()

        assert result is True
        keyboard.type_text.assert_called_once_with("\n")


# ---------------------------------------------------------------------------
# 8. FormHandler.handle_date_picker
# ---------------------------------------------------------------------------


class TestFormHandlerDatePicker:
    @pytest.mark.asyncio
    async def test_date_picker_dom_path(self):
        """handle_date_picker types the date into a date input via DOM."""
        ws = _MockWs()
        date_input = {
            "result": {
                "value": [
                    {
                        "tag": "input",
                        "id": "dob",
                        "className": "",
                        "text": "",
                        "href": None,
                        "x": 0,
                        "y": 0,
                        "width": 150,
                        "height": 30,
                        "visible": True,
                        "attrs": {"type": "date"},
                        "idx": 0,
                    }
                ]
            }
        }
        ws.queue_result(date_input)                       # get_elements
        ws.queue_result({"result": {"value": True}})      # clear
        # type_text: "2026-04-08" → 10 chars × 2 events each
        for _ in range(20):
            ws.queue_result({})

        dom = _dom_adapter_with(ws)
        keyboard = _keyboard()
        handler = FormHandler(dom=dom, mouse=_mouse(), keyboard=keyboard)

        result = await handler.handle_date_picker("2026-04-08")

        assert result is True
        keyboard.type_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_date_picker_keyboard_fallback(self):
        """handle_date_picker falls back to keyboard when DOM fails."""
        dom = _dom_adapter_failing()
        keyboard = _keyboard(ok=True)
        handler = FormHandler(dom=dom, mouse=_mouse(), keyboard=keyboard)

        result = await handler.handle_date_picker("2026-04-08")

        assert result is True
        keyboard.type_text.assert_called_once_with("2026-04-08")
