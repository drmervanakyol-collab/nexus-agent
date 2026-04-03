"""
tests/unit/test_dom_adapter.py
Unit tests for nexus/source/dom/adapter.py using mock CDP sessions.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.core.settings import NexusSettings
from nexus.core.types import Rect
from nexus.source.dom.adapter import DOMAdapter, DOMElement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(port: int = 9222) -> NexusSettings:
    return NexusSettings.model_validate({"source": {"dom_debug_port": port}})


def _make_element(
    *,
    tag: str = "button",
    node_id: int = 0,
    bounding_rect: Rect | None = Rect(10, 20, 100, 40),
) -> DOMElement:
    return DOMElement(
        tag=tag,
        id="btn1",
        class_name="primary",
        text="Click me",
        href=None,
        bounding_rect=bounding_rect,
        is_visible=True,
        attributes={},
        node_id=node_id,
    )


class _MockWs:
    """
    Mock WebSocket that echoes back CDP responses for each sent command.

    ``recv`` returns a JSON response whose ``id`` matches the most recently
    sent command, with the ``result`` provided via ``next_result``.
    """

    def __init__(self) -> None:
        self._sent: list[dict[str, Any]] = []
        self._results: list[dict[str, Any] | None] = []
        self.send_calls: list[dict[str, Any]] = []

    def queue_result(self, result: dict[str, Any] | None) -> None:
        """Queue a result dict to return for the next recv call."""
        self._results.append(result)

    async def send(self, data: str) -> None:
        msg = json.loads(data)
        self._sent.append(msg)
        self.send_calls.append(msg)

    async def recv(self) -> str:
        msg = self._sent.pop(0)
        result = self._results.pop(0) if self._results else {}
        if result is None:
            # Simulate CDP error response
            return json.dumps({"id": msg["id"], "error": {"message": "CDP error"}})
        return json.dumps({"id": msg["id"], "result": result})


def _factory_for(ws: _MockWs) -> Any:
    """Return an async-context-manager factory that yields *ws*."""

    @asynccontextmanager
    async def _factory(port: int) -> AsyncIterator[_MockWs]:
        yield ws

    return _factory


def _failing_factory(exc: Exception | None = None) -> Any:
    """Return a factory that always raises on __aenter__."""

    @asynccontextmanager
    async def _factory(port: int) -> AsyncIterator[None]:
        raise exc or ConnectionRefusedError("No browser")
        yield  # noqa: unreachable — needed for generator protocol

    return _factory


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    @pytest.mark.asyncio
    async def test_returns_true_when_session_succeeds(self):
        ws = _MockWs()
        ws.queue_result({"product": "Chrome/120"})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        assert await adapter.is_available() is True

    @pytest.mark.asyncio
    async def test_returns_false_when_factory_raises(self):
        adapter = DOMAdapter(
            _make_settings(), _session_factory=_failing_factory()
        )
        assert await adapter.is_available() is False

    @pytest.mark.asyncio
    async def test_returns_false_when_cdp_returns_error(self):
        ws = _MockWs()
        ws.queue_result(None)  # triggers CDP error response
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        assert await adapter.is_available() is False


# ---------------------------------------------------------------------------
# click() — CDP event dispatch
# ---------------------------------------------------------------------------


class TestClick:
    @pytest.mark.asyncio
    async def test_click_sends_mouse_pressed_and_released(self):
        ws = _MockWs()
        ws.queue_result({})  # mousePressed response
        ws.queue_result({})  # mouseReleased response
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        elem = _make_element()
        result = await adapter.click(elem)

        assert result is True
        methods = [c["method"] for c in ws.send_calls]
        assert methods == [
            "Input.dispatchMouseEvent",
            "Input.dispatchMouseEvent",
        ]

    @pytest.mark.asyncio
    async def test_click_sends_pressed_before_released(self):
        ws = _MockWs()
        ws.queue_result({})
        ws.queue_result({})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        await adapter.click(_make_element())

        types = [c["params"]["type"] for c in ws.send_calls]
        assert types == ["mousePressed", "mouseReleased"]

    @pytest.mark.asyncio
    async def test_click_uses_element_centre_coordinates(self):
        ws = _MockWs()
        ws.queue_result({})
        ws.queue_result({})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        # rect: x=10, y=20, w=100, h=40 → centre (60, 40)
        elem = _make_element(bounding_rect=Rect(10, 20, 100, 40))
        await adapter.click(elem)

        params = ws.send_calls[0]["params"]
        assert params["x"] == 60  # 10 + 100//2
        assert params["y"] == 40  # 20 + 40//2

    @pytest.mark.asyncio
    async def test_click_uses_left_button(self):
        ws = _MockWs()
        ws.queue_result({})
        ws.queue_result({})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        await adapter.click(_make_element())
        assert ws.send_calls[0]["params"]["button"] == "left"

    @pytest.mark.asyncio
    async def test_click_returns_false_when_no_bounding_rect(self):
        ws = _MockWs()
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        elem = _make_element(bounding_rect=None)
        assert await adapter.click(elem) is False

    @pytest.mark.asyncio
    async def test_click_returns_false_when_cdp_error(self):
        ws = _MockWs()
        ws.queue_result(None)  # mousePressed → CDP error
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        assert await adapter.click(_make_element()) is False

    @pytest.mark.asyncio
    async def test_click_returns_false_when_connection_fails(self):
        adapter = DOMAdapter(
            _make_settings(), _session_factory=_failing_factory()
        )
        assert await adapter.click(_make_element()) is False


# ---------------------------------------------------------------------------
# type_text() — character-by-character dispatch
# ---------------------------------------------------------------------------


class TestTypeText:
    @pytest.mark.asyncio
    async def test_type_text_dispatches_key_down_and_up_per_char(self):
        ws = _MockWs()
        text = "hi"
        # 2 chars × 2 events = 4 responses
        for _ in range(4):
            ws.queue_result({})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        result = await adapter.type_text(_make_element(), text)

        assert result is True
        assert all(c["method"] == "Input.dispatchKeyEvent" for c in ws.send_calls)
        event_types = [c["params"]["type"] for c in ws.send_calls]
        assert event_types == ["keyDown", "keyUp", "keyDown", "keyUp"]

    @pytest.mark.asyncio
    async def test_type_text_sends_char_in_keydown(self):
        ws = _MockWs()
        ws.queue_result({})  # keyDown 'a'
        ws.queue_result({})  # keyUp 'a'
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        await adapter.type_text(_make_element(), "a")

        keydown = ws.send_calls[0]["params"]
        keyup = ws.send_calls[1]["params"]
        assert keydown["text"] == "a"
        assert keyup["text"] == ""

    @pytest.mark.asyncio
    async def test_type_text_sends_each_character_separately(self):
        ws = _MockWs()
        text = "abc"
        for _ in range(len(text) * 2):
            ws.queue_result({})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        await adapter.type_text(_make_element(), text)

        chars = [
            c["params"]["unmodifiedText"]
            for c in ws.send_calls
            if c["params"]["type"] == "keyDown"
        ]
        assert chars == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_type_text_returns_false_on_cdp_error(self):
        ws = _MockWs()
        ws.queue_result(None)  # first keyDown → CDP error
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        assert await adapter.type_text(_make_element(), "x") is False

    @pytest.mark.asyncio
    async def test_type_text_returns_false_when_connection_fails(self):
        adapter = DOMAdapter(
            _make_settings(), _session_factory=_failing_factory()
        )
        assert await adapter.type_text(_make_element(), "hello") is False

    @pytest.mark.asyncio
    async def test_type_text_empty_string_returns_true(self):
        ws = _MockWs()
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        # No events are dispatched for empty text
        assert await adapter.type_text(_make_element(), "") is True

    @pytest.mark.asyncio
    async def test_type_text_dispatches_correct_method(self):
        ws = _MockWs()
        ws.queue_result({})
        ws.queue_result({})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        await adapter.type_text(_make_element(), "z")
        assert ws.send_calls[0]["method"] == "Input.dispatchKeyEvent"


# ---------------------------------------------------------------------------
# focus()
# ---------------------------------------------------------------------------


class TestFocus:
    @pytest.mark.asyncio
    async def test_focus_sends_runtime_evaluate(self):
        ws = _MockWs()
        ws.queue_result({"result": {"value": True}})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        result = await adapter.focus(_make_element())

        assert result is True
        assert ws.send_calls[0]["method"] == "Runtime.evaluate"

    @pytest.mark.asyncio
    async def test_focus_returns_false_on_cdp_error(self):
        ws = _MockWs()
        ws.queue_result(None)
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        assert await adapter.focus(_make_element()) is False

    @pytest.mark.asyncio
    async def test_focus_returns_false_when_connection_fails(self):
        adapter = DOMAdapter(
            _make_settings(), _session_factory=_failing_factory()
        )
        assert await adapter.focus(_make_element()) is False


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


class TestClear:
    @pytest.mark.asyncio
    async def test_clear_sends_runtime_evaluate(self):
        ws = _MockWs()
        ws.queue_result({"result": {"value": True}})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        result = await adapter.clear(_make_element())

        assert result is True
        assert ws.send_calls[0]["method"] == "Runtime.evaluate"

    @pytest.mark.asyncio
    async def test_clear_returns_false_on_cdp_error(self):
        ws = _MockWs()
        ws.queue_result(None)
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        assert await adapter.clear(_make_element()) is False

    @pytest.mark.asyncio
    async def test_clear_returns_false_when_connection_fails(self):
        adapter = DOMAdapter(
            _make_settings(), _session_factory=_failing_factory()
        )
        assert await adapter.clear(_make_element()) is False


# ---------------------------------------------------------------------------
# get_page_title / get_current_url
# ---------------------------------------------------------------------------


class TestPageInfo:
    @pytest.mark.asyncio
    async def test_get_page_title_returns_string(self):
        ws = _MockWs()
        ws.queue_result({"result": {"type": "string", "value": "My Page"}})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        title = await adapter.get_page_title()
        assert title == "My Page"

    @pytest.mark.asyncio
    async def test_get_page_title_returns_none_on_failure(self):
        adapter = DOMAdapter(
            _make_settings(), _session_factory=_failing_factory()
        )
        assert await adapter.get_page_title() is None

    @pytest.mark.asyncio
    async def test_get_page_title_returns_none_on_cdp_error(self):
        ws = _MockWs()
        ws.queue_result(None)
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        assert await adapter.get_page_title() is None

    @pytest.mark.asyncio
    async def test_get_current_url_returns_string(self):
        ws = _MockWs()
        ws.queue_result(
            {"result": {"type": "string", "value": "https://example.com"}}
        )
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        url = await adapter.get_current_url()
        assert url == "https://example.com"

    @pytest.mark.asyncio
    async def test_get_current_url_returns_none_on_failure(self):
        adapter = DOMAdapter(
            _make_settings(), _session_factory=_failing_factory()
        )
        assert await adapter.get_current_url() is None

    @pytest.mark.asyncio
    async def test_get_current_url_uses_runtime_evaluate(self):
        ws = _MockWs()
        ws.queue_result({"result": {"value": "https://x.com"}})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        await adapter.get_current_url()
        assert ws.send_calls[0]["method"] == "Runtime.evaluate"


# ---------------------------------------------------------------------------
# get_elements()
# ---------------------------------------------------------------------------


class TestGetElements:
    def _make_js_element(
        self,
        tag: str = "button",
        text: str = "OK",
        idx: int = 0,
    ) -> dict[str, Any]:
        return {
            "tag": tag,
            "id": "",
            "className": "",
            "text": text,
            "href": None,
            "x": 0,
            "y": 0,
            "width": 80,
            "height": 30,
            "visible": True,
            "attrs": {},
            "idx": idx,
        }

    @pytest.mark.asyncio
    async def test_returns_list_of_dom_elements(self):
        ws = _MockWs()
        ws.queue_result(
            {
                "result": {
                    "value": [self._make_js_element("button", "Submit", 0)]
                }
            }
        )
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        result = await adapter.get_elements("button")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].tag == "button"
        assert result[0].text == "Submit"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_matches(self):
        ws = _MockWs()
        ws.queue_result({"result": {"value": []}})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        result = await adapter.get_elements("span.nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_none_when_connection_fails(self):
        adapter = DOMAdapter(
            _make_settings(), _session_factory=_failing_factory()
        )
        assert await adapter.get_elements("button") is None

    @pytest.mark.asyncio
    async def test_maps_bounding_rect_correctly(self):
        ws = _MockWs()
        item = self._make_js_element()
        item.update({"x": 10, "y": 20, "width": 100, "height": 40})
        ws.queue_result({"result": {"value": [item]}})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        result = await adapter.get_elements("div")

        assert result is not None
        assert result[0].bounding_rect == Rect(10, 20, 100, 40)

    @pytest.mark.asyncio
    async def test_maps_visibility_correctly(self):
        ws = _MockWs()
        visible_item = {**self._make_js_element(), "visible": True}
        hidden_item = {**self._make_js_element("p", "hi", 1), "visible": False}
        ws.queue_result({"result": {"value": [visible_item, hidden_item]}})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        result = await adapter.get_elements("*")

        assert result is not None
        assert result[0].is_visible is True
        assert result[1].is_visible is False

    @pytest.mark.asyncio
    async def test_respects_timeout_ms(self):
        """get_elements with very short timeout returns None for slow sessions."""
        import asyncio as _asyncio

        @asynccontextmanager
        async def _slow_factory(port: int) -> AsyncIterator[None]:
            await _asyncio.sleep(10.0)
            yield None  # type: ignore[misc]

        adapter = DOMAdapter(_make_settings(), _session_factory=_slow_factory)
        result = await adapter.get_elements("button", timeout_ms=50)
        assert result is None

    @pytest.mark.asyncio
    async def test_uses_runtime_evaluate(self):
        ws = _MockWs()
        ws.queue_result({"result": {"value": []}})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        await adapter.get_elements("input")
        assert ws.send_calls[0]["method"] == "Runtime.evaluate"


# ---------------------------------------------------------------------------
# find_by_text()
# ---------------------------------------------------------------------------


class TestFindByText:
    @pytest.mark.asyncio
    async def test_find_by_text_returns_element(self):
        ws = _MockWs()
        ws.queue_result(
            {
                "result": {
                    "value": {
                        "tag": "span",
                        "id": "",
                        "className": "",
                        "text": "Hello",
                        "href": None,
                        "x": 0,
                        "y": 0,
                        "width": 50,
                        "height": 20,
                        "visible": True,
                        "attrs": {},
                    }
                }
            }
        )
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        elem = await adapter.find_by_text("Hello")

        assert elem is not None
        assert elem.tag == "span"
        assert elem.text == "Hello"

    @pytest.mark.asyncio
    async def test_find_by_text_returns_none_when_not_found(self):
        ws = _MockWs()
        ws.queue_result({"result": {"value": None}})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        assert await adapter.find_by_text("ghost") is None

    @pytest.mark.asyncio
    async def test_find_by_text_returns_none_on_connection_failure(self):
        adapter = DOMAdapter(
            _make_settings(), _session_factory=_failing_factory()
        )
        assert await adapter.find_by_text("anything") is None

    @pytest.mark.asyncio
    async def test_find_by_text_uses_runtime_evaluate(self):
        ws = _MockWs()
        ws.queue_result({"result": {"value": None}})
        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        await adapter.find_by_text("test")
        assert ws.send_calls[0]["method"] == "Runtime.evaluate"


# ---------------------------------------------------------------------------
# _send — command ID sequencing
# ---------------------------------------------------------------------------


class TestSendCommandId:
    @pytest.mark.asyncio
    async def test_cmd_ids_are_sequential(self):
        """Each CDP command gets a unique incrementing ID."""
        ws = _MockWs()
        # Queue three responses
        for _ in range(3):
            ws.queue_result({})

        adapter = DOMAdapter(_make_settings(), _session_factory=_factory_for(ws))
        async with _factory_for(ws)(adapter._port) as conn:
            await adapter._send(conn, "A.method")
            await adapter._send(conn, "B.method")
            await adapter._send(conn, "C.method")

        ids = [c["id"] for c in ws.send_calls]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)  # all unique
