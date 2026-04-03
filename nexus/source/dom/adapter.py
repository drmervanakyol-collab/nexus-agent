"""
nexus/source/dom/adapter.py
Chrome DevTools Protocol (CDP) adapter for Nexus Agent DOM interaction.

Architecture
------------
DOMAdapter wraps CDP over WebSocket behind a thin async Python layer.
A pluggable ``_session_factory`` is accepted at construction so that unit
tests can inject a mock WebSocket without touching a real browser.

The factory signature is::

    _session_factory(port: int) -> AsyncContextManager[ws]

where ``ws`` exposes ``send(str)`` and ``recv() -> str`` coroutines matching
the ``websockets`` library interface.

Action methods (click / type_text / focus / clear) follow the rule:
  - Return True on success.
  - Return False on connection failure, CDP error, or any exception.
  - Never raise.

CDP command IDs
---------------
Each DOMAdapter instance maintains a monotonically increasing ``_cmd_id``
counter.  This makes IDs predictable (starting at 1) for unit tests.

Key CDP methods used
--------------------
Runtime.evaluate          — run JS to collect element info / get page props
Input.dispatchMouseEvent  — synthetic mouse click at element centre
Input.dispatchKeyEvent    — synthetic keystroke (keyDown / keyUp per char)
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from nexus.core.settings import NexusSettings
from nexus.core.types import Rect
from nexus.infra.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# DOMElement
# ---------------------------------------------------------------------------


@dataclass
class DOMElement:
    """
    A snapshot of a DOM element retrieved via CDP.

    All fields are plain Python types — no live browser references are held,
    making elements safe to pass across coroutines and to serialize.
    """

    tag: str
    id: str
    class_name: str
    text: str
    href: str | None
    bounding_rect: Rect | None
    is_visible: bool
    attributes: dict[str, str]
    node_id: int  # positional index in the JS query result; used for CDP actions


# ---------------------------------------------------------------------------
# Default session factory (real websockets + CDP)
# ---------------------------------------------------------------------------

# Type alias: callable that accepts a port and returns an async context manager
_SessionFactory = Callable[[int], Any]


@asynccontextmanager
async def _default_session_factory(port: int) -> AsyncIterator[Any]:
    """
    Connect to the first available Chrome DevTools target on *port*.

    Raises on any connection error so that callers can catch and return False.
    """
    import aiohttp  # noqa: PLC0415
    import websockets  # noqa: PLC0415

    async with aiohttp.ClientSession() as http:  # noqa: SIM117
        async with http.get(f"http://localhost:{port}/json") as resp:
            targets: list[dict[str, Any]] = await resp.json()

    if not targets:
        raise RuntimeError(f"No CDP targets on port {port}")

    ws_url: str = targets[0]["webSocketDebuggerUrl"]
    async with websockets.connect(ws_url) as ws:
        yield ws


# ---------------------------------------------------------------------------
# DOMAdapter
# ---------------------------------------------------------------------------


class DOMAdapter:
    """
    Async thin wrapper over Chrome DevTools Protocol for element discovery
    and action dispatch.

    Parameters
    ----------
    settings:
        NexusSettings instance — uses ``source.dom_debug_port``.
    _session_factory:
        Optional async-context-manager factory ``(port) -> AsyncContextManager[ws]``.
        When ``None`` the real ``websockets``/``aiohttp`` stack is used.
        Pass a mock factory in unit tests.
    """

    def __init__(
        self,
        settings: NexusSettings,
        *,
        _session_factory: _SessionFactory | None = None,
    ) -> None:
        self._port: int = settings.source.dom_debug_port
        self._factory: _SessionFactory = (
            _session_factory or _default_session_factory
        )
        self._cmd_id: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._cmd_id += 1
        return self._cmd_id

    async def _send(
        self,
        ws: Any,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Send one CDP command and await the matching response.

        Returns the ``result`` dict on success, ``None`` on CDP error or timeout.
        """
        msg_id = self._next_id()
        payload = json.dumps(
            {"id": msg_id, "method": method, "params": params or {}}
        )
        await ws.send(payload)
        async with asyncio.timeout(5.0):
            while True:
                raw: str = await ws.recv()
                data: dict[str, Any] = json.loads(raw)
                if data.get("id") == msg_id:
                    if "error" in data:
                        _log.debug(
                            "cdp_command_error",
                            method=method,
                            error=data["error"],
                        )
                        return None
                    return data.get("result") or {}

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    async def is_available(self) -> bool:
        """Return True if a CDP session can be established."""
        try:
            async with self._factory(self._port) as ws:
                result = await self._send(ws, "Browser.getVersion")
                return result is not None
        except Exception as exc:
            _log.debug("cdp_availability_check_failed", error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def get_elements(
        self,
        selector: str,
        timeout_ms: int | None = None,
    ) -> list[DOMElement] | None:
        """
        Return all DOM elements matching a CSS *selector*.

        Parameters
        ----------
        selector:
            CSS selector string (e.g. ``"button"``, ``"input[type=text]"``).
        timeout_ms:
            Override default timeout for this call (default 5 000 ms).

        Returns
        -------
        list[DOMElement] on success, ``None`` on failure.
        """
        timeout_s = (timeout_ms if timeout_ms is not None else 5000) / 1000.0
        try:
            async with asyncio.timeout(timeout_s):
                async with self._factory(self._port) as ws:
                    return await self._query_selector_all(ws, selector)
        except Exception as exc:
            _log.debug(
                "cdp_get_elements_failed",
                selector=selector,
                error=str(exc),
            )
            return None

    async def _query_selector_all(
        self, ws: Any, selector: str
    ) -> list[DOMElement]:
        """Run querySelectorAll via Runtime.evaluate and map results."""
        js = (
            "(()=>{"
            f"const sel={json.dumps(selector)};"
            "const elems=Array.from(document.querySelectorAll(sel));"
            "return elems.map((el,i)=>{"
            "const r=el.getBoundingClientRect();"
            "const s=window.getComputedStyle(el);"
            "const vis=r.width>0&&r.height>0"
            "&&s.visibility!=='hidden'&&s.display!=='none';"
            "const attrs={};"
            "for(const a of el.attributes)attrs[a.name]=a.value;"
            "return{tag:el.tagName.toLowerCase(),"
            "id:el.id||'',className:el.className||'',"
            "text:(el.innerText||el.textContent||'').trim().slice(0,200),"
            "href:el.href||null,"
            "x:r.left,y:r.top,width:r.width,height:r.height,"
            "visible:vis,attrs:attrs,idx:i};});"
            "})()"
        )
        result = await self._send(
            ws,
            "Runtime.evaluate",
            {"expression": js, "returnByValue": True},
        )
        if not result:
            return []
        items: list[dict[str, Any]] = result.get("result", {}).get("value") or []
        elements: list[DOMElement] = []
        for item in items:
            rect = Rect(
                int(item.get("x", 0)),
                int(item.get("y", 0)),
                int(item.get("width", 0)),
                int(item.get("height", 0)),
            )
            elements.append(
                DOMElement(
                    tag=str(item.get("tag", "")),
                    id=str(item.get("id", "")),
                    class_name=str(item.get("className", "")),
                    text=str(item.get("text", "")),
                    href=item.get("href"),
                    bounding_rect=rect,
                    is_visible=bool(item.get("visible", False)),
                    attributes=dict(item.get("attrs", {})),
                    node_id=int(item.get("idx", 0)),
                )
            )
        return elements

    async def find_by_text(self, text: str) -> DOMElement | None:
        """Return the first element whose visible text exactly matches *text*."""
        try:
            async with self._factory(self._port) as ws:
                js = (
                    "(()=>{"
                    f"const target={json.dumps(text)};"
                    "const walker=document.createTreeWalker("
                    "document.body,NodeFilter.SHOW_ELEMENT);"
                    "let node;"
                    "while((node=walker.nextNode())){"
                    "const t=(node.innerText||node.textContent||'').trim();"
                    "if(t===target){"
                    "const r=node.getBoundingClientRect();"
                    "const s=window.getComputedStyle(node);"
                    "const vis=r.width>0&&r.height>0"
                    "&&s.visibility!=='hidden'&&s.display!=='none';"
                    "const attrs={};"
                    "for(const a of node.attributes)attrs[a.name]=a.value;"
                    "return{tag:node.tagName.toLowerCase(),"
                    "id:node.id||'',className:node.className||'',"
                    "text:t,href:node.href||null,"
                    "x:r.left,y:r.top,width:r.width,height:r.height,"
                    "visible:vis,attrs:attrs};}}"
                    "return null;})()"
                )
                result = await self._send(
                    ws,
                    "Runtime.evaluate",
                    {"expression": js, "returnByValue": True},
                )
                if not result:
                    return None
                item: dict[str, Any] | None = (
                    result.get("result", {}).get("value")
                )
                if not item:
                    return None
                rect = Rect(
                    int(item.get("x", 0)),
                    int(item.get("y", 0)),
                    int(item.get("width", 0)),
                    int(item.get("height", 0)),
                )
                return DOMElement(
                    tag=str(item.get("tag", "")),
                    id=str(item.get("id", "")),
                    class_name=str(item.get("className", "")),
                    text=str(item.get("text", "")),
                    href=item.get("href"),
                    bounding_rect=rect,
                    is_visible=bool(item.get("visible", False)),
                    attributes=dict(item.get("attrs", {})),
                    node_id=0,
                )
        except Exception as exc:
            _log.debug("cdp_find_by_text_failed", text=text, error=str(exc))
            return None

    async def get_page_title(self) -> str | None:
        """Return the current page ``document.title``, or None on failure."""
        try:
            async with self._factory(self._port) as ws:
                result = await self._send(
                    ws,
                    "Runtime.evaluate",
                    {"expression": "document.title", "returnByValue": True},
                )
                if not result:
                    return None
                value = result.get("result", {}).get("value", "")
                return str(value) if value else None
        except Exception as exc:
            _log.debug("cdp_get_page_title_failed", error=str(exc))
            return None

    async def get_current_url(self) -> str | None:
        """Return ``window.location.href``, or None on failure."""
        try:
            async with self._factory(self._port) as ws:
                result = await self._send(
                    ws,
                    "Runtime.evaluate",
                    {
                        "expression": "window.location.href",
                        "returnByValue": True,
                    },
                )
                if not result:
                    return None
                value = result.get("result", {}).get("value", "")
                return str(value) if value else None
        except Exception as exc:
            _log.debug("cdp_get_current_url_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Action methods
    # ------------------------------------------------------------------

    async def click(self, element: DOMElement) -> bool:
        """
        Click *element* via ``Input.dispatchMouseEvent`` at its centre point.

        Returns True on success, False if the element has no bounding rect,
        on CDP error, or on any connection failure.
        """
        if element.bounding_rect is None:
            return False
        rect = element.bounding_rect
        cx = rect.x + rect.width // 2
        cy = rect.y + rect.height // 2
        try:
            async with self._factory(self._port) as ws:
                for event_type in ("mousePressed", "mouseReleased"):
                    result = await self._send(
                        ws,
                        "Input.dispatchMouseEvent",
                        {
                            "type": event_type,
                            "x": cx,
                            "y": cy,
                            "button": "left",
                            "clickCount": 1,
                        },
                    )
                    if result is None:
                        return False
            _log.debug("cdp_click_ok", tag=element.tag, x=cx, y=cy)
            return True
        except Exception as exc:
            _log.debug("cdp_click_failed", tag=element.tag, error=str(exc))
            return False

    async def type_text(self, element: DOMElement, text: str) -> bool:
        """
        Type *text* into *element* character by character via
        ``Input.dispatchKeyEvent`` (keyDown + keyUp per character).

        Returns True on success, False on any CDP error or connection failure.
        """
        try:
            async with self._factory(self._port) as ws:
                for char in text:
                    for event_type in ("keyDown", "keyUp"):
                        result = await self._send(
                            ws,
                            "Input.dispatchKeyEvent",
                            {
                                "type": event_type,
                                "text": char if event_type == "keyDown" else "",
                                "unmodifiedText": char,
                                "key": char,
                            },
                        )
                        if result is None:
                            return False
            _log.debug(
                "cdp_type_text_ok", tag=element.tag, length=len(text)
            )
            return True
        except Exception as exc:
            _log.debug("cdp_type_text_failed", tag=element.tag, error=str(exc))
            return False

    async def focus(self, element: DOMElement) -> bool:
        """
        Focus *element* by evaluating ``.focus()`` via ``Runtime.evaluate``.

        Returns True on success, False on any CDP error or connection failure.
        """
        js = (
            f"(()=>{{const el=document.querySelectorAll"
            f"({json.dumps(element.tag)})[{element.node_id}];"
            "if(el){el.focus();return true;}return false;}})()"
        )
        try:
            async with self._factory(self._port) as ws:
                result = await self._send(
                    ws,
                    "Runtime.evaluate",
                    {"expression": js, "returnByValue": True},
                )
                if result is None:
                    return False
            _log.debug("cdp_focus_ok", tag=element.tag)
            return True
        except Exception as exc:
            _log.debug("cdp_focus_failed", tag=element.tag, error=str(exc))
            return False

    async def clear(self, element: DOMElement) -> bool:
        """
        Clear *element*'s value and textContent via ``Runtime.evaluate``.

        Returns True on success, False on any CDP error or connection failure.
        """
        js = (
            f"(()=>{{const el=document.querySelectorAll"
            f"({json.dumps(element.tag)})[{element.node_id}];"
            "if(el){el.value='';el.textContent='';return true;}"
            "return false;}})()"
        )
        try:
            async with self._factory(self._port) as ws:
                result = await self._send(
                    ws,
                    "Runtime.evaluate",
                    {"expression": js, "returnByValue": True},
                )
                if result is None:
                    return False
            _log.debug("cdp_clear_ok", tag=element.tag)
            return True
        except Exception as exc:
            _log.debug("cdp_clear_failed", tag=element.tag, error=str(exc))
            return False
