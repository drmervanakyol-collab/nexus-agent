"""
nexus/skills/browser/navigation.py
Browser navigation skills — tab management and dynamic-content waiting.

TabHandler
----------
Manages browser tabs via the CDP Target domain.

  switch_to_tab(index):
    List all "page" targets via Target.getTargets; activate the one at
    the given index.  Returns False when index is out of range.

  open_new_tab(url):
    Create a new browser tab via Target.createTarget.  When url is None
    "about:blank" is used.

DynamicContentWaiter
--------------------
Polls the DOM until a CSS selector produces at least one visible element
or the timeout elapses.

  wait_for_content(selector, timeout_ms, poll_interval_ms):
    Returns True as soon as a visible element appears, False on timeout.
"""
from __future__ import annotations

import asyncio

from nexus.infra.logger import get_logger
from nexus.source.dom.adapter import DOMAdapter

_log = get_logger(__name__)

_DEFAULT_POLL_MS: int = 250
_DEFAULT_TIMEOUT_MS: int = 5_000


class TabHandler:
    """
    Manages browser tabs via CDP Target domain commands.

    Parameters
    ----------
    dom:
        DOMAdapter whose session factory and port are reused for
        Target-domain CDP calls.
    """

    def __init__(self, dom: DOMAdapter) -> None:
        self._dom = dom

    async def switch_to_tab(self, index: int) -> bool:
        """
        Activate the browser tab at position *index* (0-based).

        Enumerates CDP "page" targets in discovery order and calls
        Target.activateTarget on the one at *index*.

        Returns True on success, False when *index* is out of range
        or on any CDP / connection error.
        """
        try:
            async with self._dom._factory(self._dom._port) as ws:
                result = await self._dom._send(ws, "Target.getTargets")
                if result is None:
                    return False
                targets = result.get("targetInfos", [])
                pages = [t for t in targets if t.get("type") == "page"]
                if index >= len(pages):
                    _log.debug(
                        "tab_switch_out_of_range",
                        index=index,
                        total=len(pages),
                    )
                    return False
                target_id: str = pages[index]["targetId"]
                activate = await self._dom._send(
                    ws,
                    "Target.activateTarget",
                    {"targetId": target_id},
                )
                ok = activate is not None
                _log.debug("tab_switch", index=index, target_id=target_id, ok=ok)
                return ok
        except Exception as exc:
            _log.debug("tab_switch_failed", index=index, error=str(exc))
            return False

    async def open_new_tab(self, url: str | None = None) -> bool:
        """
        Open a new browser tab, optionally navigating to *url*.

        Uses CDP Target.createTarget.  Returns True when the new target is
        created successfully, False on any error.
        """
        target_url = url or "about:blank"
        try:
            async with self._dom._factory(self._dom._port) as ws:
                result = await self._dom._send(
                    ws,
                    "Target.createTarget",
                    {"url": target_url},
                )
                ok = result is not None
                _log.debug("new_tab_opened", url=target_url, ok=ok)
                return ok
        except Exception as exc:
            _log.debug("open_new_tab_failed", url=target_url, error=str(exc))
            return False


class DynamicContentWaiter:
    """
    Waits for dynamic content to appear in the DOM.

    Parameters
    ----------
    dom:
        DOMAdapter used for CSS-selector polling.
    """

    def __init__(self, dom: DOMAdapter) -> None:
        self._dom = dom

    async def wait_for_content(
        self,
        selector: str,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        poll_interval_ms: int = _DEFAULT_POLL_MS,
    ) -> bool:
        """
        Poll the DOM until *selector* returns at least one visible element.

        Parameters
        ----------
        selector:
            CSS selector to poll (e.g. ``"#results-table"``, ``".loaded"``).
        timeout_ms:
            Maximum wait time in milliseconds (default 5 000 ms).
        poll_interval_ms:
            Sleep between polls in milliseconds (default 250 ms).

        Returns
        -------
        True when a visible element appears within the timeout, False otherwise.
        """
        deadline = asyncio.get_event_loop().time() + timeout_ms / 1000.0
        interval = poll_interval_ms / 1000.0

        while asyncio.get_event_loop().time() < deadline:
            elements = await self._dom.get_elements(selector)
            if elements and any(e.is_visible for e in elements):
                _log.debug("dynamic_content_found", selector=selector)
                return True
            remaining = deadline - asyncio.get_event_loop().time()
            await asyncio.sleep(min(interval, max(0.0, remaining)))

        _log.debug("dynamic_content_timeout", selector=selector, timeout_ms=timeout_ms)
        return False
