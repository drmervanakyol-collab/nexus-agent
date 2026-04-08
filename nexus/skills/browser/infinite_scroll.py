"""
nexus/skills/browser/infinite_scroll.py
Infinite-scroll handler — scrolls down until target text appears or max
scroll count is reached.

Architecture
------------
InfiniteScrollHandler.scroll_until_found():

  1. Check immediately via DOMAdapter.find_by_text() — return True if found.
  2. Loop up to max_scrolls times:
       a. Scroll down by one viewport via an injectable _scroll_fn.
       b. Brief settle delay (injectable _sleep_fn) to let new content render.
       c. Re-check DOMAdapter.find_by_text().
  3. Return False when max_scrolls is exhausted.

Scroll implementation
---------------------
The default _scroll_fn uses CDP Runtime.evaluate to run::

    window.scrollBy(0, Math.round(window.innerHeight * 0.8))

This makes the scroll amount proportional to the visible viewport rather than
a hard-coded pixel offset, which works across different screen sizes.  The
CDP call reuses the DOMAdapter's session factory so no extra connection is
needed.  An injectable _scroll_fn replaces this in unit tests.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from nexus.infra.logger import get_logger
from nexus.source.dom.adapter import DOMAdapter

_log = get_logger(__name__)

_SETTLE_DELAY_S: float = 0.4   # default wait after each scroll
_SCROLL_JS: str = "window.scrollBy(0, Math.round(window.innerHeight * 0.8))"


class InfiniteScrollHandler:
    """
    Scrolls down a page incrementally until target text becomes visible.

    Parameters
    ----------
    dom:
        DOMAdapter used for both text search and CDP-based scrolling.
    _scroll_fn:
        Async callable ``() -> None`` that performs one scroll step.
        When None the default CDP ``window.scrollBy`` implementation is used.
        Inject a stub in unit tests.
    _sleep_fn:
        Async callable ``(seconds: float) -> None`` used for the settle
        delay after each scroll.  Defaults to ``asyncio.sleep``.
    """

    def __init__(
        self,
        dom: DOMAdapter,
        *,
        _scroll_fn: Callable[[], Awaitable[None]] | None = None,
        _sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._dom = dom
        self._scroll_fn: Callable[[], Awaitable[None]] = (
            _scroll_fn or self._cdp_scroll
        )
        self._sleep_fn: Callable[[float], Awaitable[None]] = (
            _sleep_fn or asyncio.sleep
        )

    async def scroll_until_found(
        self,
        target_text: str,
        max_scrolls: int = 20,
    ) -> bool:
        """
        Scroll down until *target_text* is found in the DOM or *max_scrolls*
        scroll steps are exhausted.

        Parameters
        ----------
        target_text:
            Exact text to search for via DOMAdapter.find_by_text().
        max_scrolls:
            Maximum number of scroll steps before giving up (default 20).

        Returns
        -------
        True when the text is found, False when max_scrolls is exhausted.
        """
        # Pre-check — text might already be in the initial viewport
        element = await self._dom.find_by_text(target_text)
        if element is not None:
            _log.debug("infinite_scroll_found_immediately", text=target_text)
            return True

        for step in range(max_scrolls):
            await self._scroll_fn()
            await self._sleep_fn(_SETTLE_DELAY_S)
            element = await self._dom.find_by_text(target_text)
            if element is not None:
                _log.debug(
                    "infinite_scroll_found",
                    text=target_text,
                    steps=step + 1,
                )
                return True

        _log.debug(
            "infinite_scroll_max_reached",
            text=target_text,
            max_scrolls=max_scrolls,
        )
        return False

    # ------------------------------------------------------------------
    # Default CDP scroll implementation
    # ------------------------------------------------------------------

    async def _cdp_scroll(self) -> None:
        """Scroll down one viewport using CDP Runtime.evaluate."""
        try:
            async with self._dom._factory(self._dom._port) as ws:
                await self._dom._send(
                    ws,
                    "Runtime.evaluate",
                    {"expression": _SCROLL_JS, "returnByValue": False},
                )
        except Exception as exc:
            _log.debug("cdp_scroll_failed", error=str(exc))
