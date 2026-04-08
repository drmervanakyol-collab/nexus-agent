"""
nexus/skills/browser/cookie_handler.py
Cookie banner handler — DOM-first with visual fallback.

Architecture
------------
CookieBannerHandler attempts to dismiss a cookie/consent banner in two stages:

  Stage 1 — DOM (CDP):
    Query several CSS selectors known to match "accept" buttons.
    Click the first visible match via DOMAdapter.click().

  Stage 2 — Visual fallback:
    Search the PerceptionResult's SpatialGraph for nodes whose text contains
    a localised accept keyword ("Kabul Et", "Accept", "Allow", "Tamam", "OK").
    Click the node centre via MouseTransport.click().

Returns True if the banner was dismissed (a click was delivered), False when
no accept element was found through either path.
"""
from __future__ import annotations

from nexus.infra.logger import get_logger
from nexus.perception.orchestrator import PerceptionResult
from nexus.source.dom.adapter import DOMAdapter, DOMElement
from nexus.source.transport.fallback import MouseTransport

_log = get_logger(__name__)

# CSS selectors tried in order (DOM path)
_ACCEPT_SELECTORS: tuple[str, ...] = (
    "button[id*=accept]",
    "button[class*=accept]",
    "button[id*=cookie]",
    "button[class*=cookie]",
    "[id*=accept-btn]",
    "[data-testid*=accept]",
)

# Text fragments searched in the spatial graph (visual path) — case-insensitive
_ACCEPT_TEXTS: tuple[str, ...] = (
    "Kabul Et",
    "Accept All",
    "Accept",
    "Allow All",
    "Allow",
    "Tamam",
    "OK",
)


class CookieBannerHandler:
    """
    Dismisses cookie/consent banners via DOM-first, visual-fallback strategy.

    Parameters
    ----------
    dom:
        DOMAdapter for CDP-based element discovery and clicking.
    mouse:
        MouseTransport used only when the DOM path fails.
    """

    def __init__(self, dom: DOMAdapter, mouse: MouseTransport) -> None:
        self._dom = dom
        self._mouse = mouse

    async def handle(self, perception: PerceptionResult) -> bool:
        """
        Attempt to click an accept/allow button on a cookie banner.

        Parameters
        ----------
        perception:
            Latest PerceptionResult — its spatial_graph is used for the
            visual fallback when DOM discovery yields nothing.

        Returns
        -------
        True if a button was clicked, False if no accept element was found.
        """
        # ---- Stage 1: DOM path ------------------------------------------
        for selector in _ACCEPT_SELECTORS:
            elements = await self._dom.get_elements(selector)
            if not elements:
                continue
            visible = [e for e in elements if e.is_visible]
            if not visible:
                continue
            target: DOMElement = visible[0]
            ok = await self._dom.click(target)
            if ok:
                _log.debug(
                    "cookie_banner_dom_click",
                    selector=selector,
                    tag=target.tag,
                )
                return True

        # ---- Stage 2: Visual fallback ------------------------------------
        graph = perception.spatial_graph
        for text in _ACCEPT_TEXTS:
            # Exact match first, then fuzzy
            nodes = graph.find_by_text(text, fuzzy=False)
            if not nodes:
                nodes = graph.find_by_text(text, fuzzy=True)
            for node in nodes:
                center = node.element.bounding_box.center()
                ok = await self._mouse.click(center.x, center.y)
                if ok:
                    _log.debug(
                        "cookie_banner_visual_click",
                        text=text,
                        x=center.x,
                        y=center.y,
                    )
                    return True

        _log.debug("cookie_banner_not_found")
        return False
