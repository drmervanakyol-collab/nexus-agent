"""
nexus/skills/desktop/multi_panel.py
Multi-panel handler — detect and switch between application panels.

MultiPanelHandler
-----------------
Locates and activates panels in multi-pane desktop applications (e.g.
file explorers, IDEs, enterprise ERP forms) using UIA as the primary
transport and visual perception as the fallback.

find_active_panel(perception) -> UIElement | None
  Scans the SpatialGraph for nodes whose element_type is PANEL or
  CONTAINER.  Returns the node with the highest z_order_estimate (the
  topmost / focused panel), or None when no panel nodes exist.

switch_panel(description) -> bool
  Stage 1 — UIA:
    Find an element whose Name matches *description* via UIAAdapter and
    call Invoke on it.
  Stage 2 — Visual fallback:
    Use the injectable _find_visual_fn to locate the panel by text in the
    SpatialGraph and click its centre via MouseTransport.

Injectable callables
--------------------
_invoke_by_name_fn : (name: str) -> bool
    UIA path; replaces the real UIAAdapter call in tests.
_find_visual_fn    : (description: str) -> tuple[int, int] | None
    Visual fallback; returns (x, y) centre of the panel, or None.
_click_fn          : async (x: int, y: int) -> bool
    Mouse transport; replaces real MouseTransport.click() in tests.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from nexus.infra.logger import get_logger
from nexus.perception.locator.locator import ElementType, UIElement
from nexus.perception.orchestrator import PerceptionResult
from nexus.source.uia.adapter import UIAAdapter

_log = get_logger(__name__)

# Visual perception element types treated as panels
_PANEL_TYPES: frozenset[ElementType] = frozenset({
    ElementType.PANEL,
    ElementType.CONTAINER,
    ElementType.TAB,
})


class MultiPanelHandler:
    """
    Detects and switches between application panels.

    Parameters
    ----------
    uia:
        UIAAdapter for element discovery and action dispatch.
    _invoke_by_name_fn:
        Sync ``(name: str) -> bool``.  Default calls UIAAdapter.find_by_name
        + UIAAdapter.invoke.
    _find_visual_fn:
        Sync ``(description: str) -> tuple[int, int] | None``.
        Returns the centre coordinates of the target panel, or None.
    _click_fn:
        Async ``(x: int, y: int) -> bool``.  OS-level mouse click.
    """

    def __init__(
        self,
        uia: UIAAdapter,
        *,
        _invoke_by_name_fn: Callable[[str], bool] | None = None,
        _find_visual_fn: Callable[[str], tuple[int, int] | None] | None = None,
        _click_fn: Callable[[int, int], Awaitable[bool]] | None = None,
    ) -> None:
        self._uia = uia
        self._invoke_by_name: Callable[[str], bool] = (
            _invoke_by_name_fn or self._uia_invoke_by_name
        )
        self._find_visual: Callable[[str], tuple[int, int] | None] = (
            _find_visual_fn or (lambda _: None)
        )
        self._click: Callable[[int, int], Awaitable[bool]] = (
            _click_fn or _noop_click
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_active_panel(
        self, perception: PerceptionResult
    ) -> UIElement | None:
        """
        Return the topmost panel element from *perception*'s SpatialGraph.

        Scans for nodes whose element_type is PANEL, CONTAINER, or TAB.
        Selects the node with the highest z_order_estimate.

        Parameters
        ----------
        perception:
            Latest PerceptionResult containing a populated SpatialGraph.

        Returns
        -------
        The UIElement of the active panel, or None when no panels found.
        """
        best: UIElement | None = None
        best_z: int = -1

        for node in perception.spatial_graph.nodes:
            el = node.element
            if el.element_type in _PANEL_TYPES and el.z_order_estimate > best_z:
                best_z = el.z_order_estimate
                best = el

        if best is not None:
            _log.debug(
                "find_active_panel_ok",
                type=best.element_type.name,
                z=best_z,
            )
        else:
            _log.debug("find_active_panel_none")
        return best

    async def switch_panel(self, description: str) -> bool:
        """
        Activate the panel described by *description*.

        Stage 1 — UIA: invoke by name.
        Stage 2 — Visual: locate by text, click centre.

        Parameters
        ----------
        description:
            Human-readable label or name of the target panel.

        Returns
        -------
        True when the panel was activated, False on total failure.
        """
        # ---- Stage 1: UIA -----------------------------------------------
        ok = self._invoke_by_name(description)
        if ok:
            _log.debug("switch_panel_uia_ok", description=description)
            return True

        # ---- Stage 2: Visual fallback ------------------------------------
        coords = self._find_visual(description)
        if coords is not None:
            x, y = coords
            ok = await self._click(x, y)
            if ok:
                _log.debug(
                    "switch_panel_visual_ok",
                    description=description,
                    x=x,
                    y=y,
                )
                return True

        _log.debug("switch_panel_failed", description=description)
        return False

    # ------------------------------------------------------------------
    # Default UIA implementation
    # ------------------------------------------------------------------

    def _uia_invoke_by_name(self, name: str) -> bool:
        """Find an element by name via UIAAdapter and invoke it."""
        elem = self._uia.find_by_name(name)
        if elem is None:
            return False
        return self._uia.invoke(elem)


# ---------------------------------------------------------------------------
# No-op stub
# ---------------------------------------------------------------------------


async def _noop_click(_x: int, _y: int) -> bool:
    return False
