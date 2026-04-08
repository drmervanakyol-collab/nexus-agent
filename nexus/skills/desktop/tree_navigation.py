"""
nexus/skills/desktop/tree_navigation.py
Tree view navigator — UIA-first expansion and search in tree controls.

TreeNavigator
-------------
Navigates hierarchical tree controls (Windows TreeView, Explorer-style
panels, file trees) using UIA as the primary transport and visual
perception as the fallback.

expand_node(node_text) -> bool
  Stage 1 — UIA:
    Find an element whose Name matches *node_text*.
    If the element supports ExpandCollapse, call expand().
    If the element supports Invoke (double-click-style), call invoke().
  Stage 2 — Visual fallback:
    Locate the node via the injectable _find_visual_fn and double-click
    it via _double_click_fn.

find_in_tree(target_text) -> UIElement | None
  Stage 1 — UIA:
    Call the injectable _find_tree_item_fn which searches for elements
    with the TreeItem control type (50023) that match *target_text*.
    Converts the result to a visual UIElement stub.
  Stage 2 — Visual fallback:
    Use the SpatialGraph (via injectable _find_visual_node_fn) to locate
    a node matching the text.  Returns its underlying UIElement.

UIA control types
-----------------
  50023 — TreeItem
  50034 — Tree

Injectable callables
--------------------
_find_uia_tree_item_fn : (text: str) -> UIAElement | None
    UIA tree-item lookup.  Default calls UIAAdapter.find_by_name(),
    filtered by control_type == 50023.
_expand_via_uia_fn     : (text: str) -> bool
    Combined find + expand/invoke.  Default uses UIAAdapter.
_find_visual_fn        : (text: str) -> tuple[int, int] | None
    Visual centre coordinates for fallback.
_double_click_fn       : async (x: int, y: int) -> bool
    OS-level double-click for expand fallback.
_find_visual_node_fn   : (text: str) -> UIElement | None
    Visual UIElement lookup for find_in_tree fallback.
"""
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from nexus.core.types import ElementId, Rect
from nexus.infra.logger import get_logger
from nexus.perception.locator.locator import ElementType, UIElement
from nexus.source.uia.adapter import UIAAdapter, UIAElement

_log = get_logger(__name__)

# UIA control type constants for tree controls
_UIA_TREE_ITEM_CONTROL_TYPE: int = 50023


class TreeNavigator:
    """
    Navigates tree-view controls using UIA first, visual fallback second.

    Parameters
    ----------
    uia:
        UIAAdapter for UIA-based element discovery and expand/invoke.
    _expand_via_uia_fn:
        Sync ``(node_text: str) -> bool``.  Default uses UIAAdapter.
    _find_uia_tree_item_fn:
        Sync ``(text: str) -> UIAElement | None``.  Default uses UIAAdapter.
    _find_visual_fn:
        Sync ``(text: str) -> tuple[int, int] | None``.  Visual fallback
        coordinates for expand_node.
    _double_click_fn:
        Async ``(x: int, y: int) -> bool``.  OS-level double-click.
    _find_visual_node_fn:
        Sync ``(text: str) -> UIElement | None``.  Visual fallback for
        find_in_tree.
    """

    def __init__(
        self,
        uia: UIAAdapter,
        *,
        _expand_via_uia_fn: Callable[[str], bool] | None = None,
        _find_uia_tree_item_fn: Callable[[str], UIAElement | None] | None = None,
        _find_visual_fn: Callable[[str], tuple[int, int] | None] | None = None,
        _double_click_fn: Callable[[int, int], Awaitable[bool]] | None = None,
        _find_visual_node_fn: Callable[[str], UIElement | None] | None = None,
    ) -> None:
        self._uia = uia
        self._expand_via_uia: Callable[[str], bool] = (
            _expand_via_uia_fn or self._uia_expand
        )
        self._find_uia_item: Callable[[str], UIAElement | None] = (
            _find_uia_tree_item_fn or self._uia_find_tree_item
        )
        self._find_visual: Callable[[str], tuple[int, int] | None] = (
            _find_visual_fn or (lambda _: None)
        )
        self._double_click: Callable[[int, int], Awaitable[bool]] = (
            _double_click_fn or _noop_click
        )
        self._find_visual_node: Callable[[str], UIElement | None] = (
            _find_visual_node_fn or (lambda _: None)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def expand_node(self, node_text: str) -> bool:
        """
        Expand a tree node identified by *node_text*.

        Stage 1 — UIA: find element by name, expand or invoke.
        Stage 2 — Visual: locate by text, double-click.

        Parameters
        ----------
        node_text:
            The label of the tree node to expand.

        Returns
        -------
        True when expanded (or already expanded), False on failure.
        """
        # ---- Stage 1: UIA -----------------------------------------------
        ok = self._expand_via_uia(node_text)
        if ok:
            _log.debug("expand_node_uia_ok", node=node_text)
            return True

        # ---- Stage 2: Visual fallback ------------------------------------
        coords = self._find_visual(node_text)
        if coords is not None:
            x, y = coords
            ok = await self._double_click(x, y)
            if ok:
                _log.debug(
                    "expand_node_visual_ok", node=node_text, x=x, y=y
                )
                return True

        _log.debug("expand_node_failed", node=node_text)
        return False

    def find_in_tree(self, target_text: str) -> UIElement | None:
        """
        Locate a tree item matching *target_text*.

        Stage 1 — UIA:
          Search for a TreeItem-type UIA element whose Name matches
          *target_text*.  Converts the UIAElement to a UIElement stub.
        Stage 2 — Visual fallback:
          Use the injectable _find_visual_node_fn.

        Parameters
        ----------
        target_text:
            Label of the tree node to locate.

        Returns
        -------
        A UIElement snapshot of the found node, or None.
        """
        # ---- Stage 1: UIA -----------------------------------------------
        uia_elem = self._find_uia_item(target_text)
        if uia_elem is not None:
            element = _uia_to_ui_element(uia_elem)
            _log.debug("find_in_tree_uia_ok", target=target_text)
            return element

        # ---- Stage 2: Visual fallback ------------------------------------
        visual = self._find_visual_node(target_text)
        if visual is not None:
            _log.debug("find_in_tree_visual_ok", target=target_text)
            return visual

        _log.debug("find_in_tree_not_found", target=target_text)
        return None

    # ------------------------------------------------------------------
    # Default UIA implementations
    # ------------------------------------------------------------------

    def _uia_find_tree_item(self, text: str) -> UIAElement | None:
        """
        Find a UIA element by name and verify it is a TreeItem control.
        """
        elem = self._uia.find_by_name(text)
        if elem is None:
            return None
        if elem.control_type != _UIA_TREE_ITEM_CONTROL_TYPE:
            _log.debug(
                "uia_find_tree_item_wrong_type",
                name=text,
                control_type=elem.control_type,
            )
            return None
        return elem

    def _uia_expand(self, node_text: str) -> bool:
        """Find a tree item by name and expand or invoke it."""
        elem = self._uia.find_by_name(node_text)
        if elem is None:
            return False
        if elem.supports_expand_collapse:
            return self._uia.expand(elem)
        if elem.supports_invoke:
            return self._uia.invoke(elem)
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _uia_to_ui_element(uia_elem: UIAElement) -> UIElement:
    """
    Convert a UIAElement snapshot to a visual UIElement stub.

    Uses the UIA bounding rect when available, otherwise a zero-area rect.
    """
    rect = uia_elem.bounding_rect or Rect(0, 0, 0, 0)
    return UIElement(
        id=ElementId(str(uuid.uuid4())),
        element_type=ElementType.UNKNOWN,
        bounding_box=rect,
        confidence=0.9,
        is_visible=uia_elem.is_visible,
        is_occluded=False,
        occlusion_ratio=0.0,
        z_order_estimate=0,
    )


# ---------------------------------------------------------------------------
# No-op stub
# ---------------------------------------------------------------------------


async def _noop_click(_x: int, _y: int) -> bool:
    return False
