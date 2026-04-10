"""
nexus/skills/spreadsheet/navigation.py
Spreadsheet navigation — UIA-first cell and sheet navigation.

SpreadsheetNavigator
--------------------
Navigates cells and sheets in a spreadsheet application (Excel) using
UI Automation as the primary transport, falling back to keyboard shortcuts
when UIA is unavailable or returns no element.

go_to_cell(cell_ref)
  Stage 1 — UIA:
    Locate the Name Box (AutomationId "Box" or "NameBox") via UIAAdapter.
    Call ``set_value(name_box, cell_ref)`` then press Enter.
  Stage 2 — Keyboard fallback:
    Press F5 (Go To dialog), type *cell_ref*, press Enter.

get_current_cell() -> str
  Read the Name Box value via UIAAdapter.
  Returns an empty string when UIA is unavailable.

get_sheet_names() -> list[str]
  Enumerate all UIA elements under the Excel window, return the names of
  elements whose control_type is TabItem (50960).

Injectable callables
--------------------
All keyboard operations are provided via three injectable async callables
so tests never touch the real OS input stack:

_hotkey_fn  : async (keys: list[str]) -> bool
_special_key_fn : async (key: str) -> bool
_type_fn    : async (text: str) -> bool
_find_name_box_fn : () -> UIAElement | None   (sync)
_get_sheet_names_fn : () -> list[str]         (sync)
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from nexus.infra.logger import get_logger
from nexus.source.uia.adapter import UIAAdapter, UIAElement

_log = get_logger(__name__)

# Automation IDs for the Excel Name Box (tried in order)
_NAME_BOX_AIDS: tuple[str, ...] = ("Box", "NameBox")

# UIA control type for Excel sheet tab items
_UIA_TAB_ITEM_CONTROL_TYPE: int = 50960


class SpreadsheetNavigator:
    """
    UIA-first cell and sheet navigation for spreadsheet applications.

    Parameters
    ----------
    uia:
        UIAAdapter for element discovery and action dispatch.
    _hotkey_fn:
        Async ``(keys: list[str]) -> bool``.
        Calls the OS-level hotkey transport.  Default: always returns False.
    _special_key_fn:
        Async ``(key: str) -> bool``.
        Calls the OS-level special-key transport.  Default: always False.
    _type_fn:
        Async ``(text: str) -> bool``.
        Calls the OS-level text-input transport.  Default: always False.
    _find_name_box_fn:
        Sync ``() -> UIAElement | None``.  Default uses UIAAdapter.
    _get_sheet_names_fn:
        Sync ``() -> list[str]``.  Default scans UIA tree for TabItems.
    """

    def __init__(
        self,
        uia: UIAAdapter,
        *,
        _hotkey_fn: Callable[[list[str]], Awaitable[bool]] | None = None,
        _special_key_fn: Callable[[str], Awaitable[bool]] | None = None,
        _type_fn: Callable[[str], Awaitable[bool]] | None = None,
        _find_name_box_fn: Callable[[], UIAElement | None] | None = None,
        _get_sheet_names_fn: Callable[[], list[str]] | None = None,
    ) -> None:
        self._uia = uia
        self._hotkey: Callable[[list[str]], Awaitable[bool]] = (
            _hotkey_fn or _noop_bool_list
        )
        self._special_key: Callable[[str], Awaitable[bool]] = (
            _special_key_fn or _noop_bool
        )
        self._type: Callable[[str], Awaitable[bool]] = (
            _type_fn or _noop_bool
        )
        self._find_name_box: Callable[[], UIAElement | None] = (
            _find_name_box_fn or self._uia_find_name_box
        )
        self._get_sheet_names_impl: Callable[[], list[str]] = (
            _get_sheet_names_fn or self._uia_get_sheet_names
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def go_to_cell(self, cell_ref: str) -> bool:
        """
        Navigate to *cell_ref* (e.g. ``"B5"``, ``"Sheet2!A1"``).

        Stage 1 — UIA (preferred):
          Finds the Name Box, sets its value to *cell_ref*, then presses
          Enter to confirm.

        Stage 2 — Keyboard fallback:
          Sends F5 to open the Go To dialog, types *cell_ref*, presses
          Enter.

        Returns True when navigation was attempted successfully, False on
        total failure.

        Parameters
        ----------
        cell_ref:
            Excel cell reference (e.g. ``"B5"``, ``"Sheet2!C10"``).
        """
        # ---- Stage 1: UIA -----------------------------------------------
        name_box = self._find_name_box()
        if name_box is not None:
            ok = self._uia.set_value(name_box, cell_ref)
            if ok:
                await self._special_key("enter")
                _log.debug("go_to_cell_uia_ok", cell_ref=cell_ref)
                return True

        # ---- Stage 2: Keyboard fallback ---------------------------------
        await self._special_key("f5")
        await self._type(cell_ref)
        ok = await self._special_key("enter")
        _log.debug("go_to_cell_keyboard", cell_ref=cell_ref, ok=ok)
        return ok

    def get_current_cell(self) -> str:
        """
        Return the active cell reference from the Name Box.

        Returns an empty string when the Name Box cannot be located or its
        value is blank.
        """
        name_box = self._find_name_box()
        if name_box is None:
            _log.debug("get_current_cell_no_name_box")
            return ""
        val = self._uia.get_value(name_box)
        result = val or ""
        _log.debug("get_current_cell", value=result)
        return result

    def get_sheet_names(self) -> list[str]:
        """
        Return the names of all sheets in the active workbook.

        Uses UIA to enumerate TabItem control-type elements whose names
        are non-empty.  Returns an empty list when UIA is unavailable.
        """
        names = self._get_sheet_names_impl()
        _log.debug("get_sheet_names", count=len(names))
        return names

    # ------------------------------------------------------------------
    # Default UIA implementations
    # ------------------------------------------------------------------

    def _uia_find_name_box(self) -> UIAElement | None:
        """Locate the Excel Name Box element via UIAAdapter."""
        for aid in _NAME_BOX_AIDS:
            elem = self._uia.find_by_automation_id(aid)
            if elem is not None:
                return elem
        return self._uia.find_by_name("Name Box")

    def _uia_get_sheet_names(self) -> list[str]:
        """
        Return sheet names by scanning UIA elements for TabItem controls.

        Uses HWND 0 to search from the root element.  Returns an empty
        list when the tree walk fails.
        """
        elements = self._uia.get_elements(0)
        if not elements:
            return []
        return [
            e.name
            for e in elements
            if e.control_type == _UIA_TAB_ITEM_CONTROL_TYPE and e.name
        ]


# ---------------------------------------------------------------------------
# No-op async stubs (used when no keyboard callables are injected)
# ---------------------------------------------------------------------------


async def _noop_bool(_key: str) -> bool:
    return False


async def _noop_bool_list(_keys: list[str]) -> bool:
    return False
