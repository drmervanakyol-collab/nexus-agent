"""
nexus/skills/pdf/navigator.py
PDF navigation skill — page jumping and in-document text search.

PDFNavigator
------------
Navigates a PDF viewer (any application) using keyboard shortcuts and
optionally searches document content programmatically.

scroll_to_page(page_number)
  Sends a keyboard sequence to navigate to a specific page number.
  Default sequence: Ctrl+G (opens "Go to page" dialog in most PDF
  viewers) → type page number → Enter.
  Injectable _goto_page_fn replaces the whole sequence for tests.

find_text(search_text, content) -> tuple[int, int] | None
  Two modes:

  Content mode (content given):
    Search *search_text* in the DocumentContent pages list.
    Returns ``(page_index, char_offset)`` of the first occurrence,
    or None when not found.

  Viewer mode (content is None):
    Send Ctrl+F to open the viewer's search bar, type *search_text*,
    press Enter.  Returns ``(0, 0)`` when the sequence was delivered
    successfully, None on failure.  (Screen-position lookup is not
    performed in this mode; the viewer itself handles highlighting.)

Injectable callables
--------------------
_hotkey_fn      : async (keys: list[str]) -> bool
_special_key_fn : async (key: str) -> bool
_type_fn        : async (text: str) -> bool
_goto_page_fn   : async (page_number: int) -> bool
    Overrides the entire scroll_to_page implementation when provided.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from nexus.infra.logger import get_logger
from nexus.source.file.adapter import DocumentContent

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# PDFNavigator
# ---------------------------------------------------------------------------


class PDFNavigator:
    """
    Keyboard-driven PDF viewer navigation.

    Parameters
    ----------
    _hotkey_fn:
        Async ``(keys: list[str]) -> bool``.
    _special_key_fn:
        Async ``(key: str) -> bool``.
    _type_fn:
        Async ``(text: str) -> bool``.
    _goto_page_fn:
        Optional async ``(page_number: int) -> bool``.
        When provided, replaces the default Ctrl+G + type + Enter sequence.
    """

    def __init__(
        self,
        *,
        _hotkey_fn: Callable[[list[str]], Awaitable[bool]] | None = None,
        _special_key_fn: Callable[[str], Awaitable[bool]] | None = None,
        _type_fn: Callable[[str], Awaitable[bool]] | None = None,
        _goto_page_fn: Callable[[int], Awaitable[bool]] | None = None,
    ) -> None:
        self._hotkey: Callable[[list[str]], Awaitable[bool]] = (
            _hotkey_fn or _noop_bool_list
        )
        self._special_key: Callable[[str], Awaitable[bool]] = (
            _special_key_fn or _noop_bool
        )
        self._type: Callable[[str], Awaitable[bool]] = (
            _type_fn or _noop_bool
        )
        self._goto_page: Callable[[int], Awaitable[bool]] | None = _goto_page_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scroll_to_page(self, page_number: int) -> bool:
        """
        Navigate the PDF viewer to *page_number* (1-based).

        Uses an injectable _goto_page_fn when provided; otherwise sends
        Ctrl+G → type page number → Enter.

        Parameters
        ----------
        page_number:
            Target page number (1-based, as displayed in the viewer).

        Returns
        -------
        True when the key sequence was delivered, False on failure.
        """
        if self._goto_page is not None:
            ok = await self._goto_page(page_number)
            _log.debug("scroll_to_page_injected", page=page_number, ok=ok)
            return ok

        # Default: Ctrl+G → type page number → Enter
        await self._hotkey(["ctrl", "g"])
        await self._type(str(page_number))
        ok = await self._special_key("enter")
        _log.debug("scroll_to_page_keyboard", page=page_number, ok=ok)
        return ok

    def find_text(
        self,
        search_text: str,
        content: DocumentContent | None = None,
    ) -> tuple[int, int] | None:
        """
        Find *search_text* in content or signal a viewer search.

        Content mode (``content`` provided):
          Scans ``content.pages`` for *search_text* (case-sensitive substring).
          Returns ``(page_index, char_offset)`` of the first match or None.

        Viewer mode (``content`` is None):
          This method is intentionally synchronous — the caller should
          use ``find_text_in_viewer()`` for the async keyboard path.
          Returns None when no content is available.

        Parameters
        ----------
        search_text:
            The text string to locate.
        content:
            DocumentContent to search, or None for viewer mode.
        """
        if content is not None:
            return _search_in_content(search_text, content)

        _log.debug("find_text_no_content", search_text=search_text)
        return None

    async def find_text_in_viewer(self, search_text: str) -> tuple[int, int] | None:
        """
        Open the viewer's search bar via Ctrl+F and type *search_text*.

        Returns ``(0, 0)`` when the key sequence was delivered successfully,
        None on failure.

        Parameters
        ----------
        search_text:
            The text to search for in the active PDF viewer.
        """
        await self._hotkey(["ctrl", "f"])
        await self._type(search_text)
        ok = await self._special_key("enter")
        if ok:
            _log.debug("find_text_viewer_ok", search_text=search_text)
            return (0, 0)
        _log.debug("find_text_viewer_failed", search_text=search_text)
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _search_in_content(
    search_text: str,
    content: DocumentContent,
) -> tuple[int, int] | None:
    """Return (page_index, char_offset) of the first match, or None."""
    for page_idx, page_text in enumerate(content.pages):
        offset = page_text.find(search_text)
        if offset != -1:
            _log.debug(
                "find_text_content_found",
                page=page_idx,
                offset=offset,
            )
            return (page_idx, offset)
    _log.debug("find_text_content_not_found", search_text=search_text)
    return None


# ---------------------------------------------------------------------------
# No-op stubs
# ---------------------------------------------------------------------------


async def _noop_bool(_key: str) -> bool:
    return False


async def _noop_bool_list(_keys: list[str]) -> bool:
    return False
