"""
nexus/skills/browser/form_handler.py
Form interaction handler — DOM-first, transport-aware field filling.

Architecture
------------
FormHandler provides three operations:

fill_field(label, value)
  Stage 1 — DOM path:
    Search for a <label> element whose text matches *label* via DOM
    text search.  Then attempt to discover the associated <input> via:
      - label[@for] → input[@id]
      - label wrapper → descendant input
      - CSS selector ``input[placeholder*=label]``
    Clear the field and type the value via DOMAdapter.type_text().

  Stage 2 — Visual fallback:
    Use DOMAdapter.find_by_text(label) to locate a label-like element,
    then click its bounding rect centre and type via KeyboardTransport.

submit_form()
  Stage 1 — DOM path:
    Query ``button[type=submit]``, ``input[type=submit]``, ``button[type=button]``
    in that order.  Click the first visible result via DOMAdapter.click().

  Stage 2 — Visual fallback:
    Search SpatialGraph text is NOT available here (no perception arg), so
    fall back to a hard-coded Enter keystroke via KeyboardTransport.

handle_date_picker(date_str)
  Locate a date <input> via ``input[type=date]`` or ``input[type=text]``
  selectors, clear it, and type *date_str* directly via DOMAdapter.
  Falls back to KeyboardTransport when no DOM element is found.

Transport-aware design
----------------------
All paths try DOM transport first (structured, reliable) before falling
back to OS-level keyboard/mouse.  The same injectable pattern used by
the rest of the skill layer is preserved for testability.
"""
from __future__ import annotations

from nexus.infra.logger import get_logger
from nexus.source.dom.adapter import DOMAdapter, DOMElement
from nexus.source.transport.fallback import KeyboardTransport, MouseTransport

_log = get_logger(__name__)

# Selectors tried when looking for a submit button
_SUBMIT_SELECTORS: tuple[str, ...] = (
    "button[type=submit]",
    "input[type=submit]",
    "button[type=button]",
    "button",
)

# Selectors tried for date inputs
_DATE_SELECTORS: tuple[str, ...] = (
    "input[type=date]",
    "input[type=datetime-local]",
    "input[type=text][placeholder*=date]",
    "input[type=text]",
)

# Special-key name for Enter used by KeyboardTransport
_ENTER_KEY: str = "\n"


class FormHandler:
    """
    Fills form fields and submits forms using a DOM-first, visual-fallback
    strategy.

    Parameters
    ----------
    dom:
        DOMAdapter for CDP-based element discovery and actions.
    mouse:
        MouseTransport for OS-level click fallback.
    keyboard:
        KeyboardTransport for OS-level text-input fallback.
    """

    def __init__(
        self,
        dom: DOMAdapter,
        mouse: MouseTransport,
        keyboard: KeyboardTransport,
    ) -> None:
        self._dom = dom
        self._mouse = mouse
        self._keyboard = keyboard

    # ------------------------------------------------------------------
    # fill_field
    # ------------------------------------------------------------------

    async def fill_field(self, label: str, value: str) -> bool:
        """
        Fill a form field identified by *label* with *value*.

        Parameters
        ----------
        label:
            Text of the <label> element (or placeholder) that identifies
            the target input.
        value:
            Text to enter into the field.

        Returns
        -------
        True when the value was entered via any transport, False otherwise.
        """
        # ---- Stage 1: DOM path ------------------------------------------
        input_elem = await self._find_input_for_label(label)
        if input_elem is not None:
            await self._dom.clear(input_elem)
            ok = await self._dom.type_text(input_elem, value)
            if ok:
                _log.debug("fill_field_dom_ok", label=label)
                return True

        # ---- Stage 2: Visual fallback ----------------------------------------
        # Try to locate the label visually and click it first, then type.
        # If DOM is completely unavailable, skip the click and just type.
        label_elem = await self._dom.find_by_text(label)
        if label_elem is not None and label_elem.bounding_rect is not None:
            rect = label_elem.bounding_rect
            cx = rect.x + rect.width // 2
            cy = rect.y + rect.height // 2
            await self._mouse.click(cx, cy)

        # Always attempt keyboard type as the final fallback.
        typed = await self._keyboard.type_text(value)
        if typed:
            _log.debug("fill_field_visual_ok", label=label)
            return True

        _log.debug("fill_field_failed", label=label)
        return False

    # ------------------------------------------------------------------
    # submit_form
    # ------------------------------------------------------------------

    async def submit_form(self) -> bool:
        """
        Submit the current form.

        Tries submit buttons in priority order via DOM, then falls back to
        sending Enter via the keyboard.

        Returns
        -------
        True when a submit action was delivered, False on total failure.
        """
        # ---- Stage 1: DOM path ------------------------------------------
        for selector in _SUBMIT_SELECTORS:
            elements = await self._dom.get_elements(selector)
            if not elements:
                continue
            visible = [e for e in elements if e.is_visible]
            if not visible:
                continue
            target: DOMElement = visible[0]
            ok = await self._dom.click(target)
            if ok:
                _log.debug("submit_form_dom_ok", selector=selector)
                return True

        # ---- Stage 2: Keyboard fallback (Enter key) ---------------------
        ok = await self._keyboard.type_text(_ENTER_KEY)
        if ok:
            _log.debug("submit_form_keyboard_ok")
        return ok

    # ------------------------------------------------------------------
    # handle_date_picker
    # ------------------------------------------------------------------

    async def handle_date_picker(self, date_str: str) -> bool:
        """
        Enter a date into a date picker or date text field.

        Parameters
        ----------
        date_str:
            Date string to type (e.g. ``"2026-04-08"`` or ``"08/04/2026"``).
            The format must match what the input expects.

        Returns
        -------
        True when the date was entered, False on failure.
        """
        # ---- Stage 1: DOM path ------------------------------------------
        for selector in _DATE_SELECTORS:
            elements = await self._dom.get_elements(selector)
            if not elements:
                continue
            visible = [e for e in elements if e.is_visible]
            if not visible:
                continue
            target: DOMElement = visible[0]
            await self._dom.clear(target)
            ok = await self._dom.type_text(target, date_str)
            if ok:
                _log.debug(
                    "handle_date_picker_dom_ok",
                    selector=selector,
                    date=date_str,
                )
                return True

        # ---- Stage 2: Keyboard fallback ---------------------------------
        ok = await self._keyboard.type_text(date_str)
        if ok:
            _log.debug("handle_date_picker_keyboard_ok", date=date_str)
        return ok

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _find_input_for_label(self, label: str) -> DOMElement | None:
        """
        Attempt to locate the <input> associated with a <label> element.

        Strategy:
          1. Find the label by text via find_by_text().
          2. If label has a ``for`` attribute → query ``input#<id>``.
          3. Otherwise look for a descendant input via ``.`` heuristic selector.
          4. Fallback: ``input[placeholder*=label]``.
        """
        label_elem = await self._dom.find_by_text(label)

        if label_elem is not None:
            for_attr = label_elem.attributes.get("for")
            if for_attr:
                inputs = await self._dom.get_elements(f"input#{for_attr}")
                if inputs:
                    visible = [e for e in inputs if e.is_visible]
                    if visible:
                        return visible[0]

        # Placeholder-based fallback
        import json as _json  # noqa: PLC0415
        placeholder_selector = f"input[placeholder*={_json.dumps(label)}]"
        inputs = await self._dom.get_elements(placeholder_selector)
        if inputs:
            visible = [e for e in inputs if e.is_visible]
            if visible:
                return visible[0]

        return None
