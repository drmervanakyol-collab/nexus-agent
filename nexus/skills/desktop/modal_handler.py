"""
nexus/skills/desktop/modal_handler.py
Modal chain handler — detect, classify, and dismiss desktop dialog boxes.

ModalAction
-----------
Value object returned by handle_modal():

  action_type   : str  — "accepted" | "dismissed" | "no_modal" | "failed"
  button_clicked: str | None  — label of the clicked button
  modal_title   : str | None  — title of the detected dialog

ModalChainHandler
-----------------
handle_modal(perception) -> ModalAction
  1. Scan the SpatialGraph for nodes whose element_type is DIALOG.
  2. When no dialog is found, return ModalAction("no_modal").
  3. Among sibling button nodes near the dialog, match against priority
     lists (accept-buttons, dismiss-buttons).
  4. Click the best matching button via the injectable _click_fn.
  5. Return ModalAction describing the outcome.

  Priority order:
    Accept : ["OK", "Yes", "Evet", "Tamam", "Continue", "Devam", "Retry"]
    Dismiss: ["Cancel", "No", "İptal", "Hayır", "Close", "Skip", "Atla"]

wait_for_modal_close(timeout_ms) -> bool
  Poll the injectable _is_modal_present_fn at _poll_interval_ms until it
  returns False (modal gone) or timeout elapses.  Returns True when the
  modal closes within the timeout, False on timeout.

Injectable callables
--------------------
_find_dialog_fn    : () -> str | None
    Return the title of a currently open modal dialog (None = no modal).
    Default: always returns None (safe for tests without real UIA).
_click_fn          : async (x: int, y: int) -> bool
    OS-level mouse click at the button centre.
_is_modal_present_fn : () -> bool
    True while a modal dialog is visible.  Default: always False.
_sleep_fn          : async (seconds: float) -> None
    Settable sleep for testing.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from nexus.infra.logger import get_logger
from nexus.perception.locator.locator import ElementType
from nexus.perception.orchestrator import PerceptionResult
from nexus.perception.spatial_graph import SpatialNode

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Button priority lists (case-insensitive prefix match)
# ---------------------------------------------------------------------------

_ACCEPT_LABELS: tuple[str, ...] = (
    "ok", "yes", "evet", "tamam", "continue", "devam", "retry", "dene",
)
_DISMISS_LABELS: tuple[str, ...] = (
    "cancel", "no", "i̇ptal", "hayır", "close", "skip", "atla",
)

_POLL_INTERVAL_MS: int = 100


# ---------------------------------------------------------------------------
# ModalAction
# ---------------------------------------------------------------------------


@dataclass
class ModalAction:
    """
    Result of a single handle_modal() call.

    Attributes
    ----------
    action_type:
        ``"accepted"``  — an accept button was clicked.
        ``"dismissed"`` — a dismiss / cancel button was clicked.
        ``"no_modal"``  — no dialog found in the SpatialGraph.
        ``"failed"``    — a dialog was found but no button could be clicked.
    button_clicked:
        Label of the button that was activated, or None.
    modal_title:
        Text of the dialog node (used as its title), or None.
    """

    action_type: str
    button_clicked: str | None = None
    modal_title: str | None = None


# ---------------------------------------------------------------------------
# ModalChainHandler
# ---------------------------------------------------------------------------


class ModalChainHandler:
    """
    Detects and dismisses modal dialogs from desktop applications.

    Parameters
    ----------
    _find_dialog_fn:
        Sync ``() -> str | None``.  Returns dialog title or None.
    _click_fn:
        Async ``(x: int, y: int) -> bool``.  Mouse click transport.
    _is_modal_present_fn:
        Sync ``() -> bool``.  True while a modal is visible.
    _sleep_fn:
        Async ``(seconds: float) -> None``.  Injectable for tests.
    """

    def __init__(
        self,
        *,
        _find_dialog_fn: Callable[[], str | None] | None = None,
        _click_fn: Callable[[int, int], Awaitable[bool]] | None = None,
        _is_modal_present_fn: Callable[[], bool] | None = None,
        _sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._find_dialog: Callable[[], str | None] = (
            _find_dialog_fn or (lambda: None)
        )
        self._click: Callable[[int, int], Awaitable[bool]] = (
            _click_fn or _noop_click
        )
        self._is_modal_present: Callable[[], bool] = (
            _is_modal_present_fn or (lambda: False)
        )
        self._sleep: Callable[[float], Awaitable[None]] = (
            _sleep_fn or asyncio.sleep
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_modal(self, perception: PerceptionResult) -> ModalAction:
        """
        Detect and dismiss a modal dialog from *perception*.

        Scans the SpatialGraph for DIALOG nodes, then looks for adjacent
        BUTTON nodes.  Clicks the highest-priority button found.

        Parameters
        ----------
        perception:
            Latest PerceptionResult.

        Returns
        -------
        ModalAction describing what happened.
        """
        # Detect dialog node
        dialog_node = _find_dialog_node(perception)
        if dialog_node is None:
            _log.debug("handle_modal_no_dialog")
            return ModalAction(action_type="no_modal")

        modal_title = dialog_node.text or None
        _log.debug("handle_modal_dialog_found", title=modal_title)

        # Collect button nodes from the spatial graph
        buttons = _collect_buttons(perception)

        # Score and select best button
        result = _select_button(buttons)
        if result is None:
            _log.debug("handle_modal_no_button", title=modal_title)
            return ModalAction(action_type="failed", modal_title=modal_title)

        action_type, button_node = result
        center = button_node.element.bounding_box.center()
        clicked = await self._click(center.x, center.y)

        if not clicked:
            return ModalAction(action_type="failed", modal_title=modal_title)

        label = button_node.text or button_node.semantic.primary_label
        _log.debug(
            "handle_modal_clicked",
            action=action_type,
            button=label,
        )
        return ModalAction(
            action_type=action_type,
            button_clicked=label,
            modal_title=modal_title,
        )

    async def wait_for_modal_close(self, timeout_ms: int = 5000) -> bool:
        """
        Wait until the modal dialog disappears or timeout elapses.

        Polls ``_is_modal_present_fn`` every ``_POLL_INTERVAL_MS``
        milliseconds.

        Parameters
        ----------
        timeout_ms:
            Maximum wait time in milliseconds.

        Returns
        -------
        True when the modal closes within the timeout, False on timeout.
        """
        deadline = asyncio.get_event_loop().time() + timeout_ms / 1000.0
        interval = _POLL_INTERVAL_MS / 1000.0

        while asyncio.get_event_loop().time() < deadline:
            if not self._is_modal_present():
                _log.debug("wait_for_modal_close_ok")
                return True
            remaining = deadline - asyncio.get_event_loop().time()
            await self._sleep(min(interval, max(0.0, remaining)))

        _log.debug("wait_for_modal_close_timeout", timeout_ms=timeout_ms)
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_dialog_node(perception: PerceptionResult) -> SpatialNode | None:
    """Return the first DIALOG-type node in the SpatialGraph, or None."""
    for node in perception.spatial_graph.nodes:
        if node.element.element_type == ElementType.DIALOG:
            return node
    return None


def _collect_buttons(perception: PerceptionResult) -> list[SpatialNode]:
    """Return all BUTTON-type nodes from the SpatialGraph."""
    return [
        node
        for node in perception.spatial_graph.nodes
        if node.element.element_type == ElementType.BUTTON
        and node.element.is_visible
    ]


def _select_button(
    buttons: list[SpatialNode],
) -> tuple[str, SpatialNode] | None:
    """
    Select the best button to click from *buttons*.

    Priority: accept labels > dismiss labels > first available button.
    Returns (action_type, SpatialNode) or None when buttons is empty.
    """
    if not buttons:
        return None

    def _label(node: SpatialNode) -> str:
        return (node.text or node.semantic.primary_label or "").strip().lower()

    # Accept priority
    for node in buttons:
        lbl = _label(node)
        if any(lbl.startswith(a) for a in _ACCEPT_LABELS):
            return ("accepted", node)

    # Dismiss priority
    for node in buttons:
        lbl = _label(node)
        if any(lbl.startswith(d) for d in _DISMISS_LABELS):
            return ("dismissed", node)

    # Fallback — click the first button
    return ("dismissed", buttons[0])


# ---------------------------------------------------------------------------
# No-op stubs
# ---------------------------------------------------------------------------


async def _noop_click(_x: int, _y: int) -> bool:
    return False
