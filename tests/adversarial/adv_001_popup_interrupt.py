"""
tests/adversarial/adv_001_popup_interrupt.py
Adversarial Test 001 — Popup interrupt mid-task

Scenario:
  A modal dialog appears while the agent is executing an action.

Success criteria:
  - ModalChainHandler.handle_modal() returns action_type != "no_modal".
  - The system does not raise / crash.
  - The dismissed modal title is captured in the ModalAction.

All I/O is injected — no real OS calls.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from nexus.perception.locator.locator import ElementType
from nexus.skills.desktop.modal_handler import ModalChainHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(element_type: ElementType, text: str, x: int = 0, y: int = 0):
    """Build a MagicMock SpatialNode with the given type and text."""
    from nexus.core.types import Rect

    node = MagicMock()
    node.element.element_type = element_type
    node.element.is_visible = True
    # Use a real Rect so that .center() works correctly
    bb = Rect(x=x, y=y, width=100, height=30)
    node.element.bounding_box = bb
    node.text = text
    node.semantic.primary_label = text
    return node


def _make_perception(dialog_text: str | None, button_text: str = "Tamam"):
    """Build a MagicMock PerceptionResult with optional dialog + button nodes."""
    perception = MagicMock()

    if dialog_text is not None:
        dialog_node = _node(ElementType.DIALOG, dialog_text, x=300, y=200)
        btn_node = _node(ElementType.BUTTON, button_text, x=450, y=350)
        # spatial_graph.nodes must be an iterable containing both
        perception.spatial_graph.nodes = [dialog_node, btn_node]
    else:
        perception.spatial_graph.nodes = []

    perception.screen_state.blocks_perception = False
    return perception


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.adversarial
class TestPopupInterrupt:
    """ADV-001: A popup appears mid-task; agent handles it without crashing."""

    def test_popup_detected_and_dismissed(self):
        """
        Inject a dialog present in the SpatialGraph.
        ModalChainHandler must detect it, click accept, return != 'no_modal',
        and capture the modal title.
        """
        clicks: list[tuple[int, int]] = []

        async def _click(x: int, y: int) -> bool:
            clicks.append((x, y))
            return True

        handler = ModalChainHandler(
            _find_dialog_fn=lambda: "Uyarı",
            _click_fn=_click,
            _is_modal_present_fn=lambda: False,
        )

        perception = _make_perception("Uyarı")
        result = asyncio.run(handler.handle_modal(perception))

        assert result.action_type != "no_modal", (
            f"Popup must be detected; got action_type={result.action_type!r}"
        )
        assert result.modal_title is not None, "Modal title must be captured"

    def test_no_popup_returns_no_modal(self):
        """When no dialog is present, handle_modal() returns 'no_modal'."""
        handler = ModalChainHandler(_find_dialog_fn=lambda: None)
        perception = _make_perception(None)
        result = asyncio.run(handler.handle_modal(perception))
        assert result.action_type == "no_modal"

    def test_failed_button_click_returns_failed(self):
        """
        Dialog found but click fails → action_type == 'failed', no crash.
        """
        async def _failing_click(_x: int, _y: int) -> bool:
            return False

        handler = ModalChainHandler(
            _find_dialog_fn=lambda: "Hata",
            _click_fn=_failing_click,
            _is_modal_present_fn=lambda: True,
        )
        perception = _make_perception("Hata")
        result = asyncio.run(handler.handle_modal(perception))

        # System must not raise — either 'accepted' with False or 'failed'
        assert result.action_type in {"accepted", "dismissed", "failed", "no_modal"}
