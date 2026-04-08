"""
tests/unit/test_desktop_skills.py
Unit tests for nexus/skills/desktop/ — Faz 50.

TEST PLAN
---------
MultiPanelHandler:
  1.  find_active_panel — PANEL node found → UIElement returned
  2.  find_active_panel — highest z_order wins
  3.  find_active_panel — no panel nodes → None
  4.  switch_panel — UIA invoke succeeds → no visual fallback
  5.  switch_panel — UIA fails → visual fallback click
  6.  switch_panel — both fail → False

ModalChainHandler:
  7.  handle_modal — no dialog in graph → ModalAction("no_modal")
  8.  handle_modal — dialog + OK button → action_type "accepted"
  9.  handle_modal — dialog + Cancel button → action_type "dismissed"
  10. handle_modal — dialog + Turkish Tamam button → "accepted"
  11. handle_modal — dialog + no buttons → action_type "failed"
  12. handle_modal — click fails → action_type "failed"
  13. wait_for_modal_close — modal disappears before timeout → True
  14. wait_for_modal_close — timeout reached → False

TreeNavigator:
  15. expand_node — UIA expand succeeds → True, no double-click
  16. expand_node — UIA fails → visual double-click fallback
  17. find_in_tree — UIA TreeItem found → UIElement returned
  18. find_in_tree — UIA finds non-TreeItem → visual fallback
  19. find_in_tree — both fail → None
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from nexus.core.types import ElementId, Rect
from nexus.perception.arbitration.arbitrator import ArbitrationResult
from nexus.perception.locator.locator import ElementType, UIElement
from nexus.perception.matcher.matcher import Affordance, SemanticLabel
from nexus.perception.orchestrator import PerceptionResult
from nexus.perception.spatial_graph import SpatialGraph
from nexus.perception.temporal.temporal_expert import ScreenState, StateType
from nexus.skills.desktop.modal_handler import ModalChainHandler
from nexus.skills.desktop.multi_panel import MultiPanelHandler
from nexus.skills.desktop.tree_navigation import TreeNavigator
from nexus.source.resolver import SourceResult
from nexus.source.uia.adapter import UIAAdapter, UIAElement

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DEFAULT_RECT = Rect(0, 0, 200, 100)


def _ui_element(
    *,
    etype: ElementType = ElementType.BUTTON,
    rect: Rect | None = None,
    z: int = 1,
    visible: bool = True,
) -> UIElement:
    return UIElement(
        id=ElementId(str(uuid.uuid4())),
        element_type=etype,
        bounding_box=rect if rect is not None else _DEFAULT_RECT,
        confidence=0.9,
        is_visible=visible,
        is_occluded=False,
        occlusion_ratio=0.0,
        z_order_estimate=z,
    )


def _semantic(
    element_id: ElementId,
    label: str = "button",
    affordance: Affordance = Affordance.CLICKABLE,
) -> SemanticLabel:
    return SemanticLabel(
        element_id=element_id,
        primary_label=label,
        secondary_labels=[],
        confidence=0.9,
        affordance=affordance,
        is_destructive=False,
    )


def _make_perception(
    cells: list[tuple[UIElement, str]],
) -> PerceptionResult:
    """Build a PerceptionResult with given (UIElement, ocr_text) pairs."""
    elements = [c[0] for c in cells]
    labels = [_semantic(e.id) for e in elements]
    texts = {c[0].id: c[1] for c in cells}
    stable = ScreenState(
        state_type=StateType.STABLE,
        confidence=1.0,
        blocks_perception=False,
        reason="stable",
        retry_after_ms=0,
    )
    arb = ArbitrationResult(
        resolved_elements=(),
        resolved_labels=(),
        conflicts_detected=0,
        conflicts_resolved=0,
        temporal_blocked=False,
        overall_confidence=1.0,
    )
    return PerceptionResult(
        spatial_graph=SpatialGraph(elements, labels, texts),
        screen_state=stable,
        arbitration=arb,
        source_result=SourceResult(
            source_type="visual",
            data={},
            confidence=0.8,
            latency_ms=0.0,
        ),
        perception_ms=0.0,
        frame_sequence=1,
        timestamp="2026-04-08T00:00:00Z",
    )


def _empty_perception() -> PerceptionResult:
    return _make_perception([])


def _uia_stub() -> UIAAdapter:
    mock_auto = MagicMock()
    return UIAAdapter(
        MagicMock(), _automation_factory=lambda: mock_auto
    )


def _make_uia_element(
    *,
    name: str = "Node",
    control_type: int = 50023,
    supports_expand: bool = True,
    supports_invoke: bool = False,
    rect: Rect | None = None,
) -> UIAElement:
    return UIAElement(
        automation_id="",
        name=name,
        control_type=control_type,
        bounding_rect=rect or _DEFAULT_RECT,
        is_enabled=True,
        is_visible=True,
        value=None,
        supports_invoke=supports_invoke,
        supports_expand_collapse=supports_expand,
        _raw=MagicMock(),
    )


# ---------------------------------------------------------------------------
# MultiPanelHandler tests
# ---------------------------------------------------------------------------


class TestFindActivePanel:
    def test_panel_node_returned(self):
        """PANEL-type node found → its UIElement returned."""
        panel = _ui_element(etype=ElementType.PANEL, z=5)
        perception = _make_perception([(panel, "Navigation")])
        uia = _uia_stub()

        handler = MultiPanelHandler(uia)
        result = handler.find_active_panel(perception)

        assert result is panel

    def test_highest_z_order_wins(self):
        """When multiple panels exist, the one with highest z wins."""
        low = _ui_element(etype=ElementType.PANEL, z=1)
        high = _ui_element(etype=ElementType.PANEL, z=10)
        perception = _make_perception([(low, "Low"), (high, "High")])
        uia = _uia_stub()

        handler = MultiPanelHandler(uia)
        result = handler.find_active_panel(perception)

        assert result is high

    def test_no_panels_returns_none(self):
        """No PANEL/CONTAINER/TAB nodes → None."""
        btn = _ui_element(etype=ElementType.BUTTON)
        perception = _make_perception([(btn, "Click me")])
        uia = _uia_stub()

        handler = MultiPanelHandler(uia)
        result = handler.find_active_panel(perception)

        assert result is None

    def test_container_type_qualifies(self):
        """CONTAINER-type also counts as a panel."""
        container = _ui_element(etype=ElementType.CONTAINER)
        perception = _make_perception([(container, "Panel")])

        handler = MultiPanelHandler(_uia_stub())
        result = handler.find_active_panel(perception)

        assert result is container


class TestSwitchPanel:
    @pytest.mark.asyncio
    async def test_uia_invoke_succeeds_no_visual(self):
        """UIA invoke succeeds → visual fallback not called."""
        click_calls: list[tuple[int, int]] = []

        async def spy_click(x: int, y: int) -> bool:
            click_calls.append((x, y))
            return True

        handler = MultiPanelHandler(
            _uia_stub(),
            _invoke_by_name_fn=lambda name: True,
            _click_fn=spy_click,
        )
        result = await handler.switch_panel("Details Panel")

        assert result is True
        assert click_calls == []

    @pytest.mark.asyncio
    async def test_uia_fails_visual_fallback(self):
        """UIA fails → visual coordinates found → click called."""
        click_calls: list[tuple[int, int]] = []

        async def spy_click(x: int, y: int) -> bool:
            click_calls.append((x, y))
            return True

        handler = MultiPanelHandler(
            _uia_stub(),
            _invoke_by_name_fn=lambda _: False,
            _find_visual_fn=lambda _: (150, 75),
            _click_fn=spy_click,
        )
        result = await handler.switch_panel("Details Panel")

        assert result is True
        assert (150, 75) in click_calls

    @pytest.mark.asyncio
    async def test_both_fail_returns_false(self):
        handler = MultiPanelHandler(
            _uia_stub(),
            _invoke_by_name_fn=lambda _: False,
            _find_visual_fn=lambda _: None,
        )
        result = await handler.switch_panel("Missing Panel")

        assert result is False


# ---------------------------------------------------------------------------
# ModalChainHandler tests
# ---------------------------------------------------------------------------


class TestHandleModal:
    @pytest.mark.asyncio
    async def test_no_dialog_returns_no_modal(self):
        """No DIALOG element in graph → no_modal."""
        perception = _empty_perception()
        handler = ModalChainHandler()

        result = await handler.handle_modal(perception)

        assert result.action_type == "no_modal"

    @pytest.mark.asyncio
    async def test_ok_button_accepted(self):
        """Dialog present + OK button → action_type 'accepted'."""
        dialog = _ui_element(etype=ElementType.DIALOG)
        ok_btn = _ui_element(etype=ElementType.BUTTON, rect=Rect(100, 200, 80, 30))
        perception = _make_perception([
            (dialog, "Confirm"),
            (ok_btn, "OK"),
        ])

        click_calls: list[tuple[int, int]] = []

        async def spy_click(x: int, y: int) -> bool:
            click_calls.append((x, y))
            return True

        handler = ModalChainHandler(_click_fn=spy_click)
        result = await handler.handle_modal(perception)

        assert result.action_type == "accepted"
        assert result.button_clicked is not None
        assert "ok" in result.button_clicked.lower()
        assert click_calls  # button was clicked

    @pytest.mark.asyncio
    async def test_cancel_button_dismissed(self):
        """Dialog + Cancel button → 'dismissed'."""
        dialog = _ui_element(etype=ElementType.DIALOG)
        cancel = _ui_element(etype=ElementType.BUTTON, rect=Rect(100, 200, 80, 30))
        perception = _make_perception([
            (dialog, "Delete?"),
            (cancel, "Cancel"),
        ])

        async def spy_click(x: int, y: int) -> bool:
            return True

        handler = ModalChainHandler(_click_fn=spy_click)
        result = await handler.handle_modal(perception)

        assert result.action_type == "dismissed"

    @pytest.mark.asyncio
    async def test_turkish_tamam_accepted(self):
        """Turkish 'Tamam' button → 'accepted'."""
        dialog = _ui_element(etype=ElementType.DIALOG)
        tamam = _ui_element(etype=ElementType.BUTTON, rect=Rect(10, 10, 60, 30))
        perception = _make_perception([
            (dialog, "Onay"),
            (tamam, "Tamam"),
        ])

        async def spy_click(x: int, y: int) -> bool:
            return True

        handler = ModalChainHandler(_click_fn=spy_click)
        result = await handler.handle_modal(perception)

        assert result.action_type == "accepted"

    @pytest.mark.asyncio
    async def test_no_buttons_returns_failed(self):
        """Dialog present but no buttons → 'failed'."""
        dialog = _ui_element(etype=ElementType.DIALOG)
        perception = _make_perception([(dialog, "Wait...")])

        handler = ModalChainHandler()
        result = await handler.handle_modal(perception)

        assert result.action_type == "failed"

    @pytest.mark.asyncio
    async def test_click_fails_returns_failed(self):
        """Click returns False → 'failed'."""
        dialog = _ui_element(etype=ElementType.DIALOG)
        ok_btn = _ui_element(etype=ElementType.BUTTON)
        perception = _make_perception([
            (dialog, "Error"),
            (ok_btn, "OK"),
        ])

        async def failing_click(_x: int, _y: int) -> bool:
            return False

        handler = ModalChainHandler(_click_fn=failing_click)
        result = await handler.handle_modal(perception)

        assert result.action_type == "failed"


class TestWaitForModalClose:
    @pytest.mark.asyncio
    async def test_modal_closes_before_timeout(self):
        """_is_modal_present returns False → returns True."""
        call_count = 0

        def is_present() -> bool:
            nonlocal call_count
            call_count += 1
            return call_count < 3   # present twice, then gone

        async def fast_sleep(_s: float) -> None:
            pass

        handler = ModalChainHandler(
            _is_modal_present_fn=is_present,
            _sleep_fn=fast_sleep,
        )
        result = await handler.wait_for_modal_close(timeout_ms=5000)

        assert result is True

    @pytest.mark.asyncio
    async def test_timeout_returns_false(self):
        """Modal never closes → False after timeout."""
        async def fast_sleep(s: float) -> None:
            pass

        handler = ModalChainHandler(
            _is_modal_present_fn=lambda: True,   # always present
            _sleep_fn=fast_sleep,
        )
        result = await handler.wait_for_modal_close(timeout_ms=50)

        assert result is False


# ---------------------------------------------------------------------------
# TreeNavigator tests
# ---------------------------------------------------------------------------


class TestExpandNode:
    @pytest.mark.asyncio
    async def test_uia_expand_succeeds_no_visual(self):
        """UIA expand returns True → double-click NOT called."""
        dbl_click_calls: list[tuple[int, int]] = []

        async def spy_dbl(x: int, y: int) -> bool:
            dbl_click_calls.append((x, y))
            return True

        nav = TreeNavigator(
            _uia_stub(),
            _expand_via_uia_fn=lambda _: True,
            _double_click_fn=spy_dbl,
        )
        result = await nav.expand_node("Root")

        assert result is True
        assert dbl_click_calls == []

    @pytest.mark.asyncio
    async def test_uia_fails_visual_double_click(self):
        """UIA fails → visual coords found → double-click called."""
        dbl_click_calls: list[tuple[int, int]] = []

        async def spy_dbl(x: int, y: int) -> bool:
            dbl_click_calls.append((x, y))
            return True

        nav = TreeNavigator(
            _uia_stub(),
            _expand_via_uia_fn=lambda _: False,
            _find_visual_fn=lambda _: (50, 80),
            _double_click_fn=spy_dbl,
        )
        result = await nav.expand_node("Root")

        assert result is True
        assert (50, 80) in dbl_click_calls


class TestFindInTree:
    def test_uia_tree_item_found(self):
        """UIA returns TreeItem (50023) → UIElement stub returned."""
        tree_elem = _make_uia_element(
            name="Documents",
            control_type=50023,
        )

        nav = TreeNavigator(
            _uia_stub(),
            _find_uia_tree_item_fn=lambda _: tree_elem,
        )
        result = nav.find_in_tree("Documents")

        assert result is not None
        assert result.is_visible is True

    def test_uia_non_tree_item_uses_visual_fallback(self):
        """UIA returns None (wrong type filtered) → visual fallback."""
        visual_elem = _ui_element(etype=ElementType.UNKNOWN)

        nav = TreeNavigator(
            _uia_stub(),
            _find_uia_tree_item_fn=lambda _: None,
            _find_visual_node_fn=lambda _: visual_elem,
        )
        result = nav.find_in_tree("Documents")

        assert result is visual_elem

    def test_both_fail_returns_none(self):
        """UIA None + visual None → None."""
        nav = TreeNavigator(
            _uia_stub(),
            _find_uia_tree_item_fn=lambda _: None,
            _find_visual_node_fn=lambda _: None,
        )
        result = nav.find_in_tree("Missing")

        assert result is None
