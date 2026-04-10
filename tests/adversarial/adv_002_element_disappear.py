"""
tests/adversarial/adv_002_element_disappear.py
Adversarial Test 002 — Element disappears before click (Preflight catch)

Scenario:
  An ActionSpec targets element "el-ghost" which is NOT present in the
  SpatialGraph at preflight time (simulating an element that vanished between
  perception and action dispatch).

Success criteria:
  - PreflightChecker.check() returns passed=False.
  - failed_check == CHECK_ELEMENT_VISIBLE.
  - No exception raised.

All I/O is injected — no real OS calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nexus.action.preflight import (
    CHECK_ELEMENT_VISIBLE,
    PreflightChecker,
    PreflightContext,
)
from nexus.action.registry import ActionSpec
from nexus.core.types import Rect

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perception_without_element():
    """SpatialGraph returns None for any element_id lookup."""
    p = MagicMock()
    p.spatial_graph.get_node.return_value = None
    p.screen_state.blocks_perception = False
    p.source_result.source_type = "visual"
    return p


def _make_perception_with_element(element_id: str):
    """SpatialGraph contains the element."""
    p = MagicMock()
    node = MagicMock()
    node.element.bounding_box = Rect(x=100, y=100, width=120, height=40)
    p.spatial_graph.get_node.return_value = node
    p.screen_state.blocks_perception = False
    p.source_result.source_type = "visual"
    return p


_CHECKER = PreflightChecker()


def _ctx(**kwargs) -> PreflightContext:
    return PreflightContext(screen_width=1920, screen_height=1080, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.adversarial
class TestElementDisappear:
    """ADV-002: Element gone at preflight time → graceful failure, no crash."""

    def test_missing_element_fails_preflight(self):
        """
        Target element 'el-ghost' absent from graph →
        preflight fails with CHECK_ELEMENT_VISIBLE.
        """
        spec = ActionSpec(
            action_type="click",
            target_element_id="el-ghost",
            preferred_transport="uia",
        )
        perception = _make_perception_without_element()
        result = _CHECKER.check(spec, perception, _ctx())

        assert result.passed is False, "Preflight must fail for missing element"
        assert result.failed_check == CHECK_ELEMENT_VISIBLE, (
            f"Expected CHECK_ELEMENT_VISIBLE; got {result.failed_check!r}"
        )

    def test_present_element_passes_preflight(self):
        """Element present → preflight passes (baseline)."""
        spec = ActionSpec(
            action_type="click",
            target_element_id="el-real",
            coordinates=(110, 120),
            preferred_transport="mouse",
        )
        perception = _make_perception_with_element("el-real")
        result = _CHECKER.check(spec, perception, _ctx())

        assert result.passed is True, (
            f"Preflight must pass for present element; "
            f"failed_check={result.failed_check!r}"
        )

    def test_no_element_id_skips_visibility_check(self):
        """
        ActionSpec without target_element_id skips CHECK_ELEMENT_VISIBLE,
        so preflight can still pass for coordinate-only actions.
        """
        spec = ActionSpec(
            action_type="click",
            target_element_id=None,
            coordinates=(200, 300),
            preferred_transport="mouse",
        )
        perception = _make_perception_without_element()
        result = _CHECKER.check(spec, perception, _ctx())

        # CHECK_ELEMENT_VISIBLE is skipped when no element_id;
        # result might pass or fail on another check — but must NOT be
        # CHECK_ELEMENT_VISIBLE.
        assert result.failed_check != CHECK_ELEMENT_VISIBLE, (
            "CHECK_ELEMENT_VISIBLE must not trigger when no element_id is set"
        )
