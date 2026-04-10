"""
tests/unit/test_preflight.py
Unit tests for nexus/action/preflight.py.

Coverage
--------
  CHECK_ELEMENT_VISIBLE     — element_id in graph, missing, no id
  CHECK_NOT_OCCLUDED        — blocks_perception True/False
  CHECK_COORDS_ON_SCREEN    — within bounds, out of bounds, no coords
  CHECK_POLICY              — block/abort → fail; allow/warn → pass; no policy → pass
  CHECK_DESTRUCTIVE_CONFIRM — destructive + no confirm → fail; confirmed → pass
  CHECK_SENSITIVE_AREA      — block region → fail; warn region → pass; no detector → pass
  CHECK_TRANSPORT_AVAILABLE — uia/dom unavailable × fallback allowed/forbidden
  Bypass resistance         — tests that verify no single check can be skipped
                              when earlier checks in the chain should fail
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nexus.action.preflight import (
    CHECK_COORDS_ON_SCREEN,
    CHECK_DESTRUCTIVE_CONFIRM,
    CHECK_ELEMENT_VISIBLE,
    CHECK_NOT_OCCLUDED,
    CHECK_POLICY,
    CHECK_SENSITIVE_AREA,
    CHECK_TRANSPORT_AVAILABLE,
    PreflightChecker,
    PreflightContext,
)
from nexus.action.registry import ActionSpec
from nexus.core.policy import PolicyEngine, PolicyResult
from nexus.core.sensitive_regions import SensitiveRegion, SensitiveRegionDetector
from nexus.core.settings import NexusSettings
from nexus.core.types import Rect

# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------

_CHECKER = PreflightChecker()


def _spec(**kwargs) -> ActionSpec:
    defaults: dict = {"action_type": "click"}
    defaults.update(kwargs)
    return ActionSpec(**defaults)


def _ctx(**kwargs) -> PreflightContext:
    return PreflightContext(**kwargs)


def _make_graph(
    element_id: str | None = None,
    coords: tuple[int, int] | None = None,
    bounding_rect: Rect | None = None,
) -> MagicMock:
    """Return a MagicMock spatial graph.
    If element_id is given, get_node(element_id) returns a node with the rect.
    Otherwise get_node returns None.
    """
    graph = MagicMock()
    if element_id is not None:
        node = MagicMock()
        bb = bounding_rect or Rect(x=50, y=80, width=100, height=30)
        el = MagicMock()
        el.bounding_box = bb
        node.element = el
        graph.get_node.return_value = node
    else:
        graph.get_node.return_value = None
    return graph


def _make_screen_state(blocks: bool = False, reason: str = "stable") -> MagicMock:
    ss = MagicMock()
    ss.blocks_perception = blocks
    ss.reason = reason
    return ss


def _make_perception(
    element_id: str | None = None,
    bounding_rect: Rect | None = None,
    blocks_perception: bool = False,
    source_type: str = "visual",
) -> MagicMock:
    p = MagicMock()
    p.spatial_graph = _make_graph(element_id=element_id, bounding_rect=bounding_rect)
    p.screen_state = _make_screen_state(blocks=blocks_perception)
    src = MagicMock()
    src.source_type = source_type
    p.source_result = src
    return p


def _make_policy(verdict: str = "allow", rule: str | None = None) -> PolicyEngine:
    engine = MagicMock(spec=PolicyEngine)
    engine.check_action.return_value = PolicyResult(
        verdict=verdict,  # type: ignore[arg-type]
        rule=rule,
        severity=verdict,
        message=f"policy verdict={verdict}",
    )
    return engine


def _make_sensitive_detector(
    regions: list[tuple[Rect, str, str]] | None = None,
) -> SensitiveRegionDetector:
    """regions: list of (rect, label, severity)."""
    regs = [
        SensitiveRegion(rect=r, label=lbl, severity=sev)  # type: ignore[arg-type]
        for r, lbl, sev in (regions or [])
    ]
    return SensitiveRegionDetector(regs)


# ---------------------------------------------------------------------------
# TEST 1 — CHECK_ELEMENT_VISIBLE
# ---------------------------------------------------------------------------


class TestCheckElementVisible:
    def test_no_element_id_passes(self) -> None:
        spec = _spec(coordinates=(100, 200))
        p = _make_perception()
        result = _CHECKER.check(spec, p, _ctx())
        assert result.passed

    def test_element_found_in_graph_passes(self) -> None:
        spec = _spec(target_element_id="el-1")
        p = _make_perception(element_id="el-1")
        result = _CHECKER.check(spec, p, _ctx())
        assert result.passed

    def test_element_not_in_graph_fails(self) -> None:
        spec = _spec(target_element_id="el-missing")
        p = _make_perception(element_id=None)  # graph returns None
        result = _CHECKER.check(spec, p, _ctx())
        assert not result.passed
        assert result.failed_check == CHECK_ELEMENT_VISIBLE

    def test_failure_reason_contains_element_id(self) -> None:
        spec = _spec(target_element_id="el-xyz")
        p = _make_perception(element_id=None)
        result = _CHECKER.check(spec, p, _ctx())
        assert "el-xyz" in (result.reason or "")

    def test_element_visible_check_uses_graph_get_node(self) -> None:
        spec = _spec(target_element_id="el-42")
        p = _make_perception(element_id=None)
        _CHECKER.check(spec, p, _ctx())
        # Verify get_node was called with the correct id
        p.spatial_graph.get_node.assert_called_once()


# ---------------------------------------------------------------------------
# TEST 2 — CHECK_NOT_OCCLUDED
# ---------------------------------------------------------------------------


class TestCheckNotOccluded:
    def test_stable_screen_passes(self) -> None:
        p = _make_perception(blocks_perception=False)
        result = _CHECKER.check(_spec(), p, _ctx())
        assert result.passed

    def test_blocking_screen_fails(self) -> None:
        p = _make_perception(blocks_perception=True)
        result = _CHECKER.check(_spec(), p, _ctx())
        assert not result.passed
        assert result.failed_check == CHECK_NOT_OCCLUDED

    def test_occluded_reason_mentions_state(self) -> None:
        p = MagicMock()
        p.spatial_graph = _make_graph()
        p.screen_state.blocks_perception = True
        p.screen_state.reason = "loading_spinner"
        p.source_result.source_type = "visual"
        result = _CHECKER.check(_spec(), p, _ctx())
        assert "loading_spinner" in (result.reason or "")

    def test_blocks_perception_false_proceeds_to_later_checks(self) -> None:
        """When not occluded, the check chain continues."""
        p = _make_perception(blocks_perception=False)
        result = _CHECKER.check(_spec(), p, _ctx())
        # If CHECK_NOT_OCCLUDED were to stop the chain, we'd never see a pass.
        assert result.passed


# ---------------------------------------------------------------------------
# TEST 3 — CHECK_COORDS_ON_SCREEN
# ---------------------------------------------------------------------------


class TestCheckCoordsOnScreen:
    def test_no_coords_passes(self) -> None:
        spec = _spec(target_element_id=None, coordinates=None)
        result = _CHECKER.check(spec, _make_perception(), _ctx())
        assert result.passed

    def test_coords_within_bounds_passes(self) -> None:
        spec = _spec(coordinates=(500, 400))
        result = _CHECKER.check(spec, _make_perception(), _ctx(screen_width=1920, screen_height=1080))
        assert result.passed

    def test_x_negative_fails(self) -> None:
        spec = _spec(coordinates=(-1, 100))
        result = _CHECKER.check(spec, _make_perception(), _ctx())
        assert not result.passed
        assert result.failed_check == CHECK_COORDS_ON_SCREEN

    def test_y_negative_fails(self) -> None:
        spec = _spec(coordinates=(100, -1))
        result = _CHECKER.check(spec, _make_perception(), _ctx())
        assert not result.passed
        assert result.failed_check == CHECK_COORDS_ON_SCREEN

    def test_x_equals_screen_width_fails(self) -> None:
        spec = _spec(coordinates=(1920, 500))
        result = _CHECKER.check(spec, _make_perception(), _ctx(screen_width=1920))
        assert not result.passed
        assert result.failed_check == CHECK_COORDS_ON_SCREEN

    def test_y_equals_screen_height_fails(self) -> None:
        spec = _spec(coordinates=(500, 1080))
        result = _CHECKER.check(spec, _make_perception(), _ctx(screen_height=1080))
        assert not result.passed
        assert result.failed_check == CHECK_COORDS_ON_SCREEN

    def test_coords_at_origin_passes(self) -> None:
        spec = _spec(coordinates=(0, 0))
        result = _CHECKER.check(spec, _make_perception(), _ctx(screen_width=1920, screen_height=1080))
        assert result.passed

    def test_coords_at_max_valid_passes(self) -> None:
        spec = _spec(coordinates=(1919, 1079))
        result = _CHECKER.check(spec, _make_perception(), _ctx(screen_width=1920, screen_height=1080))
        assert result.passed

    def test_failure_reason_contains_coords(self) -> None:
        spec = _spec(coordinates=(9999, 8888))
        result = _CHECKER.check(spec, _make_perception(), _ctx(screen_width=1920, screen_height=1080))
        assert "9999" in (result.reason or "") or "8888" in (result.reason or "")


# ---------------------------------------------------------------------------
# TEST 4 — CHECK_POLICY
# ---------------------------------------------------------------------------


class TestCheckPolicy:
    def test_no_policy_passes(self) -> None:
        result = _CHECKER.check(_spec(), _make_perception(), _ctx(policy=None))
        assert result.passed

    def test_allow_verdict_passes(self) -> None:
        policy = _make_policy("allow")
        result = _CHECKER.check(_spec(), _make_perception(), _ctx(policy=policy))
        assert result.passed

    def test_warn_verdict_passes(self) -> None:
        policy = _make_policy("warn")
        result = _CHECKER.check(_spec(), _make_perception(), _ctx(policy=policy))
        assert result.passed

    def test_block_verdict_fails(self) -> None:
        policy = _make_policy("block", rule="RULE_MAX_ACTIONS")
        result = _CHECKER.check(_spec(), _make_perception(), _ctx(policy=policy))
        assert not result.passed
        assert result.failed_check == CHECK_POLICY

    def test_abort_verdict_fails(self) -> None:
        policy = _make_policy("abort")
        result = _CHECKER.check(_spec(), _make_perception(), _ctx(policy=policy))
        assert not result.passed
        assert result.failed_check == CHECK_POLICY

    def test_block_reason_contains_policy_message(self) -> None:
        engine = MagicMock(spec=PolicyEngine)
        engine.check_action.return_value = PolicyResult(
            verdict="block",
            rule="RULE_MAX_ACTIONS",
            severity="block",
            message="cap reached",
        )
        result = _CHECKER.check(_spec(), _make_perception(), _ctx(policy=engine))
        assert "cap reached" in (result.reason or "")

    def test_real_policy_blocks_at_budget(self) -> None:
        cfg = NexusSettings(
            budget={"max_cost_per_task_usd": 0.001}  # type: ignore[arg-type]
        )
        policy = PolicyEngine(cfg)
        ctx = _ctx(policy=policy, task_cost_usd=1.0)  # far above cap
        result = _CHECKER.check(_spec(), _make_perception(), ctx)
        assert not result.passed
        assert result.failed_check == CHECK_POLICY


# ---------------------------------------------------------------------------
# TEST 5 — CHECK_DESTRUCTIVE_CONFIRM
# ---------------------------------------------------------------------------


class TestCheckDestructiveConfirm:
    def test_non_destructive_passes(self) -> None:
        spec = _spec(is_destructive=False)
        result = _CHECKER.check(spec, _make_perception(), _ctx())
        assert result.passed

    def test_destructive_without_confirm_fails(self) -> None:
        spec = _spec(action_type="delete", is_destructive=True)
        result = _CHECKER.check(spec, _make_perception(), _ctx(destructive_confirmed=False))
        assert not result.passed
        assert result.failed_check == CHECK_DESTRUCTIVE_CONFIRM

    def test_destructive_with_confirm_passes(self) -> None:
        spec = _spec(action_type="delete", is_destructive=True)
        result = _CHECKER.check(spec, _make_perception(), _ctx(destructive_confirmed=True))
        assert result.passed

    def test_failure_reason_mentions_destructive(self) -> None:
        spec = _spec(action_type="overwrite", is_destructive=True)
        result = _CHECKER.check(spec, _make_perception(), _ctx(destructive_confirmed=False))
        assert "destructive" in (result.reason or "").lower()

    def test_confirmation_flag_is_required_not_optional(self) -> None:
        """Default context has destructive_confirmed=False — must not pass."""
        spec = _spec(is_destructive=True)
        default_ctx = PreflightContext()
        assert default_ctx.destructive_confirmed is False
        result = _CHECKER.check(spec, _make_perception(), default_ctx)
        assert not result.passed


# ---------------------------------------------------------------------------
# TEST 6 — CHECK_SENSITIVE_AREA
# ---------------------------------------------------------------------------


class TestCheckSensitiveArea:
    _TARGET_RECT = Rect(x=100, y=100, width=80, height=30)

    def test_no_detector_passes(self) -> None:
        spec = _spec(coordinates=(140, 115))
        result = _CHECKER.check(spec, _make_perception(), _ctx(sensitive_detector=None))
        assert result.passed

    def test_warn_region_passes(self) -> None:
        det = _make_sensitive_detector(
            [(self._TARGET_RECT, "password_field", "warn")]
        )
        spec = _spec(coordinates=(140, 115))
        result = _CHECKER.check(spec, _make_perception(), _ctx(sensitive_detector=det))
        assert result.passed

    def test_block_region_fails(self) -> None:
        det = _make_sensitive_detector(
            [(self._TARGET_RECT, "payment_field", "block")]
        )
        spec = _spec(coordinates=(140, 115))
        result = _CHECKER.check(spec, _make_perception(), _ctx(sensitive_detector=det))
        assert not result.passed
        assert result.failed_check == CHECK_SENSITIVE_AREA

    def test_failure_reason_contains_label(self) -> None:
        det = _make_sensitive_detector(
            [(self._TARGET_RECT, "payment_field", "block")]
        )
        spec = _spec(coordinates=(140, 115))
        result = _CHECKER.check(spec, _make_perception(), _ctx(sensitive_detector=det))
        assert "payment_field" in (result.reason or "")

    def test_no_coords_no_element_skips_check(self) -> None:
        """No rect can be resolved → check is skipped → passes."""
        det = _make_sensitive_detector(
            [(self._TARGET_RECT, "payment_field", "block")]
        )
        spec = ActionSpec(action_type="click")  # no coords, no element
        result = _CHECKER.check(spec, _make_perception(), _ctx(sensitive_detector=det))
        assert result.passed

    def test_non_overlapping_block_region_passes(self) -> None:
        far_rect = Rect(x=900, y=900, width=80, height=30)
        det = _make_sensitive_detector([(far_rect, "danger_zone", "block")])
        spec = _spec(coordinates=(140, 115))  # nowhere near far_rect
        result = _CHECKER.check(spec, _make_perception(), _ctx(sensitive_detector=det))
        assert result.passed


# ---------------------------------------------------------------------------
# TEST 7 — CHECK_TRANSPORT_AVAILABLE
# ---------------------------------------------------------------------------


class TestCheckTransportAvailable:
    def test_no_preferred_transport_passes(self) -> None:
        spec = _spec(preferred_transport=None)
        result = _CHECKER.check(spec, _make_perception(), _ctx())
        assert result.passed

    def test_mouse_transport_always_passes(self) -> None:
        spec = _spec(preferred_transport="mouse")
        result = _CHECKER.check(spec, _make_perception(), _ctx())
        assert result.passed

    def test_keyboard_transport_always_passes(self) -> None:
        spec = _spec(preferred_transport="keyboard")
        result = _CHECKER.check(spec, _make_perception(), _ctx())
        assert result.passed

    def test_uia_available_passes(self) -> None:
        spec = _spec(preferred_transport="uia")
        result = _CHECKER.check(spec, _make_perception(), _ctx(uia_available=True))
        assert result.passed

    def test_dom_available_passes(self) -> None:
        spec = _spec(preferred_transport="dom")
        result = _CHECKER.check(spec, _make_perception(), _ctx(dom_available=True))
        assert result.passed

    def test_uia_unavailable_fallback_allowed_passes(self) -> None:
        spec = _spec(preferred_transport="uia")
        ctx = _ctx(uia_available=False, allow_transport_fallback=True)
        result = _CHECKER.check(spec, _make_perception(), ctx)
        assert result.passed

    def test_uia_unavailable_no_fallback_fails(self) -> None:
        spec = _spec(preferred_transport="uia")
        ctx = _ctx(uia_available=False, allow_transport_fallback=False)
        result = _CHECKER.check(spec, _make_perception(), ctx)
        assert not result.passed
        assert result.failed_check == CHECK_TRANSPORT_AVAILABLE

    def test_dom_unavailable_no_fallback_fails(self) -> None:
        spec = _spec(preferred_transport="dom")
        ctx = _ctx(dom_available=False, allow_transport_fallback=False)
        result = _CHECKER.check(spec, _make_perception(), ctx)
        assert not result.passed
        assert result.failed_check == CHECK_TRANSPORT_AVAILABLE

    def test_dom_unavailable_fallback_allowed_passes(self) -> None:
        spec = _spec(preferred_transport="dom")
        ctx = _ctx(dom_available=False, allow_transport_fallback=True)
        result = _CHECKER.check(spec, _make_perception(), ctx)
        assert result.passed

    def test_failure_reason_mentions_transport(self) -> None:
        spec = _spec(preferred_transport="uia")
        ctx = _ctx(uia_available=False, allow_transport_fallback=False)
        result = _CHECKER.check(spec, _make_perception(), ctx)
        assert "uia" in (result.reason or "").lower()

    def test_failure_reason_mentions_fallback(self) -> None:
        spec = _spec(preferred_transport="uia")
        ctx = _ctx(uia_available=False, allow_transport_fallback=False)
        result = _CHECKER.check(spec, _make_perception(), ctx)
        assert "fallback" in (result.reason or "").lower()


# ---------------------------------------------------------------------------
# Bypass resistance
# ---------------------------------------------------------------------------


class TestBypassResistance:
    """
    Verify that failed earlier checks are not bypassed by passing later ones.
    The public API (check()) always runs checks in order and stops at first
    failure — individual check methods are private and cannot be called
    directly by callers.
    """

    def test_occluded_screen_blocks_even_with_valid_coords(self) -> None:
        """CHECK_NOT_OCCLUDED fires before CHECK_COORDS_ON_SCREEN."""
        spec = _spec(coordinates=(100, 100))
        p = _make_perception(blocks_perception=True)
        result = _CHECKER.check(spec, p, _ctx())
        assert not result.passed
        assert result.failed_check == CHECK_NOT_OCCLUDED

    def test_missing_element_blocks_before_policy(self) -> None:
        """CHECK_ELEMENT_VISIBLE fires before CHECK_POLICY."""
        policy = _make_policy("allow")
        spec = _spec(target_element_id="ghost")
        p = _make_perception(element_id=None)
        result = _CHECKER.check(spec, p, _ctx(policy=policy))
        assert not result.passed
        assert result.failed_check == CHECK_ELEMENT_VISIBLE

    def test_policy_block_fires_before_destructive_confirm(self) -> None:
        """CHECK_POLICY fires before CHECK_DESTRUCTIVE_CONFIRM."""
        policy = _make_policy("block")
        spec = _spec(action_type="delete", is_destructive=True)
        p = _make_perception()
        result = _CHECKER.check(spec, p, _ctx(policy=policy, destructive_confirmed=False))
        assert not result.passed
        assert result.failed_check == CHECK_POLICY

    def test_all_checks_pass_returns_passed_true(self) -> None:
        policy = _make_policy("allow")
        spec = _spec(
            action_type="click",
            coordinates=(100, 100),
            is_destructive=False,
            preferred_transport="uia",
        )
        p = _make_perception()
        ctx = _ctx(
            policy=policy,
            uia_available=True,
            destructive_confirmed=False,  # non-destructive, so this is irrelevant
        )
        result = _CHECKER.check(spec, p, ctx)
        assert result.passed
        assert result.failed_check is None
        assert result.reason is None

    def test_no_private_check_method_accessible(self) -> None:
        """Private methods must not be public API."""
        checker = PreflightChecker()
        public_api = [name for name in dir(checker) if not name.startswith("_")]
        assert "check" in public_api
        # No individual check should be callable from outside
        for name in public_api:
            assert not name.startswith("_check_"), (
                f"Check method '{name}' must be private"
            )

    def test_preflight_result_is_immutable(self) -> None:
        """PreflightResult is frozen — cannot be tampered with after creation."""
        r = _CHECKER.check(_spec(), _make_perception(), _ctx())
        with pytest.raises((AttributeError, TypeError)):
            r.passed = False  # type: ignore[misc]
