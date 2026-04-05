"""
tests/unit/test_action_registry.py
Unit tests for nexus/action/registry.py.

Coverage
--------
  ActionSpec    — field defaults, explicit values
  ActionResult  — field defaults, explicit values
  ActionHandler — isinstance check via runtime_checkable Protocol
  ActionRegistry — register, get, list_types, overwrite, empty state
"""
from __future__ import annotations

import pytest

from nexus.action.registry import (
    ActionHandler,
    ActionRegistry,
    ActionResult,
    ActionSpec,
)


# ---------------------------------------------------------------------------
# Concrete handler for tests
# ---------------------------------------------------------------------------


class _ClickHandler:
    async def execute(self, spec: ActionSpec) -> ActionResult:
        return ActionResult(success=True, transport_used="mouse")

    def validate(self, spec: ActionSpec) -> bool:
        return spec.action_type == "click"


class _TypeHandler:
    async def execute(self, spec: ActionSpec) -> ActionResult:
        return ActionResult(success=True, transport_used="keyboard")

    def validate(self, spec: ActionSpec) -> bool:
        return spec.action_type == "type" and spec.value is not None


# ---------------------------------------------------------------------------
# ActionSpec
# ---------------------------------------------------------------------------


class TestActionSpec:
    def test_required_field(self) -> None:
        spec = ActionSpec(action_type="click")
        assert spec.action_type == "click"

    def test_defaults(self) -> None:
        spec = ActionSpec(action_type="click")
        assert spec.target_element_id is None
        assert spec.coordinates is None
        assert spec.value is None
        assert spec.is_destructive is False
        assert spec.preferred_transport is None
        assert spec.metadata == {}

    def test_all_fields(self) -> None:
        spec = ActionSpec(
            action_type="type",
            target_element_id="el-1",
            coordinates=(100, 200),
            value="hello",
            is_destructive=True,
            preferred_transport="uia",
            metadata={"source": "cloud"},
        )
        assert spec.action_type == "type"
        assert spec.target_element_id == "el-1"
        assert spec.coordinates == (100, 200)
        assert spec.value == "hello"
        assert spec.is_destructive is True
        assert spec.preferred_transport == "uia"
        assert spec.metadata == {"source": "cloud"}

    def test_metadata_is_independent_per_instance(self) -> None:
        a = ActionSpec(action_type="click")
        b = ActionSpec(action_type="click")
        a.metadata["k"] = "v"
        assert "k" not in b.metadata


# ---------------------------------------------------------------------------
# ActionResult
# ---------------------------------------------------------------------------


class TestActionResult:
    def test_required_field(self) -> None:
        r = ActionResult(success=True)
        assert r.success is True

    def test_failure_defaults(self) -> None:
        r = ActionResult(success=False)
        assert r.duration_ms == 0.0
        assert r.error is None
        assert r.partial_completion is False
        assert r.side_effects == []
        assert r.transport_used is None

    def test_all_fields(self) -> None:
        r = ActionResult(
            success=False,
            duration_ms=42.5,
            error="timeout",
            partial_completion=True,
            side_effects=["dialog_opened"],
            transport_used="uia",
        )
        assert r.success is False
        assert r.duration_ms == 42.5
        assert r.error == "timeout"
        assert r.partial_completion is True
        assert r.side_effects == ["dialog_opened"]
        assert r.transport_used == "uia"

    def test_side_effects_independent_per_instance(self) -> None:
        a = ActionResult(success=True)
        b = ActionResult(success=True)
        a.side_effects.append("x")
        assert "x" not in b.side_effects


# ---------------------------------------------------------------------------
# ActionHandler protocol
# ---------------------------------------------------------------------------


class TestActionHandlerProtocol:
    def test_concrete_handler_satisfies_protocol(self) -> None:
        handler = _ClickHandler()
        assert isinstance(handler, ActionHandler)

    def test_object_missing_execute_does_not_satisfy(self) -> None:
        class _Bad:
            def validate(self, spec: ActionSpec) -> bool:
                return True

        assert not isinstance(_Bad(), ActionHandler)

    def test_object_missing_validate_does_not_satisfy(self) -> None:
        class _Bad:
            async def execute(self, spec: ActionSpec) -> ActionResult:
                return ActionResult(success=True)

        assert not isinstance(_Bad(), ActionHandler)

    def test_type_handler_satisfies_protocol(self) -> None:
        assert isinstance(_TypeHandler(), ActionHandler)


# ---------------------------------------------------------------------------
# ActionRegistry
# ---------------------------------------------------------------------------


class TestActionRegistry:
    def test_empty_registry_get_returns_none(self) -> None:
        reg = ActionRegistry()
        assert reg.get("click") is None

    def test_empty_registry_list_types_is_empty(self) -> None:
        reg = ActionRegistry()
        assert reg.list_types() == []

    def test_register_and_get(self) -> None:
        reg = ActionRegistry()
        h = _ClickHandler()
        reg.register("click", h)
        assert reg.get("click") is h

    def test_get_unknown_type_returns_none(self) -> None:
        reg = ActionRegistry()
        reg.register("click", _ClickHandler())
        assert reg.get("type") is None

    def test_list_types_sorted(self) -> None:
        reg = ActionRegistry()
        reg.register("type", _TypeHandler())
        reg.register("click", _ClickHandler())
        assert reg.list_types() == ["click", "type"]

    def test_list_types_single(self) -> None:
        reg = ActionRegistry()
        reg.register("scroll", _ClickHandler())
        assert reg.list_types() == ["scroll"]

    def test_register_overwrites_existing(self) -> None:
        reg = ActionRegistry()
        h1 = _ClickHandler()
        h2 = _ClickHandler()
        reg.register("click", h1)
        reg.register("click", h2)
        assert reg.get("click") is h2

    def test_multiple_handlers_independent(self) -> None:
        reg = ActionRegistry()
        hc = _ClickHandler()
        ht = _TypeHandler()
        reg.register("click", hc)
        reg.register("type", ht)
        assert reg.get("click") is hc
        assert reg.get("type") is ht

    def test_list_types_reflects_all_registrations(self) -> None:
        reg = ActionRegistry()
        for name in ("scroll", "click", "type", "focus"):
            reg.register(name, _ClickHandler())
        assert reg.list_types() == ["click", "focus", "scroll", "type"]

    def test_registries_are_independent(self) -> None:
        r1 = ActionRegistry()
        r2 = ActionRegistry()
        r1.register("click", _ClickHandler())
        assert r2.get("click") is None
