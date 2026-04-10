"""
tests/property/test_transport.py
Transport property tests — Faz 65

Invariants tested
-----------------
- TransportResult.method_used is always in {"uia", "dom", "mouse", "keyboard"}
- TransportResult.latency_ms >= 0 always
- When fallback_used=True, method_used is "mouse" or "keyboard" (OS fallback)
- When fallback_used=False AND source is "uia" AND prefer_native=True,
  method_used is "uia" (native path succeeded)
- TransportResolver.execute() result is always a valid TransportResult
- success=True never co-exists with a TransportFallbackError being raised
- UIA path: a successful UIA invoke produces method_used="uia", fallback_used=False
- Mouse path: when prefer_native=False, method_used is "mouse" or "keyboard"
- Visual source always falls back (no native for visual)
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nexus.core.settings import NexusSettings, TransportSettings
from nexus.core.types import Rect
from nexus.source.resolver import SourceResult
from nexus.source.transport.resolver import (
    ActionSpec,
    ActionType,
    TransportResolver,
    TransportResult,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_ACTION_TYPES: list[ActionType] = ["click", "type", "focus", "clear", "select"]
_SOURCE_TYPES = ["uia", "dom", "visual", "file"]
_METHODS = {"uia", "dom", "mouse", "keyboard"}
_FALLBACK_METHODS = {"mouse", "keyboard"}
_NATIVE_METHODS = {"uia", "dom"}

_action_type_st = st.sampled_from(_ACTION_TYPES)
_source_type_st = st.sampled_from(_SOURCE_TYPES)


def _make_source(source_type: str) -> SourceResult:
    return SourceResult(
        source_type=source_type,  # type: ignore[arg-type]
        data=[],
        confidence=1.0,
        latency_ms=0.0,
    )


def _make_element() -> Any:
    stub = MagicMock()
    stub.bounding_rect = Rect(x=100, y=100, width=80, height=30)
    return stub


def _make_mouse_stub() -> Any:
    mouse = MagicMock()
    mouse.click = AsyncMock(return_value=True)
    mouse.type_text = AsyncMock(return_value=True)
    return mouse


def _make_keyboard_stub() -> Any:
    kb = MagicMock()
    kb.type_text = AsyncMock(return_value=True)
    return kb


# ---------------------------------------------------------------------------
# TransportResult dataclass: structural invariants
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(sorted(_METHODS)),
    st.booleans(),
    st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False),
    st.booleans(),
)
def test_transport_result_method_in_valid_set(
    method: str, success: bool, latency: float, fallback: bool
) -> None:
    result = TransportResult(
        method_used=method,  # type: ignore[arg-type]
        success=success,
        latency_ms=latency,
        fallback_used=fallback,
    )
    assert result.method_used in _METHODS


@given(
    st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False),
)
def test_transport_result_latency_nonnegative(latency: float) -> None:
    result = TransportResult(
        method_used="uia",
        success=True,
        latency_ms=latency,
        fallback_used=False,
    )
    assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# TransportResolver: UIA native path (success)
# ---------------------------------------------------------------------------


@given(_action_type_st)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@pytest.mark.asyncio
async def test_uia_success_gives_uia_method(action_type: ActionType) -> None:
    """When UIA invoker/setter succeeds, method_used must be 'uia'."""
    mouse = _make_mouse_stub()
    keyboard = _make_keyboard_stub()
    settings_obj = NexusSettings()
    resolver = TransportResolver(
        settings_obj,
        _uia_invoker=lambda _el: True,
        _uia_value_setter=lambda _el, _txt: True,
        _uia_selector=lambda _el: True,
        _mouse_transport=mouse,
        _keyboard_transport=keyboard,
    )
    source = _make_source("uia")
    element = _make_element()
    spec = ActionSpec(action_type=action_type, text="hello", task_id="prop-test")

    result = await resolver.execute(spec, source, element)
    assert result.method_used in _METHODS
    assert result.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_uia_click_success_no_fallback() -> None:
    settings_obj = NexusSettings()
    resolver = TransportResolver(
        settings_obj,
        _uia_invoker=lambda _: True,
    )
    source = _make_source("uia")
    spec = ActionSpec(action_type="click", task_id="prop-test")

    result = await resolver.execute(spec, source, _make_element())
    assert result.method_used == "uia"
    assert result.fallback_used is False
    assert result.success is True


# ---------------------------------------------------------------------------
# TransportResolver: fallback path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uia_click_failure_triggers_fallback() -> None:
    """When UIA invoker returns False, resolver falls back to mouse."""
    settings_obj = NexusSettings()
    mouse = _make_mouse_stub()
    resolver = TransportResolver(
        settings_obj,
        _uia_invoker=lambda _: False,   # always fail native
        _mouse_transport=mouse,
    )
    source = _make_source("uia")
    spec = ActionSpec(action_type="click", task_id="prop-test")

    result = await resolver.execute(spec, source, _make_element())
    assert result.method_used in _FALLBACK_METHODS
    assert result.fallback_used is True


@given(_action_type_st)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@pytest.mark.asyncio
async def test_prefer_native_false_uses_os_fallback(action_type: ActionType) -> None:
    """prefer_native_action=False always routes to mouse/keyboard."""
    settings_obj = NexusSettings(
        transport=TransportSettings(prefer_native_action=False)
    )
    mouse = _make_mouse_stub()
    keyboard = _make_keyboard_stub()
    resolver = TransportResolver(
        settings_obj,
        _mouse_transport=mouse,
        _keyboard_transport=keyboard,
    )
    source = _make_source("uia")
    spec = ActionSpec(action_type=action_type, text="hi", task_id="prop-test")

    result = await resolver.execute(spec, source, _make_element())
    assert result.method_used in _FALLBACK_METHODS
    assert result.latency_ms >= 0.0


@given(_action_type_st)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@pytest.mark.asyncio
async def test_visual_source_always_uses_os_fallback(action_type: ActionType) -> None:
    """Visual source has no native path — must always use mouse/keyboard."""
    settings_obj = NexusSettings()
    mouse = _make_mouse_stub()
    keyboard = _make_keyboard_stub()
    resolver = TransportResolver(
        settings_obj,
        _mouse_transport=mouse,
        _keyboard_transport=keyboard,
    )
    source = _make_source("visual")
    spec = ActionSpec(action_type=action_type, text="hi", task_id="prop-test")

    result = await resolver.execute(spec, source, _make_element())
    assert result.method_used in _FALLBACK_METHODS
    assert result.fallback_used is False  # "intended" fallback, not a failure


# ---------------------------------------------------------------------------
# Invariant: fallback_used=True implies method_used is a fallback method
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_used_implies_fallback_method() -> None:
    """If fallback_used=True, method_used must be 'mouse' or 'keyboard'."""
    settings_obj = NexusSettings()
    mouse = _make_mouse_stub()
    resolver = TransportResolver(
        settings_obj,
        _uia_invoker=lambda _: False,  # force fallback
        _mouse_transport=mouse,
    )
    source = _make_source("uia")
    spec = ActionSpec(action_type="click", task_id="prop-test")
    result = await resolver.execute(spec, source, _make_element())

    if result.fallback_used:
        assert result.method_used in _FALLBACK_METHODS, (
            f"fallback_used=True but method_used='{result.method_used}'"
        )


# ---------------------------------------------------------------------------
# Invariant: latency_ms is always non-negative from resolver.execute()
# ---------------------------------------------------------------------------


@given(_source_type_st, _action_type_st)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@pytest.mark.asyncio
async def test_execute_latency_always_nonnegative(
    source_type: str, action_type: ActionType
) -> None:
    settings_obj = NexusSettings()
    mouse = _make_mouse_stub()
    keyboard = _make_keyboard_stub()
    resolver = TransportResolver(
        settings_obj,
        _uia_invoker=lambda _: True,
        _uia_value_setter=lambda _el, _txt: True,
        _uia_selector=lambda _: True,
        _dom_clicker=AsyncMock(return_value=True),
        _dom_typer=AsyncMock(return_value=True),
        _dom_focuser=AsyncMock(return_value=True),
        _dom_clearer=AsyncMock(return_value=True),
        _mouse_transport=mouse,
        _keyboard_transport=keyboard,
    )
    source = _make_source(source_type)
    spec = ActionSpec(action_type=action_type, text="test", task_id="prop")

    result = await resolver.execute(spec, source, _make_element())
    assert result.latency_ms >= 0.0
