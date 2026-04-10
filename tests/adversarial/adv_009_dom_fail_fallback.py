"""
tests/adversarial/adv_009_dom_fail_fallback.py
Adversarial Test 009 — CDP connection severed → visual fallback

Scenario:
  DOMAdapter.is_available() returns False because the WebSocket session
  factory raises ConnectionRefusedError (simulating Chrome DevTools being
  disconnected).

Success criteria:
  - DOMAdapter.is_available() returns False (no crash).
  - PreflightChecker with dom_available=False and allow_transport_fallback=True
    does NOT fail on CHECK_TRANSPORT_AVAILABLE.
  - PreflightChecker with dom_available=False and allow_transport_fallback=False
    DOES fail on CHECK_TRANSPORT_AVAILABLE.

All I/O is injected — no real Chrome/CDP connections.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from nexus.action.preflight import (
    CHECK_TRANSPORT_AVAILABLE,
    PreflightChecker,
    PreflightContext,
)
from nexus.action.registry import ActionSpec
from nexus.core.settings import NexusSettings
from nexus.source.dom.adapter import DOMAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _refusing_factory(_port: int):
    """Simulate a refused CDP connection."""
    raise ConnectionRefusedError("Chrome DevTools not available")
    yield  # unreachable; required for @asynccontextmanager typing


def _make_perception():
    p = MagicMock()
    p.spatial_graph.get_node.return_value = None   # no element_id needed
    p.screen_state.blocks_perception = False
    p.source_result.source_type = "dom"
    return p


_CHECKER = PreflightChecker()


def _ctx(**kwargs) -> PreflightContext:
    return PreflightContext(screen_width=1920, screen_height=1080, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.adversarial
class TestDOMFailFallback:
    """ADV-009: CDP disconnect — DOMAdapter graceful failure + preflight fallback logic."""

    def test_dom_adapter_returns_false_when_connection_refused(self):
        """
        DOMAdapter.is_available() returns False when the session factory
        raises ConnectionRefusedError.  Must not raise.
        """
        settings = NexusSettings()
        adapter = DOMAdapter(settings, _session_factory=_refusing_factory)

        result = asyncio.run(adapter.is_available())

        assert result is False, (
            "is_available() must return False on CDP connection failure"
        )

    def test_fallback_allowed_skips_transport_check(self):
        """
        dom_available=False but allow_transport_fallback=True →
        CHECK_TRANSPORT_AVAILABLE does NOT fail (preflight passes).
        """
        spec = ActionSpec(
            action_type="click",
            target_element_id=None,
            coordinates=(200, 300),
            preferred_transport="dom",
        )
        perception = _make_perception()
        ctx = _ctx(dom_available=False, allow_transport_fallback=True)
        result = _CHECKER.check(spec, perception, ctx)

        assert result.failed_check != CHECK_TRANSPORT_AVAILABLE, (
            "Transport check must be skipped when fallback is allowed"
        )

    def test_fallback_not_allowed_fails_transport_check(self):
        """
        dom_available=False and allow_transport_fallback=False →
        CHECK_TRANSPORT_AVAILABLE fails.
        """
        spec = ActionSpec(
            action_type="click",
            target_element_id=None,
            coordinates=(200, 300),
            preferred_transport="dom",
        )
        perception = _make_perception()
        ctx = _ctx(dom_available=False, allow_transport_fallback=False)
        result = _CHECKER.check(spec, perception, ctx)

        assert result.passed is False
        assert result.failed_check == CHECK_TRANSPORT_AVAILABLE, (
            f"Expected CHECK_TRANSPORT_AVAILABLE; got {result.failed_check!r}"
        )

    def test_dom_available_passes_transport_check(self):
        """When DOM is available, transport check passes (baseline)."""
        spec = ActionSpec(
            action_type="click",
            target_element_id=None,
            coordinates=(200, 300),
            preferred_transport="dom",
        )
        perception = _make_perception()
        ctx = _ctx(dom_available=True, allow_transport_fallback=False)
        result = _CHECKER.check(spec, perception, ctx)

        assert result.failed_check != CHECK_TRANSPORT_AVAILABLE, (
            "Transport check must pass when DOM is available"
        )
