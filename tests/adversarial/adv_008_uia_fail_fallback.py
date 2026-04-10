"""
tests/adversarial/adv_008_uia_fail_fallback.py
Adversarial Test 008 — UIA crash → mouse fallback + audit record

Scenario:
  SafeFieldReplace is configured with preferred_transport="uia".
  The native UIA click function raises an exception (simulating a UIA crash).

Success criteria:
  - Mouse fallback is invoked.
  - MacroActionResult.success is True (fallback succeeded).
  - step_results[0].transport_used == "mouse" (audit: fallback recorded).
  - No unhandled exception propagates.

All I/O is injected — no real UIA calls.
"""
from __future__ import annotations

import asyncio

import pytest

from nexus.action.macroactions import SafeFieldReplace

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.adversarial
class TestUIAFailFallback:
    """ADV-008: UIA crash triggers mouse fallback; audit trail reflects it."""

    def test_uia_crash_triggers_mouse_fallback(self):
        """
        Native UIA click raises RuntimeError →
        fallback click used → success=True, transport_used='mouse'.
        """
        fallback_calls: list[tuple[int, int]] = []

        async def _crashing_native(_element_id) -> bool:
            raise RuntimeError("UIA COM crash simulated")

        async def _mouse_fallback(coords: tuple[int, int]) -> bool:
            fallback_calls.append(coords)
            return True

        async def _type(_text: str) -> bool:
            return True

        async def _hotkey(_keys) -> bool:
            return True

        async def _ocr_read(_coords) -> str | None:
            return "test_value"

        field = SafeFieldReplace(
            _native_click_fn=_crashing_native,
            _fallback_click_fn=_mouse_fallback,
            _type_fn=_type,
            _hotkey_fn=_hotkey,
            _ocr_read_fn=_ocr_read,
            preferred_transport="uia",
            max_retries=1,
        )

        result = asyncio.run(
            field.execute((200, 300), "test_value", element_id="el-1")
        )

        assert result.success is True, (
            f"Fallback must succeed; got success={result.success!r}, "
            f"error={result.error!r}"
        )
        assert len(fallback_calls) >= 1, "Mouse fallback must be invoked"

        # Audit trail: first click step must record mouse transport
        first_click = result.step_results[0]
        assert first_click.transport_used == "mouse", (
            f"Audit trail must record 'mouse'; got {first_click.transport_used!r}"
        )

    def test_uia_success_no_fallback(self):
        """
        When native UIA click succeeds, mouse fallback must NOT be called.
        """
        fallback_calls: list[tuple[int, int]] = []

        async def _native_ok(_element_id) -> bool:
            return True

        async def _mouse_fallback(coords: tuple[int, int]) -> bool:
            fallback_calls.append(coords)
            return True

        async def _type(_text: str) -> bool:
            return True

        async def _hotkey(_keys) -> bool:
            return True

        async def _ocr_read(_coords) -> str | None:
            return "value"

        field = SafeFieldReplace(
            _native_click_fn=_native_ok,
            _fallback_click_fn=_mouse_fallback,
            _type_fn=_type,
            _hotkey_fn=_hotkey,
            _ocr_read_fn=_ocr_read,
            preferred_transport="uia",
            max_retries=1,
        )

        result = asyncio.run(
            field.execute((100, 200), "value", element_id="el-2")
        )

        assert result.success is True
        assert len(fallback_calls) == 0, (
            "Mouse fallback must NOT be called when UIA succeeds"
        )
        # Audit: transport should be 'uia'
        assert result.step_results[0].transport_used == "uia", (
            f"Expected transport_used='uia'; got {result.step_results[0].transport_used!r}"
        )

    def test_both_transports_fail_returns_failure(self):
        """
        UIA crashes and mouse fallback also fails →
        result.success == False (no crash, graceful failure).
        """
        async def _crashing_native(_eid) -> bool:
            raise RuntimeError("UIA crash")

        async def _failing_mouse(_coords) -> bool:
            return False

        async def _type(_t) -> bool:
            return True

        async def _hotkey(_k) -> bool:
            return True

        field = SafeFieldReplace(
            _native_click_fn=_crashing_native,
            _fallback_click_fn=_failing_mouse,
            _type_fn=_type,
            _hotkey_fn=_hotkey,
            preferred_transport="uia",
            max_retries=1,
        )

        result = asyncio.run(field.execute((10, 20), "val"))

        assert result.success is False, (
            "Both transports failed → result must be failure (no exception)"
        )
