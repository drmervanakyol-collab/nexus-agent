"""
tests/unit/source/test_transport.py
Unit tests for nexus/source/transport/resolver.py and
nexus/source/transport/fallback.py.

All tests inject mock callables — no real COM, CDP, or OS input.

Coverage targets
----------------
1. UIA source + click  → uia_invoker called
2. UIA invoke fails    → mouse fallback used
3. DOM source + type   → dom_typer called
4. DOM type fails      → keyboard fallback used
5. visual source       → direct mouse/keyboard (no native attempt)
6. prefer_native=False → direct OS transport regardless of source
7. Audit writer called after every execute()
8. TransportFallbackError raised when all transports fail
9. MouseTransport / KeyboardTransport happy-path and failure
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from nexus.core.errors import TransportFallbackError
from nexus.core.settings import NexusSettings
from nexus.core.types import Rect
from nexus.source.resolver import SourceResult
from nexus.source.transport.fallback import KeyboardTransport, MouseTransport
from nexus.source.transport.resolver import (
    ActionSpec,
    TransportResolver,
    TransportResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(*, prefer_native: bool = True, fallback_to_mouse: bool = True) -> NexusSettings:
    return NexusSettings.model_validate(
        {
            "transport": {
                "prefer_native_action": prefer_native,
                "fallback_to_mouse": fallback_to_mouse,
            }
        }
    )


def _source(source_type: str = "uia", confidence: float = 1.0) -> SourceResult:
    return SourceResult(
        source_type=source_type,  # type: ignore[arg-type]
        data={"ok": True},
        confidence=confidence,
        latency_ms=1.0,
    )


def _element(*, x: int = 10, y: int = 20, w: int = 100, h: int = 40) -> MagicMock:
    elem = MagicMock()
    elem.bounding_rect = Rect(x, y, w, h)
    return elem


def _element_no_rect() -> MagicMock:
    elem = MagicMock()
    elem.bounding_rect = None
    return elem


def _click_spec(task_id: str = "t1") -> ActionSpec:
    return ActionSpec(action_type="click", task_id=task_id)


def _type_spec(text: str = "hello", task_id: str = "t1") -> ActionSpec:
    return ActionSpec(action_type="type", text=text, task_id=task_id)


async def _ok_audit(result: TransportResult, spec: ActionSpec) -> None:
    pass


# ---------------------------------------------------------------------------
# MouseTransport unit tests
# ---------------------------------------------------------------------------


class TestMouseTransport:
    @pytest.mark.asyncio
    async def test_click_calls_click_fn(self):
        calls: list[tuple[int, int]] = []

        def click_fn(x: int, y: int) -> None:
            calls.append((x, y))

        mt = MouseTransport(_click_fn=click_fn)
        ok = await mt.click(50, 60)
        assert ok is True
        assert calls == [(50, 60)]

    @pytest.mark.asyncio
    async def test_click_returns_false_on_exception(self):
        def bad_click(x: int, y: int) -> None:
            raise OSError("no mouse")

        mt = MouseTransport(_click_fn=bad_click)
        assert await mt.click(0, 0) is False

    @pytest.mark.asyncio
    async def test_click_coordinates_forwarded(self):
        received: list[tuple[int, int]] = []

        mt = MouseTransport(_click_fn=lambda x, y: received.append((x, y)))
        await mt.click(123, 456)
        assert received == [(123, 456)]


# ---------------------------------------------------------------------------
# KeyboardTransport unit tests
# ---------------------------------------------------------------------------


class TestKeyboardTransport:
    @pytest.mark.asyncio
    async def test_type_calls_type_fn(self):
        typed: list[str] = []
        kt = KeyboardTransport(_type_fn=lambda t: typed.append(t))
        ok = await kt.type_text("abc")
        assert ok is True
        assert typed == ["abc"]

    @pytest.mark.asyncio
    async def test_type_returns_false_on_exception(self):
        kt = KeyboardTransport(_type_fn=lambda t: (_ for _ in ()).throw(OSError("no kb")))
        assert await kt.type_text("x") is False

    @pytest.mark.asyncio
    async def test_type_empty_string(self):
        typed: list[str] = []
        kt = KeyboardTransport(_type_fn=lambda t: typed.append(t))
        ok = await kt.type_text("")
        assert ok is True
        assert typed == [""]


# ---------------------------------------------------------------------------
# TransportResolver — UIA click path
# ---------------------------------------------------------------------------


class TestUiaClickPath:
    @pytest.mark.asyncio
    async def test_uia_click_calls_invoker(self):
        """UIA source + click → invoker is called with the element."""
        invoke_calls: list[Any] = []

        def invoker(elem: Any) -> bool:
            invoke_calls.append(elem)
            return True

        resolver = TransportResolver(
            _settings(),
            _uia_invoker=invoker,
            _audit_writer=_ok_audit,
        )
        elem = _element()
        result = await resolver.execute(_click_spec(), _source("uia"), elem)

        assert result.method_used == "uia"
        assert result.success is True
        assert result.fallback_used is False
        assert len(invoke_calls) == 1
        assert invoke_calls[0] is elem

    @pytest.mark.asyncio
    async def test_uia_invoke_fail_uses_mouse_fallback(self):
        """UIA invoke returns False → mouse fallback activated."""
        mouse_clicks: list[tuple[int, int]] = []
        mouse = MouseTransport(_click_fn=lambda x, y: mouse_clicks.append((x, y)))

        resolver = TransportResolver(
            _settings(),
            _uia_invoker=lambda e: False,
            _mouse_transport=mouse,
            _audit_writer=_ok_audit,
        )
        # Element centre: x=10+100//2=60, y=20+40//2=40
        elem = _element(x=10, y=20, w=100, h=40)
        result = await resolver.execute(_click_spec(), _source("uia"), elem)

        assert result.method_used == "mouse"
        assert result.success is True
        assert result.fallback_used is True
        assert mouse_clicks == [(60, 40)]

    @pytest.mark.asyncio
    async def test_uia_no_invoker_falls_to_mouse(self):
        """No UIA invoker wired → mouse fallback without native attempt."""
        mouse_clicks: list[tuple] = []
        mouse = MouseTransport(_click_fn=lambda x, y: mouse_clicks.append((x, y)))

        resolver = TransportResolver(
            _settings(),
            _mouse_transport=mouse,
            _audit_writer=_ok_audit,
        )
        result = await resolver.execute(_click_spec(), _source("uia"), _element())
        assert result.method_used == "mouse"
        assert len(mouse_clicks) == 1


# ---------------------------------------------------------------------------
# TransportResolver — UIA type path
# ---------------------------------------------------------------------------


class TestUiaTypePath:
    @pytest.mark.asyncio
    async def test_uia_type_calls_value_setter(self):
        set_calls: list[tuple[Any, str]] = []

        def setter(elem: Any, text: str) -> bool:
            set_calls.append((elem, text))
            return True

        resolver = TransportResolver(
            _settings(),
            _uia_value_setter=setter,
            _audit_writer=_ok_audit,
        )
        elem = _element()
        result = await resolver.execute(_type_spec("world"), _source("uia"), elem)

        assert result.method_used == "uia"
        assert result.success is True
        assert set_calls[0][1] == "world"

    @pytest.mark.asyncio
    async def test_uia_type_fail_uses_keyboard_fallback(self):
        typed: list[str] = []
        kb = KeyboardTransport(_type_fn=lambda t: typed.append(t))

        resolver = TransportResolver(
            _settings(),
            _uia_value_setter=lambda e, t: False,
            _keyboard_transport=kb,
            _audit_writer=_ok_audit,
        )
        result = await resolver.execute(_type_spec("hi"), _source("uia"), _element())

        assert result.method_used == "keyboard"
        assert result.success is True
        assert result.fallback_used is True
        assert typed == ["hi"]


# ---------------------------------------------------------------------------
# TransportResolver — DOM paths
# ---------------------------------------------------------------------------


class TestDomPaths:
    @pytest.mark.asyncio
    async def test_dom_click_calls_dom_clicker(self):
        click_calls: list[Any] = []

        async def dom_clicker(elem: Any) -> bool:
            click_calls.append(elem)
            return True

        resolver = TransportResolver(
            _settings(),
            _dom_clicker=dom_clicker,
            _audit_writer=_ok_audit,
        )
        elem = _element()
        result = await resolver.execute(_click_spec(), _source("dom"), elem)

        assert result.method_used == "dom"
        assert result.success is True
        assert click_calls[0] is elem

    @pytest.mark.asyncio
    async def test_dom_type_calls_dom_typer(self):
        typed: list[tuple[Any, str]] = []

        async def dom_typer(elem: Any, text: str) -> bool:
            typed.append((elem, text))
            return True

        resolver = TransportResolver(
            _settings(),
            _dom_typer=dom_typer,
            _audit_writer=_ok_audit,
        )
        elem = _element()
        result = await resolver.execute(_type_spec("nexus"), _source("dom"), elem)

        assert result.method_used == "dom"
        assert typed[0][1] == "nexus"

    @pytest.mark.asyncio
    async def test_dom_click_fail_uses_mouse_fallback(self):
        mouse_clicks: list[tuple] = []
        mouse = MouseTransport(_click_fn=lambda x, y: mouse_clicks.append((x, y)))

        async def failing_clicker(elem: Any) -> bool:
            return False

        resolver = TransportResolver(
            _settings(),
            _dom_clicker=failing_clicker,
            _mouse_transport=mouse,
            _audit_writer=_ok_audit,
        )
        result = await resolver.execute(_click_spec(), _source("dom"), _element())
        assert result.method_used == "mouse"
        assert result.fallback_used is True

    @pytest.mark.asyncio
    async def test_dom_type_fail_uses_keyboard_fallback(self):
        typed: list[str] = []
        kb = KeyboardTransport(_type_fn=lambda t: typed.append(t))

        async def failing_typer(elem: Any, text: str) -> bool:
            return False

        resolver = TransportResolver(
            _settings(),
            _dom_typer=failing_typer,
            _keyboard_transport=kb,
            _audit_writer=_ok_audit,
        )
        result = await resolver.execute(_type_spec("abc"), _source("dom"), _element())
        assert result.method_used == "keyboard"
        assert result.fallback_used is True
        assert typed == ["abc"]


# ---------------------------------------------------------------------------
# TransportResolver — visual source
# ---------------------------------------------------------------------------


class TestVisualSource:
    @pytest.mark.asyncio
    async def test_visual_click_uses_mouse_directly(self):
        """visual source → no UIA/DOM attempt → direct mouse."""
        invoke_calls: list[Any] = []
        mouse_clicks: list[tuple] = []
        mouse = MouseTransport(_click_fn=lambda x, y: mouse_clicks.append((x, y)))

        resolver = TransportResolver(
            _settings(),
            _uia_invoker=lambda e: invoke_calls.append(e) or True,
            _mouse_transport=mouse,
            _audit_writer=_ok_audit,
        )
        result = await resolver.execute(_click_spec(), _source("visual"), _element())

        assert result.method_used == "mouse"
        assert result.fallback_used is False
        assert invoke_calls == []  # native never called
        assert len(mouse_clicks) == 1

    @pytest.mark.asyncio
    async def test_visual_type_uses_keyboard_directly(self):
        typed: list[str] = []
        kb = KeyboardTransport(_type_fn=lambda t: typed.append(t))

        resolver = TransportResolver(
            _settings(),
            _keyboard_transport=kb,
            _audit_writer=_ok_audit,
        )
        result = await resolver.execute(_type_spec("vision"), _source("visual"), None)
        assert result.method_used == "keyboard"
        assert typed == ["vision"]


# ---------------------------------------------------------------------------
# TransportResolver — prefer_native_action=False
# ---------------------------------------------------------------------------


class TestPreferNativeFalse:
    @pytest.mark.asyncio
    async def test_prefer_native_false_click_uses_mouse(self):
        """prefer_native=False → mouse directly, even for UIA source."""
        invoke_calls: list[Any] = []
        mouse_clicks: list[tuple] = []
        mouse = MouseTransport(_click_fn=lambda x, y: mouse_clicks.append((x, y)))

        resolver = TransportResolver(
            _settings(prefer_native=False),
            _uia_invoker=lambda e: invoke_calls.append(e) or True,
            _mouse_transport=mouse,
            _audit_writer=_ok_audit,
        )
        result = await resolver.execute(_click_spec(), _source("uia"), _element())

        assert result.method_used == "mouse"
        assert invoke_calls == []
        assert len(mouse_clicks) == 1

    @pytest.mark.asyncio
    async def test_prefer_native_false_type_uses_keyboard(self):
        typed: list[str] = []
        kb = KeyboardTransport(_type_fn=lambda t: typed.append(t))
        setter_calls: list[Any] = []

        resolver = TransportResolver(
            _settings(prefer_native=False),
            _uia_value_setter=lambda e, t: setter_calls.append(t) or True,
            _keyboard_transport=kb,
            _audit_writer=_ok_audit,
        )
        result = await resolver.execute(_type_spec("skip"), _source("uia"), _element())

        assert result.method_used == "keyboard"
        assert setter_calls == []
        assert typed == ["skip"]


# ---------------------------------------------------------------------------
# TransportResolver — audit
# ---------------------------------------------------------------------------


class TestAudit:
    @pytest.mark.asyncio
    async def test_audit_writer_called_on_success(self):
        audit_calls: list[tuple[TransportResult, ActionSpec]] = []

        async def audit(result: TransportResult, spec: ActionSpec) -> None:
            audit_calls.append((result, spec))

        resolver = TransportResolver(
            _settings(),
            _uia_invoker=lambda e: True,
            _audit_writer=audit,
        )
        spec = _click_spec("task-42")
        await resolver.execute(spec, _source("uia"), _element())

        assert len(audit_calls) == 1
        assert audit_calls[0][1].task_id == "task-42"
        assert audit_calls[0][0].success is True

    @pytest.mark.asyncio
    async def test_audit_writer_called_on_fallback(self):
        audit_calls: list[tuple[TransportResult, ActionSpec]] = []

        async def audit(result: TransportResult, spec: ActionSpec) -> None:
            audit_calls.append((result, spec))

        mouse = MouseTransport(_click_fn=lambda x, y: None)

        resolver = TransportResolver(
            _settings(),
            _uia_invoker=lambda e: False,
            _mouse_transport=mouse,
            _audit_writer=audit,
        )
        await resolver.execute(_click_spec(), _source("uia"), _element())

        assert len(audit_calls) == 1
        assert audit_calls[0][0].fallback_used is True

    @pytest.mark.asyncio
    async def test_audit_failure_does_not_mask_success(self):
        """If the audit writer raises, the TransportResult is still returned."""

        async def bad_audit(result: TransportResult, spec: ActionSpec) -> None:
            raise RuntimeError("db gone")

        resolver = TransportResolver(
            _settings(),
            _uia_invoker=lambda e: True,
            _audit_writer=bad_audit,
        )
        result = await resolver.execute(_click_spec(), _source("uia"), _element())
        assert result.success is True


# ---------------------------------------------------------------------------
# TransportResolver — TransportFallbackError
# ---------------------------------------------------------------------------


class TestTransportFallbackError:
    @pytest.mark.asyncio
    async def test_raises_when_all_fail(self):
        """Native + fallback both fail → TransportFallbackError raised."""
        resolver = TransportResolver(
            _settings(),
            _uia_invoker=lambda e: False,
            _mouse_transport=MouseTransport(_click_fn=lambda x, y: (_ for _ in ()).throw(OSError())),
            _audit_writer=_ok_audit,
        )
        with pytest.raises(TransportFallbackError):
            await resolver.execute(_click_spec(), _source("uia"), _element())

    @pytest.mark.asyncio
    async def test_raises_when_mouse_has_no_coordinates(self):
        """Mouse fallback for click with no bounding_rect → coords=None → fail."""
        resolver = TransportResolver(
            _settings(),
            _uia_invoker=lambda e: False,
            _mouse_transport=MouseTransport(_click_fn=lambda x, y: None),
            _audit_writer=_ok_audit,
        )
        # Element with no bounding_rect → coords=None → mouse.click not called
        with pytest.raises(TransportFallbackError):
            await resolver.execute(_click_spec(), _source("uia"), _element_no_rect())

    @pytest.mark.asyncio
    async def test_error_code_is_transport_fallback(self):
        resolver = TransportResolver(
            _settings(),
            _uia_invoker=lambda e: False,
            _mouse_transport=MouseTransport(
                _click_fn=lambda x, y: (_ for _ in ()).throw(OSError())
            ),
            _audit_writer=_ok_audit,
        )
        with pytest.raises(TransportFallbackError) as exc_info:
            await resolver.execute(_click_spec(), _source("uia"), _element())
        assert exc_info.value.code == "transport_fallback_error"


# ---------------------------------------------------------------------------
# TransportResult data model
# ---------------------------------------------------------------------------


class TestTransportResult:
    def test_can_construct(self):
        r = TransportResult(
            method_used="uia",
            success=True,
            latency_ms=2.5,
            fallback_used=False,
        )
        assert r.method_used == "uia"
        assert r.latency_ms == pytest.approx(2.5)

    def test_all_method_types_accepted(self):
        for m in ("uia", "dom", "mouse", "keyboard"):
            r = TransportResult(
                method_used=m,  # type: ignore[arg-type]
                success=True,
                latency_ms=0.0,
                fallback_used=False,
            )
            assert r.method_used == m


# ---------------------------------------------------------------------------
# ActionSpec data model
# ---------------------------------------------------------------------------


class TestActionSpec:
    def test_defaults(self):
        spec = ActionSpec(action_type="click")
        assert spec.text is None
        assert spec.task_id == ""
        assert spec.action_id is None

    def test_with_text(self):
        spec = ActionSpec(action_type="type", text="hello")
        assert spec.text == "hello"

    def test_all_action_types(self):
        for at in ("click", "type", "focus", "clear", "select"):
            spec = ActionSpec(action_type=at)  # type: ignore[arg-type]
            assert spec.action_type == at
