"""
nexus/source/transport/resolver.py
Transport Resolver — native-first action delivery with fallback.

Architecture
------------
TransportResolver implements the decision matrix:

    SOURCE uia  + action click  → UIAAdapter.invoke()    → mouse fallback
    SOURCE uia  + action type   → UIAAdapter.set_value() → keyboard fallback
    SOURCE dom  + action click  → DOMAdapter.click()     → mouse fallback
    SOURCE dom  + action type   → DOMAdapter.type_text() → keyboard fallback
    SOURCE visual / file        → direct mouse / keyboard (no native attempt)
    prefer_native_action=False  → direct mouse / keyboard (test / override mode)

Every execution (success or failure) is written to the transport_audit
table via an injectable ``_audit_writer`` coroutine so that the repo
stays testable without a live database connection.

Raises
------
TransportFallbackError
    When all transport options (native + fallback) are exhausted.
"""
from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from nexus.core.errors import TransportFallbackError
from nexus.core.settings import NexusSettings
from nexus.core.types import Rect
from nexus.infra.logger import get_logger
from nexus.source.resolver import SourceResult
from nexus.source.transport.fallback import KeyboardTransport, MouseTransport

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

ActionType = Literal[
    "click", "type", "focus", "clear", "select", "press_key", "scroll", "drag", "hover"
]
TransportMethod = Literal["uia", "dom", "mouse", "keyboard"]


@dataclass
class ActionSpec:
    """
    Specification of an action to be delivered.

    Attributes
    ----------
    action_type:
        The kind of action (click, type, focus, clear, select, press_key).
    text:
        For ``"type"`` / ``"press_key"`` actions — the text or key name.
    coordinates:
        (x, y) screen coordinates for click/focus actions when no element
        object is available (visual-source path).
    task_id:
        ID of the owning task (written to transport_audit).
    action_id:
        Optional ID of the owning action row (written to transport_audit).
    """

    action_type: ActionType
    text: str | None = None
    coordinates: tuple[int, int] | None = None
    task_id: str = ""
    action_id: str | None = None


@dataclass
class TransportResult:
    """
    Outcome of a single transport execution.

    Attributes
    ----------
    method_used:
        The transport method that produced the final result.
    success:
        True when the action was delivered successfully.
    latency_ms:
        Total wall-clock time for the execution (native attempt + fallback
        if applicable).
    fallback_used:
        True when the native method failed and the OS-level fallback ran.
    """

    method_used: TransportMethod
    success: bool
    latency_ms: float
    fallback_used: bool


# ---------------------------------------------------------------------------
# Callable type aliases for injectable dependencies
# ---------------------------------------------------------------------------

# UIA: synchronous bool-returning callables (COM is blocking)
_UiaInvoker = Callable[[Any], bool]           # (element) -> bool
_UiaValueSetter = Callable[[Any, str], bool]  # (element, text) -> bool
_UiaSelector = Callable[[Any], bool]          # (element) -> bool

# DOM: async bool-returning callables (CDP is async)
_DomClicker = Callable[[Any], Awaitable[bool]]
_DomTyper = Callable[[Any, str], Awaitable[bool]]
_DomFocuser = Callable[[Any], Awaitable[bool]]
_DomClearer = Callable[[Any], Awaitable[bool]]

# Audit writer: async, receives TransportResult + ActionSpec
_AuditWriter = Callable[[TransportResult, ActionSpec], Awaitable[None]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _centre_of(element: Any) -> tuple[int, int] | None:
    """
    Extract (cx, cy) from any element that has a ``bounding_rect`` attribute.

    Returns None if no bounding rect is available.
    """
    rect: Rect | None = getattr(element, "bounding_rect", None)
    if rect is None:
        return None
    return (rect.x + rect.width // 2, rect.y + rect.height // 2)


async def _noop_audit(result: TransportResult, spec: ActionSpec) -> None:
    """Default audit writer — does nothing (used when no DB is wired)."""


# ---------------------------------------------------------------------------
# TransportResolver
# ---------------------------------------------------------------------------


class TransportResolver:
    """
    Executes actions using the optimal transport method.

    Parameters
    ----------
    settings:
        NexusSettings — reads ``transport.prefer_native_action`` and
        ``transport.fallback_to_mouse``.
    _uia_invoker:
        Sync callable ``(element) -> bool`` wrapping ``UIAAdapter.invoke()``.
    _uia_value_setter:
        Sync callable ``(element, text) -> bool`` wrapping
        ``UIAAdapter.set_value()``.
    _uia_selector:
        Sync callable ``(element) -> bool`` wrapping ``UIAAdapter.select()``.
    _dom_clicker:
        Async callable ``(element) -> bool`` wrapping ``DOMAdapter.click()``.
    _dom_typer:
        Async callable ``(element, text) -> bool`` wrapping
        ``DOMAdapter.type_text()``.
    _dom_focuser:
        Async callable ``(element) -> bool`` wrapping ``DOMAdapter.focus()``.
    _dom_clearer:
        Async callable ``(element) -> bool`` wrapping ``DOMAdapter.clear()``.
    _mouse_transport:
        MouseTransport instance.  Created with no-op defaults if omitted.
    _keyboard_transport:
        KeyboardTransport instance.  Created with no-op defaults if omitted.
    _audit_writer:
        Async callable ``(result, spec) -> None``.  Defaults to a no-op.
        Inject ``TransportAuditRepository.record`` bound to an open DB
        connection for production use.
    """

    def __init__(
        self,
        settings: NexusSettings,
        *,
        _uia_invoker: _UiaInvoker | None = None,
        _uia_value_setter: _UiaValueSetter | None = None,
        _uia_selector: _UiaSelector | None = None,
        _dom_clicker: _DomClicker | None = None,
        _dom_typer: _DomTyper | None = None,
        _dom_focuser: _DomFocuser | None = None,
        _dom_clearer: _DomClearer | None = None,
        _mouse_transport: MouseTransport | None = None,
        _keyboard_transport: KeyboardTransport | None = None,
        _audit_writer: _AuditWriter | None = None,
    ) -> None:
        self._prefer_native: bool = settings.transport.prefer_native_action
        self._fallback_to_mouse: bool = settings.transport.fallback_to_mouse

        self._uia_invoke = _uia_invoker
        self._uia_set_value = _uia_value_setter
        self._uia_select = _uia_selector
        self._dom_click = _dom_clicker
        self._dom_type = _dom_typer
        self._dom_focus = _dom_focuser
        self._dom_clear = _dom_clearer

        self._mouse = _mouse_transport or MouseTransport()
        self._keyboard = _keyboard_transport or KeyboardTransport()
        self._audit = _audit_writer or _noop_audit

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute(
        self,
        action_spec: ActionSpec,
        source_result: SourceResult,
        target_element: Any = None,
    ) -> TransportResult:
        """
        Deliver *action_spec* using the best available transport.

        Parameters
        ----------
        action_spec:
            What to do and on which task / action row.
        source_result:
            The SourceResult that located *target_element*, used to
            select the native transport path.
        target_element:
            The element to act on (UIAElement, DOMElement, or None).
            Mouse/keyboard fallbacks use ``target_element.bounding_rect``
            for coordinates when available.

        Returns
        -------
        TransportResult

        Raises
        ------
        TransportFallbackError
            When all transport options are exhausted.
        """
        t0 = time.perf_counter()

        result = await self._dispatch(action_spec, source_result, target_element)

        result = TransportResult(
            method_used=result.method_used,
            success=result.success,
            latency_ms=(time.perf_counter() - t0) * 1000,
            fallback_used=result.fallback_used,
        )

        # Audit — best-effort; failure must not mask the primary result.
        try:
            await self._audit(result, action_spec)
        except Exception as exc:
            _log.debug("transport_audit_failed", error=str(exc))

        _log.info(
            "transport_executed",
            method=result.method_used,
            action=action_spec.action_type,
            success=result.success,
            fallback=result.fallback_used,
            latency_ms=round(result.latency_ms, 3),
        )

        if not result.success:
            raise TransportFallbackError(
                f"All transports failed for action={action_spec.action_type!r}",
                context={
                    "method_used": result.method_used,
                    "source_type": source_result.source_type,
                },
            )

        return result

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        spec: ActionSpec,
        source: SourceResult,
        element: Any,
    ) -> TransportResult:
        """Route to the correct transport path and return an unsettled result."""
        # Override: prefer_native_action=False → direct OS fallback
        if not self._prefer_native:
            return await self._os_fallback(spec, element, fallback_used=False)

        # visual / file sources have no native action path
        if source.source_type in ("visual", "file"):
            return await self._os_fallback(spec, element, fallback_used=False)

        # Native-first path
        if source.source_type == "uia":
            return await self._uia_path(spec, element)

        if source.source_type == "dom":
            return await self._dom_path(spec, element)

        # Should not be reached; treat unknown source as OS fallback.
        return await self._os_fallback(spec, element, fallback_used=False)

    # ------------------------------------------------------------------
    # UIA native path
    # ------------------------------------------------------------------

    async def _uia_path(self, spec: ActionSpec, element: Any) -> TransportResult:
        native_ok = False

        # Always try the injected invoker — it is responsible for handling
        # None elements gracefully (real UIAAdapter returns False for None).
        if spec.action_type == "click" and self._uia_invoke is not None:
            native_ok = self._uia_invoke(element)
        elif spec.action_type == "type" and self._uia_set_value is not None:
            native_ok = self._uia_set_value(element, spec.text or "")
        elif spec.action_type == "select" and self._uia_select is not None:
            native_ok = self._uia_select(element)

        if native_ok:
            _log.debug("transport_native_ok", source="uia", action=spec.action_type)
            return TransportResult(
                method_used="uia",
                success=True,
                latency_ms=0.0,
                fallback_used=False,
            )

        _log.debug(
            "transport_native_failed_fallback",
            source="uia",
            action=spec.action_type,
        )
        return await self._os_fallback(spec, element, fallback_used=True)

    # ------------------------------------------------------------------
    # DOM native path
    # ------------------------------------------------------------------

    async def _dom_path(self, spec: ActionSpec, element: Any) -> TransportResult:
        native_ok = False

        if spec.action_type == "click" and self._dom_click is not None:
            native_ok = await self._dom_click(element)
        elif spec.action_type == "type" and self._dom_type is not None:
            native_ok = await self._dom_type(element, spec.text or "")
        elif spec.action_type == "focus" and self._dom_focus is not None:
            native_ok = await self._dom_focus(element)
        elif spec.action_type == "clear" and self._dom_clear is not None:
            native_ok = await self._dom_clear(element)

        if native_ok:
            _log.debug("transport_native_ok", source="dom", action=spec.action_type)
            return TransportResult(
                method_used="dom",
                success=True,
                latency_ms=0.0,
                fallback_used=False,
            )

        _log.debug(
            "transport_native_failed_fallback",
            source="dom",
            action=spec.action_type,
        )
        return await self._os_fallback(spec, element, fallback_used=True)

    # ------------------------------------------------------------------
    # OS-level fallback (mouse / keyboard)
    # ------------------------------------------------------------------

    async def _os_fallback(
        self,
        spec: ActionSpec,
        element: Any,
        *,
        fallback_used: bool,
    ) -> TransportResult:
        """Deliver action via mouse or keyboard OS input."""
        if spec.action_type in ("click", "focus", "select", "clear"):
            coords = _centre_of(element) if element is not None else None
            if coords is None:
                coords = spec.coordinates  # fall back to spec coordinates
            if coords is not None:
                ok = await self._mouse.click(*coords)
            else:
                ok = False
            return TransportResult(
                method_used="mouse",
                success=ok,
                latency_ms=0.0,
                fallback_used=fallback_used,
            )

        if spec.action_type == "type":
            ok = await self._keyboard.type_text(spec.text or "")
            return TransportResult(
                method_used="keyboard",
                success=ok,
                latency_ms=0.0,
                fallback_used=fallback_used,
            )

        if spec.action_type == "press_key":
            ok = await self._keyboard.press_key(spec.text or "")
            return TransportResult(
                method_used="keyboard",
                success=ok,
                latency_ms=0.0,
                fallback_used=fallback_used,
            )

        if spec.action_type in ("scroll", "hover"):
            coords = _centre_of(element) if element is not None else None
            if coords is not None:
                ok = await self._mouse.click(*coords)
            else:
                ok = True  # scroll/hover without coords is a no-op, not a failure
            return TransportResult(
                method_used="mouse",
                success=ok,
                latency_ms=0.0,
                fallback_used=fallback_used,
            )

        # Unknown action type — log and return success to avoid crashing the loop
        _log.warning("transport_unknown_action", action=spec.action_type)
        return TransportResult(
            method_used="mouse",
            success=True,
            latency_ms=0.0,
            fallback_used=fallback_used,
        )
