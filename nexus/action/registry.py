"""
nexus/action/registry.py
Action Registry — ActionHandler protocol, ActionSpec, ActionResult, ActionRegistry.

ActionHandler
-------------
Protocol for all action handler implementations.  Handlers must provide:
  execute(spec)  — async, delivers the action and returns ActionResult.
  validate(spec) — sync, checks structural validity of the spec.

ActionSpec
----------
Agent-level action specification (richer than the transport-layer ActionSpec).
Fields intentionally mirror the output of DecisionEngine.Decision so that
the execution layer can consume Decision output directly.

ActionResult
------------
Outcome of a single executed action.  Handlers must populate at minimum
``success`` and ``transport_used``.

ActionRegistry
--------------
Maps action_type strings to ActionHandler instances.
  register(type, handler) — add / replace a handler.
  get(type)               — return handler or None.
  list_types()            — sorted list of registered types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# ActionSpec
# ---------------------------------------------------------------------------


@dataclass
class ActionSpec:
    """
    Agent-level specification of an action to be executed.

    Attributes
    ----------
    action_type:
        Verb for the action, e.g. ``"click"``, ``"type"``, ``"scroll"``.
    target_element_id:
        Stable element identifier from the perception layer, or ``None``.
    coordinates:
        ``(x, y)`` screen coordinates targeting the action, or ``None``.
    value:
        Text / key value for type/press_key actions, or ``None``.
    is_destructive:
        True when the action may irreversibly modify data.
    preferred_transport:
        Caller-requested delivery channel: ``"uia"``, ``"dom"``,
        ``"mouse"``, ``"keyboard"``, or ``None`` (auto-select).
    metadata:
        Arbitrary key/value pairs for handler-specific extensions.
    """

    action_type: str
    target_element_id: str | None = None
    coordinates: tuple[int, int] | None = None
    value: str | None = None
    is_destructive: bool = False
    preferred_transport: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ActionResult
# ---------------------------------------------------------------------------


@dataclass
class ActionResult:
    """
    Outcome of a single executed action.

    Attributes
    ----------
    success:
        True when the action completed without error.
    duration_ms:
        Wall-clock time for the execution (milliseconds).
    error:
        Human-readable error description, or ``None`` on success.
    partial_completion:
        True when the action started but did not fully finish
        (e.g. text was partially typed before a timeout).
    side_effects:
        List of observable side-effect descriptors recorded by the handler
        (e.g. ``["dialog_opened"]``).
    transport_used:
        The transport channel that delivered the action, or ``None``.
    """

    success: bool
    duration_ms: float = 0.0
    error: str | None = None
    partial_completion: bool = False
    side_effects: list[str] = field(default_factory=list)
    transport_used: str | None = None


# ---------------------------------------------------------------------------
# ActionHandler protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ActionHandler(Protocol):
    """
    Protocol that every action handler must satisfy.

    All concrete handlers are discovered via ``isinstance(obj, ActionHandler)``
    at registration time — use ``@runtime_checkable`` for this check.
    """

    async def execute(self, spec: ActionSpec) -> ActionResult:
        """
        Deliver the action described by *spec* and return the result.

        Must never raise — return ``ActionResult(success=False, error=...)``
        instead of propagating exceptions.
        """
        ...

    def validate(self, spec: ActionSpec) -> bool:
        """
        Return True if *spec* is structurally valid for this handler.

        Validation is fast and synchronous.  It must not have side effects
        and must not perform I/O.
        """
        ...


# ---------------------------------------------------------------------------
# ActionRegistry
# ---------------------------------------------------------------------------


class ActionRegistry:
    """
    Registry that maps action_type strings to :class:`ActionHandler` instances.

    Thread-safety: not guaranteed.  Registrations should be completed at
    startup before concurrent ``get`` calls begin.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, ActionHandler] = {}

    def register(self, action_type: str, handler: ActionHandler) -> None:
        """
        Associate *action_type* with *handler*.

        If *action_type* was already registered, the old handler is replaced.

        Parameters
        ----------
        action_type:
            The verb string (e.g. ``"click"``).
        handler:
            Any object that satisfies the :class:`ActionHandler` protocol.
        """
        self._handlers[action_type] = handler

    def get(self, action_type: str) -> ActionHandler | None:
        """
        Return the handler for *action_type*, or ``None`` if not registered.
        """
        return self._handlers.get(action_type)

    def list_types(self) -> list[str]:
        """Return a sorted list of all registered action type strings."""
        return sorted(self._handlers.keys())
