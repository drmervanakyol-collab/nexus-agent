"""
nexus/source/resolver.py
Source Priority Resolver for Nexus Agent.

Architecture
------------
SourcePriorityResolver orchestrates the four information sources in a
fixed priority chain:

    1. UIA   — Windows UI Automation (direct native API, highest fidelity)
    2. DOM   — Chrome DevTools Protocol (browser element tree)
    3. File  — PDF / XLSX document extraction
    4. Visual — Screenshot + OCR / vision perception (final fallback)

Each attempt is:
  - Timed with ``time.perf_counter()``.
  - Logged at INFO level: source, latency_ms, success flag.
  - Abandoned immediately on the first non-None result.

The "visual" source acts as a guaranteed fallback: its default probe
returns a sentinel ``{"visual_pending": True}`` so that ``resolve()``
always returns a ``SourceResult`` — never raises.

Pluggable probes
----------------
All four probes accept a ``dict[str, Any]`` context and return either
non-None data (success) or ``None`` (failure).  Pass mock callables at
construction time to inject test behaviour without touching real adapters.

Priority override
-----------------
``prefer_source`` moves the nominated source to the head of the attempt
list.  The remaining sources follow in the default UIA → DOM → File →
Visual order (excluding the preferred one to avoid double-attempts).
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from nexus.core.settings import NexusSettings
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

SourceType = Literal["uia", "dom", "file", "visual"]

_SOURCE_CONFIDENCE: dict[str, float] = {
    "uia": 1.0,
    "dom": 0.95,
    "file": 0.90,
    "visual": 0.70,
}


@dataclass
class SourceResult:
    """
    The outcome of a successful source resolution.

    Attributes
    ----------
    source_type:
        Which source provided the data.
    data:
        Raw payload from the probe — element list, DocumentContent,
        screenshot bytes, etc.  Type depends on the winning source.
    confidence:
        Inherent reliability of the source (1.0 for UIA, 0.7 for visual).
    latency_ms:
        Wall-clock time spent in the winning probe.
    """

    source_type: SourceType
    data: Any
    confidence: float
    latency_ms: float


# ---------------------------------------------------------------------------
# Probe type alias and defaults
# ---------------------------------------------------------------------------

_ProbeFunc = Callable[[dict[str, Any]], Any]
# Contract: returns non-None data on success, None on failure / unavailable.


def _null_probe(ctx: dict[str, Any]) -> Any:
    """Default probe for UIA / DOM / File: always unavailable."""
    return None


def _visual_fallback_probe(ctx: dict[str, Any]) -> Any:
    """Default visual probe: always succeeds with a 'pending' sentinel."""
    return {"visual_pending": True, "context": ctx}


# ---------------------------------------------------------------------------
# Priority helpers
# ---------------------------------------------------------------------------

_DEFAULT_ORDER: tuple[SourceType, ...] = ("uia", "dom", "file", "visual")


def _build_order(
    prefer_source: SourceType | None,
    default: tuple[SourceType, ...],
) -> list[SourceType]:
    """
    Return the probe order with *prefer_source* moved to position 0.

    If *prefer_source* is None or already first, return a copy of *default*.
    """
    if prefer_source is None or prefer_source == default[0]:
        return list(default)
    rest = [s for s in default if s != prefer_source]
    return [prefer_source, *rest]


# ---------------------------------------------------------------------------
# SourcePriorityResolver
# ---------------------------------------------------------------------------


class SourcePriorityResolver:
    """
    Tries sources in priority order and returns the first successful result.

    Parameters
    ----------
    settings:
        NexusSettings — reserved for future per-source timeout config.
    _uia_probe:
        Callable ``(ctx) -> data | None``.  Defaults to always-None.
    _dom_probe:
        Callable ``(ctx) -> data | None``.  Defaults to always-None.
    _file_probe:
        Callable ``(ctx) -> data | None``.  Defaults to always-None.
    _visual_probe:
        Callable ``(ctx) -> data | None``.  Defaults to visual-sentinel.
    """

    def __init__(
        self,
        settings: NexusSettings,
        *,
        _uia_probe: _ProbeFunc | None = None,
        _dom_probe: _ProbeFunc | None = None,
        _file_probe: _ProbeFunc | None = None,
        _visual_probe: _ProbeFunc | None = None,
    ) -> None:
        self._settings = settings
        self._probes: dict[SourceType, _ProbeFunc] = {
            "uia": _uia_probe or _null_probe,
            "dom": _dom_probe or _null_probe,
            "file": _file_probe or _null_probe,
            "visual": _visual_probe or _visual_fallback_probe,
        }

    # ------------------------------------------------------------------
    # Core resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        task_context: dict[str, Any],
        prefer_source: SourceType | None = None,
    ) -> SourceResult:
        """
        Try sources in priority order and return the first success.

        Each attempt is logged at INFO with: source, latency_ms, success.
        The visual source is the guaranteed final fallback.

        Parameters
        ----------
        task_context:
            Arbitrary dict forwarded to each probe callable.
        prefer_source:
            When set, that source is tried first regardless of the
            default order.

        Returns
        -------
        SourceResult — always non-None (visual fallback ensures this).
        """
        order = _build_order(prefer_source, _DEFAULT_ORDER)

        for source in order:
            probe = self._probes[source]
            t0 = time.perf_counter()
            try:
                data = probe(task_context)
            except Exception as exc:
                data = None
                _log.debug(
                    "source_probe_exception",
                    source=source,
                    error=str(exc),
                )
            latency_ms = (time.perf_counter() - t0) * 1000

            success = data is not None
            _log.info(
                "source_tried",
                source=source,
                latency_ms=round(latency_ms, 3),
                success=success,
            )

            if success:
                return SourceResult(
                    source_type=source,
                    data=data,
                    confidence=_SOURCE_CONFIDENCE[source],
                    latency_ms=latency_ms,
                )

        # Unreachable: visual probe never returns None by contract.
        raise RuntimeError(  # pragma: no cover
            "All sources including visual returned None — "
            "visual probe must not return None."
        )

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def resolve_for_read(
        self,
        description: str,
        context: dict[str, Any] | None = None,
    ) -> SourceResult:
        """
        Resolve for a *read* intent (element discovery / document access).

        Builds a context dict with ``intent="read"`` and ``description``
        then delegates to :meth:`resolve`.

        Parameters
        ----------
        description:
            Human-readable description of what to read (e.g. selector,
            element name, file path).
        context:
            Optional extra keys merged into the task context.
        """
        task_context: dict[str, Any] = {
            "intent": "read",
            "description": description,
            **(context or {}),
        }
        return self.resolve(task_context)

    def resolve_for_action(
        self,
        action_type: str,
        target: str,
        context: dict[str, Any] | None = None,
    ) -> SourceResult:
        """
        Resolve for an *action* intent (click, type, focus, …).

        Builds a context dict with ``intent="action"``, ``action_type``,
        and ``target`` then delegates to :meth:`resolve`.

        Parameters
        ----------
        action_type:
            Kind of action (e.g. ``"click"``, ``"type"``, ``"focus"``).
        target:
            CSS selector, element name, or file path being acted on.
        context:
            Optional extra keys merged into the task context.
        """
        task_context: dict[str, Any] = {
            "intent": "action",
            "action_type": action_type,
            "target": target,
            **(context or {}),
        }
        return self.resolve(task_context)
