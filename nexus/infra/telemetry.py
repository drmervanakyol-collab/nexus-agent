"""
nexus/infra/telemetry.py
In-process telemetry collector for a single task execution.

Collected metrics
-----------------
- Phase timings and success flags
- Cloud call tokens and cost
- Action outcomes
- Transport method usage, success, and latency

All methods are synchronous and thread-safe via a simple lock.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

TransportMethod = Literal["uia", "dom", "file", "mouse", "keyboard"]

# ---------------------------------------------------------------------------
# Per-record dataclasses (immutable snapshots)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseRecord:
    phase: str
    duration_ms: float
    success: bool


@dataclass(frozen=True)
class CloudCallRecord:
    provider: str
    tokens: int
    cost: float


@dataclass(frozen=True)
class ActionRecord:
    action_type: str
    success: bool


@dataclass(frozen=True)
class TransportRecord:
    method: TransportMethod
    success: bool
    latency_ms: float


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------


@dataclass
class TaskTelemetry:
    """Aggregated view of a task's telemetry data."""

    # Phase
    phases: list[PhaseRecord] = field(default_factory=list)
    total_duration_ms: float = 0.0
    phase_success_rate: float = 0.0  # 0.0–1.0

    # Cloud
    cloud_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    # Actions
    actions_total: int = 0
    actions_succeeded: int = 0
    action_success_rate: float = 0.0  # 0.0–1.0

    # Transport
    transport_records: list[TransportRecord] = field(default_factory=list)
    transport_success_rate: float = 0.0  # 0.0–1.0
    transport_method_counts: dict[str, int] = field(default_factory=dict)
    native_calls: int = 0   # uia + dom + file
    fallback_calls: int = 0  # mouse + keyboard
    native_ratio: float = 0.0  # native / total transport calls
    avg_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

_NATIVE_METHODS: frozenset[TransportMethod] = frozenset({"uia", "dom", "file"})
_FALLBACK_METHODS: frozenset[TransportMethod] = frozenset({"mouse", "keyboard"})


class TelemetryCollector:
    """Collects and aggregates telemetry for one task execution."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phases: list[PhaseRecord] = []
        self._cloud_calls: list[CloudCallRecord] = []
        self._actions: list[ActionRecord] = []
        self._transports: list[TransportRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_phase(
        self, phase: str, duration_ms: float, *, success: bool
    ) -> None:
        """Record the outcome of one pipeline phase."""
        with self._lock:
            self._phases.append(PhaseRecord(phase, duration_ms, success))

    def record_cloud_call(
        self, provider: str, tokens: int, cost: float
    ) -> None:
        """Record one LLM provider call."""
        with self._lock:
            self._cloud_calls.append(CloudCallRecord(provider, tokens, cost))

    def record_action(self, action_type: str, *, success: bool) -> None:
        """Record one atomic action."""
        with self._lock:
            self._actions.append(ActionRecord(action_type, success))

    def record_transport(
        self,
        method: TransportMethod,
        *,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Record one transport attempt."""
        with self._lock:
            self._transports.append(TransportRecord(method, success, latency_ms))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> TaskTelemetry:
        """Return an aggregated TaskTelemetry snapshot."""
        with self._lock:
            phases = list(self._phases)
            cloud = list(self._cloud_calls)
            actions = list(self._actions)
            transports = list(self._transports)

        summary = TaskTelemetry()

        # --- phases ---
        summary.phases = phases
        summary.total_duration_ms = sum(p.duration_ms for p in phases)
        summary.phase_success_rate = (
            sum(1 for p in phases if p.success) / len(phases) if phases else 0.0
        )

        # --- cloud ---
        summary.cloud_calls = len(cloud)
        summary.total_tokens = sum(c.tokens for c in cloud)
        summary.total_cost = sum(c.cost for c in cloud)

        # --- actions ---
        summary.actions_total = len(actions)
        summary.actions_succeeded = sum(1 for a in actions if a.success)
        summary.action_success_rate = (
            summary.actions_succeeded / summary.actions_total
            if actions
            else 0.0
        )

        # --- transport ---
        summary.transport_records = transports
        summary.transport_success_rate = (
            sum(1 for t in transports if t.success) / len(transports)
            if transports
            else 0.0
        )
        method_counts: dict[str, int] = {}
        for t in transports:
            method_counts[t.method] = method_counts.get(t.method, 0) + 1
        summary.transport_method_counts = method_counts

        summary.native_calls = sum(
            c for m, c in method_counts.items() if m in _NATIVE_METHODS
        )
        summary.fallback_calls = sum(
            c for m, c in method_counts.items() if m in _FALLBACK_METHODS
        )
        total_transport = len(transports)
        summary.native_ratio = (
            summary.native_calls / total_transport if total_transport else 0.0
        )
        summary.avg_latency_ms = (
            sum(t.latency_ms for t in transports) / total_transport
            if total_transport
            else 0.0
        )

        return summary

    # ------------------------------------------------------------------
    # Reset (for reuse across sub-tasks)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self._phases.clear()
            self._cloud_calls.clear()
            self._actions.clear()
            self._transports.clear()
