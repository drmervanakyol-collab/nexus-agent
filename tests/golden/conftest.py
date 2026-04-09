"""
tests/golden/conftest.py
Shared fixtures and auto-skip logic for golden (end-to-end) scenarios.

Golden tests require a real Windows desktop environment — they are
automatically skipped when the CI environment variable is set or when
NEXUS_GOLDEN_TESTS is absent.

Set NEXUS_GOLDEN_TESTS=1 (or run `make golden`) to enable execution.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from nexus.core.task_executor import TaskResult, TransportStats

# ---------------------------------------------------------------------------
# Auto-skip: CI or missing opt-in flag
# ---------------------------------------------------------------------------

_IN_CI = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))
_GOLDEN_ENABLED = bool(os.environ.get("NEXUS_GOLDEN_TESTS"))


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Skip all @pytest.mark.golden tests unless NEXUS_GOLDEN_TESTS=1."""
    skip_reason = pytest.mark.skip(
        reason="Golden scenarios require NEXUS_GOLDEN_TESTS=1 (not set in CI)"
    )
    for item in items:
        if item.get_closest_marker("golden") and not _GOLDEN_ENABLED:
            item.add_marker(skip_reason)


# ---------------------------------------------------------------------------
# ScenarioReport — captures metrics from each golden run
# ---------------------------------------------------------------------------


@dataclass
class ScenarioReport:
    """Timing, cost, and transport metrics collected during a golden run."""

    scenario_id: str
    duration_ms: float = 0.0
    cost_usd: float = 0.0
    steps: int = 0
    native_count: int = 0
    fallback_count: int = 0
    assertions: list[str] = field(default_factory=list)

    def record_result(self, result: TaskResult) -> None:
        self.duration_ms = result.duration_ms
        self.cost_usd = result.total_cost_usd
        self.steps = result.steps_completed
        self.native_count = result.transport_stats.native_count
        self.fallback_count = result.transport_stats.fallback_count

    def assert_ok(self, condition: bool, message: str) -> None:
        self.assertions.append(message)
        assert condition, message

    def summary(self) -> str:
        total = self.native_count + self.fallback_count
        native_pct = (
            round(self.native_count / total * 100)
            if total > 0 else 0
        )
        return (
            f"[{self.scenario_id}] "
            f"{self.duration_ms:.0f} ms | "
            f"${self.cost_usd:.4f} | "
            f"{self.steps} steps | "
            f"native {self.native_count} ({native_pct}%) "
            f"fallback {self.fallback_count}"
        )


# ---------------------------------------------------------------------------
# Fixture: scenario_report
# ---------------------------------------------------------------------------


@pytest.fixture()
def scenario_report(request: Any) -> ScenarioReport:
    """Provide a ScenarioReport and print its summary after the test."""
    report = ScenarioReport(scenario_id=request.node.name)
    yield report
    print(f"\n  {report.summary()}")
