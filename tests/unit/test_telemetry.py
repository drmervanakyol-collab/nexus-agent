"""Unit tests for nexus/infra/telemetry.py — TelemetryCollector."""
from __future__ import annotations

import threading

import pytest

from nexus.infra.telemetry import (
    ActionRecord,
    CloudCallRecord,
    PhaseRecord,
    TaskTelemetry,
    TelemetryCollector,
    TransportRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fresh() -> TelemetryCollector:
    return TelemetryCollector()


# ---------------------------------------------------------------------------
# Empty collector
# ---------------------------------------------------------------------------


class TestEmptyCollector:
    def test_summary_all_zeros(self) -> None:
        s = fresh().get_summary()
        assert s.total_duration_ms == 0.0
        assert s.phase_success_rate == 0.0
        assert s.cloud_calls == 0
        assert s.total_tokens == 0
        assert s.total_cost == 0.0
        assert s.actions_total == 0
        assert s.action_success_rate == 0.0
        assert s.transport_success_rate == 0.0
        assert s.native_ratio == 0.0
        assert s.avg_latency_ms == 0.0
        assert s.transport_method_counts == {}

    def test_summary_lists_empty(self) -> None:
        s = fresh().get_summary()
        assert s.phases == []
        assert s.transport_records == []


# ---------------------------------------------------------------------------
# record_phase
# ---------------------------------------------------------------------------


class TestRecordPhase:
    def test_single_phase(self) -> None:
        c = fresh()
        c.record_phase("capture", 100.0, success=True)
        s = c.get_summary()
        assert len(s.phases) == 1
        assert s.phases[0] == PhaseRecord("capture", 100.0, True)
        assert s.total_duration_ms == pytest.approx(100.0)
        assert s.phase_success_rate == pytest.approx(1.0)

    def test_multiple_phases(self) -> None:
        c = fresh()
        c.record_phase("a", 50.0, success=True)
        c.record_phase("b", 150.0, success=False)
        s = c.get_summary()
        assert len(s.phases) == 2
        assert s.total_duration_ms == pytest.approx(200.0)
        assert s.phase_success_rate == pytest.approx(0.5)

    def test_all_phases_fail(self) -> None:
        c = fresh()
        c.record_phase("x", 10.0, success=False)
        c.record_phase("y", 20.0, success=False)
        assert c.get_summary().phase_success_rate == pytest.approx(0.0)

    def test_all_phases_succeed(self) -> None:
        c = fresh()
        for i in range(5):
            c.record_phase(f"p{i}", float(i), success=True)
        assert c.get_summary().phase_success_rate == pytest.approx(1.0)

    def test_zero_duration(self) -> None:
        c = fresh()
        c.record_phase("fast", 0.0, success=True)
        assert c.get_summary().total_duration_ms == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# record_cloud_call
# ---------------------------------------------------------------------------


class TestRecordCloudCall:
    def test_single_call(self) -> None:
        c = fresh()
        c.record_cloud_call("anthropic", tokens=500, cost=0.01)
        s = c.get_summary()
        assert s.cloud_calls == 1
        assert s.total_tokens == 500
        assert s.total_cost == pytest.approx(0.01)

    def test_multiple_providers(self) -> None:
        c = fresh()
        c.record_cloud_call("anthropic", tokens=1000, cost=0.02)
        c.record_cloud_call("openai", tokens=500, cost=0.01)
        s = c.get_summary()
        assert s.cloud_calls == 2
        assert s.total_tokens == 1500
        assert s.total_cost == pytest.approx(0.03)

    def test_zero_cost(self) -> None:
        c = fresh()
        c.record_cloud_call("local", tokens=100, cost=0.0)
        assert c.get_summary().total_cost == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# record_action
# ---------------------------------------------------------------------------


class TestRecordAction:
    def test_single_success(self) -> None:
        c = fresh()
        c.record_action("click", success=True)
        s = c.get_summary()
        assert s.actions_total == 1
        assert s.actions_succeeded == 1
        assert s.action_success_rate == pytest.approx(1.0)

    def test_single_failure(self) -> None:
        c = fresh()
        c.record_action("type", success=False)
        s = c.get_summary()
        assert s.actions_succeeded == 0
        assert s.action_success_rate == pytest.approx(0.0)

    def test_mixed_actions(self) -> None:
        c = fresh()
        c.record_action("click", success=True)
        c.record_action("scroll", success=True)
        c.record_action("type", success=False)
        s = c.get_summary()
        assert s.actions_total == 3
        assert s.actions_succeeded == 2
        assert s.action_success_rate == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# record_transport
# ---------------------------------------------------------------------------


class TestRecordTransport:
    def test_uia_recorded(self) -> None:
        c = fresh()
        c.record_transport("uia", success=True, latency_ms=10.0)
        s = c.get_summary()
        assert len(s.transport_records) == 1
        assert s.transport_records[0] == TransportRecord("uia", True, 10.0)
        assert s.transport_method_counts == {"uia": 1}

    def test_native_vs_fallback_ratio(self) -> None:
        c = fresh()
        # 3 native
        c.record_transport("uia", success=True, latency_ms=5.0)
        c.record_transport("dom", success=True, latency_ms=6.0)
        c.record_transport("file", success=True, latency_ms=4.0)
        # 1 fallback
        c.record_transport("mouse", success=False, latency_ms=20.0)
        s = c.get_summary()
        assert s.native_calls == 3
        assert s.fallback_calls == 1
        assert s.native_ratio == pytest.approx(0.75)

    def test_all_fallback(self) -> None:
        c = fresh()
        c.record_transport("mouse", success=True, latency_ms=15.0)
        c.record_transport("keyboard", success=True, latency_ms=12.0)
        s = c.get_summary()
        assert s.native_calls == 0
        assert s.fallback_calls == 2
        assert s.native_ratio == pytest.approx(0.0)

    def test_all_native(self) -> None:
        c = fresh()
        c.record_transport("uia", success=True, latency_ms=8.0)
        c.record_transport("dom", success=True, latency_ms=9.0)
        s = c.get_summary()
        assert s.fallback_calls == 0
        assert s.native_ratio == pytest.approx(1.0)

    def test_avg_latency(self) -> None:
        c = fresh()
        c.record_transport("uia", success=True, latency_ms=10.0)
        c.record_transport("mouse", success=False, latency_ms=30.0)
        assert c.get_summary().avg_latency_ms == pytest.approx(20.0)

    def test_transport_success_rate(self) -> None:
        c = fresh()
        c.record_transport("uia", success=True, latency_ms=5.0)
        c.record_transport("uia", success=True, latency_ms=5.0)
        c.record_transport("mouse", success=False, latency_ms=20.0)
        assert c.get_summary().transport_success_rate == pytest.approx(2 / 3)

    def test_method_counts_multiple(self) -> None:
        c = fresh()
        for _ in range(3):
            c.record_transport("uia", success=True, latency_ms=1.0)
        for _ in range(2):
            c.record_transport("mouse", success=False, latency_ms=10.0)
        s = c.get_summary()
        assert s.transport_method_counts["uia"] == 3
        assert s.transport_method_counts["mouse"] == 2

    def test_keyboard_counted_as_fallback(self) -> None:
        c = fresh()
        c.record_transport("keyboard", success=True, latency_ms=5.0)
        s = c.get_summary()
        assert s.fallback_calls == 1
        assert s.native_calls == 0


# ---------------------------------------------------------------------------
# Full summary correctness
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_full_scenario(self) -> None:
        c = fresh()
        c.record_phase("capture", 50.0, success=True)
        c.record_phase("perception", 80.0, success=True)
        c.record_phase("decision", 30.0, success=False)

        c.record_cloud_call("anthropic", tokens=1000, cost=0.02)
        c.record_cloud_call("anthropic", tokens=500, cost=0.01)

        c.record_action("click", success=True)
        c.record_action("type", success=False)

        c.record_transport("uia", success=True, latency_ms=10.0)
        c.record_transport("mouse", success=False, latency_ms=50.0)

        s = c.get_summary()

        assert s.total_duration_ms == pytest.approx(160.0)
        assert s.phase_success_rate == pytest.approx(2 / 3)
        assert s.cloud_calls == 2
        assert s.total_tokens == 1500
        assert s.total_cost == pytest.approx(0.03)
        assert s.actions_total == 2
        assert s.action_success_rate == pytest.approx(0.5)
        assert s.native_calls == 1
        assert s.fallback_calls == 1
        assert s.native_ratio == pytest.approx(0.5)
        assert s.avg_latency_ms == pytest.approx(30.0)

    def test_summary_returns_task_telemetry_type(self) -> None:
        assert isinstance(fresh().get_summary(), TaskTelemetry)

    def test_summary_snapshot_independence(self) -> None:
        c = fresh()
        c.record_phase("p", 10.0, success=True)
        s1 = c.get_summary()
        c.record_phase("q", 20.0, success=False)
        s2 = c.get_summary()
        # s1 must not be affected by later recordings
        assert len(s1.phases) == 1
        assert len(s2.phases) == 2


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all(self) -> None:
        c = fresh()
        c.record_phase("p", 10.0, success=True)
        c.record_cloud_call("anthropic", tokens=100, cost=0.01)
        c.record_action("click", success=True)
        c.record_transport("uia", success=True, latency_ms=5.0)
        c.reset()
        s = c.get_summary()
        assert s.phases == []
        assert s.cloud_calls == 0
        assert s.actions_total == 0
        assert s.transport_records == []

    def test_record_after_reset(self) -> None:
        c = fresh()
        c.record_phase("old", 100.0, success=False)
        c.reset()
        c.record_phase("new", 20.0, success=True)
        s = c.get_summary()
        assert len(s.phases) == 1
        assert s.phases[0].phase == "new"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record_phase(self) -> None:
        c = fresh()
        n = 200

        def worker() -> None:
            for i in range(n):
                c.record_phase(f"p{i}", float(i), success=True)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(c.get_summary().phases) == 4 * n

    def test_concurrent_mixed_records(self) -> None:
        c = fresh()

        def worker() -> None:
            for _ in range(50):
                c.record_phase("p", 1.0, success=True)
                c.record_cloud_call("x", tokens=1, cost=0.0)
                c.record_action("click", success=True)
                c.record_transport("uia", success=True, latency_ms=1.0)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        s = c.get_summary()
        assert len(s.phases) == 200
        assert s.cloud_calls == 200
        assert s.actions_total == 200
        assert len(s.transport_records) == 200
