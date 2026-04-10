"""
tests/integration/test_block0.py
Blok 0 Integration Tests — Faz 10

TEST 1 — Settings + Logger
TEST 2 — Trace + Telemetry (transport metrikleri dahil)
TEST 3 — DB + Repository (transport_audit dahil)
TEST 4 — Policy + Budget entegrasyonu
TEST 5 — Health check gerçek sistem
TEST 6 — Settings override zinciri
TEST 7 — Native action policy block
"""
from __future__ import annotations

import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# TEST 1 — Settings + Logger
# ---------------------------------------------------------------------------


class TestSettingsAndLogger:
    """
    Settings yüklenip yüklenmediğini ve logger'ın JSON ürettiğini
    uçtan uca doğrular.
    """

    def test_default_settings_load(self) -> None:
        from nexus.core.settings import NexusSettings

        s = NexusSettings()
        assert s.safety.max_actions_per_task == 100
        assert s.budget.max_cost_per_task_usd == 1.0
        assert s.capture.fps == 15

    def test_logger_produces_json(self) -> None:
        from nexus.infra.logger import configure_logging, get_logger

        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)
        log = get_logger("nexus.integration.test1")
        log.info("block0_test1", component="settings_logger")

        stream.seek(0)
        lines = [ln.strip() for ln in stream.read().splitlines() if ln.strip()]
        assert lines, "No log output"
        record = json.loads(lines[-1])
        assert record["event"] == "block0_test1"
        assert record["component"] == "settings_logger"
        assert "timestamp" in record

    def test_settings_fields_reach_logger_context(self) -> None:
        from nexus.core.settings import NexusSettings
        from nexus.core.types import TaskId, TraceId
        from nexus.infra.logger import configure_logging, get_logger
        from nexus.infra.trace import traced

        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)

        s = NexusSettings()
        log = get_logger("nexus.integration")

        with traced("capture", trace_id=TraceId("t1-trace"), task_id=TaskId("t1")):
            log.info(
                "settings_reached",
                dry_run=s.safety.dry_run_mode,
                max_actions=s.safety.max_actions_per_task,
            )

        stream.seek(0)
        lines = [ln.strip() for ln in stream.read().splitlines() if ln.strip()]
        record = json.loads(lines[-1])
        assert record["trace_id"] == "t1-trace"
        assert record["task_id"] == "t1"
        assert record["dry_run"] is False


# ---------------------------------------------------------------------------
# TEST 2 — Trace + Telemetry (transport metrikleri dahil)
# ---------------------------------------------------------------------------


class TestTraceAndTelemetry:
    """Trace context ve TelemetryCollector birlikte çalışır mı?"""

    async def test_async_trace_sets_ids(self) -> None:
        from nexus.core.types import TaskId, TraceId
        from nexus.infra.trace import TraceContext, async_traced

        async with async_traced(
            "perception", trace_id=TraceId("tr-2"), task_id=TaskId("tk-2")
        ) as ctx:
            current = TraceContext.current()
            assert current is ctx
            assert ctx.trace_id == "tr-2"
            assert ctx.task_id == "tk-2"

        assert TraceContext.current() is None

    def test_telemetry_transport_breakdown(self) -> None:
        from nexus.infra.telemetry import TelemetryCollector

        col = TelemetryCollector()
        col.record_phase("capture", 50.0, success=True)
        col.record_phase("perception", 80.0, success=True)
        col.record_phase("decision", 30.0, success=False)

        col.record_cloud_call("anthropic", tokens=1000, cost=0.015)
        col.record_cloud_call("openai", tokens=500, cost=0.005)

        col.record_action("click", success=True)
        col.record_action("type", success=False)

        col.record_transport("uia", success=True, latency_ms=8.0)
        col.record_transport("dom", success=True, latency_ms=6.0)
        col.record_transport("mouse", success=False, latency_ms=40.0)

        s = col.get_summary()
        assert s.total_duration_ms == pytest.approx(160.0)
        assert s.phase_success_rate == pytest.approx(2 / 3)
        assert s.cloud_calls == 2
        assert s.total_tokens == 1500
        assert s.native_calls == 2
        assert s.fallback_calls == 1
        assert s.native_ratio == pytest.approx(2 / 3)
        assert s.avg_latency_ms == pytest.approx((8 + 6 + 40) / 3)

    async def test_trace_context_isolated_across_concurrent_tasks(self) -> None:
        import asyncio

        from nexus.core.types import TraceId
        from nexus.infra.trace import TraceContext, async_traced

        results: dict[str, str] = {}

        async def worker(name: str) -> None:
            async with async_traced(trace_id=TraceId(name)):
                await asyncio.sleep(0)
                ctx = TraceContext.current()
                assert ctx is not None
                results[name] = ctx.trace_id

        await asyncio.gather(worker("alpha"), worker("beta"), worker("gamma"))
        assert results == {"alpha": "alpha", "beta": "beta", "gamma": "gamma"}


# ---------------------------------------------------------------------------
# TEST 3 — DB + Repository (transport_audit dahil)
# ---------------------------------------------------------------------------


class TestDatabaseAndRepositories:
    """Database katmanı ile tüm repository'lerin uçtan uca entegrasyonu."""

    @pytest.fixture
    async def db(self, tmp_path: Path):  # type: ignore[type-arg]
        from nexus.infra.database import Database

        d = Database(str(tmp_path / "test_block0.db"))
        await d.init()
        return d

    async def test_task_action_transport_workflow(self, db) -> None:  # type: ignore[type-arg]
        from nexus.infra.repositories import (
            ActionRepository,
            TaskRepository,
            TransportAuditRepository,
        )

        task_repo = TaskRepository()
        action_repo = ActionRepository()
        transport_repo = TransportAuditRepository()

        async with db.connection() as conn:
            await task_repo.create(conn, id="task-blk0", goal="open browser")
            await action_repo.create(
                conn, id="act-001", task_id="task-blk0", type="click"
            )
            await transport_repo.record(
                conn,
                task_id="task-blk0",
                action_id="act-001",
                attempted_transport="uia",
                fallback_used=False,
                success=True,
                latency_ms=12.0,
            )
            await transport_repo.record(
                conn,
                task_id="task-blk0",
                action_id="act-001",
                attempted_transport="mouse",
                fallback_used=True,
                success=False,
                latency_ms=50.0,
            )
            ratio = await transport_repo.native_ratio_for_task(conn, "task-blk0")
            rows = await transport_repo.list_for_task(conn, "task-blk0")

        assert ratio == pytest.approx(0.5)
        assert len(rows) == 2
        assert rows[0].action_id == "act-001"

    async def test_cost_ledger_accumulates(self, db) -> None:  # type: ignore[type-arg]
        from nexus.infra.repositories import CostRepository, TaskRepository

        task_repo = TaskRepository()
        cost_repo = CostRepository()

        async with db.connection() as conn:
            await task_repo.create(conn, id="task-cost", goal="llm task")
            await cost_repo.record(
                conn, task_id="task-cost", provider="anthropic",
                tokens=1000, cost_usd=0.015,
            )
            await cost_repo.record(
                conn, task_id="task-cost", provider="openai",
                tokens=500, cost_usd=0.005,
            )
            total = await cost_repo.total_for_task(conn, "task-cost")

        assert total == pytest.approx(0.020)

    async def test_fk_cascade_delete(self, db) -> None:  # type: ignore[type-arg]
        from nexus.infra.repositories import (
            ActionRepository,
            TaskRepository,
            TransportAuditRepository,
        )

        task_repo = TaskRepository()
        action_repo = ActionRepository()
        transport_repo = TransportAuditRepository()

        async with db.connection() as conn:
            await task_repo.create(conn, id="task-del", goal="to delete")
            await action_repo.create(
                conn, id="act-del", task_id="task-del", type="click"
            )
            await transport_repo.record(
                conn,
                task_id="task-del",
                action_id="act-del",
                attempted_transport="uia",
                fallback_used=False,
                success=True,
                latency_ms=5.0,
            )
            await task_repo.delete(conn, "task-del")
            # Cascade: action and transport_audit rows must be gone
            action = await action_repo.get(conn, "act-del")
            transport_rows = await transport_repo.list_for_task(conn, "task-del")

        assert action is None
        assert transport_rows == []

    async def test_wal_mode_and_fk_active(self, db) -> None:  # type: ignore[type-arg]
        async with db.connection() as conn:
            async with conn.execute("PRAGMA journal_mode;") as cur:
                row = await cur.fetchone()
            assert row[0] == "wal"

            async with conn.execute("PRAGMA foreign_keys;") as cur:
                row = await cur.fetchone()
            assert row[0] == 1


# ---------------------------------------------------------------------------
# TEST 4 — Policy + Budget entegrasyonu
# ---------------------------------------------------------------------------


class TestPolicyAndBudget:
    """PolicyEngine ile CostTracker birlikte kullanılıyor mu?"""

    def _make_engine_and_tracker(
        self,
        max_task: float = 1.0,
        max_daily: float = 10.0,
        dry_run: bool = False,
    ):  # type: ignore[type-arg]
        from nexus.core.policy import PolicyEngine
        from nexus.core.settings import BudgetSettings, NexusSettings, SafetySettings
        from nexus.infra.cost_tracker import CostTracker

        s = NexusSettings(
            safety=SafetySettings(
                dry_run_mode=dry_run,
                max_actions_per_task=50,
            ),
            budget=BudgetSettings(
                max_cost_per_task_usd=max_task,
                max_cost_per_day_usd=max_daily,
            ),
        )
        engine = PolicyEngine(s)
        tracker = CostTracker(s, clock=lambda: datetime.now(UTC))
        return engine, tracker, s

    def test_budget_block_triggers_policy(self) -> None:
        from nexus.core.policy import RULE_TASK_BUDGET, ActionContext

        engine, tracker, s = self._make_engine_and_tracker(max_task=0.01)
        # Spend more than the cap
        tracker.record("t1", "gpt-4o", input_tokens=10_000, output_tokens=0)
        # 10000/1000 * 0.005 = 0.05 > 0.01

        task_cost = tracker.get_task_cost("t1")
        ctx = ActionContext(
            action_type="click",
            transport="mouse",
            task_cost_usd=task_cost,
            daily_cost_usd=tracker.get_daily_cost(),
        )
        result = engine.check_action(ctx)
        assert result.verdict == "block"
        assert result.rule == RULE_TASK_BUDGET

    def test_within_budget_allows_action(self) -> None:
        from nexus.core.policy import ActionContext

        engine, tracker, s = self._make_engine_and_tracker(max_task=1.0)
        tracker.record("t1", "gpt-4o-mini", input_tokens=100, output_tokens=0)

        task_cost = tracker.get_task_cost("t1")
        ctx = ActionContext(
            action_type="click",
            transport="uia",
            task_cost_usd=task_cost,
            daily_cost_usd=tracker.get_daily_cost(),
        )
        result = engine.check_action(ctx)
        assert result.verdict in ("allow", "warn")

    def test_cost_tracker_alert_level_matches_policy(self) -> None:
        from nexus.core.policy import ActionContext

        engine, tracker, s = self._make_engine_and_tracker(max_task=0.005)
        # Exactly at cap: 1000 input with gpt-4o → 0.005
        alert = tracker.record("t1", "gpt-4o", input_tokens=1000, output_tokens=0)
        assert alert.level == "block"

        ctx = ActionContext(
            action_type="delete",
            transport="mouse",
            is_destructive=True,
            task_cost_usd=alert.task_cost_usd,
            daily_cost_usd=alert.daily_cost_usd,
        )
        result = engine.check_action(ctx)
        assert result.verdict == "block"

    def test_transport_breakdown_visible_in_dashboard(self) -> None:
        _, tracker, _ = self._make_engine_and_tracker()
        tracker.record("t1", "gpt-4o-mini", input_tokens=100, output_tokens=0)
        tracker.record_transport("t1", "uia")
        tracker.record_transport("t1", "uia")
        tracker.record_transport("t1", "mouse")

        d = tracker.get_dashboard_data()
        assert d.transport_breakdown.native_calls == 2
        assert d.transport_breakdown.fallback_calls == 1
        assert d.transport_breakdown.native_ratio == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# TEST 5 — Health check gerçek sistem
# ---------------------------------------------------------------------------


class TestHealthCheckRealSystem:
    """
    HealthChecker'ı gerçek sistem üzerinde çalıştırır.
    Her check'in bir sonuç döndürdüğünü ve her non-ok sonucun
    fix_hint içerdiğini doğrular.
    (Actual pass/fail durumu sisteme bağlı; mevcut geliştirme
    ortamında bazı check'ler fail olabilir — bu beklenen bir durum.)
    """

    def test_run_all_returns_report(self, tmp_path: Path) -> None:
        from nexus.infra.health import HealthChecker, HealthReport

        checker = HealthChecker(
            db_path=str(tmp_path / "health_test.db"),
            write_dir=str(tmp_path),
        )
        report = checker.run_all()
        assert isinstance(report, HealthReport)

    def test_all_10_checks_present(self, tmp_path: Path) -> None:
        from nexus.infra.health import (
            CHECK_CREDENTIAL_MANAGER,
            CHECK_DB_ACCESSIBLE,
            CHECK_DISK_SPACE,
            CHECK_DPI_AWARENESS,
            CHECK_DXCAM,
            CHECK_PYTHON_VERSION,
            CHECK_RAM,
            CHECK_TESSERACT_BINARY,
            CHECK_WINDOWS_VERSION,
            CHECK_WRITE_PERMISSION,
            HealthChecker,
        )

        expected = {
            CHECK_PYTHON_VERSION,
            CHECK_WINDOWS_VERSION,
            CHECK_DISK_SPACE,
            CHECK_RAM,
            CHECK_DPI_AWARENESS,
            CHECK_DB_ACCESSIBLE,
            CHECK_TESSERACT_BINARY,
            CHECK_DXCAM,
            CHECK_WRITE_PERMISSION,
            CHECK_CREDENTIAL_MANAGER,
        }
        checker = HealthChecker(
            db_path=str(tmp_path / "hc.db"),
            write_dir=str(tmp_path),
        )
        report = checker.run_all()
        assert {r.name for r in report.checks} == expected

    def test_non_ok_checks_have_fix_hints(self, tmp_path: Path) -> None:
        from nexus.infra.health import HealthChecker

        checker = HealthChecker(
            db_path=str(tmp_path / "hc2.db"),
            write_dir=str(tmp_path),
        )
        report = checker.run_all()
        for r in report.checks:
            if r.status != "ok":
                assert r.fix_hint, (
                    f"Check {r.name} is {r.status} but has no fix_hint"
                )

    def test_exit_code_is_int(self, tmp_path: Path) -> None:
        from nexus.infra.health import HealthChecker

        checker = HealthChecker(
            db_path=str(tmp_path / "hc3.db"),
            write_dir=str(tmp_path),
        )
        report = checker.run_all()
        assert report.exit_code in (0, 1, 2)

    def test_python_version_ok_on_current_interpreter(self) -> None:
        from nexus.infra.health import CHECK_PYTHON_VERSION, HealthChecker

        checker = HealthChecker()
        result = checker.run_one(CHECK_PYTHON_VERSION)
        # The test suite itself runs on Python 3.11+, so this must pass.
        assert result.status == "ok"

    def test_write_permission_ok_on_tmpdir(self, tmp_path: Path) -> None:
        from nexus.infra.health import CHECK_WRITE_PERMISSION, HealthChecker

        checker = HealthChecker(write_dir=str(tmp_path))
        result = checker.run_one(CHECK_WRITE_PERMISSION)
        assert result.status == "ok"

    def test_db_accessible_with_memory_db(self) -> None:
        from nexus.infra.health import CHECK_DB_ACCESSIBLE, HealthChecker

        checker = HealthChecker(db_path=":memory:")
        result = checker.run_one(CHECK_DB_ACCESSIBLE)
        assert result.status == "ok"


# ---------------------------------------------------------------------------
# TEST 6 — Settings override zinciri
# ---------------------------------------------------------------------------


class TestSettingsOverrideChain:
    """
    Settings'in TOML dosyası → env var → Python default öncelik zincirini
    doğrular.  Gerçek dosya I/O kullanır.
    """

    def test_toml_overrides_defaults(self, tmp_path: Path) -> None:
        from nexus.core.settings import load_settings

        toml = tmp_path / "nexus.toml"
        toml.write_text(
            '[safety]\nmax_actions_per_task = 42\ndry_run_mode = true\n',
            encoding="utf-8",
        )
        s = load_settings(toml)
        assert s.safety.max_actions_per_task == 42
        assert s.safety.dry_run_mode is True

    def test_python_kwargs_override_toml(self, tmp_path: Path) -> None:
        from nexus.core.settings import load_settings

        toml = tmp_path / "nexus.toml"
        toml.write_text(
            '[budget]\nmax_cost_per_task_usd = 0.50\n',
            encoding="utf-8",
        )
        s = load_settings(toml, budget={"max_cost_per_task_usd": 2.0})
        assert s.budget.max_cost_per_task_usd == pytest.approx(2.0)

    def test_defaults_used_when_no_toml(self) -> None:
        from nexus.core.settings import NexusSettings

        s = NexusSettings()
        assert s.capture.fps == 15
        assert s.transport.prefer_native_action is True
        assert s.cloud.primary_provider == "openai"

    def test_nested_settings_accessible(self, tmp_path: Path) -> None:
        from nexus.core.settings import load_settings

        toml = tmp_path / "nexus.toml"
        toml.write_text(
            '[cloud]\nprimary_provider = "anthropic"\nmax_tokens = 2000\n',
            encoding="utf-8",
        )
        s = load_settings(toml)
        assert s.cloud.primary_provider == "anthropic"
        assert s.cloud.max_tokens == 2000

    def test_budget_pricing_for_model(self) -> None:
        from nexus.core.settings import NexusSettings

        s = NexusSettings()
        p = s.budget.pricing_for("gpt-4o")
        assert p.input_per_1k == pytest.approx(0.005)
        assert p.output_per_1k == pytest.approx(0.015)

    def test_unknown_model_raises(self) -> None:
        from nexus.core.settings import NexusSettings

        s = NexusSettings()
        with pytest.raises(KeyError):
            s.budget.pricing_for("gpt-999-ultra")


# ---------------------------------------------------------------------------
# TEST 7 — Native action policy block
# ---------------------------------------------------------------------------


class TestNativeActionPolicyBlock:
    """
    dry_run=True + destructive native action → RULE_NATIVE_ACTION_SAFETY veya
    RULE_DRY_RUN (ikisi de block verir).  Hiçbir kombinasyonun bypass
    edilemediğini doğrular.
    """

    def _dry_run_engine(self):  # type: ignore[type-arg]
        from nexus.core.policy import PolicyEngine
        from nexus.core.settings import NexusSettings, SafetySettings

        s = NexusSettings(safety=SafetySettings(dry_run_mode=True))
        return PolicyEngine(s)

    def _live_engine(self):  # type: ignore[type-arg]
        from nexus.core.policy import PolicyEngine
        from nexus.core.settings import NexusSettings, SafetySettings

        s = NexusSettings(safety=SafetySettings(dry_run_mode=False))
        return PolicyEngine(s)

    def test_dry_run_native_destructive_is_blocked(self) -> None:
        from nexus.core.policy import ActionContext

        engine = self._dry_run_engine()
        for transport in ("uia", "dom", "file"):
            ctx = ActionContext(
                action_type="delete_record",
                transport=transport,
                is_destructive=True,
            )
            result = engine.check_action(ctx)
            assert result.verdict == "block", (
                f"transport={transport} must be blocked in dry-run"
            )

    def test_dry_run_non_destructive_native_is_allowed(self) -> None:
        from nexus.core.policy import ActionContext

        engine = self._dry_run_engine()
        for transport in ("uia", "dom", "file"):
            ctx = ActionContext(
                action_type="read_element",
                transport=transport,
                is_destructive=False,
            )
            result = engine.check_action(ctx)
            assert result.verdict == "allow", (
                "Non-destructive native in dry-run should be allowed"
            )

    def test_live_mode_native_destructive_warns(self) -> None:
        from nexus.core.policy import RULE_NATIVE_ACTION_SAFETY, ActionContext

        engine = self._live_engine()
        for transport in ("uia", "dom", "file"):
            ctx = ActionContext(
                action_type="delete_record",
                transport=transport,
                is_destructive=True,
            )
            result = engine.check_action(ctx)
            assert result.verdict == "warn"
            assert result.rule == RULE_NATIVE_ACTION_SAFETY

    def test_dry_run_fallback_non_destructive_is_allowed(self) -> None:
        from nexus.core.policy import ActionContext

        engine = self._dry_run_engine()
        ctx = ActionContext(
            action_type="click",
            transport="mouse",
            is_destructive=False,
        )
        result = engine.check_action(ctx)
        assert result.verdict == "allow"

    def test_dry_run_fallback_destructive_is_blocked(self) -> None:
        from nexus.core.policy import RULE_DRY_RUN, ActionContext

        engine = self._dry_run_engine()
        ctx = ActionContext(
            action_type="overwrite_file",
            transport="keyboard",
            is_destructive=True,
        )
        result = engine.check_action(ctx)
        assert result.verdict == "block"
        assert result.rule == RULE_DRY_RUN

    def test_policy_result_message_is_meaningful(self) -> None:
        from nexus.core.policy import ActionContext

        engine = self._dry_run_engine()
        ctx = ActionContext(
            action_type="drop_table",
            transport="uia",
            is_destructive=True,
        )
        result = engine.check_action(ctx)
        assert len(result.message) > 20
        assert result.rule is not None

    def test_bypass_attempt_with_zero_cost_still_blocked(self) -> None:
        """Sıfır maliyetli, sıfır action sayılı senaryo: dry-run bloklar."""
        from nexus.core.policy import ActionContext

        engine = self._dry_run_engine()
        ctx = ActionContext(
            action_type="rm_rf",
            transport="uia",
            is_destructive=True,
            actions_so_far=0,
            elapsed_seconds=0.0,
            task_cost_usd=0.0,
            daily_cost_usd=0.0,
        )
        result = engine.check_action(ctx)
        assert result.verdict == "block"

    def test_all_native_transports_blocked_in_dry_run(self) -> None:
        """Her native transport kombinasyonu dry-run'da bloklanmalı."""
        from nexus.core.policy import ActionContext

        engine = self._dry_run_engine()
        action_types = ["delete", "overwrite", "truncate", "drop", "format"]
        transports = ["uia", "dom", "file"]

        for action in action_types:
            for transport in transports:
                ctx = ActionContext(
                    action_type=action,
                    transport=transport,
                    is_destructive=True,
                )
                result = engine.check_action(ctx)
                assert result.verdict == "block", (
                    f"action={action}, transport={transport} must be blocked"
                )
