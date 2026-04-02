"""
Unit tests for nexus/infra/database.py and nexus/infra/repositories.py.

All tests use an in-memory SQLite DB (":memory:") so they are fast,
isolated, and leave no files on disk.
"""
from __future__ import annotations

import asyncio
import pytest

from nexus.infra.database import Database, MAX_CONNECTIONS
from nexus.infra.repositories import (
    ActionRepository,
    CostRepository,
    MemoryRepository,
    TaskRepository,
    TransportAuditRepository,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path) -> Database:  # type: ignore[type-arg]
    """Return an initialised in-memory Database."""
    d = Database(str(tmp_path / "test.db"))
    await d.init()
    return d


@pytest.fixture
def task_repo() -> TaskRepository:
    return TaskRepository()


@pytest.fixture
def action_repo() -> ActionRepository:
    return ActionRepository()


@pytest.fixture
def cost_repo() -> CostRepository:
    return CostRepository()


@pytest.fixture
def memory_repo() -> MemoryRepository:
    return MemoryRepository()


@pytest.fixture
def transport_repo() -> TransportAuditRepository:
    return TransportAuditRepository()


# ---------------------------------------------------------------------------
# Database — initialisation & PRAGMAs
# ---------------------------------------------------------------------------


class TestDatabaseInit:
    async def test_init_idempotent(self, db: Database) -> None:
        """Calling init() twice must not raise."""
        await db.init()
        await db.init()

    async def test_wal_mode(self, db: Database) -> None:
        async with db.connection() as conn:
            async with conn.execute("PRAGMA journal_mode;") as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row[0] == "wal"

    async def test_foreign_keys_on(self, db: Database) -> None:
        async with db.connection() as conn:
            async with conn.execute("PRAGMA foreign_keys;") as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row[0] == 1

    async def test_schema_migration_recorded(self, db: Database) -> None:
        async with db.connection() as conn:
            async with conn.execute(
                "SELECT version FROM schema_migrations WHERE version = 1;"
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row[0] == 1

    async def test_all_tables_exist(self, db: Database) -> None:
        expected = {
            "tasks", "actions", "cost_ledger",
            "memory_fingerprints", "corrections",
            "user_consent", "transport_audit",
            "schema_migrations",
        }
        async with db.connection() as conn:
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ) as cur:
                rows = await cur.fetchall()
        tables = {r[0] for r in rows}
        assert expected <= tables

    async def test_connection_returns_context(self, db: Database) -> None:
        async with db.connection() as conn:
            async with conn.execute("SELECT 1;") as cur:
                row = await cur.fetchone()
        assert row[0] == 1

    async def test_close_is_noop(self, db: Database) -> None:
        await db.close()  # must not raise


# ---------------------------------------------------------------------------
# FK constraint
# ---------------------------------------------------------------------------


class TestForeignKeyConstraint:
    async def test_action_requires_valid_task(self, db: Database) -> None:
        with pytest.raises(Exception):
            async with db.connection() as conn:
                await conn.execute(
                    "INSERT INTO actions (id, task_id, type) VALUES ('a1', 'nonexistent', 'click');"
                )

    async def test_cost_requires_valid_task(self, db: Database) -> None:
        with pytest.raises(Exception):
            async with db.connection() as conn:
                await conn.execute(
                    "INSERT INTO cost_ledger (task_id, provider, tokens, cost_usd) "
                    "VALUES ('ghost', 'openai', 100, 0.01);"
                )

    async def test_transport_audit_requires_valid_task(self, db: Database) -> None:
        with pytest.raises(Exception):
            async with db.connection() as conn:
                await conn.execute(
                    "INSERT INTO transport_audit "
                    "(task_id, attempted_transport, fallback_used, success, latency_ms) "
                    "VALUES ('ghost', 'uia', 0, 1, 5.0);"
                )


# ---------------------------------------------------------------------------
# TaskRepository
# ---------------------------------------------------------------------------


class TestTaskRepository:
    async def test_create_and_get(
        self, db: Database, task_repo: TaskRepository
    ) -> None:
        async with db.connection() as conn:
            await task_repo.create(conn, id="t1", goal="open browser")
            row = await task_repo.get(conn, "t1")
        assert row is not None
        assert row.id == "t1"
        assert row.goal == "open browser"
        assert row.status == "pending"

    async def test_get_missing_returns_none(
        self, db: Database, task_repo: TaskRepository
    ) -> None:
        async with db.connection() as conn:
            row = await task_repo.get(conn, "does-not-exist")
        assert row is None

    async def test_update_status(
        self, db: Database, task_repo: TaskRepository
    ) -> None:
        async with db.connection() as conn:
            await task_repo.create(conn, id="t2", goal="fill form")
            await task_repo.update_status(conn, "t2", "running")
            row = await task_repo.get(conn, "t2")
        assert row is not None
        assert row.status == "running"

    async def test_list_by_status(
        self, db: Database, task_repo: TaskRepository
    ) -> None:
        async with db.connection() as conn:
            await task_repo.create(conn, id="t3", goal="a", status="success")
            await task_repo.create(conn, id="t4", goal="b", status="pending")
            await task_repo.create(conn, id="t5", goal="c", status="success")
            rows = await task_repo.list_by_status(conn, "success")
        ids = {r.id for r in rows}
        assert "t3" in ids and "t5" in ids
        assert "t4" not in ids

    async def test_delete_cascades_actions(
        self,
        db: Database,
        task_repo: TaskRepository,
        action_repo: ActionRepository,
    ) -> None:
        async with db.connection() as conn:
            await task_repo.create(conn, id="t6", goal="cascade test")
            await action_repo.create(
                conn, id="a6", task_id="t6", type="click"
            )
            await task_repo.delete(conn, "t6")
            act = await action_repo.get(conn, "a6")
        assert act is None

    async def test_metadata_stored(
        self, db: Database, task_repo: TaskRepository
    ) -> None:
        async with db.connection() as conn:
            await task_repo.create(conn, id="t7", goal="meta", metadata='{"k":"v"}')
            row = await task_repo.get(conn, "t7")
        assert row is not None
        assert row.metadata == '{"k":"v"}'

    async def test_invalid_status_rejected(
        self, db: Database, task_repo: TaskRepository
    ) -> None:
        with pytest.raises(Exception):
            async with db.connection() as conn:
                await task_repo.create(conn, id="t8", goal="x", status="bogus")


# ---------------------------------------------------------------------------
# ActionRepository
# ---------------------------------------------------------------------------


class TestActionRepository:
    async def _seed_task(
        self, db: Database, task_repo: TaskRepository, tid: str = "ta1"
    ) -> None:
        async with db.connection() as conn:
            await task_repo.create(conn, id=tid, goal="test")

    async def test_create_and_get(
        self,
        db: Database,
        task_repo: TaskRepository,
        action_repo: ActionRepository,
    ) -> None:
        await self._seed_task(db, task_repo)
        async with db.connection() as conn:
            await action_repo.create(
                conn, id="a1", task_id="ta1", type="click", payload='{"x":5}'
            )
            row = await action_repo.get(conn, "a1")
        assert row is not None
        assert row.type == "click"
        assert row.payload == '{"x":5}'
        assert row.task_id == "ta1"

    async def test_list_for_task(
        self,
        db: Database,
        task_repo: TaskRepository,
        action_repo: ActionRepository,
    ) -> None:
        await self._seed_task(db, task_repo, "ta2")
        async with db.connection() as conn:
            await action_repo.create(conn, id="a2", task_id="ta2", type="click")
            await action_repo.create(conn, id="a3", task_id="ta2", type="type")
            rows = await action_repo.list_for_task(conn, "ta2")
        assert len(rows) == 2

    async def test_update_status(
        self,
        db: Database,
        task_repo: TaskRepository,
        action_repo: ActionRepository,
    ) -> None:
        await self._seed_task(db, task_repo, "ta3")
        async with db.connection() as conn:
            await action_repo.create(conn, id="a4", task_id="ta3", type="scroll")
            await action_repo.update_status(conn, "a4", "success")
            row = await action_repo.get(conn, "a4")
        assert row is not None
        assert row.status == "success"

    async def test_get_missing_returns_none(
        self,
        db: Database,
        action_repo: ActionRepository,
    ) -> None:
        async with db.connection() as conn:
            assert await action_repo.get(conn, "nope") is None

    async def test_delete(
        self,
        db: Database,
        task_repo: TaskRepository,
        action_repo: ActionRepository,
    ) -> None:
        await self._seed_task(db, task_repo, "ta4")
        async with db.connection() as conn:
            await action_repo.create(conn, id="a5", task_id="ta4", type="click")
            await action_repo.delete(conn, "a5")
            assert await action_repo.get(conn, "a5") is None


# ---------------------------------------------------------------------------
# CostRepository
# ---------------------------------------------------------------------------


class TestCostRepository:
    async def _seed(self, db: Database, tid: str = "tc1") -> None:
        async with db.connection() as conn:
            await TaskRepository().create(conn, id=tid, goal="cost test")

    async def test_record_and_total(
        self, db: Database, cost_repo: CostRepository
    ) -> None:
        await self._seed(db)
        async with db.connection() as conn:
            await cost_repo.record(
                conn, task_id="tc1", provider="anthropic", tokens=1000, cost_usd=0.02
            )
            await cost_repo.record(
                conn, task_id="tc1", provider="openai", tokens=500, cost_usd=0.01
            )
            total = await cost_repo.total_for_task(conn, "tc1")
        assert total == pytest.approx(0.03)

    async def test_total_empty_task(
        self, db: Database, cost_repo: CostRepository
    ) -> None:
        await self._seed(db, "tc2")
        async with db.connection() as conn:
            total = await cost_repo.total_for_task(conn, "tc2")
        assert total == pytest.approx(0.0)

    async def test_list_for_task(
        self, db: Database, cost_repo: CostRepository
    ) -> None:
        await self._seed(db, "tc3")
        async with db.connection() as conn:
            await cost_repo.record(
                conn, task_id="tc3", provider="anthropic", tokens=100, cost_usd=0.005
            )
            rows = await cost_repo.list_for_task(conn, "tc3")
        assert len(rows) == 1
        assert rows[0].provider == "anthropic"

    async def test_multiple_providers_summed(
        self, db: Database, cost_repo: CostRepository
    ) -> None:
        await self._seed(db, "tc4")
        async with db.connection() as conn:
            for i in range(5):
                await cost_repo.record(
                    conn, task_id="tc4", provider="x", tokens=10, cost_usd=0.001
                )
            total = await cost_repo.total_for_task(conn, "tc4")
        assert total == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# MemoryRepository
# ---------------------------------------------------------------------------


class TestMemoryRepository:
    async def test_upsert_and_get_by_fingerprint(
        self, db: Database, memory_repo: MemoryRepository
    ) -> None:
        async with db.connection() as conn:
            await memory_repo.upsert(
                conn,
                id="m1",
                fingerprint="fp-abc",
                label="login_button",
                confidence=0.95,
            )
            row = await memory_repo.get_by_fingerprint(conn, "fp-abc")
        assert row is not None
        assert row.label == "login_button"
        assert row.confidence == pytest.approx(0.95)

    async def test_upsert_updates_existing(
        self, db: Database, memory_repo: MemoryRepository
    ) -> None:
        async with db.connection() as conn:
            await memory_repo.upsert(
                conn, id="m2", fingerprint="fp-dup", label="old", confidence=0.5
            )
            await memory_repo.upsert(
                conn, id="m2b", fingerprint="fp-dup", label="new", confidence=0.9
            )
            row = await memory_repo.get_by_fingerprint(conn, "fp-dup")
        assert row is not None
        assert row.label == "new"
        assert row.confidence == pytest.approx(0.9)

    async def test_get_by_id(
        self, db: Database, memory_repo: MemoryRepository
    ) -> None:
        async with db.connection() as conn:
            await memory_repo.upsert(
                conn, id="m3", fingerprint="fp-xyz", label="submit"
            )
            row = await memory_repo.get(conn, "m3")
        assert row is not None
        assert row.fingerprint == "fp-xyz"

    async def test_get_missing_returns_none(
        self, db: Database, memory_repo: MemoryRepository
    ) -> None:
        async with db.connection() as conn:
            assert await memory_repo.get(conn, "nope") is None

    async def test_delete(
        self, db: Database, memory_repo: MemoryRepository
    ) -> None:
        async with db.connection() as conn:
            await memory_repo.upsert(
                conn, id="m4", fingerprint="fp-del", label="x"
            )
            await memory_repo.delete(conn, "m4")
            assert await memory_repo.get(conn, "m4") is None


# ---------------------------------------------------------------------------
# TransportAuditRepository
# ---------------------------------------------------------------------------


class TestTransportAuditRepository:
    async def _seed_task(self, db: Database, tid: str = "tt1") -> None:
        async with db.connection() as conn:
            await TaskRepository().create(conn, id=tid, goal="transport test")

    async def test_record_and_list(
        self, db: Database, transport_repo: TransportAuditRepository
    ) -> None:
        await self._seed_task(db)
        async with db.connection() as conn:
            await transport_repo.record(
                conn,
                task_id="tt1",
                action_id=None,
                attempted_transport="uia",
                fallback_used=False,
                success=True,
                latency_ms=12.5,
            )
            rows = await transport_repo.list_for_task(conn, "tt1")
        assert len(rows) == 1
        r = rows[0]
        assert r.attempted_transport == "uia"
        assert r.fallback_used is False
        assert r.success is True
        assert r.latency_ms == pytest.approx(12.5)

    async def test_fallback_recorded(
        self, db: Database, transport_repo: TransportAuditRepository
    ) -> None:
        await self._seed_task(db, "tt2")
        async with db.connection() as conn:
            await transport_repo.record(
                conn,
                task_id="tt2",
                action_id=None,
                attempted_transport="mouse",
                fallback_used=True,
                success=False,
                latency_ms=50.0,
            )
            rows = await transport_repo.list_for_task(conn, "tt2")
        assert rows[0].fallback_used is True
        assert rows[0].success is False

    async def test_native_ratio_all_native(
        self, db: Database, transport_repo: TransportAuditRepository
    ) -> None:
        await self._seed_task(db, "tt3")
        async with db.connection() as conn:
            for m in ("uia", "dom", "file"):
                await transport_repo.record(
                    conn,
                    task_id="tt3",
                    action_id=None,
                    attempted_transport=m,
                    fallback_used=False,
                    success=True,
                    latency_ms=5.0,
                )
            ratio = await transport_repo.native_ratio_for_task(conn, "tt3")
        assert ratio == pytest.approx(1.0)

    async def test_native_ratio_mixed(
        self, db: Database, transport_repo: TransportAuditRepository
    ) -> None:
        await self._seed_task(db, "tt4")
        async with db.connection() as conn:
            await transport_repo.record(
                conn, task_id="tt4", action_id=None,
                attempted_transport="uia", fallback_used=False,
                success=True, latency_ms=5.0,
            )
            await transport_repo.record(
                conn, task_id="tt4", action_id=None,
                attempted_transport="mouse", fallback_used=True,
                success=False, latency_ms=30.0,
            )
            ratio = await transport_repo.native_ratio_for_task(conn, "tt4")
        assert ratio == pytest.approx(0.5)

    async def test_native_ratio_empty_task(
        self, db: Database, transport_repo: TransportAuditRepository
    ) -> None:
        await self._seed_task(db, "tt5")
        async with db.connection() as conn:
            ratio = await transport_repo.native_ratio_for_task(conn, "tt5")
        assert ratio == pytest.approx(0.0)

    async def test_invalid_transport_rejected(
        self, db: Database, transport_repo: TransportAuditRepository
    ) -> None:
        await self._seed_task(db, "tt6")
        with pytest.raises(Exception):
            async with db.connection() as conn:
                await transport_repo.record(
                    conn,
                    task_id="tt6",
                    action_id=None,
                    attempted_transport="telepathy",  # invalid
                    fallback_used=False,
                    success=True,
                    latency_ms=1.0,
                )

    async def test_with_action_id(
        self,
        db: Database,
        task_repo: TaskRepository,
        action_repo: ActionRepository,
        transport_repo: TransportAuditRepository,
    ) -> None:
        await self._seed_task(db, "tt7")
        async with db.connection() as conn:
            await action_repo.create(
                conn, id="act-tt7", task_id="tt7", type="click"
            )
            await transport_repo.record(
                conn,
                task_id="tt7",
                action_id="act-tt7",
                attempted_transport="dom",
                fallback_used=False,
                success=True,
                latency_ms=8.0,
            )
            rows = await transport_repo.list_for_task(conn, "tt7")
        assert rows[0].action_id == "act-tt7"

    async def test_list_empty_task(
        self, db: Database, transport_repo: TransportAuditRepository
    ) -> None:
        await self._seed_task(db, "tt8")
        async with db.connection() as conn:
            rows = await transport_repo.list_for_task(conn, "tt8")
        assert rows == []


# ---------------------------------------------------------------------------
# Concurrency — semaphore cap
# ---------------------------------------------------------------------------


class TestConcurrency:
    async def test_concurrent_connections_within_limit(
        self, db: Database
    ) -> None:
        """MAX_CONNECTIONS concurrent acquisitions must all succeed."""

        async def one_query() -> int:
            async with db.connection() as conn:
                async with conn.execute("SELECT 42;") as cur:
                    row = await cur.fetchone()
                return row[0]

        results = await asyncio.gather(
            *(one_query() for _ in range(MAX_CONNECTIONS))
        )
        assert all(r == 42 for r in results)

    async def test_rollback_on_error(self, db: Database) -> None:
        """A connection that raises inside the context must be rolled back."""
        async with db.connection() as conn:
            await TaskRepository().create(conn, id="rollback-task", goal="test")

        with pytest.raises(RuntimeError):
            async with db.connection() as conn:
                await conn.execute(
                    "UPDATE tasks SET goal = 'mutated' WHERE id = 'rollback-task';"
                )
                raise RuntimeError("simulate error")

        async with db.connection() as conn:
            async with conn.execute(
                "SELECT goal FROM tasks WHERE id = 'rollback-task';"
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row[0] == "test"  # original value preserved
