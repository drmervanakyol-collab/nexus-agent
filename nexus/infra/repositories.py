"""
nexus/infra/repositories.py
Data-access layer for all Nexus Agent domain objects.

Repositories
------------
TaskRepository          — tasks table
ActionRepository        — actions table
CostRepository          — cost_ledger table
MemoryRepository        — memory_fingerprints table
TransportAuditRepository — transport_audit table

All methods are async and receive an active aiosqlite.Connection so that
callers can compose multiple operations inside a single transaction.
"""
from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

# ---------------------------------------------------------------------------
# Row dataclasses (lightweight — no ORM)
# ---------------------------------------------------------------------------


@dataclass
class TaskRow:
    id: str
    goal: str
    status: str
    created_at: str
    updated_at: str
    metadata: str | None


@dataclass
class ActionRow:
    id: str
    task_id: str
    type: str
    payload: str
    status: str
    created_at: str


@dataclass
class CostRow:
    id: int
    task_id: str
    provider: str
    tokens: int
    cost_usd: float
    recorded_at: str


@dataclass
class MemoryFingerprintRow:
    id: str
    task_id: str | None
    fingerprint: str
    label: str
    confidence: float
    created_at: str


@dataclass
class TransportAuditRow:
    id: int
    task_id: str
    action_id: str | None
    attempted_transport: str
    fallback_used: bool
    success: bool
    latency_ms: float
    created_at: str


# ---------------------------------------------------------------------------
# TaskRepository
# ---------------------------------------------------------------------------


class TaskRepository:
    async def create(
        self,
        conn: aiosqlite.Connection,
        *,
        id: str,
        goal: str,
        status: str = "pending",
        metadata: str | None = None,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO tasks (id, goal, status, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (id, goal, status, metadata),
        )

    async def get(
        self, conn: aiosqlite.Connection, task_id: str
    ) -> TaskRow | None:
        async with conn.execute(
            "SELECT id, goal, status, created_at, updated_at, metadata "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return TaskRow(*row)

    async def update_status(
        self, conn: aiosqlite.Connection, task_id: str, status: str
    ) -> None:
        await conn.execute(
            """
            UPDATE tasks
               SET status = ?,
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = ?
            """,
            (status, task_id),
        )

    async def list_by_status(
        self, conn: aiosqlite.Connection, status: str
    ) -> list[TaskRow]:
        async with conn.execute(
            "SELECT id, goal, status, created_at, updated_at, metadata "
            "FROM tasks WHERE status = ? ORDER BY created_at",
            (status,),
        ) as cur:
            rows = await cur.fetchall()
        return [TaskRow(*r) for r in rows]

    async def delete(self, conn: aiosqlite.Connection, task_id: str) -> None:
        await conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))


# ---------------------------------------------------------------------------
# ActionRepository
# ---------------------------------------------------------------------------


class ActionRepository:
    async def create(
        self,
        conn: aiosqlite.Connection,
        *,
        id: str,
        task_id: str,
        type: str,
        payload: str = "{}",
        status: str = "pending",
    ) -> None:
        await conn.execute(
            """
            INSERT INTO actions (id, task_id, type, payload, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (id, task_id, type, payload, status),
        )

    async def get(
        self, conn: aiosqlite.Connection, action_id: str
    ) -> ActionRow | None:
        async with conn.execute(
            "SELECT id, task_id, type, payload, status, created_at "
            "FROM actions WHERE id = ?",
            (action_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return ActionRow(*row)

    async def list_for_task(
        self, conn: aiosqlite.Connection, task_id: str
    ) -> list[ActionRow]:
        async with conn.execute(
            "SELECT id, task_id, type, payload, status, created_at "
            "FROM actions WHERE task_id = ? ORDER BY created_at",
            (task_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [ActionRow(*r) for r in rows]

    async def update_status(
        self, conn: aiosqlite.Connection, action_id: str, status: str
    ) -> None:
        await conn.execute(
            "UPDATE actions SET status = ? WHERE id = ?",
            (status, action_id),
        )

    async def delete(self, conn: aiosqlite.Connection, action_id: str) -> None:
        await conn.execute("DELETE FROM actions WHERE id = ?", (action_id,))


# ---------------------------------------------------------------------------
# CostRepository
# ---------------------------------------------------------------------------


class CostRepository:
    async def record(
        self,
        conn: aiosqlite.Connection,
        *,
        task_id: str,
        provider: str,
        tokens: int,
        cost_usd: float,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO cost_ledger (task_id, provider, tokens, cost_usd)
            VALUES (?, ?, ?, ?)
            """,
            (task_id, provider, tokens, cost_usd),
        )

    async def total_for_task(
        self, conn: aiosqlite.Connection, task_id: str
    ) -> float:
        async with conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM cost_ledger WHERE task_id = ?",
            (task_id,),
        ) as cur:
            row = await cur.fetchone()
        return float(row[0]) if row else 0.0

    async def list_for_task(
        self, conn: aiosqlite.Connection, task_id: str
    ) -> list[CostRow]:
        async with conn.execute(
            "SELECT id, task_id, provider, tokens, cost_usd, recorded_at "
            "FROM cost_ledger WHERE task_id = ? ORDER BY recorded_at",
            (task_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [CostRow(*r) for r in rows]


# ---------------------------------------------------------------------------
# MemoryRepository
# ---------------------------------------------------------------------------


class MemoryRepository:
    async def upsert(
        self,
        conn: aiosqlite.Connection,
        *,
        id: str,
        fingerprint: str,
        label: str = "",
        confidence: float = 1.0,
        task_id: str | None = None,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO memory_fingerprints
                (id, task_id, fingerprint, label, confidence)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(fingerprint) DO UPDATE SET
                label      = excluded.label,
                confidence = excluded.confidence,
                task_id    = excluded.task_id
            """,
            (id, task_id, fingerprint, label, confidence),
        )

    async def get_by_fingerprint(
        self, conn: aiosqlite.Connection, fingerprint: str
    ) -> MemoryFingerprintRow | None:
        async with conn.execute(
            "SELECT id, task_id, fingerprint, label, confidence, created_at "
            "FROM memory_fingerprints WHERE fingerprint = ?",
            (fingerprint,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return MemoryFingerprintRow(*row)

    async def get(
        self, conn: aiosqlite.Connection, id: str
    ) -> MemoryFingerprintRow | None:
        async with conn.execute(
            "SELECT id, task_id, fingerprint, label, confidence, created_at "
            "FROM memory_fingerprints WHERE id = ?",
            (id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return MemoryFingerprintRow(*row)

    async def delete(self, conn: aiosqlite.Connection, id: str) -> None:
        await conn.execute(
            "DELETE FROM memory_fingerprints WHERE id = ?", (id,)
        )


# ---------------------------------------------------------------------------
# TransportAuditRepository
# ---------------------------------------------------------------------------


class TransportAuditRepository:
    async def record(
        self,
        conn: aiosqlite.Connection,
        *,
        task_id: str,
        action_id: str | None,
        attempted_transport: str,
        fallback_used: bool,
        success: bool,
        latency_ms: float,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO transport_audit
                (task_id, action_id, attempted_transport,
                 fallback_used, success, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                action_id,
                attempted_transport,
                int(fallback_used),
                int(success),
                latency_ms,
            ),
        )

    async def list_for_task(
        self, conn: aiosqlite.Connection, task_id: str
    ) -> list[TransportAuditRow]:
        async with conn.execute(
            """
            SELECT id, task_id, action_id, attempted_transport,
                   fallback_used, success, latency_ms, created_at
            FROM transport_audit
            WHERE task_id = ?
            ORDER BY created_at
            """,
            (task_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [
            TransportAuditRow(
                id=r[0],
                task_id=r[1],
                action_id=r[2],
                attempted_transport=r[3],
                fallback_used=bool(r[4]),
                success=bool(r[5]),
                latency_ms=r[6],
                created_at=r[7],
            )
            for r in rows
        ]

    async def native_ratio_for_task(
        self, conn: aiosqlite.Connection, task_id: str
    ) -> float:
        """Return fraction of transport calls that used a native method."""
        async with conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN attempted_transport IN ('uia','dom','file')
                         THEN 1 ELSE 0 END) AS native
            FROM transport_audit
            WHERE task_id = ?
            """,
            (task_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None or row[0] == 0:
            return 0.0
        return float(row[1]) / float(row[0])
