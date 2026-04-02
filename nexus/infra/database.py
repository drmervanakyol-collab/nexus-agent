"""
nexus/infra/database.py
Async SQLite connection pool for Nexus Agent.

Features
--------
- WAL journal mode + FK enforcement enforced per-connection
- busy_timeout = 5 000 ms (built into PRAGMA, reflected in connect timeout)
- Soft cap of MAX_CONNECTIONS concurrent connections
- Auto-migration on first acquire (idempotent SQL)
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

MAX_CONNECTIONS = 5
_MIGRATION_FILE = Path(__file__).parent.parent.parent / "migrations" / "001_initial.sql"


async def _configure(conn: aiosqlite.Connection) -> None:
    """Apply PRAGMAs that must be set on every new connection."""
    await conn.execute("PRAGMA journal_mode = WAL;")
    await conn.execute("PRAGMA foreign_keys = ON;")
    await conn.execute("PRAGMA busy_timeout = 5000;")
    conn.row_factory = aiosqlite.Row


async def _run_migrations(conn: aiosqlite.Connection) -> None:
    """Execute the initial migration SQL (idempotent)."""
    sql = _MIGRATION_FILE.read_text(encoding="utf-8")
    await conn.executescript(sql)
    await conn.commit()


class Database:
    """
    Manages a bounded pool of aiosqlite connections.

    Usage
    -----
    db = Database("nexus.db")
    await db.init()
    async with db.connection() as conn:
        await conn.execute(...)
    await db.close()
    """

    def __init__(self, db_path: str = "nexus.db") -> None:
        self._db_path = db_path
        self._semaphore = asyncio.Semaphore(MAX_CONNECTIONS)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def init(self) -> None:
        """Run migrations (idempotent — safe to call multiple times)."""
        async with self._init_lock:
            if self._initialized:
                return
            async with aiosqlite.connect(self._db_path) as conn:
                await _configure(conn)
                await _run_migrations(conn)
            self._initialized = True

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """
        Yield a configured connection, respecting the MAX_CONNECTIONS cap.
        The connection is committed and closed automatically.
        """
        if not self._initialized:
            await self.init()
        async with self._semaphore, aiosqlite.connect(self._db_path) as conn:
            await _configure(conn)
            try:
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def close(self) -> None:
        """No-op — aiosqlite connections are per-context; kept for API symmetry."""
        pass
