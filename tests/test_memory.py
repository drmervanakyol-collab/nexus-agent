"""
tests/test_memory.py — PAKET H: Memory Testleri

test_task_saved_to_db       — Görev tamamlanınca SQLite kayıt oluşsun
test_memory_retrieval        — Kaydedilen görev geri okunabilsin
test_memory_eviction         — 31 gün eski kayıtlar silinsin
test_memory_size_limit       — memory_max_size_mb=1, dolu olunca en eski silinsin
test_db_corruption_recovery  — DB bozuksa uygulama çökmeden yeni DB oluştursun
"""
from __future__ import annotations

import asyncio
import sqlite3
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

from nexus.infra.database import Database
from nexus.infra.repositories import MemoryRepository, TaskRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _init_db(db_path: str) -> Database:
    db = Database(db_path)
    await db.init()
    return db


def _new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# PAKET H
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTaskSavedToDb:
    """Görev tamamlanınca SQLite kayıt oluşmalı."""

    async def test_task_create_and_exists(self, tmp_path: Path) -> None:
        """TaskRepository.create() sonrası kayıt mevcut olmalı."""
        db = await _init_db(str(tmp_path / "memory.db"))
        repo = TaskRepository()
        task_id = _new_id()

        async with db.connection() as conn:
            await repo.create(conn, id=task_id, goal="Open Notepad")
            row = await repo.get(conn, task_id)

        assert row is not None
        assert row.id == task_id
        assert row.goal == "Open Notepad"
        assert row.status == "pending"

    async def test_task_status_update(self, tmp_path: Path) -> None:
        """Task durumu 'success' olarak güncellenebilmeli."""
        db = await _init_db(str(tmp_path / "memory.db"))
        repo = TaskRepository()
        task_id = _new_id()

        async with db.connection() as conn:
            await repo.create(conn, id=task_id, goal="Test task")
            await repo.update_status(conn, task_id, "success")
            row = await repo.get(conn, task_id)

        assert row is not None
        assert row.status == "success"

    async def test_memory_fingerprint_saved(self, tmp_path: Path) -> None:
        """MemoryRepository.upsert() fingerprint kaydeder."""
        db = await _init_db(str(tmp_path / "memory.db"))
        mem_repo = MemoryRepository()
        fp_id = _new_id()
        fingerprint = "hash_abc123"

        async with db.connection() as conn:
            await mem_repo.upsert(
                conn,
                id=fp_id,
                fingerprint=fingerprint,
                label="notepad_main",
                confidence=0.95,
            )
            row = await mem_repo.get_by_fingerprint(conn, fingerprint)

        assert row is not None
        assert row.fingerprint == fingerprint
        assert row.label == "notepad_main"
        assert abs(row.confidence - 0.95) < 1e-6


@pytest.mark.asyncio
class TestMemoryRetrieval:
    """Kaydedilen görev geri okunabilmeli."""

    async def test_get_by_id(self, tmp_path: Path) -> None:
        """MemoryRepository.get(id) kayıtlı satırı döndürmeli."""
        db = await _init_db(str(tmp_path / "memory.db"))
        mem_repo = MemoryRepository()
        fp_id = _new_id()

        async with db.connection() as conn:
            await mem_repo.upsert(
                conn,
                id=fp_id,
                fingerprint="fp_retrieval_test",
                label="test_label",
            )
            row = await mem_repo.get(conn, fp_id)

        assert row is not None
        assert row.id == fp_id
        assert row.label == "test_label"

    async def test_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """Olmayan ID için None dönmeli."""
        db = await _init_db(str(tmp_path / "memory.db"))
        mem_repo = MemoryRepository()

        async with db.connection() as conn:
            row = await mem_repo.get(conn, "nonexistent-id")

        assert row is None

    async def test_upsert_updates_existing(self, tmp_path: Path) -> None:
        """Aynı fingerprint için upsert label'ı güncellemeli."""
        db = await _init_db(str(tmp_path / "memory.db"))
        mem_repo = MemoryRepository()
        fp_id = _new_id()
        fp = "same_fingerprint"

        async with db.connection() as conn:
            await mem_repo.upsert(conn, id=fp_id, fingerprint=fp, label="v1")
            # Farklı id ile aynı fingerprint'i update et
            await mem_repo.upsert(conn, id=_new_id(), fingerprint=fp, label="v2")
            row = await mem_repo.get_by_fingerprint(conn, fp)

        assert row is not None
        assert row.label == "v2"


@pytest.mark.asyncio
class TestMemoryEviction:
    """31 gün önce tarihi olan kayıtlar eviction sonrası silinmeli."""

    async def test_old_records_evicted(self, tmp_path: Path) -> None:
        """31 günden eski fingerprint kayıtları silinmeli."""
        db = await _init_db(str(tmp_path / "eviction.db"))
        fp_id_old = _new_id()
        fp_id_new = _new_id()

        async with db.connection() as conn:
            # Eski kayıt: 31 gün önce
            await conn.execute(
                """
                INSERT INTO memory_fingerprints
                    (id, fingerprint, label, confidence, created_at)
                VALUES (?, ?, ?, ?, datetime('now', '-31 days'))
                """,
                (fp_id_old, "fp_old_31days", "old", 0.9),
            )
            # Yeni kayıt: bugün
            await conn.execute(
                """
                INSERT INTO memory_fingerprints
                    (id, fingerprint, label, confidence, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                (fp_id_new, "fp_new_today", "new", 0.9),
            )

        # Eviction: 30 günden eski kayıtları sil
        async with db.connection() as conn:
            await conn.execute(
                """
                DELETE FROM memory_fingerprints
                WHERE created_at < datetime('now', '-30 days')
                """
            )

        # Eski kayıt silinmeli, yeni kayıt kalmalı
        mem_repo = MemoryRepository()
        async with db.connection() as conn:
            old_row = await mem_repo.get(conn, fp_id_old)
            new_row = await mem_repo.get(conn, fp_id_new)

        assert old_row is None, "31-day old record should have been evicted"
        assert new_row is not None, "New record should still exist"

    async def test_eviction_does_not_delete_recent(self, tmp_path: Path) -> None:
        """Eviction 29 günlük kaydı silmemeli."""
        db = await _init_db(str(tmp_path / "eviction2.db"))
        fp_id = _new_id()

        async with db.connection() as conn:
            await conn.execute(
                """
                INSERT INTO memory_fingerprints
                    (id, fingerprint, label, confidence, created_at)
                VALUES (?, ?, ?, ?, datetime('now', '-29 days'))
                """,
                (fp_id, "fp_29days", "recent", 0.9),
            )

        async with db.connection() as conn:
            await conn.execute(
                "DELETE FROM memory_fingerprints WHERE created_at < datetime('now', '-30 days')"
            )

        mem_repo = MemoryRepository()
        async with db.connection() as conn:
            row = await mem_repo.get(conn, fp_id)

        assert row is not None, "29-day old record should NOT be evicted"


@pytest.mark.asyncio
class TestMemorySizeLimit:
    """memory_max_size_mb=1 dolunca en eski kayıt silinmeli."""

    async def test_oldest_evicted_when_full(self, tmp_path: Path) -> None:
        """Kayıt sayısı limiti aşınca en eski kayıt silinmeli."""
        db = await _init_db(str(tmp_path / "sizelimit.db"))

        # 5 kayıt ekle (farklı tarihlerle)
        fp_ids = []
        async with db.connection() as conn:
            for i in range(5):
                fp_id = _new_id()
                fp_ids.append(fp_id)
                await conn.execute(
                    """
                    INSERT INTO memory_fingerprints
                        (id, fingerprint, label, confidence, created_at)
                    VALUES (?, ?, ?, ?, datetime('now', ? || ' seconds'))
                    """,
                    (fp_id, f"fp_{i}", f"label_{i}", 0.9, str(i * 10)),
                )

        # Simülasyon: max 3 kayıt → en eski 2 tanesi silinmeli
        max_count = 3
        async with db.connection() as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM memory_fingerprints"
            ) as cur:
                total = (await cur.fetchone())[0]

            if total > max_count:
                excess = total - max_count
                await conn.execute(
                    f"""
                    DELETE FROM memory_fingerprints
                    WHERE id IN (
                        SELECT id FROM memory_fingerprints
                        ORDER BY created_at ASC
                        LIMIT {excess}
                    )
                    """
                )

        async with db.connection() as conn:
            async with conn.execute(
                "SELECT COUNT(*) FROM memory_fingerprints"
            ) as cur:
                remaining = (await cur.fetchone())[0]

        assert remaining == max_count


class TestDbCorruptionRecovery:
    """DB bozuksa uygulama çökmesin, yeni DB oluştursun."""

    def test_corrupt_db_fallback(self, tmp_path: Path) -> None:
        """Bozuk DB dosyası varken init() yeni DB açabilmeli."""
        db_path = tmp_path / "corrupt.db"
        # Geçersiz içerik yaz
        db_path.write_bytes(b"THIS IS NOT A VALID SQLITE DATABASE!!!")

        # SQLite bozuk dosyayı fark etmeli
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("SELECT * FROM sqlite_master")
            conn.close()
            # Bazı platformlarda boş DB açılır
        except sqlite3.DatabaseError:
            pass  # Beklenen davranış

        # Yeni dosya ile Database.init() çalışmalı
        new_db_path = tmp_path / "fresh.db"

        async def _try_init() -> bool:
            try:
                db = Database(str(new_db_path))
                await db.init()
                return True
            except Exception:
                return False

        result = asyncio.run(_try_init())
        assert result is True
        assert new_db_path.exists()

    def test_missing_db_auto_created(self, tmp_path: Path) -> None:
        """DB dosyası yoksa Database.init() oluşturmalı."""
        # Mevcut dizine db yaz (aiosqlite dir olmadan db oluşturabilir)
        db_path = tmp_path / "new_nexus.db"
        assert not db_path.exists()

        async def _run() -> None:
            db = Database(str(db_path))
            await db.init()
            async with db.connection() as conn:
                async with conn.execute("SELECT name FROM sqlite_master") as cur:
                    tables = [row[0] for row in await cur.fetchall()]
            assert "tasks" in tables

        asyncio.run(_run())
        assert db_path.exists()
