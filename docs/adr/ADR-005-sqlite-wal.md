# ADR-005: SQLite WAL Modu

## Status
Accepted

## Context
Nexus Agent; correction memory, cost ledger ve task geçmişini kalıcı olarak saklar.
Eş zamanlı okuma/yazma (async görevler + HITL UI) için veritabanı kilitleme stratejisi seçilmeli.

Adaylar:
- **SQLite (default journal)** — seri yazma, okuma sırasında kilitlenme
- **SQLite WAL** — eş zamanlı okuma + tek yazar, dosya tabanlı, sıfır kurulum
- **PostgreSQL** — güçlü, ancak lokal agent için fazla karmaşık

## Decision
SQLite **WAL (Write-Ahead Logging)** modu kullanılır.

Bağlantı açılırken:
```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
```

`aiosqlite` async erişim için kullanılır.

## Consequences
- Birden fazla async reader aynı anda çalışabilir; tek writer kilitleme yapar.
- WAL dosyası (`*.db-wal`) deploy artifact'larına dahil edilmez.
- Checkpoint stratejisi: her 1000 write'ta otomatik checkpoint, kapatmada tam checkpoint.
- Şema migrasyonları için `alembic` V2'de değerlendirilir; V1'de `CREATE TABLE IF NOT EXISTS`.
