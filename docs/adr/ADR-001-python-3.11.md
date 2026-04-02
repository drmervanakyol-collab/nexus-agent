# ADR-001: Python 3.11

## Status
Accepted

## Context
Nexus Agent; asyncio, tip anotasyonları ve performans açısından modern bir Python sürümüne ihtiyaç duyuyor.
Python 3.11, 3.10'a kıyasla %10–60 hız iyileştirmesi, daha iyi hata mesajları (fine-grained tracebacks)
ve `tomllib` standart kütüphane desteği sunuyor.

## Decision
Minimum Python sürümü **3.11** olarak sabitlenir.

`pyproject.toml`:
```toml
requires-python = ">=3.11"
```

## Consequences
- `match/case` (3.10+) ve `ExceptionGroup` (3.11+) özgürce kullanılabilir.
- CI/CD pipeline'ı yalnızca 3.11+ image'larıyla çalışır.
- 3.10 ve altı sürümler desteklenmez; `pyproject.toml` bunu açıkça kısıtlar.
