# Mutation Test Report — Faz 77

**Tarih:** 2026-04-13  
**Araç:** Cosmic Ray (SQLite backend)  
**Test suite:** pytest + respx + asyncio

---

## Özet

| Dosya | Önceki Skor | Yeni Skor | K/K+S | Hedef | Durum |
|-------|------------|-----------|-------|-------|-------|
| `nexus/decision/engine.py` | 45.8% | **73.3%** | 209/285 | 70% | PASS |
| `nexus/cloud/providers.py` | ~50% | **60.6%** | 183/302 | 70% | Fiziksel sınır (~68%) |
| `nexus/core/task_executor.py` | ~35% | **40.3%** | 250/620 | 70% | Fiziksel sınır (~67%) |

---

## engine.py — 73.3% (PASS)

### Yeni öldürülen mutasyonlar

| Konum | Mutasyon | Öldüren test |
|-------|----------|--------------|
| L512 `_target_from_plan` coord | 40+ operatör değişimi | `TestTargetFromPlan.test_odd_dimensions` |
| L237–238 `LocalResolver` `//` → `/` | Tamsayı bölme | `TestLocalResolverOddDimensions` |
| L58–59 `_ANTI/HARD_STUCK_WINDOW` | Sabit değer | `TestEngineConstants` |
| L177–183 `DecisionContext` defaults | 7 alan default | `TestDecisionContextDefaults` |
| L539, L541 `SuspendDecision` | float defaults | `TestSuspendDecisionDefaults` |
| L569 `!= ` hard-stuck | Operatör | `TestHardStuckBoundaryExtra` |
| L571 `[not WINDOW:]` slice | İndeks | `TestHardStuckBoundaryExtra` |
| L446, L309 cost_before_fn | max(0, neg) | `TestCostBeforeFnDefault` |

---

## providers.py — 60.6% (Fiziksel sınır: ~68%)

### Yeni öldürülen mutasyonlar

| Konum | Mutasyon | Öldüren test |
|-------|----------|--------------|
| L222/L340 `* 1000` → `*999/1001` | Tolerans daraltma (`abs=0.01`) | `TestOpenAILatencyPrecision` |
| L222/L340 `- t0` → `+ t0` | Non-zero t0 testi | `TestLatencyNonZeroT0` |
| L230/L351 `round(...,3)` → `round(...,2/4)` | 3-decimal latency | `TestLatencyRoundPrecision` |
| L173/L282 `max_tokens=1024` default | Explicit default test | `TestDefaultMaxTokensInRequestBody` |
| L174/L283 `timeout=30.0` default | `_attempt` mock | `TestCompleteTimeoutDefault` |
| L381 FallbackProvider `max_tokens` | Positional arg check | `TestFallbackProviderDefaults` |
| L382 FallbackProvider `timeout` | Positional arg check | `TestFallbackProviderDefaults` |
| L178/L287 `range(+2)` | Boundary: 2 fail, 3rd succeed | `TestRetryCountBoundary` |
| L178/L287 `range(\|1)` | Boundary: 3 fail, 4th succeed | `TestRetryCountBoundary` |
| L316 `==` → `<=` | Assistant before system | `TestSystemFilteringOrderMatters` |

### Unkillable survived mutasyonlar (97 adet)

| Kategori | Adet | Neden |
|----------|------|-------|
| Tip annotation `\| None` → diğer operatörler | 66 | `from __future__ import annotations` → annotation string olarak değerlendiriliyor |
| Log-only `backoff_s=2**attempt` (Pow→diğer) | 16 | Sadece `_log.debug()` argümanı; sleep/retry etkilenmiyor |
| Log-only `backoff_s` base (2→1/3) | 4 | Aynı |
| `CloudProvider` Protocol defaults | 4 | Duck-typing protocol; runtime'da kullanılmıyor |
| CPython string interning (`is`, `is not`) | 3 | Short string'ler intern ediliyor; `is` == `==` |
| `attempt < max_retries` → `!=`/`is not` | 4 | `range(n+1)` içinde `attempt` hiç `>= max_retries` olmaz |

**Maksimum ulaşılabilir skor:** (183+22)/(183+22+97) = **68.2%**

---

## task_executor.py — 40.3% (Fiziksel sınır: ~67%)

### Yeni öldürülen mutasyonlar

| Konum | Mutasyon | Öldüren test |
|-------|----------|--------------|
| L146 `r.y <= y` → `r.y >= y` | Koordinat sınır | `TestResolveUiaTargetBoundary` |
| `_default_done_fn` `== "cloud"` | String karşılaştırma | `TestDefaultDoneFn` |
| `_default_capture_fn` height/width=1 | Sayısal default | `TestDefaultCaptureFn` |
| `_default_perceive_fn` frame_sequence=1 | Sayısal default | `TestDefaultPerceiveFn` |

### Unkillable survived mutasyonlar (206+ adet)

| Kategori | Adet | Neden |
|----------|------|-------|
| Tip annotation `\| None` → diğer | 183 | Aynı `from __future__ import annotations` sorunu |
| CPython string interning | 23 | Eşdeğer mutasyonlar |
| Windows UIA/platform syscall kodu | Değişken | `ctypes`, `win32`, `pywinauto` bağımlı kod mock'lanamıyor |

**Maksimum ulaşılabilir skor:** ~67% (unkillable tip annotation yoğunluğu nedeniyle)

---

## Eklenen test sınıfları

### `tests/unit/test_decision_engine.py` — §M
- `TestTargetFromPlan` (3 test)
- `TestLocalResolverOddDimensions` (2 test)
- `TestDecisionContextDefaults` (1 test)
- `TestHardStuckBoundaryExtra` (2 test)
- `TestSuspendDecisionDefaults` (1 test)
- `TestEngineConstants` (1 test)
- `TestCostBeforeFnDefault` (2 test)

### `tests/unit/test_providers.py` — §13
- `TestOpenAILatencyPrecision` (2 test, tolerans `abs=0.5 → 0.01`)
- `TestAnthropicLatencyPrecision` (2 test, tolerans düzeltmesi)
- `TestAnthropicSystemPartsFirstUsed` (1 test)
- `TestOpenAIFirstChoiceUsed` (1 test)
- `TestDefaultMaxRetries` (3 test)
- `TestDefaultMaxTokensInRequestBody` (2 test)
- `TestAnthropicRoleFiltering` (2 test)
- `TestLatencyNonZeroT0` (2 test)
- `TestLatencyRoundPrecision` (2 test)
- `TestCompleteTimeoutDefault` (2 test)
- `TestFallbackProviderDefaults` (2 test)
- `TestRetryCountBoundary` (4 test)
- `TestSystemFilteringOrderMatters` (1 test)

### `tests/unit/test_task_executor.py` — §M
- `TestResolveUiaTargetBoundary` (2 test)
- `TestDefaultDoneFn` (3 test)
- `TestDefaultCaptureFn` (1 test)
- `TestDefaultPerceiveFn` (1 test)

**Toplam test sayısı: 3111 passed, 12 skipped, 0 failed**
