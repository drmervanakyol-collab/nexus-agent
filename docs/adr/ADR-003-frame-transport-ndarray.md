# ADR-003: Frame Transport = Raw ndarray

## Status
Accepted

## Context
Ekran yakalama pipeline'ı, capture subprocess ile perception modülü arasında
frame'leri iletmek için bir format seçmek zorundadır.

Adaylar:
- **PNG/JPEG** — evrensel, ancak encode/decode maliyeti yüksek (~5–15ms/frame)
- **raw ndarray** — sıfır kopya, doğrudan NumPy/OpenCV işlemi, ~0.1ms overhead
- **shared memory** — en hızlı, ancak platform bağımlı ve karmaşık lifecycle

## Decision
Hot path'te (capture → perception) **raw `numpy.ndarray`** kullanılır.
Format: `uint8`, shape `(H, W, 3)`, renk sırası **BGR** (OpenCV standardı).

PNG yalnızca şu durumlarda üretilir:
- Kalıcı kayıt (golden test artifact'ları)
- HITL görsel kanıt
- Hata raporlama

## Consequences
- Tüm perception modülleri `np.ndarray` girdi bekler; tip annotasyonu zorunlu.
- Subprocess sınırında `multiprocessing.shared_memory` veya pickle kullanılır
  (ADR-010 ile koordineli).
- OpenCV BGR → RGB dönüşümü yalnızca UI gösterimi için yapılır.
- Bellek kullanımı: 1920×1080×3 ≈ 6MB/frame; pipeline tamponu max 3 frame tutar.
