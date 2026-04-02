# ADR-002: API-first, Visual Fallback

## Status
Accepted

## Context
Masaüstü/web otomasyonunda üç farklı erişim katmanı vardır:
1. **Structured** — UIA (Accessibility API), DOM (CDP/WebDriver), File API
2. **Visual** — Ekran görüntüsü + OCR + koordinat tabanlı işlem
3. **Hybrid** — İkisinin kombinasyonu

Sadece visual yaklaşım, koordinat kayması ve DPI değişimlerinde kırılgandır.
Sadece API yaklaşımı, erişilebilirlik ağacı olmayan uygulamalarda çalışmaz.

## Decision
Her READ ve ACTION işlemi için **önce structured kaynak** denenir:

```
UIA / DOM / File API
        │
   başarılı mı?
   ┌─────┴─────┐
  Evet         Hayır
   │            │
  kullan    capture + OCR
             + mouse/keyboard
```

Bu kural hem kaynak (source) katmanı hem de aksiyon (action) katmanı için geçerlidir.

## Consequences
- `source_layer` her zaman bir `transport_resolver` üzerinden erişir.
- `visual_fallback` ikinci-sınıf değil, birinci-sınıf kod yoludur; tamamen test edilir.
- Yeni bir uygulama adaptörü eklemek için önce structured adapter yazılır,
  visual fallback otomatik devreye girer.
- Performans: structured yol ~2ms, visual fallback ~80–200ms (OCR dahil).
