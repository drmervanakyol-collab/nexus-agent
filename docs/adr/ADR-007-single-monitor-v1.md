# ADR-007: Tek Aktif Monitör — V1

## Status
Accepted

## Context
Çoklu monitör desteği; koordinat uzayı, pencere yönetimi ve capture pipeline'ını karmaşıklaştırır.
`dxcam` kütüphanesi çoklu monitörü destekler, ancak ekran seçimi ve pencere-monitör eşlemesi
edge case'ler içerir.

## Decision
V1'de yalnızca **birincil (primary) monitör** desteklenir.

- `dxcam` her zaman `output=0` (primary display) ile başlatılır.
- Çoklu monitör konfigürasyonu tespit edilirse kullanıcıya bilgi mesajı gösterilir.
- Tüm koordinatlar birincil monitörün koordinat uzayında tutulur.

## Consequences
- Capture modülü `monitor_index` parametresi almaz; V1'de sabit `0`.
- Çoklu monitör konfigürasyonunda agent çalışır, ancak yalnızca birincil ekranı okur.
- V2: `output` parametresi yapılandırılabilir hale gelir; pencere-monitör eşlemesi eklenir.
- Test ortamı: sanal tek monitör varsayılır.
