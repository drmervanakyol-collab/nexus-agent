# Nexus Agent — V1 Scope

Bu belge hangi özelliklerin hangi versiyona ait olduğunu tanımlar.
Her faz tag'i bu belgeyle tutarlı olmalıdır.

---

## V1-core (Zorunlu — V1 bitmeden ship edilmez)

Temel pipeline'ın uçtan uca çalışması için gereken minimum özellik seti.

### Altyapı
- [ ] Proje iskeleti ve araç zinciri (Faz 1)
- [ ] ADR ve Glossary (Faz 2)
- [ ] `structlog` tabanlı yapılandırılmış loglama
- [ ] `pydantic-settings` ile yapılandırma yönetimi
- [ ] SQLite WAL veritabanı şeması (ADR-005)
- [ ] Windows Credential Manager entegrasyonu (ADR-009)

### Capture
- [ ] `dxcam` tabanlı birincil monitör yakalama (ADR-007)
- [ ] Dedicated subprocess mimarisi (ADR-010)
- [ ] `stabilization_gate` implementasyonu
- [ ] `dirty_region` hesaplama

### Source Layer
- [ ] `transport_resolver` ve öncelik mantığı (ADR-002)
- [ ] `uia_adapter` — temel pencere/kontrol okuma
- [ ] `dom_adapter` — CDP bağlantısı, DOM sorgusu (ADR-011)
- [ ] `file_adapter` — metin dosyası okuma
- [ ] `visual_fallback` yolu

### Perception
- [ ] Tesseract OCR entegrasyonu (ADR-004)
- [ ] `locator` — bounding box tespiti
- [ ] `reader` — metin/değer okuma
- [ ] `arbitration` — çelişki çözümü
- [ ] `confidence_score` ve `ambiguity_score` hesaplama

### Decision
- [ ] OpenAI ve Anthropic LLM entegrasyonu (ADR-006)
- [ ] `action_spec` üretimi
- [ ] `cost_ledger` kaydı
- [ ] `budget_cap` kontrolü

### Action
- [ ] `transport_layer` — UIA aksiyon yürütme
- [ ] `transport_layer` — CDP aksiyon yürütme
- [ ] `transport_layer` — keyboard/mouse (visual fallback)
- [ ] `preflight` kontrolü
- [ ] `macroaction` zinciri

### Verification
- [ ] `visual_verification`
- [ ] `source_verification`
- [ ] `verification_policy` yapılandırması

### Memory
- [ ] `fingerprint` üretimi ve eşleştirmesi
- [ ] `correction_memory` kayıt ve sorgu

### HITL
- [ ] Terminal tabanlı suspend/resume (ADR-008)
- [ ] Headless mod (CI) için otomatik politika

### Skills
- [ ] `browser` skill — temel sayfa navigasyonu
- [ ] `desktop` skill — pencere yönetimi

---

## V1-nice (İstenen — Zaman kalırsa V1'e girer)

- [ ] `temporal_expert` — animasyon/yükleme tespiti
- [ ] `spatial_graph` — element konumsal ilişkileri
- [ ] `semantic_verification` — LLM tabanlı doğrulama
- [ ] `pdf` skill — PDF okuma
- [ ] `spreadsheet` skill — Excel/Sheets temel okuma
- [ ] `structlog` + `rich` formatlamalı terminal çıktısı
- [ ] Basit onboarding akışı (API key kurulumu)

---

## V2 (Planlı — V1 sonrası)

### Capture & Perception
- [ ] Çoklu monitör desteği (ADR-007 kısıtı kaldırılır)
- [ ] GPU destekli OCR (özel Türkçe model, ADR-004)
- [ ] Template matching yerine özellik tabanlı eşleştirme

### Source Layer
- [ ] Mevcut çalışan browser'a bağlanma (ADR-011 V2 notu)
- [ ] Firefox CDP desteği

### HITL & UI
- [ ] WebSocket tabanlı uzak HITL paneli (ADR-008 V2 notu)
- [ ] Görev geçmişi ve maliyet dashboard'u

### Entegrasyonlar
- [ ] Google Workspace entegrasyonu (`nexus/integrations/google/`)
- [ ] Azure OpenAI desteği (ADR-006 V2 notu)
- [ ] Local Ollama desteği

### Altyapı
- [ ] Alembic tabanlı şema migrasyonları (ADR-005 notu)
- [ ] CPU affinity yapılandırması (ADR-010 notu)
- [ ] Paketlenmiş dağıtım (PyInstaller veya MSIX)
