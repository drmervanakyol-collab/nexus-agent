# ADR-008: Terminal HITL — V1, Uzak Panel — V2

## Status
Accepted

## Context
Human-in-the-Loop (HITL) mekanizması, agent'ın belirsiz veya riskli durumlarda insandan
onay almasını sağlar. UI seçimi; geliştirme süresi, güvenilirlik ve erişilebilirlik arasında
denge kurar.

Adaylar:
- Terminal (stdin/stdout) — en basit, her ortamda çalışır
- Rich TUI (textual/rich) — güzel, ancak terminal bağımlı
- Uzak web paneli — en esnek, ancak web sunucusu gerektirir

## Decision
V1'de HITL **terminal tabanlı** çalışır:

```
[NEXUS HITL] Belirsiz eylem tespit edildi.
Devam etmek için [y], atlamak için [s], iptal için [q]: _
```

Agent `suspend` moduna geçer, kullanıcı yanıt verene kadar bekler.

## Consequences
- `hitl` modülü `input()` sarmalayıcısı + async event loop entegrasyonu içerir.
- Headless ortamlarda (CI) HITL otomatik olarak `skip` veya `abort` moduna geçer;
  bu davranış `configs/` ile yapılandırılır.
- V2: WebSocket tabanlı uzak panel; aynı `HitlHandler` protokolü implementasyonu.
- `suspend` / `resume` sinyalleri tüm pipeline katmanlarına yayılır.
