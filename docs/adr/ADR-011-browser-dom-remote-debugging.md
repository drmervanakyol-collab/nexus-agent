# ADR-011: Browser DOM Access — Remote Debugging Port

## Status
Accepted

## Context
Browser DOM'una erişim için üç yaklaşım değerlendirildi:

1. **Browser extension** — yüksek erişim, ancak kurulum gerektirir; sandboxed
2. **WebDriver (Selenium/Playwright)** — güçlü, ancak ayrı browser instance başlatır;
   kullanıcının mevcut oturumunu göremez
3. **Chrome DevTools Protocol (CDP) — remote debugging** — mevcut browser'a bağlanır,
   tam DOM/JS erişimi, standart protokol

## Decision
V1'de Chrome ve Edge **`--remote-debugging-port=9222`** flag'i ile başlatılır.

Agent başlangıçta browser'ı bu modda kendisi başlatır:
```
chrome.exe --remote-debugging-port=9222 --user-data-dir=<profile>
```

CDP üzerinden WebSocket bağlantısı: `ws://localhost:9222/json`

**Mevcut çalışan browser'a bağlanma V2'ye ertelendi.**

## Consequences
- `nexus/source/dom/` CDP istemcisi içerir; `aiohttp` + `websockets` kullanır.
- Port `9222` varsayılan; `configs/` ile değiştirilebilir.
- Güvenlik: remote debugging port yalnızca `localhost`'a bind edilir.
- Kullanıcı mevcut browser oturumunu kapatmak zorunda kalabilir (V1 kısıtı);
  bu durum onboarding belgelerinde açıkça belirtilir.
- V2: çalışan browser tespit edilirse `--remote-debugging-port` ile yeniden başlatma
  veya mevcut porta bağlanma desteklenir.
