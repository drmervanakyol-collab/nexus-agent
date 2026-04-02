# ADR-010: Capture = Dedicated Subprocess

## Status
Accepted

## Context
Ekran yakalama (dxcam / DXGI) GIL'i etkiler ve ana event loop'u bloke edebilir.
İzolasyon seçenekleri:

- **asyncio.run_in_executor (ThreadPool)** — GIL nedeniyle CPU-bound işlerde yetersiz
- **ProcessPoolExecutor** — genel amaçlı; ancak task scheduling overhead'i yüksek,
  shared memory yönetimi karmaşık
- **Dedicated subprocess** — tek sorumluluk, bağımsız restart, belirli CPU affinity

## Decision
Capture modülü **ayrı bir Python subprocess** olarak çalışır.

```
ana process  ──IPC──►  capture_subprocess
(asyncio)    ◄──ndarray─  (dxcam loop)
```

IPC mekanizması: `multiprocessing.shared_memory` (frame buffer) +
`multiprocessing.Queue` (kontrol mesajları).

ProcessPoolExecutor **kullanılmaz**.

## Consequences
- `nexus/capture/` subprocess entry point içerir (`__main__.py`).
- Ana process capture'ı `start()` / `stop()` sinyalleriyle yönetir.
- Capture subprocess çökmesi ana process'i düşürmez; watchdog ile otomatik restart.
- CPU affinity: Windows'ta `SetProcessAffinityMask` ile ayrı çekirdek atanabilir (V2).
- Test: subprocess başlatmak yerine `CaptureAdapter` mock'lanır.
