# ADR-009: Credential Storage = Windows Credential Manager

## Status
Accepted

## Context
API anahtarları (OpenAI, Anthropic) ve diğer hassas bilgilerin güvenli saklanması gerekir.

Adaylar:
- **Düz dosya** (`.env`) — kolay, ancak disk üzerinde şifresiz; git'e kaçma riski
- **Windows Registry** — programatik erişim kolay, ancak ACL karmaşık; şifreleme yok
- **Windows Credential Manager** — DPAPI-backed, kullanıcı oturumuna bağlı şifreleme;
  `keyring` kütüphanesi ile erişim
- **Azure Key Vault** — güçlü, ancak internet bağımlılığı ve kurulum maliyeti

## Decision
Tüm credential'lar **Windows Credential Manager**'da saklanır.
Erişim `keyring` kütüphanesi üzerinden yapılır (`pywin32` DPAPI doğrudan değil).

```python
import keyring
keyring.set_password("nexus-agent", "openai_api_key", value)
key = keyring.get_password("nexus-agent", "openai_api_key")
```

Registry kullanılmaz.

## Consequences
- `keyring` production bağımlılığına eklenir (V1 implementasyon fazında).
- Credential'lar kullanıcı oturumu dışında (başka kullanıcı, farklı makine) okunamaz.
- CI/CD: test ortamında `keyring` mock'lanır; gerçek credential kullanılmaz.
- Servis hesabı ile çalışma: headless modda environment variable fallback desteklenir.
- `pywin32` dolaylı bağımlılık olarak kalır (dxcam ve diğer Windows API'leri için).
