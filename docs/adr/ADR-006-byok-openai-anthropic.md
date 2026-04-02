# ADR-006: BYOK — OpenAI + Anthropic

## Status
Accepted

## Context
LLM sağlayıcısını sabitlemek, kullanıcıyı vendor lock-in'e sokar ve maliyet kontrolünü kısıtlar.
BYOK (Bring Your Own Key) modeli, kullanıcının kendi API anahtarlarını getirmesine izin verir.

## Decision
V1'de **OpenAI** ve **Anthropic** desteklenir. Her ikisi de eş zamanlı aktif olabilir.

- API anahtarları ADR-009 uyarınca Windows Credential Manager'da saklanır.
- `LLMProvider` protokolü, sağlayıcıyı soyutlar:

```python
class LLMProvider(Protocol):
    async def complete(self, messages: list[Message], **kwargs: Any) -> str: ...
```

- `cost_ledger`: her çağrı token sayısı + tahmini maliyet kaydedilir.
- `budget_cap`: günlük/aylık harcama limiti aşılırsa işlem duraklatılır (HITL).

## Consequences
- `openai>=1.0` ve `anthropic>=0.20` production bağımlılığı.
- Hangi modelin hangi görev için kullanılacağı `configs/` altında yapılandırılır.
- Kullanıcı key girmeden agent başlatılamaz; onboarding akışı bunu doğrular.
- V2: Azure OpenAI, local Ollama desteği değerlendirilir.
