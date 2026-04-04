"""
nexus/cloud/providers.py
Cloud LLM Provider abstraction — ADR-010: multi-provider with fallback.

Public API
----------
  CloudMessage           Input message (role, content, optional image bytes).
  CloudResponse          Normalised response from any provider.
  CloudProvider          Protocol — any object with async complete().
  OpenAIProvider         Wraps AsyncOpenAI with error mapping + retry.
  AnthropicProvider      Wraps AsyncAnthropic with error mapping + retry.
  FallbackProvider       Tries primary, falls back to secondary on CloudError.

Error mapping
-------------
  openai / anthropic exceptions → Nexus CloudError hierarchy:
    AuthenticationError  → InvalidAPIKeyError      (no retry)
    RateLimitError       → CloudQuotaExceededError  (no retry)
    APITimeoutError      → CloudTimeoutError        (retried)
    APIConnectionError   → CloudUnavailableError    (retried)

  NOTE: APITimeoutError must be caught *before* APIConnectionError because
  it is a subclass of APIConnectionError in both SDK error hierarchies.

Retry policy
------------
  max_retries=3 (default): up to 3 additional attempts after the first,
  for 4 total calls.  Backoff: sleep(2^attempt) seconds after each failure
  (attempt index is 0-based, so sleeps are 1s, 2s, 4s).
  No retry for InvalidAPIKeyError or CloudQuotaExceededError.

Image support
-------------
  CloudMessage.image (bytes | None): PNG bytes are base64-encoded and
  sent as a multipart content block in both provider request formats.
"""
from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from nexus.core.errors import (
    CloudError,
    CloudQuotaExceededError,
    CloudTimeoutError,
    CloudUnavailableError,
    InvalidAPIKeyError,
)
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

_DEFAULT_MAX_RETRIES: int = 3


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloudMessage:
    """Single message in a conversation with a cloud LLM provider.

    Attributes
    ----------
    role:
        Message author: ``"user"``, ``"assistant"``, or ``"system"``.
    content:
        Text content of the message.
    image:
        Optional PNG image bytes.  Base64-encoded before transmission.
    """

    role: str
    content: str
    image: bytes | None = None


@dataclass(frozen=True)
class CloudResponse:
    """Normalised response from a cloud LLM provider.

    Attributes
    ----------
    content:        Text content of the completion.
    tokens_input:   Number of tokens in the prompt.
    tokens_output:  Number of tokens in the completion.
    model_used:     Model identifier as returned by the provider.
    provider:       Provider name: ``"openai"`` or ``"anthropic"``.
    latency_ms:     Wall-clock time for the API call (milliseconds).
    finish_reason:  Provider finish reason (e.g. ``"stop"``, ``"end_turn"``).
    """

    content: str
    tokens_input: int
    tokens_output: int
    model_used: str
    provider: str
    latency_ms: float
    finish_reason: str


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CloudProvider(Protocol):
    """Duck-typing interface for any LLM provider wrapper."""

    async def complete(
        self,
        messages: list[CloudMessage],
        model: str,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> CloudResponse: ...


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class OpenAIProvider:
    """
    OpenAI chat-completion provider with error mapping and retry.

    Parameters
    ----------
    api_key:
        OpenAI secret key.  Ignored when *client* is supplied.
    client:
        Injected ``AsyncOpenAI`` instance (used in tests).
    max_retries:
        Number of retry attempts after the first failure.
    _sleep_fn:
        Injectable async sleep; defaults to ``asyncio.sleep``.
    _time_fn:
        Injectable clock; defaults to ``time.monotonic``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        client: Any | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        _sleep_fn: Callable[[float], Any] | None = None,
        _time_fn: Callable[[], float] | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            from openai import AsyncOpenAI  # noqa: PLC0415

            # Disable SDK retries — our retry loop handles all backoff logic.
            self._client = AsyncOpenAI(api_key=api_key, max_retries=0)
        self._max_retries = max_retries
        self._sleep: Callable[[float], Any] = _sleep_fn or asyncio.sleep
        self._time: Callable[[], float] = _time_fn or time.monotonic

    async def complete(
        self,
        messages: list[CloudMessage],
        model: str,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> CloudResponse:
        """Send *messages* to OpenAI and return a normalised CloudResponse."""
        last_exc: CloudError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._attempt(messages, model, max_tokens, timeout)
            except (InvalidAPIKeyError, CloudQuotaExceededError):
                raise
            except (CloudTimeoutError, CloudUnavailableError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    _log.debug(
                        "openai_retry",
                        attempt=attempt,
                        backoff_s=2**attempt,
                        error=str(exc),
                    )
                    await self._sleep(2**attempt)
        raise last_exc  # type: ignore[misc]

    async def _attempt(
        self,
        messages: list[CloudMessage],
        model: str,
        max_tokens: int,
        timeout: float,
    ) -> CloudResponse:
        import openai  # noqa: PLC0415

        oai_messages = [_to_openai_message(m) for m in messages]
        t0 = self._time()
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=oai_messages,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except openai.AuthenticationError as exc:
            raise InvalidAPIKeyError(str(exc)) from exc
        except openai.RateLimitError as exc:
            raise CloudQuotaExceededError(str(exc)) from exc
        except openai.APITimeoutError as exc:  # must precede APIConnectionError
            raise CloudTimeoutError(str(exc)) from exc
        except openai.APIConnectionError as exc:
            raise CloudUnavailableError(str(exc)) from exc

        latency_ms = (self._time() - t0) * 1000
        choice = response.choices[0]
        return CloudResponse(
            content=choice.message.content or "",
            tokens_input=response.usage.prompt_tokens,
            tokens_output=response.usage.completion_tokens,
            model_used=response.model,
            provider="openai",
            latency_ms=round(latency_ms, 3),
            finish_reason=choice.finish_reason or "",
        )


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """
    Anthropic Messages provider with error mapping and retry.

    Parameters
    ----------
    api_key:
        Anthropic secret key.  Ignored when *client* is supplied.
    client:
        Injected ``AsyncAnthropic`` instance (used in tests).
    max_retries:
        Number of retry attempts after the first failure.
    _sleep_fn:
        Injectable async sleep; defaults to ``asyncio.sleep``.
    _time_fn:
        Injectable clock; defaults to ``time.monotonic``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        client: Any | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        _sleep_fn: Callable[[float], Any] | None = None,
        _time_fn: Callable[[], float] | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            from anthropic import AsyncAnthropic  # noqa: PLC0415

            # Disable SDK retries — our retry loop handles all backoff logic.
            self._client = AsyncAnthropic(api_key=api_key, max_retries=0)
        self._max_retries = max_retries
        self._sleep: Callable[[float], Any] = _sleep_fn or asyncio.sleep
        self._time: Callable[[], float] = _time_fn or time.monotonic

    async def complete(
        self,
        messages: list[CloudMessage],
        model: str,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> CloudResponse:
        """Send *messages* to Anthropic and return a normalised CloudResponse."""
        last_exc: CloudError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._attempt(messages, model, max_tokens, timeout)
            except (InvalidAPIKeyError, CloudQuotaExceededError):
                raise
            except (CloudTimeoutError, CloudUnavailableError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    _log.debug(
                        "anthropic_retry",
                        attempt=attempt,
                        backoff_s=2**attempt,
                        error=str(exc),
                    )
                    await self._sleep(2**attempt)
        raise last_exc  # type: ignore[misc]

    async def _attempt(
        self,
        messages: list[CloudMessage],
        model: str,
        max_tokens: int,
        timeout: float,
    ) -> CloudResponse:
        import anthropic  # noqa: PLC0415

        ant_messages = [
            _to_anthropic_message(m) for m in messages if m.role != "system"
        ]
        system_parts = [m.content for m in messages if m.role == "system"]

        kwargs: dict[str, Any] = {}
        if system_parts:
            kwargs["system"] = system_parts[0]

        t0 = self._time()
        try:
            response = await self._client.messages.create(
                model=model,
                messages=ant_messages,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )
        except anthropic.AuthenticationError as exc:
            raise InvalidAPIKeyError(str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise CloudQuotaExceededError(str(exc)) from exc
        except anthropic.APITimeoutError as exc:  # must precede APIConnectionError
            raise CloudTimeoutError(str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise CloudUnavailableError(str(exc)) from exc

        latency_ms = (self._time() - t0) * 1000
        text_content = next(
            (block.text for block in response.content if hasattr(block, "text")),
            "",
        )
        return CloudResponse(
            content=text_content,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            model_used=response.model,
            provider="anthropic",
            latency_ms=round(latency_ms, 3),
            finish_reason=response.stop_reason or "",
        )


# ---------------------------------------------------------------------------
# FallbackProvider
# ---------------------------------------------------------------------------


class FallbackProvider:
    """
    Try *primary*; on any CloudError fall back to *secondary*.

    Parameters
    ----------
    primary:
        First provider to try.
    secondary:
        Provider used when *primary* raises a CloudError.
    """

    def __init__(self, primary: CloudProvider, secondary: CloudProvider) -> None:
        self._primary = primary
        self._secondary = secondary

    async def complete(
        self,
        messages: list[CloudMessage],
        model: str,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ) -> CloudResponse:
        """Complete with primary; on CloudError retry with secondary."""
        try:
            return await self._primary.complete(messages, model, max_tokens, timeout)
        except CloudError as exc:
            _log.warning(
                "fallback_triggered",
                primary_error=type(exc).__name__,
                detail=str(exc),
            )
            return await self._secondary.complete(messages, model, max_tokens, timeout)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_openai_message(msg: CloudMessage) -> dict[str, Any]:
    """Convert a CloudMessage to the OpenAI messages-list format."""
    if msg.image is None:
        return {"role": msg.role, "content": msg.content}
    b64 = base64.b64encode(msg.image).decode()
    return {
        "role": msg.role,
        "content": [
            {"type": "text", "text": msg.content},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
        ],
    }


def _to_anthropic_message(msg: CloudMessage) -> dict[str, Any]:
    """Convert a CloudMessage to the Anthropic messages-list format."""
    if msg.image is None:
        return {"role": msg.role, "content": msg.content}
    b64 = base64.b64encode(msg.image).decode()
    return {
        "role": msg.role,
        "content": [
            {"type": "text", "text": msg.content},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                },
            },
        ],
    }
