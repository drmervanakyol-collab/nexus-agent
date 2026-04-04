"""
tests/unit/test_providers.py
Unit tests for nexus/cloud/providers.py — FAZ 33.

Coverage
--------
  §1  CloudMessage / CloudResponse field defaults
  §2  _to_openai_message / _to_anthropic_message helpers
  §3  OpenAI response parse — all CloudResponse fields
  §4  OpenAI error mapping — AuthN, RateLimit, Timeout, Connection
  §5  Anthropic response parse — all CloudResponse fields
  §6  Anthropic error mapping — AuthN, RateLimit, Timeout, Connection
  §7  Retry logic — success after failures, exhausted retries
  §8  QuotaExceeded — no retry
  §9  InvalidAPIKey — no retry
  §10 FallbackProvider — primary fail → secondary, primary success
  §11 Image support — base64 in OpenAI and Anthropic requests
  §12 CloudProvider Protocol isinstance check
"""
from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from nexus.cloud.providers import (
    AnthropicProvider,
    CloudMessage,
    CloudProvider,
    CloudResponse,
    FallbackProvider,
    OpenAIProvider,
    _to_anthropic_message,
    _to_openai_message,
)
from nexus.core.errors import (
    CloudQuotaExceededError,
    CloudTimeoutError,
    CloudUnavailableError,
    InvalidAPIKeyError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OAI_URL = "https://api.openai.com/v1/chat/completions"
_ANT_URL = "https://api.anthropic.com/v1/messages"
_OAI_MODEL = "gpt-4o"
_ANT_MODEL = "claude-3-5-sonnet-20241022"
_OAI_KEY = "sk-test-key-openai"
_ANT_KEY = "sk-ant-test-key"


# ---------------------------------------------------------------------------
# Response body factories
# ---------------------------------------------------------------------------


def _oai_body(
    content: str,
    model: str = _OAI_MODEL,
    prompt_tokens: int = 5,
    completion_tokens: int = 10,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "choices": [
            {
                "finish_reason": finish_reason,
                "index": 0,
                "message": {"content": content, "role": "assistant"},
                "logprobs": None,
            }
        ],
        "created": 1234567890,
        "model": model,
        "object": "chat.completion",
        "usage": {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _ant_body(
    content: str,
    model: str = _ANT_MODEL,
    input_tokens: int = 5,
    output_tokens: int = 10,
    stop_reason: str = "end_turn",
) -> dict[str, Any]:
    return {
        "id": "msg-test",
        "content": [{"type": "text", "text": content}],
        "model": model,
        "role": "assistant",
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _oai_error_body(message: str = "error", code: str = "error") -> dict[str, Any]:
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


def _ant_error_body(message: str = "error", error_type: str = "error") -> dict[str, Any]:
    return {"type": "error", "error": {"type": error_type, "message": message}}


# ---------------------------------------------------------------------------
# Provider factories
# ---------------------------------------------------------------------------


def _oai_provider(**kwargs: Any) -> OpenAIProvider:
    """Create an OpenAIProvider with no sleeps and a no-op time function."""
    sleep_calls: list[float] = []

    async def _fake_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    defaults: dict[str, Any] = {
        "api_key": _OAI_KEY,
        "_sleep_fn": _fake_sleep,
        "_time_fn": _fixed_clock(0.0, 0.05),
    }
    defaults.update(kwargs)
    provider = OpenAIProvider(**defaults)
    provider._test_sleep_calls = sleep_calls  # type: ignore[attr-defined]
    return provider


def _ant_provider(**kwargs: Any) -> AnthropicProvider:
    """Create an AnthropicProvider with no sleeps and a no-op time function."""
    sleep_calls: list[float] = []

    async def _fake_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    defaults: dict[str, Any] = {
        "api_key": _ANT_KEY,
        "_sleep_fn": _fake_sleep,
        "_time_fn": _fixed_clock(0.0, 0.05),
    }
    defaults.update(kwargs)
    provider = AnthropicProvider(**defaults)
    provider._test_sleep_calls = sleep_calls  # type: ignore[attr-defined]
    return provider


def _fixed_clock(*values: float) -> Callable[[], float]:
    """Return a clock that cycles through *values* (last value repeats)."""
    seq = list(values)
    idx = 0

    def _clock() -> float:
        nonlocal idx
        v = seq[min(idx, len(seq) - 1)]
        idx += 1
        return v

    return _clock


# ---------------------------------------------------------------------------
# §1 — CloudMessage / CloudResponse field defaults
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_cloud_message_no_image(self) -> None:
        msg = CloudMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.image is None

    def test_cloud_message_with_image(self) -> None:
        img = b"\x89PNG\r\n"
        msg = CloudMessage(role="user", content="describe", image=img)
        assert msg.image == img

    def test_cloud_message_frozen(self) -> None:
        msg = CloudMessage(role="user", content="x")
        with pytest.raises((AttributeError, TypeError)):
            msg.role = "assistant"  # type: ignore[misc]

    def test_cloud_response_fields(self) -> None:
        resp = CloudResponse(
            content="hi",
            tokens_input=3,
            tokens_output=7,
            model_used="gpt-4o",
            provider="openai",
            latency_ms=42.0,
            finish_reason="stop",
        )
        assert resp.content == "hi"
        assert resp.tokens_input == 3
        assert resp.tokens_output == 7
        assert resp.model_used == "gpt-4o"
        assert resp.provider == "openai"
        assert resp.latency_ms == 42.0
        assert resp.finish_reason == "stop"

    def test_cloud_response_frozen(self) -> None:
        resp = CloudResponse(
            content="x",
            tokens_input=1,
            tokens_output=1,
            model_used="m",
            provider="openai",
            latency_ms=1.0,
            finish_reason="stop",
        )
        with pytest.raises((AttributeError, TypeError)):
            resp.content = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# §2 — Message conversion helpers
# ---------------------------------------------------------------------------


class TestMessageHelpers:
    def test_openai_text_only(self) -> None:
        msg = CloudMessage(role="user", content="hello")
        result = _to_openai_message(msg)
        assert result == {"role": "user", "content": "hello"}

    def test_openai_with_image(self) -> None:
        img = b"FAKE_PNG"
        msg = CloudMessage(role="user", content="describe", image=img)
        result = _to_openai_message(msg)
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert result["content"][0] == {"type": "text", "text": "describe"}
        url = result["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        decoded = base64.b64decode(url.split(",", 1)[1])
        assert decoded == img

    def test_anthropic_text_only(self) -> None:
        msg = CloudMessage(role="user", content="hello")
        result = _to_anthropic_message(msg)
        assert result == {"role": "user", "content": "hello"}

    def test_anthropic_with_image(self) -> None:
        img = b"FAKE_PNG"
        msg = CloudMessage(role="user", content="describe", image=img)
        result = _to_anthropic_message(msg)
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert result["content"][0] == {"type": "text", "text": "describe"}
        img_block = result["content"][1]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/png"
        decoded = base64.b64decode(img_block["source"]["data"])
        assert decoded == img


# ---------------------------------------------------------------------------
# §3 — OpenAI response parse
# ---------------------------------------------------------------------------


class TestOpenAIResponseParse:
    async def test_all_fields_correct(self) -> None:
        with respx.mock:
            respx.post(_OAI_URL).mock(
                return_value=httpx.Response(
                    200,
                    json=_oai_body(
                        "The answer is 42",
                        model="gpt-4o-mini",
                        prompt_tokens=8,
                        completion_tokens=15,
                        finish_reason="stop",
                    ),
                )
            )
            provider = _oai_provider()
            result = await provider.complete(
                [CloudMessage(role="user", content="What is 42?")],
                model=_OAI_MODEL,
            )

        assert result.content == "The answer is 42"
        assert result.tokens_input == 8
        assert result.tokens_output == 15
        assert result.model_used == "gpt-4o-mini"
        assert result.provider == "openai"
        assert result.finish_reason == "stop"
        assert result.latency_ms >= 0

    async def test_latency_ms_is_positive(self) -> None:
        with respx.mock:
            respx.post(_OAI_URL).mock(
                return_value=httpx.Response(200, json=_oai_body("ok"))
            )
            provider = OpenAIProvider(api_key=_OAI_KEY)
            result = await provider.complete(
                [CloudMessage(role="user", content="hi")], model=_OAI_MODEL
            )
        assert result.latency_ms >= 0

    async def test_empty_content_normalised(self) -> None:
        body = _oai_body("Hello")
        body["choices"][0]["message"]["content"] = None  # type: ignore[index]
        with respx.mock:
            respx.post(_OAI_URL).mock(return_value=httpx.Response(200, json=body))
            provider = _oai_provider()
            result = await provider.complete(
                [CloudMessage(role="user", content="hi")], model=_OAI_MODEL
            )
        assert result.content == ""

    async def test_system_message_forwarded(self) -> None:
        """System messages are passed through in the messages list for OpenAI."""
        captured: list[Any] = []

        def capture(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_oai_body("ok"))

        with respx.mock:
            respx.post(_OAI_URL).mock(side_effect=capture)
            provider = _oai_provider()
            await provider.complete(
                [
                    CloudMessage(role="system", content="Be concise"),
                    CloudMessage(role="user", content="hello"),
                ],
                model=_OAI_MODEL,
            )

        roles = [m["role"] for m in captured[0]["messages"]]
        assert "system" in roles


# ---------------------------------------------------------------------------
# §4 — OpenAI error mapping
# ---------------------------------------------------------------------------


class TestOpenAIErrorMapping:
    async def test_401_raises_invalid_api_key(self) -> None:
        with respx.mock:
            respx.post(_OAI_URL).mock(
                return_value=httpx.Response(
                    401, json=_oai_error_body("Invalid key", "invalid_api_key")
                )
            )
            provider = _oai_provider(max_retries=0)
            with pytest.raises(InvalidAPIKeyError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_OAI_MODEL
                )

    async def test_429_raises_quota_exceeded(self) -> None:
        with respx.mock:
            respx.post(_OAI_URL).mock(
                return_value=httpx.Response(
                    429, json=_oai_error_body("Rate limit", "rate_limit_exceeded")
                )
            )
            provider = _oai_provider(max_retries=0)
            with pytest.raises(CloudQuotaExceededError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_OAI_MODEL
                )

    async def test_read_timeout_raises_cloud_timeout(self) -> None:
        with respx.mock:
            respx.post(_OAI_URL).mock(
                side_effect=httpx.ReadTimeout("timed out", request=None)
            )
            provider = _oai_provider(max_retries=0)
            with pytest.raises(CloudTimeoutError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_OAI_MODEL
                )

    async def test_connect_error_raises_cloud_unavailable(self) -> None:
        with respx.mock:
            respx.post(_OAI_URL).mock(
                side_effect=httpx.ConnectError("connection refused")
            )
            provider = _oai_provider(max_retries=0)
            with pytest.raises(CloudUnavailableError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_OAI_MODEL
                )


# ---------------------------------------------------------------------------
# §5 — Anthropic response parse
# ---------------------------------------------------------------------------


class TestAnthropicResponseParse:
    async def test_all_fields_correct(self) -> None:
        with respx.mock:
            respx.post(_ANT_URL).mock(
                return_value=httpx.Response(
                    200,
                    json=_ant_body(
                        "Merhaba!",
                        model="claude-3-haiku-20240307",
                        input_tokens=7,
                        output_tokens=12,
                        stop_reason="end_turn",
                    ),
                )
            )
            provider = _ant_provider()
            result = await provider.complete(
                [CloudMessage(role="user", content="Merhaba")],
                model=_ANT_MODEL,
            )

        assert result.content == "Merhaba!"
        assert result.tokens_input == 7
        assert result.tokens_output == 12
        assert result.model_used == "claude-3-haiku-20240307"
        assert result.provider == "anthropic"
        assert result.finish_reason == "end_turn"
        assert result.latency_ms >= 0

    async def test_system_message_extracted(self) -> None:
        """System messages are extracted and sent as top-level 'system' param."""
        captured: list[Any] = []

        def capture(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_ant_body("ok"))

        with respx.mock:
            respx.post(_ANT_URL).mock(side_effect=capture)
            provider = _ant_provider()
            await provider.complete(
                [
                    CloudMessage(role="system", content="You are helpful"),
                    CloudMessage(role="user", content="hello"),
                ],
                model=_ANT_MODEL,
            )

        body = captured[0]
        assert body.get("system") == "You are helpful"
        roles = [m["role"] for m in body["messages"]]
        assert "system" not in roles

    async def test_no_system_message_omits_system_param(self) -> None:
        captured: list[Any] = []

        def capture(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_ant_body("ok"))

        with respx.mock:
            respx.post(_ANT_URL).mock(side_effect=capture)
            provider = _ant_provider()
            await provider.complete(
                [CloudMessage(role="user", content="hello")], model=_ANT_MODEL
            )

        assert "system" not in captured[0]

    async def test_multiple_content_blocks_first_text_used(self) -> None:
        body = _ant_body("first text")
        body["content"] = [
            {"type": "text", "text": "first text"},
            {"type": "text", "text": "second text"},
        ]
        with respx.mock:
            respx.post(_ANT_URL).mock(return_value=httpx.Response(200, json=body))
            provider = _ant_provider()
            result = await provider.complete(
                [CloudMessage(role="user", content="hi")], model=_ANT_MODEL
            )
        assert result.content == "first text"


# ---------------------------------------------------------------------------
# §6 — Anthropic error mapping
# ---------------------------------------------------------------------------


class TestAnthropicErrorMapping:
    async def test_401_raises_invalid_api_key(self) -> None:
        with respx.mock:
            respx.post(_ANT_URL).mock(
                return_value=httpx.Response(
                    401,
                    json=_ant_error_body("Invalid key", "authentication_error"),
                )
            )
            provider = _ant_provider(max_retries=0)
            with pytest.raises(InvalidAPIKeyError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_ANT_MODEL
                )

    async def test_429_raises_quota_exceeded(self) -> None:
        with respx.mock:
            respx.post(_ANT_URL).mock(
                return_value=httpx.Response(
                    429,
                    json=_ant_error_body("Rate limit", "rate_limit_error"),
                )
            )
            provider = _ant_provider(max_retries=0)
            with pytest.raises(CloudQuotaExceededError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_ANT_MODEL
                )

    async def test_read_timeout_raises_cloud_timeout(self) -> None:
        with respx.mock:
            respx.post(_ANT_URL).mock(
                side_effect=httpx.ReadTimeout("timed out", request=None)
            )
            provider = _ant_provider(max_retries=0)
            with pytest.raises(CloudTimeoutError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_ANT_MODEL
                )

    async def test_connect_error_raises_cloud_unavailable(self) -> None:
        with respx.mock:
            respx.post(_ANT_URL).mock(
                side_effect=httpx.ConnectError("connection refused")
            )
            provider = _ant_provider(max_retries=0)
            with pytest.raises(CloudUnavailableError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_ANT_MODEL
                )


# ---------------------------------------------------------------------------
# §7 — Retry logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    async def test_openai_succeeds_after_two_failures(self) -> None:
        """Two connection failures then success: 2 sleep calls with backoff."""
        call_n = 0

        def _side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal call_n
            call_n += 1
            if call_n <= 2:
                raise httpx.ConnectError("fail")
            return httpx.Response(200, json=_oai_body("ok"))

        sleep_calls: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with respx.mock:
            respx.post(_OAI_URL).mock(side_effect=_side_effect)
            provider = OpenAIProvider(
                api_key=_OAI_KEY,
                max_retries=3,
                _sleep_fn=_fake_sleep,
            )
            result = await provider.complete(
                [CloudMessage(role="user", content="hi")], model=_OAI_MODEL
            )

        assert result.content == "ok"
        assert sleep_calls == [1.0, 2.0]

    async def test_openai_retry_exhausted_raises(self) -> None:
        """All 4 attempts (1 + 3 retries) fail → CloudUnavailableError raised."""
        sleep_calls: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with respx.mock:
            respx.post(_OAI_URL).mock(
                side_effect=httpx.ConnectError("always fail")
            )
            provider = OpenAIProvider(
                api_key=_OAI_KEY,
                max_retries=3,
                _sleep_fn=_fake_sleep,
            )
            with pytest.raises(CloudUnavailableError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_OAI_MODEL
                )

        # 3 sleeps: after attempt 0 (1s), 1 (2s), 2 (4s); no sleep after last
        assert sleep_calls == [1.0, 2.0, 4.0]

    async def test_anthropic_succeeds_after_timeout_retry(self) -> None:
        """Timeout on first call, success on retry."""
        call_n = 0

        def _side_effect(request: httpx.Request) -> httpx.Response:
            nonlocal call_n
            call_n += 1
            if call_n == 1:
                raise httpx.ReadTimeout("timeout", request=None)
            return httpx.Response(200, json=_ant_body("retry ok"))

        sleep_calls: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with respx.mock:
            respx.post(_ANT_URL).mock(side_effect=_side_effect)
            provider = AnthropicProvider(
                api_key=_ANT_KEY,
                max_retries=3,
                _sleep_fn=_fake_sleep,
            )
            result = await provider.complete(
                [CloudMessage(role="user", content="hi")], model=_ANT_MODEL
            )

        assert result.content == "retry ok"
        assert sleep_calls == [1.0]

    async def test_anthropic_retry_exhausted_raises(self) -> None:
        """All 4 attempts fail with timeout → CloudTimeoutError raised."""
        sleep_calls: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with respx.mock:
            respx.post(_ANT_URL).mock(
                side_effect=httpx.ReadTimeout("timeout", request=None)
            )
            provider = AnthropicProvider(
                api_key=_ANT_KEY,
                max_retries=3,
                _sleep_fn=_fake_sleep,
            )
            with pytest.raises(CloudTimeoutError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_ANT_MODEL
                )

        assert sleep_calls == [1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# §8 — QuotaExceeded no retry
# ---------------------------------------------------------------------------


class TestQuotaExceededNoRetry:
    async def test_openai_quota_exceeded_not_retried(self) -> None:
        """429 on first attempt → immediately raises, sleep never called."""
        sleep_calls: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with respx.mock:
            respx.post(_OAI_URL).mock(
                return_value=httpx.Response(
                    429, json=_oai_error_body("quota", "rate_limit_exceeded")
                )
            )
            provider = OpenAIProvider(
                api_key=_OAI_KEY, max_retries=3, _sleep_fn=_fake_sleep
            )
            with pytest.raises(CloudQuotaExceededError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_OAI_MODEL
                )

        assert sleep_calls == []

    async def test_anthropic_quota_exceeded_not_retried(self) -> None:
        sleep_calls: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with respx.mock:
            respx.post(_ANT_URL).mock(
                return_value=httpx.Response(
                    429, json=_ant_error_body("quota", "rate_limit_error")
                )
            )
            provider = AnthropicProvider(
                api_key=_ANT_KEY, max_retries=3, _sleep_fn=_fake_sleep
            )
            with pytest.raises(CloudQuotaExceededError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_ANT_MODEL
                )

        assert sleep_calls == []


# ---------------------------------------------------------------------------
# §9 — InvalidAPIKey no retry
# ---------------------------------------------------------------------------


class TestInvalidAPIKeyNoRetry:
    async def test_openai_401_not_retried(self) -> None:
        sleep_calls: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with respx.mock:
            respx.post(_OAI_URL).mock(
                return_value=httpx.Response(
                    401, json=_oai_error_body("invalid", "invalid_api_key")
                )
            )
            provider = OpenAIProvider(
                api_key=_OAI_KEY, max_retries=3, _sleep_fn=_fake_sleep
            )
            with pytest.raises(InvalidAPIKeyError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_OAI_MODEL
                )

        assert sleep_calls == []

    async def test_anthropic_401_not_retried(self) -> None:
        sleep_calls: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with respx.mock:
            respx.post(_ANT_URL).mock(
                return_value=httpx.Response(
                    401, json=_ant_error_body("invalid", "authentication_error")
                )
            )
            provider = AnthropicProvider(
                api_key=_ANT_KEY, max_retries=3, _sleep_fn=_fake_sleep
            )
            with pytest.raises(InvalidAPIKeyError):
                await provider.complete(
                    [CloudMessage(role="user", content="hi")], model=_ANT_MODEL
                )

        assert sleep_calls == []


# ---------------------------------------------------------------------------
# §10 — FallbackProvider
# ---------------------------------------------------------------------------


class TestFallbackProvider:
    async def test_primary_success_no_fallback(self) -> None:
        primary = AsyncMock()
        secondary = AsyncMock()
        primary.complete.return_value = CloudResponse(
            content="primary ok",
            tokens_input=1,
            tokens_output=1,
            model_used="m",
            provider="openai",
            latency_ms=5.0,
            finish_reason="stop",
        )
        fallback = FallbackProvider(primary=primary, secondary=secondary)  # type: ignore[arg-type]

        result = await fallback.complete(
            [CloudMessage(role="user", content="hi")], model="gpt-4o"
        )

        assert result.content == "primary ok"
        secondary.complete.assert_not_called()

    async def test_primary_fail_secondary_used(self) -> None:
        primary = AsyncMock()
        secondary = AsyncMock()
        primary.complete.side_effect = CloudUnavailableError("primary down")
        secondary.complete.return_value = CloudResponse(
            content="secondary ok",
            tokens_input=2,
            tokens_output=3,
            model_used="m2",
            provider="anthropic",
            latency_ms=10.0,
            finish_reason="end_turn",
        )
        fallback = FallbackProvider(primary=primary, secondary=secondary)  # type: ignore[arg-type]

        result = await fallback.complete(
            [CloudMessage(role="user", content="hi")], model="claude-3-5-sonnet-20241022"
        )

        assert result.content == "secondary ok"
        assert result.provider == "anthropic"
        primary.complete.assert_called_once()
        secondary.complete.assert_called_once()

    async def test_primary_quota_error_fallback_to_secondary(self) -> None:
        primary = AsyncMock()
        secondary = AsyncMock()
        primary.complete.side_effect = CloudQuotaExceededError("quota")
        secondary.complete.return_value = CloudResponse(
            content="fallback content",
            tokens_input=1,
            tokens_output=1,
            model_used="m",
            provider="anthropic",
            latency_ms=5.0,
            finish_reason="end_turn",
        )
        fallback = FallbackProvider(primary=primary, secondary=secondary)  # type: ignore[arg-type]

        result = await fallback.complete(
            [CloudMessage(role="user", content="hi")], model="gpt-4o"
        )

        assert result.content == "fallback content"

    async def test_both_fail_propagates_secondary_error(self) -> None:
        primary = AsyncMock()
        secondary = AsyncMock()
        primary.complete.side_effect = CloudUnavailableError("primary down")
        secondary.complete.side_effect = CloudTimeoutError("secondary timeout")
        fallback = FallbackProvider(primary=primary, secondary=secondary)  # type: ignore[arg-type]

        with pytest.raises(CloudTimeoutError):
            await fallback.complete(
                [CloudMessage(role="user", content="hi")], model="gpt-4o"
            )

    async def test_fallback_passes_same_arguments(self) -> None:
        primary = AsyncMock()
        secondary = AsyncMock()
        primary.complete.side_effect = CloudUnavailableError("down")
        secondary.complete.return_value = CloudResponse(
            content="ok",
            tokens_input=1,
            tokens_output=1,
            model_used="m",
            provider="anthropic",
            latency_ms=1.0,
            finish_reason="end_turn",
        )
        fallback = FallbackProvider(primary=primary, secondary=secondary)  # type: ignore[arg-type]
        msgs = [CloudMessage(role="user", content="test")]

        await fallback.complete(msgs, model="gpt-4o", max_tokens=512, timeout=15.0)

        secondary.complete.assert_called_once_with(msgs, "gpt-4o", 512, 15.0)


# ---------------------------------------------------------------------------
# §11 — Image support
# ---------------------------------------------------------------------------


class TestImageSupport:
    async def test_openai_image_in_request(self) -> None:
        """Image bytes are base64-encoded and sent in content array."""
        img_bytes = b"\x89PNG\r\nfake_png_data"
        captured: list[Any] = []

        def _capture(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_oai_body("ok"))

        with respx.mock:
            respx.post(_OAI_URL).mock(side_effect=_capture)
            provider = _oai_provider()
            await provider.complete(
                [CloudMessage(role="user", content="describe", image=img_bytes)],
                model=_OAI_MODEL,
            )

        msg = captured[0]["messages"][0]
        assert isinstance(msg["content"], list)
        text_part = next(p for p in msg["content"] if p["type"] == "text")
        img_part = next(p for p in msg["content"] if p["type"] == "image_url")
        assert text_part["text"] == "describe"
        url = img_part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == img_bytes

    async def test_anthropic_image_in_request(self) -> None:
        """Image bytes are base64-encoded and sent as Anthropic image block."""
        img_bytes = b"\x89PNG\r\nfake_png_data"
        captured: list[Any] = []

        def _capture(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_ant_body("ok"))

        with respx.mock:
            respx.post(_ANT_URL).mock(side_effect=_capture)
            provider = _ant_provider()
            await provider.complete(
                [CloudMessage(role="user", content="describe", image=img_bytes)],
                model=_ANT_MODEL,
            )

        msg = captured[0]["messages"][0]
        assert isinstance(msg["content"], list)
        text_part = next(p for p in msg["content"] if p["type"] == "text")
        img_part = next(p for p in msg["content"] if p["type"] == "image")
        assert text_part["text"] == "describe"
        assert img_part["source"]["type"] == "base64"
        assert img_part["source"]["media_type"] == "image/png"
        assert base64.b64decode(img_part["source"]["data"]) == img_bytes

    async def test_openai_no_image_sends_plain_string(self) -> None:
        captured: list[Any] = []

        def _capture(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=_oai_body("ok"))

        with respx.mock:
            respx.post(_OAI_URL).mock(side_effect=_capture)
            provider = _oai_provider()
            await provider.complete(
                [CloudMessage(role="user", content="hello")], model=_OAI_MODEL
            )

        msg = captured[0]["messages"][0]
        assert isinstance(msg["content"], str)
        assert msg["content"] == "hello"


# ---------------------------------------------------------------------------
# §12 — CloudProvider Protocol
# ---------------------------------------------------------------------------


class TestCloudProviderProtocol:
    def test_openai_provider_is_cloud_provider(self) -> None:
        assert isinstance(OpenAIProvider(api_key=_OAI_KEY), CloudProvider)

    def test_anthropic_provider_is_cloud_provider(self) -> None:
        assert isinstance(AnthropicProvider(api_key=_ANT_KEY), CloudProvider)

    def test_fallback_provider_is_cloud_provider(self) -> None:
        primary = OpenAIProvider(api_key=_OAI_KEY)
        secondary = AnthropicProvider(api_key=_ANT_KEY)
        fallback = FallbackProvider(primary=primary, secondary=secondary)  # type: ignore[arg-type]
        assert isinstance(fallback, CloudProvider)
