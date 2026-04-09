"""
tests/adversarial/adv_003_cloud_timeout.py
Adversarial Test 003 — Cloud timeout → retry → eventual CloudTimeoutError

Scenario:
  The cloud provider's _attempt() always raises CloudTimeoutError
  (simulating a 35-second network hang).

Success criteria:
  - All (max_retries + 1) attempts are made.
  - Final exception is CloudTimeoutError (not a crash / unexpected type).
  - Sleep backoff sequence is [1, 2, 4] (2^0, 2^1, 2^2) for 3 retries.

All I/O is injected — no real HTTP calls.
"""
from __future__ import annotations

import asyncio

import pytest

from nexus.cloud.providers import AnthropicProvider, CloudMessage
from nexus.core.errors import CloudTimeoutError


@pytest.mark.adversarial
class TestCloudTimeout:
    """ADV-003: Sustained cloud timeout exhausts retries and raises CloudTimeoutError."""

    def test_all_retries_exhausted_raises_cloud_timeout(self):
        """
        Every attempt raises CloudTimeoutError →
        provider retries max_retries times then re-raises.
        """
        attempt_count = [0]
        sleeps: list[float] = []

        async def _always_timeout(*_args, **_kwargs):
            attempt_count[0] += 1
            raise CloudTimeoutError("simulated 35 s timeout")

        async def _fake_sleep(s: float) -> None:
            sleeps.append(s)

        provider = AnthropicProvider(
            api_key="sk-fake",
            max_retries=3,
            _sleep_fn=_fake_sleep,
        )
        # Patch _attempt directly — skip real HTTP
        provider._attempt = _always_timeout  # type: ignore[method-assign]

        msgs = [CloudMessage(role="user", content="ping")]

        with pytest.raises(CloudTimeoutError):
            asyncio.run(provider.complete(msgs, "claude-3-5-sonnet-20241022"))

        assert attempt_count[0] == 4, (
            f"Expected 4 attempts (1 + 3 retries); got {attempt_count[0]}"
        )
        assert sleeps == [1.0, 2.0, 4.0], (
            f"Expected backoff [1, 2, 4]; got {sleeps}"
        )

    def test_first_timeout_then_success_returns_response(self):
        """
        First attempt times out, second succeeds →
        CloudTimeoutError is NOT raised; valid response returned.
        """
        from nexus.cloud.providers import CloudResponse

        calls = [0]

        async def _flaky(*_args, **_kwargs) -> CloudResponse:
            calls[0] += 1
            if calls[0] == 1:
                raise CloudTimeoutError("first attempt timeout")
            return CloudResponse(
                content="pong",
                tokens_input=10,
                tokens_output=5,
                model_used="claude-3-5-sonnet-20241022",
                provider="anthropic",
                latency_ms=120.0,
                finish_reason="end_turn",
            )

        sleeps: list[float] = []

        async def _fake_sleep(s: float) -> None:
            sleeps.append(s)

        provider = AnthropicProvider(
            api_key="sk-fake",
            max_retries=3,
            _sleep_fn=_fake_sleep,
        )
        provider._attempt = _flaky  # type: ignore[method-assign]

        msgs = [CloudMessage(role="user", content="ping")]
        response = asyncio.run(provider.complete(msgs, "claude-3-5-sonnet-20241022"))

        assert response.content == "pong"
        assert calls[0] == 2
        assert sleeps == [1.0]  # one backoff sleep after first failure
