"""
tests/unit/test_source_resolver.py
Unit tests for nexus/source/resolver.py.

All tests use injected mock probes — no real adapters, file I/O, or
browser connections are required.

Coverage targets
----------------
1. Priority order: UIA → DOM → File → Visual
2. Short-circuit: first success stops remaining probes
3. prefer_source: preferred source is tried before the rest
4. Visual fallback: all-None → SourceResult with source_type="visual"
5. Logging: each attempt emits one INFO log record
6. resolve_for_read / resolve_for_action: context keys are forwarded
7. probe exception: treated as None (failure), not propagated
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from nexus.core.settings import NexusSettings
from nexus.source.resolver import (
    SourcePriorityResolver,
    SourceResult,
    _build_order,
    _DEFAULT_ORDER,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings() -> NexusSettings:
    return NexusSettings.model_validate({})


def _success_probe(data: Any = "ok") -> Any:
    """Return a probe callable that always succeeds with *data*."""
    return lambda ctx: data


def _failing_probe() -> Any:
    """Return a probe callable that always returns None."""
    return lambda ctx: None


def _counting_probe(store: list[Any], data: Any = "ok") -> Any:
    """Return a probe that appends the context to *store* and returns *data*."""

    def _probe(ctx: dict[str, Any]) -> Any:
        store.append(ctx)
        return data

    return _probe


def _raising_probe(exc: Exception | None = None) -> Any:
    """Return a probe that always raises."""

    def _probe(ctx: dict[str, Any]) -> Any:
        raise exc or RuntimeError("probe error")

    return _probe


# ---------------------------------------------------------------------------
# _build_order unit tests
# ---------------------------------------------------------------------------


class TestBuildOrder:
    def test_no_prefer_returns_default(self):
        assert _build_order(None, _DEFAULT_ORDER) == list(_DEFAULT_ORDER)

    def test_prefer_uia_is_default(self):
        assert _build_order("uia", _DEFAULT_ORDER) == ["uia", "dom", "file", "visual"]

    def test_prefer_dom_moves_to_front(self):
        order = _build_order("dom", _DEFAULT_ORDER)
        assert order == ["dom", "uia", "file", "visual"]

    def test_prefer_file_moves_to_front(self):
        order = _build_order("file", _DEFAULT_ORDER)
        assert order == ["file", "uia", "dom", "visual"]

    def test_prefer_visual_moves_to_front(self):
        order = _build_order("visual", _DEFAULT_ORDER)
        assert order == ["visual", "uia", "dom", "file"]

    def test_preferred_source_not_duplicated(self):
        order = _build_order("dom", _DEFAULT_ORDER)
        assert order.count("dom") == 1


# ---------------------------------------------------------------------------
# SourceResult data model
# ---------------------------------------------------------------------------


class TestSourceResult:
    def test_can_construct(self):
        r = SourceResult(
            source_type="uia",
            data={"elements": []},
            confidence=1.0,
            latency_ms=2.5,
        )
        assert r.source_type == "uia"
        assert r.confidence == 1.0

    def test_all_source_types_accepted(self):
        for st in ("uia", "dom", "file", "visual"):
            r = SourceResult(
                source_type=st,  # type: ignore[arg-type]
                data=None,
                confidence=0.5,
                latency_ms=0.0,
            )
            assert r.source_type == st


# ---------------------------------------------------------------------------
# resolve() — priority order
# ---------------------------------------------------------------------------


class TestResolvePriorityOrder:
    def test_uia_first_returns_uia_result(self):
        """UIA probe succeeds → source_type is 'uia'."""
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_success_probe("uia_data"),
        )
        result = adapter.resolve({})
        assert result.source_type == "uia"
        assert result.data == "uia_data"

    def test_uia_success_skips_dom(self):
        """UIA success → DOM probe must not be called."""
        dom_calls: list[Any] = []
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_success_probe(),
            _dom_probe=_counting_probe(dom_calls),
        )
        adapter.resolve({})
        assert dom_calls == []

    def test_uia_success_skips_file_and_visual(self):
        """UIA success → File and Visual probes not called."""
        file_calls: list[Any] = []
        visual_calls: list[Any] = []
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_success_probe(),
            _file_probe=_counting_probe(file_calls),
            _visual_probe=_counting_probe(visual_calls),
        )
        adapter.resolve({})
        assert file_calls == []
        assert visual_calls == []

    def test_uia_fails_dom_tried_next(self):
        """UIA None → DOM is next candidate."""
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_failing_probe(),
            _dom_probe=_success_probe("dom_data"),
        )
        result = adapter.resolve({})
        assert result.source_type == "dom"
        assert result.data == "dom_data"

    def test_uia_dom_fail_file_tried(self):
        """UIA + DOM None → File is next."""
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_failing_probe(),
            _dom_probe=_failing_probe(),
            _file_probe=_success_probe("file_data"),
        )
        result = adapter.resolve({})
        assert result.source_type == "file"

    def test_all_none_returns_visual(self):
        """All probes return None → visual fallback."""
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_failing_probe(),
            _dom_probe=_failing_probe(),
            _file_probe=_failing_probe(),
            _visual_probe=_success_probe("visual_data"),
        )
        result = adapter.resolve({})
        assert result.source_type == "visual"
        assert result.data == "visual_data"

    def test_default_visual_probe_always_succeeds(self):
        """Default visual probe returns non-None sentinel."""
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_failing_probe(),
            _dom_probe=_failing_probe(),
            _file_probe=_failing_probe(),
            # No _visual_probe → default fallback used
        )
        result = adapter.resolve({})
        assert result.source_type == "visual"
        assert result.data is not None


# ---------------------------------------------------------------------------
# resolve() — confidence values
# ---------------------------------------------------------------------------


class TestResolveConfidence:
    def test_uia_confidence_is_1(self):
        adapter = SourcePriorityResolver(
            _make_settings(), _uia_probe=_success_probe()
        )
        assert adapter.resolve({}).confidence == pytest.approx(1.0)

    def test_dom_confidence_is_0_95(self):
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_failing_probe(),
            _dom_probe=_success_probe(),
        )
        assert adapter.resolve({}).confidence == pytest.approx(0.95)

    def test_file_confidence_is_0_90(self):
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_failing_probe(),
            _dom_probe=_failing_probe(),
            _file_probe=_success_probe(),
        )
        assert adapter.resolve({}).confidence == pytest.approx(0.90)

    def test_visual_confidence_is_0_70(self):
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_failing_probe(),
            _dom_probe=_failing_probe(),
            _file_probe=_failing_probe(),
            _visual_probe=_success_probe(),
        )
        assert adapter.resolve({}).confidence == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# resolve() — prefer_source
# ---------------------------------------------------------------------------


class TestResolvePreferSource:
    def test_prefer_dom_tries_dom_first(self):
        """prefer_source='dom' → DOM probe called before UIA."""
        order: list[str] = []

        def uia(ctx: dict[str, Any]) -> Any:
            order.append("uia")
            return None

        def dom(ctx: dict[str, Any]) -> Any:
            order.append("dom")
            return "dom_data"

        adapter = SourcePriorityResolver(
            _make_settings(), _uia_probe=uia, _dom_probe=dom
        )
        result = adapter.resolve({}, prefer_source="dom")
        assert result.source_type == "dom"
        assert order[0] == "dom"

    def test_prefer_file_tries_file_first(self):
        order: list[str] = []

        def file_probe(ctx: dict[str, Any]) -> Any:
            order.append("file")
            return "file_data"

        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=lambda ctx: (order.append("uia") or None),
            _file_probe=file_probe,
        )
        result = adapter.resolve({}, prefer_source="file")
        assert result.source_type == "file"
        assert order[0] == "file"
        assert "uia" not in order  # UIA never called since file succeeded

    def test_prefer_source_falls_back_when_preferred_fails(self):
        """prefer_source fails → next in default order tried."""
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_success_probe("uia_data"),
            _dom_probe=_failing_probe(),
        )
        result = adapter.resolve({}, prefer_source="dom")
        # DOM fails, falls back to UIA
        assert result.source_type == "uia"

    def test_prefer_uia_is_same_as_default(self):
        uia_calls: list[Any] = []
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_counting_probe(uia_calls, "data"),
        )
        result = adapter.resolve({}, prefer_source="uia")
        assert result.source_type == "uia"
        assert len(uia_calls) == 1

    def test_prefer_none_uses_default_order(self):
        """No preference → UIA is tried first."""
        calls: list[str] = []

        def uia(ctx: dict[str, Any]) -> Any:
            calls.append("uia")
            return "data"

        adapter = SourcePriorityResolver(_make_settings(), _uia_probe=uia)
        adapter.resolve({}, prefer_source=None)
        assert calls == ["uia"]


# ---------------------------------------------------------------------------
# resolve() — context forwarding
# ---------------------------------------------------------------------------


class TestResolveContextForwarding:
    def test_task_context_forwarded_to_probe(self):
        received: list[dict[str, Any]] = []

        def probe(ctx: dict[str, Any]) -> Any:
            received.append(ctx)
            return "data"

        adapter = SourcePriorityResolver(_make_settings(), _uia_probe=probe)
        adapter.resolve({"key": "value"})
        assert received[0]["key"] == "value"

    def test_latency_ms_is_non_negative(self):
        adapter = SourcePriorityResolver(
            _make_settings(), _uia_probe=_success_probe()
        )
        result = adapter.resolve({})
        assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# resolve() — probe exception handling
# ---------------------------------------------------------------------------


class TestResolveExceptionHandling:
    def test_raising_probe_treated_as_failure(self):
        """A probe that raises is treated as None — next source tried."""
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_raising_probe(),
            _dom_probe=_success_probe("dom_data"),
        )
        result = adapter.resolve({})
        assert result.source_type == "dom"

    def test_raising_probe_does_not_propagate(self):
        """Exceptions from probes must never escape resolve()."""
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_raising_probe(ValueError("oops")),
        )
        # Should not raise — should fall through to visual
        result = adapter.resolve({})
        assert result is not None
        assert result.source_type == "visual"


# ---------------------------------------------------------------------------
# resolve() — logging
# ---------------------------------------------------------------------------


class TestResolveLogging:
    def test_each_attempt_emits_log_event(self):
        """Every source tried → one structured log event per attempt."""
        from structlog.testing import capture_logs

        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_failing_probe(),
            _dom_probe=_failing_probe(),
            _file_probe=_failing_probe(),
            _visual_probe=_success_probe(),
        )
        with capture_logs() as logs:
            adapter.resolve({})

        # 4 attempts: uia (fail), dom (fail), file (fail), visual (success)
        tried = [e for e in logs if e.get("event") == "source_tried"]
        assert len(tried) == 4

    def test_log_events_contain_source_key(self):
        """Each log event includes a 'source' key."""
        from structlog.testing import capture_logs

        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_success_probe(),
        )
        with capture_logs() as logs:
            adapter.resolve({})

        tried = [e for e in logs if e.get("event") == "source_tried"]
        assert all("source" in e for e in tried)
        assert tried[0]["source"] == "uia"

    def test_failed_attempts_logged_before_success(self):
        """UIA failure event logged before DOM success event."""
        from structlog.testing import capture_logs

        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_failing_probe(),
            _dom_probe=_success_probe(),
        )
        with capture_logs() as logs:
            adapter.resolve({})

        tried = [e for e in logs if e.get("event") == "source_tried"]
        # Exactly 2: uia fail + dom success
        assert len(tried) == 2
        assert tried[0]["source"] == "uia"
        assert tried[0]["success"] is False
        assert tried[1]["source"] == "dom"
        assert tried[1]["success"] is True


# ---------------------------------------------------------------------------
# resolve_for_read
# ---------------------------------------------------------------------------


class TestResolveForRead:
    def test_returns_source_result(self):
        adapter = SourcePriorityResolver(
            _make_settings(), _uia_probe=_success_probe()
        )
        result = adapter.resolve_for_read("button text")
        assert isinstance(result, SourceResult)

    def test_intent_is_read(self):
        received: list[dict[str, Any]] = []
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_counting_probe(received, "data"),
        )
        adapter.resolve_for_read("some label")
        assert received[0]["intent"] == "read"

    def test_description_is_in_context(self):
        received: list[dict[str, Any]] = []
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_counting_probe(received, "data"),
        )
        adapter.resolve_for_read("the description")
        assert received[0]["description"] == "the description"

    def test_extra_context_merged(self):
        received: list[dict[str, Any]] = []
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_counting_probe(received, "data"),
        )
        adapter.resolve_for_read("desc", context={"window_handle": 42})
        assert received[0]["window_handle"] == 42

    def test_no_extra_context_defaults_to_empty(self):
        received: list[dict[str, Any]] = []
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_counting_probe(received, "data"),
        )
        adapter.resolve_for_read("desc")
        assert "intent" in received[0]


# ---------------------------------------------------------------------------
# resolve_for_action
# ---------------------------------------------------------------------------


class TestResolveForAction:
    def test_returns_source_result(self):
        adapter = SourcePriorityResolver(
            _make_settings(), _uia_probe=_success_probe()
        )
        result = adapter.resolve_for_action("click", "Submit")
        assert isinstance(result, SourceResult)

    def test_intent_is_action(self):
        received: list[dict[str, Any]] = []
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_counting_probe(received, "data"),
        )
        adapter.resolve_for_action("click", "OK")
        assert received[0]["intent"] == "action"

    def test_action_type_in_context(self):
        received: list[dict[str, Any]] = []
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_counting_probe(received, "data"),
        )
        adapter.resolve_for_action("type", "input#name")
        assert received[0]["action_type"] == "type"

    def test_target_in_context(self):
        received: list[dict[str, Any]] = []
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_counting_probe(received, "data"),
        )
        adapter.resolve_for_action("focus", "input#search")
        assert received[0]["target"] == "input#search"

    def test_extra_context_merged(self):
        received: list[dict[str, Any]] = []
        adapter = SourcePriorityResolver(
            _make_settings(),
            _uia_probe=_counting_probe(received, "data"),
        )
        adapter.resolve_for_action("click", "btn", context={"hwnd": 999})
        assert received[0]["hwnd"] == 999
