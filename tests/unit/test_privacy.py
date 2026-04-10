"""
tests/unit/test_privacy.py
Unit tests for nexus/ui/privacy.py and transport-aware masking — Faz 59.

TEST PLAN
---------
PrivacyTransparencyScreen.show_cloud_call_info:
  1.  sensitive_count in output.
  2.  provider name in output.
  3.  estimated cost in output.
  4.  native transport → "veri gönderilmedi" in output.
  5.  native transport → "Ekran görüntüsü buluta gönderilmedi." in output.
  6.  visual transport → "ekran görüntüsü gönderildi" in output.
  7.  visual transport with path → path shown in output.
  8.  uia transport treated as native.
  9.  dom transport treated as native.
  10. unknown transport shows raw name.

PrivacyTransparencyScreen.show_masking_preview:
  11. is_first_call=False → True immediately, no prompt issued.
  12. is_first_call=True, user consents ("e") → True.
  13. is_first_call=True, user declines ("h") → False.
  14. is_first_call=True, any "yes"/"y" variant → True.

ScreenshotMasker.mask transport-aware logging:
  15. visual transport → log event "screenshot_masked", screenshot_sent=True.
  16. native ("uia") transport → log event "screenshot_not_sent", screenshot_sent=False.
  17. native ("dom") transport → log event "screenshot_not_sent".
  18. native ("file") transport → log event "screenshot_not_sent".
  19. native ("native") transport → log event "screenshot_not_sent".
  20. transport default ("visual") → screenshot_sent=True (backward compat).

_is_visual_transport (unit):
  21. "visual" → True.
  22. "mouse" → True.
  23. "keyboard" → True.
  24. "native" → False.
  25. "uia" → False.
  26. "dom" → False.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nexus.core.screenshot_masker import ScreenshotMasker
from nexus.ui.privacy import PrivacyTransparencyScreen, _is_visual_transport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _screen(
    *,
    prompts: list[str] | None = None,
    printed: list[str] | None = None,
) -> tuple[PrivacyTransparencyScreen, list[str]]:
    output: list[str] = printed if printed is not None else []
    _prompts = list(prompts or [])

    ps = PrivacyTransparencyScreen(
        _print_fn=lambda t: output.append(t),
        _prompt_fn=lambda _: _prompts.pop(0) if _prompts else "",
    )
    return ps, output


def _lines(output: list[str]) -> str:
    return "\n".join(output)


def _small_image() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# show_cloud_call_info
# ---------------------------------------------------------------------------


class TestShowCloudCallInfo:
    def test_sensitive_count_in_output(self):
        ps, out = _screen()
        ps.show_cloud_call_info("img.png", 3, "anthropic", "visual")
        assert any("3 hassas bölge" in line for line in out)

    def test_provider_in_output(self):
        ps, out = _screen()
        ps.show_cloud_call_info("img.png", 0, "openai", "visual")
        assert any("openai" in line for line in out)

    def test_cost_in_output(self):
        ps, out = _screen()
        ps.show_cloud_call_info("img.png", 0, "anthropic", "visual", 0.0032)
        assert any("0.0032" in line for line in out)

    def test_native_transport_no_data_sent_message(self):
        ps, out = _screen()
        ps.show_cloud_call_info("img.png", 2, "anthropic", "native")
        text = _lines(out)
        assert "veri gönderilmedi" in text

    def test_native_transport_no_screenshot_sent_line(self):
        ps, out = _screen()
        ps.show_cloud_call_info("img.png", 0, "anthropic", "native")
        text = _lines(out)
        assert "Ekran görüntüsü buluta gönderilmedi" in text

    def test_visual_transport_screenshot_sent_message(self):
        ps, out = _screen()
        ps.show_cloud_call_info("img.png", 1, "openai", "visual")
        text = _lines(out)
        assert "ekran görüntüsü gönderildi" in text

    def test_visual_transport_shows_path(self):
        ps, out = _screen()
        ps.show_cloud_call_info("/tmp/masked.png", 0, "anthropic", "visual")
        assert any("/tmp/masked.png" in line for line in out)

    def test_uia_treated_as_native(self):
        ps, out = _screen()
        ps.show_cloud_call_info("", 0, "anthropic", "uia")
        text = _lines(out)
        assert "veri gönderilmedi" in text

    def test_dom_treated_as_native(self):
        ps, out = _screen()
        ps.show_cloud_call_info("", 0, "anthropic", "dom")
        text = _lines(out)
        assert "veri gönderilmedi" in text

    def test_unknown_transport_shows_raw(self):
        ps, out = _screen()
        ps.show_cloud_call_info("", 0, "anthropic", "ftp")
        text = _lines(out)
        assert "ftp" in text


# ---------------------------------------------------------------------------
# show_masking_preview
# ---------------------------------------------------------------------------


class TestShowMaskingPreview:
    def test_not_first_call_returns_true_no_prompt(self):
        prompts_issued: list[str] = []
        ps = PrivacyTransparencyScreen(
            _print_fn=lambda _: None,
            _prompt_fn=lambda label: prompts_issued.append(label) or "",  # type: ignore[func-returns-value]
        )
        result = ps.show_masking_preview(object(), is_first_call=False)

        assert result is True
        assert prompts_issued == [], "No prompt should be issued when not first call"

    def test_first_call_consent_granted(self):
        ps, _ = _screen(prompts=["e"])
        result = ps.show_masking_preview(object(), is_first_call=True)
        assert result is True

    def test_first_call_consent_denied(self):
        ps, _ = _screen(prompts=["h"])
        result = ps.show_masking_preview(object(), is_first_call=True)
        assert result is False

    @pytest.mark.parametrize("answer", ["y", "yes", "e", "evet"])
    def test_first_call_consent_variants(self, answer: str):
        ps, _ = _screen(prompts=[answer])
        result = ps.show_masking_preview(object(), is_first_call=True)
        assert result is True

    def test_first_call_shows_consent_prompt(self):
        printed: list[str] = []
        ps, out = _screen(prompts=["e"], printed=printed)
        ps.show_masking_preview(object(), is_first_call=True)
        text = _lines(out)
        assert "ONAY" in text or "onay" in text.lower()


# ---------------------------------------------------------------------------
# ScreenshotMasker transport-aware logging
# ---------------------------------------------------------------------------


class TestMaskerTransportLogging:
    """Verify that ScreenshotMasker.mask() emits the correct log event."""

    def _run_mask(self, transport: str) -> list[dict]:
        """Run mask() and capture structlog events."""
        logged: list[dict] = []

        import nexus.core.screenshot_masker as _mod  # noqa: PLC0415

        original = _mod._log

        class _CapLog:
            def info(self, event: str, **kw: Any) -> None:
                logged.append({"event": event, **kw})

            def warning(self, *a: Any, **kw: Any) -> None:
                pass

        _mod._log = _CapLog()  # type: ignore[assignment]
        try:
            masker = ScreenshotMasker()
            img = _small_image()
            masker.mask(img, [], transport=transport)
        finally:
            _mod._log = original

        return logged

    def test_visual_transport_logs_screenshot_masked(self):
        logs = self._run_mask("visual")
        assert any(e["event"] == "screenshot_masked" for e in logs)
        assert any(e.get("screenshot_sent") is True for e in logs)

    def test_uia_transport_logs_not_sent(self):
        logs = self._run_mask("uia")
        assert any(e["event"] == "screenshot_not_sent" for e in logs)
        assert any(e.get("screenshot_sent") is False for e in logs)

    def test_dom_transport_logs_not_sent(self):
        logs = self._run_mask("dom")
        assert any(e["event"] == "screenshot_not_sent" for e in logs)

    def test_file_transport_logs_not_sent(self):
        logs = self._run_mask("file")
        assert any(e["event"] == "screenshot_not_sent" for e in logs)

    def test_native_transport_logs_not_sent(self):
        logs = self._run_mask("native")
        assert any(e["event"] == "screenshot_not_sent" for e in logs)

    def test_default_transport_is_visual(self):
        """mask() with no transport kwarg behaves as visual (backward compat)."""
        logged: list[dict] = []

        import nexus.core.screenshot_masker as _mod  # noqa: PLC0415

        original = _mod._log

        class _CapLog:
            def info(self, event: str, **kw: Any) -> None:
                logged.append({"event": event, **kw})

            def warning(self, *a: Any, **kw: Any) -> None:
                pass

        _mod._log = _CapLog()  # type: ignore[assignment]
        try:
            masker = ScreenshotMasker()
            masker.mask(_small_image(), [])
        finally:
            _mod._log = original

        assert any(e.get("screenshot_sent") is True for e in logged)


# ---------------------------------------------------------------------------
# _is_visual_transport (unit)
# ---------------------------------------------------------------------------


class TestIsVisualTransport:
    @pytest.mark.parametrize("transport", ["visual", "mouse", "keyboard"])
    def test_visual_transports(self, transport: str):
        assert _is_visual_transport(transport) is True

    @pytest.mark.parametrize("transport", ["native", "uia", "dom", "file"])
    def test_native_transports(self, transport: str):
        assert _is_visual_transport(transport) is False
