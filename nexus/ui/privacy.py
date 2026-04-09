"""
nexus/ui/privacy.py
Privacy transparency screen for Nexus Agent.

PrivacyTransparencyScreen
--------------------------
Renders transport-aware privacy notices before and after cloud calls.

show_cloud_call_info(masked_screenshot_path, sensitive_count,
                     provider, transport, estimated_cost_usd)
  Displays a one-panel privacy summary:
    • How many regions were masked.
    • Which provider will receive data.
    • Estimated cost of the call.
    • Transport line:
        native  → "Transport: native (veri gönderilmedi)"
        visual  → "Transport: visual (ekran görüntüsü gönderildi)"

show_masking_preview(frame, *, is_first_call)
  When is_first_call=True, prompts the user for explicit consent before
  the first cloud call that involves a screenshot.
  Returns True when the user consents (or is_first_call=False).

Injectable callables
--------------------
_print_fn  : (text: str) -> None
_prompt_fn : (label: str) -> str
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# Transport labels used in the privacy panel
_TRANSPORT_LABELS: dict[str, str] = {
    "native": "native (veri gönderilmedi)",
    "uia":    "native (veri gönderilmedi)",
    "dom":    "native (veri gönderilmedi)",
    "file":   "native (veri gönderilmedi)",
    "visual": "visual (ekran görüntüsü gönderildi)",
    "mouse":  "visual (ekran görüntüsü gönderildi)",
    "keyboard": "visual (ekran görüntüsü gönderildi)",
}

_BORDER = "─" * 46


class PrivacyTransparencyScreen:
    """
    Renders transport-aware privacy notices.

    Parameters
    ----------
    _print_fn:
        Output callable.  Default: ``print``.
    _prompt_fn:
        Input callable.  Default: ``input``.
    """

    def __init__(
        self,
        *,
        _print_fn: Callable[[str], None] | None = None,
        _prompt_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._print = _print_fn or print
        self._prompt = _prompt_fn or input

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_cloud_call_info(
        self,
        masked_screenshot_path: str,
        sensitive_count: int,
        provider: str,
        transport: str = "visual",
        estimated_cost_usd: float = 0.001,
    ) -> None:
        """
        Display a privacy summary panel for an imminent cloud call.

        Parameters
        ----------
        masked_screenshot_path:
            Path to the masked screenshot that will be (or was) sent.
            Shown as a reference; empty string hides the path line.
        sensitive_count:
            Number of sensitive regions that were blacked out.
        provider:
            Cloud provider name (e.g. "anthropic", "openai").
        transport:
            Transport method used: "native", "uia", "dom", "file",
            "visual", "mouse", or "keyboard".
        estimated_cost_usd:
            Estimated cost of this cloud call in USD.
        """
        transport_label = _TRANSPORT_LABELS.get(
            transport.lower(), f"{transport} (bilinmiyor)"
        )
        is_visual = _is_visual_transport(transport)

        p = self._print
        p(f"\n  {_BORDER}")
        p("  GİZLİLİK BİLGİSİ")
        p(f"  {_BORDER}")
        p(f"  {sensitive_count} hassas bölge maskelendi")
        p(f"  Provider    : {provider}")
        p(f"  Tahmini maliyet: ${estimated_cost_usd:.4f}")
        p(f"  Transport   : {transport_label}")
        if masked_screenshot_path and is_visual:
            p(f"  Maskeli görüntü: {masked_screenshot_path}")
        elif not is_visual:
            p("  Ekran görüntüsü buluta gönderilmedi.")
        p(f"  {_BORDER}")

        _log.info(
            "privacy_panel_shown",
            sensitive_count=sensitive_count,
            provider=provider,
            transport=transport,
            estimated_cost_usd=estimated_cost_usd,
            screenshot_sent=is_visual,
        )

    def show_masking_preview(
        self,
        frame: Any,
        *,
        is_first_call: bool = False,
    ) -> bool:
        """
        Optionally prompt the user for consent before the first cloud call
        that includes a screenshot.

        Parameters
        ----------
        frame:
            The current screen frame (not rendered to terminal; kept for
            future GUI use).
        is_first_call:
            When True, display a consent prompt and return False if the
            user declines.  When False, return True immediately.

        Returns
        -------
        True when the call may proceed, False when the user blocks it.
        """
        if not is_first_call:
            return True

        p = self._print
        p(f"\n  {_BORDER}")
        p("  İLK BULUT ÇAĞRISI — ONAY GEREKLİ")
        p(f"  {_BORDER}")
        p("  Nexus Agent, görevi tamamlamak için bir ekran görüntüsü")
        p("  yapay zeka sağlayıcısına gönderecek.")
        p("  Hassas bölgeler önceden maskelenmektedir.")
        p(f"  {_BORDER}")

        answer = self._prompt("  Devam etmek istiyor musunuz? [e/h]: ")
        allowed = answer.strip().lower() in ("e", "evet", "y", "yes")

        if allowed:
            _log.info("first_call_consent_granted")
        else:
            _log.info("first_call_consent_denied")

        return allowed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_visual_transport(transport: str) -> bool:
    """Return True when *transport* involves sending a screenshot to the cloud."""
    return transport.lower() in {"visual", "mouse", "keyboard"}
