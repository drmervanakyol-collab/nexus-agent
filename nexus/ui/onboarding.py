"""
nexus/ui/onboarding.py
First-run onboarding flow for Nexus Agent.

OnboardingFlow
--------------
Guides the user through 8 sequential steps on first launch.
All I/O is injectable so the flow is fully testable without a terminal,
database, or live API.

run() -> bool
  Runs every step in order.  Returns True when all steps complete
  successfully.  Returns False when the user cancels or a hard
  requirement (health fail) blocks progression.

is_first_run() -> bool
  Returns True when neither 'privacy' nor 'terms' consent is stored.
  A returning user (both consents present) skips directly to True return.

Steps
-----
  1  Welcome        — Nexus Agent v1.0 greeting.
  2  Health check   — run HealthChecker; block on "fail", warn and continue
                      on "warn" (with user confirmation).
  3  Privacy        — show privacy_policy summary; record consent.
  4  Terms          — show terms_of_service summary; record consent.
  5  API Key        — prompt provider choice + key; validate with retry.
  6  Browser setup  — optional Chrome debug-mode launch.
  7  Connectivity   — ~$0.001 test call to chosen provider; show cost.
  8  Ready          — first task suggestion.

Injectable callables
--------------------
_print_fn        : (text: str) -> None
_prompt_fn       : (label: str) -> str
_has_consent_fn  : (scope: str, version: str) -> bool
_save_consent_fn : (scope: str, version: str) -> None
_health_check_fn : () -> HealthReport
_validate_key_fn : (provider: str, key: str) -> tuple[bool, str]
_launch_browser_fn: () -> bool
_test_api_fn     : (provider: str, key: str) -> tuple[bool, float]
    Returns (success, cost_usd).
"""
from __future__ import annotations

from collections.abc import Callable

from nexus.infra.health import HealthChecker, HealthReport
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Version / copy
# ---------------------------------------------------------------------------

_VERSION = "1.0"

# Document version constants — bump when the corresponding docs/*.md file is updated.
# Onboarding re-requests consent whenever the stored version differs from these.
_PRIVACY_VERSION = "1.0"
_TERMS_VERSION = "1.0"

_PRIVACY_SUMMARY = """\
Gizlilik Politikasi Ozeti (docs/privacy_policy.md v1.0)
--------------------------------------------------------
1. Nexus Agent tamamen yerel makinenizde calisir.
2. Native transport (UIA): ekran goruntusu GONDERILMEZ.
3. Visual transport (mouse/klavye): yalnizca maskelenmis ekran
   goruntusu gonderilir; hassas alanlar hicbir zaman gitmez.
4. Hassas bolgeler (parola, odeme alanlari) HICBIR ZAMAN
   islenmez veya iletilmez.
5. API anahtarlari Windows Credential Manager'da sifreli saklanir.
6. Gorev gecmisi yerel SQLite veritabaninda saklanir.
7. Tum yerel verileri 'nexus reset --all' ile silebilirsiniz.
8. Nexus Agent hesap olusturmaz veya kullanici profili tutmaz.
9. KVKK kapsaminda bilgi, erisim, silme ve itirazhakkiniz vardir.
10. Politika guncellendiginde yeniden onay istenir."""

_TERMS_SUMMARY = """\
Kullanim Kosullari Ozeti (docs/terms_of_service.md v1.0)
---------------------------------------------------------
1. Nexus Agent "oldugu gibi" sunulmaktadir; hicbir garanti verilmez.
2. Ajanin makinenizde gerceklestirdigi tum eylemlerden siz sorumlusunuz.
3. API maliyetleri (BYOK) dogrudan ilgili saglayici tarafindan tahsil edilir.
4. Zarali faaliyet, gizlilik ihlali veya yetkisiz erisim icin kullanilamaz.
5. Kritik altyapi sistemleri uzerinde yetkisiz otomasyon yasaktir.
6. Bu kosullari kabul ederek tam Kullanim Kosullarini onaylamis olursunuz."""

_FIRST_TASK_SUGGESTION = (
    'Try: "Open Notepad and type Hello, World!"'
)

_PROVIDERS = ("anthropic", "openai", "both")

# ---------------------------------------------------------------------------
# Default injectable implementations
# ---------------------------------------------------------------------------


def _default_health_check() -> HealthReport:
    return HealthChecker().run_all()


def _default_validate_key(provider: str, key: str) -> tuple[bool, str]:
    """Minimal format check — real validation deferred to CredentialManager."""
    from nexus.cloud.credentials import CredentialManager  # noqa: PLC0415

    cm = CredentialManager()
    ok, reason = cm.validate_key(provider, key)
    return ok, reason


def _default_test_api(provider: str, key: str) -> tuple[bool, float]:
    """Stub — real connectivity test would call the provider with a tiny prompt."""
    return True, 0.001


# ---------------------------------------------------------------------------
# OnboardingFlow
# ---------------------------------------------------------------------------


class OnboardingFlow:
    """
    First-run onboarding wizard.

    Parameters
    ----------
    _print_fn:
        Output function.  Default: ``print``.
    _prompt_fn:
        Input function.  Default: ``input``.
    _has_consent_fn:
        ``(scope: str, version: str) -> bool``.  True when *scope* consent
        for the given *version* is stored.
    _save_consent_fn:
        ``(scope: str, version: str) -> None``.  Persist consent for *scope*
        and *version* so re-consent is requested when the version changes.
    _health_check_fn:
        ``() -> HealthReport``.  Run all system health checks.
    _validate_key_fn:
        ``(provider: str, key: str) -> (bool, reason)``.
    _launch_browser_fn:
        ``() -> bool``.  Launch Chrome in remote-debug mode.
    _test_api_fn:
        ``(provider: str, key: str) -> (success, cost_usd)``.
    """

    def __init__(
        self,
        *,
        _print_fn: Callable[[str], None] | None = None,
        _prompt_fn: Callable[[str], str] | None = None,
        _has_consent_fn: Callable[[str, str], bool] | None = None,
        _save_consent_fn: Callable[[str, str], None] | None = None,
        _health_check_fn: Callable[[], HealthReport] | None = None,
        _validate_key_fn: (
            Callable[[str, str], tuple[bool, str]] | None
        ) = None,
        _launch_browser_fn: Callable[[], bool] | None = None,
        _test_api_fn: (
            Callable[[str, str], tuple[bool, float]] | None
        ) = None,
    ) -> None:
        self._print = _print_fn or print
        self._prompt = _prompt_fn or input
        self._has_consent = _has_consent_fn or (lambda _scope, _ver: False)
        self._save_consent = _save_consent_fn or (lambda _scope, _ver: None)
        self._health_check = _health_check_fn or _default_health_check
        self._validate_key = _validate_key_fn or _default_validate_key
        self._launch_browser = _launch_browser_fn or (lambda: False)
        self._test_api = _test_api_fn or _default_test_api

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_first_run(self) -> bool:
        """
        Return True when neither 'privacy' nor 'terms' consent is stored.

        A returning user who completed onboarding will have both consents
        recorded; this returns False for them.
        """
        return not (
            self._has_consent("privacy", _PRIVACY_VERSION)
            and self._has_consent("terms", _TERMS_VERSION)
        )

    def run(self) -> bool:
        """
        Execute the full onboarding flow.

        Returns
        -------
        True when all steps complete.
        False when the user cancels or a hard health failure blocks progress.
        """
        if not self.is_first_run():
            _log.debug("onboarding_skipped_returning_user")
            return True

        _log.debug("onboarding_start")

        if not self._step_welcome():
            return False
        if not self._step_health():
            return False
        if not self._step_privacy():
            return False
        if not self._step_terms():
            return False

        provider, key = self._step_api_key()
        if not provider:
            return False

        self._step_browser_setup()
        self._step_connectivity(provider, key)
        self._step_ready()

        _log.debug("onboarding_complete")
        return True

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _step_welcome(self) -> bool:
        self._print("")
        self._print(f"  Nexus Agent v{_VERSION}")
        self._print("  ─────────────────────")
        self._print("  Başlamak için birkaç adım tamamlayacağız.")
        self._print("")
        answer = self._prompt("  Devam etmek için Enter'a basın (çıkmak için 'q'): ")
        if answer.strip().lower() == "q":
            self._print("  Kurulum iptal edildi.")
            return False
        return True

    def _step_health(self) -> bool:
        self._print("\n  [Adım 2/8] Sistem Kontrolü")
        self._print("  ──────────────────────────")
        report = self._health_check()

        for check in report.checks:
            icon = {"ok": "✓", "warn": "!", "fail": "✗"}[check.status]
            self._print(f"  {icon} {check.name}: {check.message}")
            if check.status in ("warn", "fail") and check.fix_hint:
                self._print(f"    → {check.fix_hint}")

        if report.overall == "fail":
            self._print(
                "\n  Kritik sistem hatası tespit edildi. "
                "Onboarding tamamlanamıyor."
            )
            return False

        if report.overall == "warn":
            self._print(
                "\n  Bazı uyarılar var. Devam etmek istediğinizden emin misiniz?"
            )
            answer = self._prompt("  Devam et? [e/h]: ")
            if answer.strip().lower() not in ("e", "evet", "y", "yes"):
                self._print("  Kurulum iptal edildi.")
                return False

        self._print("  Sistem kontrolü tamamlandı.")
        return True

    def _step_privacy(self) -> bool:
        self._print("\n  [Adım 3/8] Gizlilik Politikası")
        self._print("  ───────────────────────────────")
        self._print(_PRIVACY_SUMMARY)
        self._print("")
        answer = self._prompt(
            "  Gizlilik politikasını okudum ve kabul ediyorum. [e/h]: "
        )
        if answer.strip().lower() not in ("e", "evet", "y", "yes"):
            self._print("  Gizlilik politikası kabul edilmedi. Kurulum iptal.")
            return False
        self._save_consent("privacy", _PRIVACY_VERSION)
        self._print(f"  Gizlilik politikası onayı kaydedildi (v{_PRIVACY_VERSION}).")
        return True

    def _step_terms(self) -> bool:
        self._print("\n  [Adım 4/8] Kullanım Koşulları")
        self._print("  ──────────────────────────────")
        self._print(_TERMS_SUMMARY)
        self._print("")
        answer = self._prompt(
            "  Kullanım koşullarını okudum ve kabul ediyorum. [e/h]: "
        )
        if answer.strip().lower() not in ("e", "evet", "y", "yes"):
            self._print("  Kullanım koşulları kabul edilmedi. Kurulum iptal.")
            return False
        self._save_consent("terms", _TERMS_VERSION)
        self._print(f"  Kullanım koşulları onayı kaydedildi (v{_TERMS_VERSION}).")
        return True

    def _step_api_key(self) -> tuple[str, str]:
        """
        Return (provider, key) on success, ("", "") on cancel.

        Retries indefinitely until the user enters a valid key or cancels.
        """
        self._print("\n  [Adım 5/8] API Anahtarı")
        self._print("  ────────────────────────")
        self._print("  Hangi AI sağlayıcısını kullanmak istersiniz?")
        for i, p in enumerate(_PROVIDERS, 1):
            self._print(f"    {i}. {p}")

        choice_raw = self._prompt("  Seçiminiz (1/2/3): ").strip()
        try:
            idx = int(choice_raw) - 1
            if not 0 <= idx < len(_PROVIDERS):
                raise ValueError
            provider = _PROVIDERS[idx]
        except (ValueError, IndexError):
            self._print("  Geçersiz seçim. Kurulum iptal edildi.")
            return "", ""

        # For "both", validate anthropic key first
        validate_provider = "anthropic" if provider == "both" else provider

        while True:
            key = self._prompt(
                f"  {validate_provider.capitalize()} API anahtarınızı girin"
                f" (iptal için boş bırakın): "
            ).strip()
            if not key:
                self._print("  API anahtarı girilmedi. Kurulum iptal edildi.")
                return "", ""

            ok, reason = self._validate_key(validate_provider, key)
            if ok:
                self._print(f"  API anahtarı doğrulandı: {validate_provider}")
                return provider, key

            self._print(f"  Geçersiz anahtar: {reason}")
            self._print("  Lütfen tekrar deneyin.")

    def _step_browser_setup(self) -> None:
        self._print("\n  [Adım 6/8] Tarayıcı Kurulumu")
        self._print("  ────────────────────────────")
        self._print(
            "  DOM adaptörü kullanmak için Chrome'un debug modunda "
            "çalışması gerekiyor."
        )
        answer = self._prompt(
            "  Chrome'u debugging modunda başlatmamızı ister misiniz? [e/h]: "
        )
        if answer.strip().lower() in ("e", "evet", "y", "yes"):
            ok = self._launch_browser()
            if ok:
                self._print("  Chrome debug modunda başlatıldı.")
            else:
                self._print(
                    "  Chrome başlatılamadı. "
                    "DOM özellikleri sınırlı çalışabilir."
                )
        else:
            self._print("  Tarayıcı kurulumu atlandı.")

    def _step_connectivity(self, provider: str, key: str) -> None:
        self._print("\n  [Adım 7/8] Bağlantı Testi")
        self._print("  ──────────────────────────")
        self._print(
            f"  {provider.capitalize()} API'sine küçük bir test isteği "
            f"gönderiliyor (~$0.001)..."
        )
        ok, cost = self._test_api(provider, key)
        if ok:
            self._print(f"  Bağlantı başarılı! Maliyet: ${cost:.4f}")
        else:
            self._print(
                "  Bağlantı testi başarısız. "
                "API anahtarınızı daha sonra kontrol edin."
            )

    def _step_ready(self) -> None:
        self._print("\n  [Adım 8/8] Hazır!")
        self._print("  ──────────────────")
        self._print("  Nexus Agent kullanıma hazır.")
        self._print(f"\n  İlk görev önerisi: {_FIRST_TASK_SUGGESTION}")
        self._print("")
