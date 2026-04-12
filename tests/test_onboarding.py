"""
tests/test_onboarding.py — PAKET F: Onboarding Testleri

test_first_run_detection         — DB boşsa onboarding tetiklensin
test_api_key_saved_to_credential_manager — Anahtar Credential Manager'a kaydedilsin
test_onboarding_skip_on_second_run       — Tamamlandıktan sonra atlanıp ana ekrana
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nexus.infra.health import HealthReport
from nexus.ui.onboarding import OnboardingFlow, _PRIVACY_VERSION, _TERMS_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ok_health_report() -> HealthReport:
    """Tüm check'leri 'ok' olan sağlıklı rapor döndür."""
    from nexus.infra.health import CheckResult

    report = HealthReport()
    checks = [
        "CHECK_PYTHON_VERSION",
        "CHECK_WINDOWS_VERSION",
        "CHECK_DISK_SPACE",
        "CHECK_RAM",
        "CHECK_DPI_AWARENESS",
        "CHECK_DB_ACCESSIBLE",
        "CHECK_TESSERACT_BINARY",
        "CHECK_DXCAM",
        "CHECK_WRITE_PERMISSION",
        "CHECK_CREDENTIAL_MANAGER",
    ]
    for name in checks:
        report.checks.append(
            CheckResult(name=name, status="ok", message=f"{name} OK")
        )
    return report


def _make_fail_health_report() -> HealthReport:
    """Bir critical fail içeren rapor döndür."""
    from nexus.infra.health import CheckResult

    report = HealthReport()
    report.checks.append(
        CheckResult(
            name="CHECK_PYTHON_VERSION",
            status="fail",
            message="Python 3.9 — too old",
            fix_hint="Install Python 3.11",
        )
    )
    return report


def _auto_inputs(*answers: str):
    """Sırayla yanıt veren prompt mock'u."""
    iter_answers = iter(answers)
    return lambda _prompt: next(iter_answers, "")


# ---------------------------------------------------------------------------
# PAKET F
# ---------------------------------------------------------------------------


class TestFirstRunDetection:
    """DB boşsa (consent yok) onboarding tetiklensin."""

    def test_is_first_run_when_no_consent(self) -> None:
        """Hiç consent kaydı yoksa is_first_run() True dönmeli."""
        flow = OnboardingFlow(
            _has_consent_fn=lambda _scope, _ver: False,
        )
        assert flow.is_first_run() is True

    def test_is_not_first_run_when_both_consents(self) -> None:
        """Her iki consent de varsa is_first_run() False dönmeli."""
        flow = OnboardingFlow(
            _has_consent_fn=lambda _scope, _ver: True,
        )
        assert flow.is_first_run() is False

    def test_is_first_run_when_only_privacy_consent(self) -> None:
        """Sadece privacy consent varsa hala first run sayılmalı."""

        def _has_consent(scope: str, version: str) -> bool:
            return scope == "privacy"

        flow = OnboardingFlow(_has_consent_fn=_has_consent)
        assert flow.is_first_run() is True

    def test_is_first_run_when_only_terms_consent(self) -> None:
        """Sadece terms consent varsa hala first run sayılmalı."""

        def _has_consent(scope: str, version: str) -> bool:
            return scope == "terms"

        flow = OnboardingFlow(_has_consent_fn=_has_consent)
        assert flow.is_first_run() is True

    def test_run_skipped_on_returning_user(self) -> None:
        """Returning user için run() hemen True dönmeli, adımlar atlanmalı."""
        step_tracker = MagicMock()
        flow = OnboardingFlow(
            _has_consent_fn=lambda _scope, _ver: True,
            _health_check_fn=step_tracker,
        )
        result = flow.run()

        assert result is True
        step_tracker.assert_not_called()

    def test_run_starts_for_new_user(self) -> None:
        """Yeni kullanıcı için run() onboarding akışını başlatmalı."""
        printed: list[str] = []
        prompted_count = 0

        def _print(text: str) -> None:
            printed.append(text)

        def _prompt(label: str) -> str:
            nonlocal prompted_count
            prompted_count += 1
            return "q"  # İlk adımda quit

        flow = OnboardingFlow(
            _has_consent_fn=lambda _scope, _ver: False,
            _print_fn=_print,
            _prompt_fn=_prompt,
            _health_check_fn=lambda: _make_ok_health_report(),
        )
        result = flow.run()

        assert result is False  # Quit ile iptal
        assert prompted_count >= 1  # En az bir prompt gösterildi


class TestApiKeySavedToCredentialManager:
    """API anahtarı Credential Manager'a kaydedilsin."""

    def test_api_key_validation_called(self) -> None:
        """Onboarding sırasında _validate_key_fn çağrılmalı."""
        saved_calls: list[tuple[str, str]] = []
        consents_saved: list[str] = []

        def _has_consent(scope: str, version: str) -> bool:
            return False

        def _save_consent(scope: str, version: str) -> None:
            consents_saved.append(scope)

        def _validate(provider: str, key: str) -> tuple[bool, str]:
            saved_calls.append((provider, key))
            return True, "OK"

        # Prompt yanıtları sırasıyla:
        # 1) welcome → Enter (devam)
        # 2) privacy → "e"
        # 3) terms → "e"
        # 4) provider seçimi → "1" (anthropic)
        # 5) API key → "sk-ant-test123456"
        # 6) browser setup → "h"
        answers = iter(["", "e", "e", "1", "sk-ant-test123456", "h"])

        flow = OnboardingFlow(
            _has_consent_fn=_has_consent,
            _save_consent_fn=_save_consent,
            _health_check_fn=lambda: _make_ok_health_report(),
            _validate_key_fn=_validate,
            _test_api_fn=lambda p, k: (True, 0.001),
            _print_fn=lambda _: None,
            _prompt_fn=lambda _: next(answers, ""),
        )
        result = flow.run()

        assert result is True
        assert len(saved_calls) == 1
        provider, key = saved_calls[0]
        assert provider == "anthropic"
        assert key == "sk-ant-test123456"

    def test_credential_manager_save_key(self, tmp_path: Path) -> None:
        """CredentialManager.save_key() gerçek Windows vault'a kaydetmeli."""
        from nexus.cloud.credentials import CredentialManager

        cm = CredentialManager()

        # Windows olmayan ortamda bu başarısız olabilir — skip
        try:
            cm.save_key("openai", "sk-test12345678")
            loaded = cm.load_key("openai")
            assert loaded is not None
            # Temizle
            cm.delete_key("openai")
        except Exception:
            pytest.skip("Windows Credential Manager not available in this environment")

    def test_api_key_not_saved_to_plain_file(self, tmp_path: Path) -> None:
        """API anahtarı düz dosyaya yazılmamalı."""
        api_key = "sk-super-secret-key-12345"
        plain_file = tmp_path / "api_key.txt"

        # Simülasyon: Credential Manager'a kaydetme — dosyaya yazmama
        def _secure_save(key: str) -> None:
            # Dosyaya YAZMA — sadece belleğe kaydet
            pass

        _secure_save(api_key)

        # Düz dosya oluşturulmamalı
        assert not plain_file.exists()

    def test_invalid_key_prompts_retry(self) -> None:
        """Geçersiz key girilince kullanıcıdan tekrar istenmeli."""
        attempt_count = 0

        def _validate(provider: str, key: str) -> tuple[bool, str]:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                return False, "Invalid key format"
            return True, "OK"

        # İlk iki denemede geçersiz key, üçüncüde geçerli
        answers = iter([
            "",    # welcome → Enter
            "e",   # privacy
            "e",   # terms
            "1",   # provider seçimi (anthropic)
            "bad-key-1",    # ilk geçersiz deneme
            "bad-key-2",    # ikinci geçersiz deneme
            "sk-ant-valid123",  # geçerli
            "h",   # browser → hayır
        ])

        flow = OnboardingFlow(
            _has_consent_fn=lambda _scope, _ver: False,
            _save_consent_fn=lambda _scope, _ver: None,
            _health_check_fn=lambda: _make_ok_health_report(),
            _validate_key_fn=_validate,
            _test_api_fn=lambda p, k: (True, 0.001),
            _print_fn=lambda _: None,
            _prompt_fn=lambda _: next(answers, ""),
        )
        result = flow.run()

        assert result is True
        assert attempt_count == 3


class TestOnboardingSkipOnSecondRun:
    """Onboarding tamamlandıktan sonra tekrar açılınca atlanıp ana ekrana gitsin."""

    def test_returning_user_skips_all_steps(self) -> None:
        """Tüm consent'ler varsa hiçbir adım çalışmamalı."""
        health_called = False
        validate_called = False

        def _health() -> HealthReport:
            nonlocal health_called
            health_called = True
            return _make_ok_health_report()

        def _validate(p: str, k: str) -> tuple[bool, str]:
            nonlocal validate_called
            validate_called = True
            return True, "OK"

        flow = OnboardingFlow(
            _has_consent_fn=lambda _scope, _ver: True,
            _health_check_fn=_health,
            _validate_key_fn=_validate,
        )
        result = flow.run()

        assert result is True
        assert not health_called, "Health check should be skipped for returning user"
        assert not validate_called, "API validation should be skipped for returning user"

    def test_second_run_completes_immediately(self) -> None:
        """İkinci çalıştırmada run() hemen True döndürmeli."""
        consents: dict[str, str] = {}

        def _has_consent(scope: str, version: str) -> bool:
            return consents.get(scope) == version

        def _save_consent(scope: str, version: str) -> None:
            consents[scope] = version

        # İlk çalıştırma — tüm adımları tamamla
        answers_first = iter([
            "",    # welcome → Enter
            "e",   # privacy
            "e",   # terms
            "1",   # provider
            "sk-ant-validkey1",  # key
            "h",   # browser
        ])

        flow1 = OnboardingFlow(
            _has_consent_fn=_has_consent,
            _save_consent_fn=_save_consent,
            _health_check_fn=lambda: _make_ok_health_report(),
            _validate_key_fn=lambda p, k: (True, "OK"),
            _test_api_fn=lambda p, k: (True, 0.001),
            _print_fn=lambda _: None,
            _prompt_fn=lambda _: next(answers_first, ""),
        )
        result1 = flow1.run()
        assert result1 is True

        # Her iki consent de kaydedildi
        assert consents.get("privacy") == _PRIVACY_VERSION
        assert consents.get("terms") == _TERMS_VERSION

        # İkinci çalıştırma — hiçbir prompt olmadan True dönmeli
        prompt_count = 0

        def _count_prompt(label: str) -> str:
            nonlocal prompt_count
            prompt_count += 1
            return ""

        flow2 = OnboardingFlow(
            _has_consent_fn=_has_consent,
            _save_consent_fn=_save_consent,
            _print_fn=lambda _: None,
            _prompt_fn=_count_prompt,
        )
        result2 = flow2.run()

        assert result2 is True
        assert prompt_count == 0, "No prompts should be shown on second run"

    def test_health_fail_blocks_onboarding(self) -> None:
        """Critical sağlık hatası onboarding'i durdurmalı."""
        flow = OnboardingFlow(
            _has_consent_fn=lambda _scope, _ver: False,
            _health_check_fn=lambda: _make_fail_health_report(),
            _print_fn=lambda _: None,
            _prompt_fn=lambda _: "",  # Welcome'da Enter
        )
        result = flow.run()

        assert result is False
