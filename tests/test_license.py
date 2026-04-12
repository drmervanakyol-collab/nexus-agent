"""
tests/test_license.py — PAKET E: Lisans Testleri

test_valid_license           — Geçerli HMAC secret ile doğrulama geçsin
test_invalid_license         — Yanlış secret ile uygulama açılmasın
test_missing_license         — NEXUS_LICENSE_SECRET yoksa çökmesin, net hata versin
test_sensitive_fields_not_logged — Lisans anahtarı loglara yazılmasın
"""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from nexus.release.license_manager import (
    LicenseManager,
    LicenseType,
    _DEV_SECRET,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager_with_machine_id(machine_id: str) -> LicenseManager:
    """LicenseManager ile önceden belirlenmiş machine_id döndür."""
    lm = LicenseManager()
    lm._machine_id = machine_id
    return lm


# ---------------------------------------------------------------------------
# PAKET E
# ---------------------------------------------------------------------------


class TestValidLicense:
    """Geçerli HMAC secret ile doğrulama geçmeli."""

    def test_full_license_validates(self) -> None:
        """generate_key() + validate_license() döngüsü geçmeli."""
        machine_id = "abcdef1234567890"
        lm = _make_manager_with_machine_id(machine_id)

        key = LicenseManager.generate_key(machine_id, LicenseType.FULL)
        result = lm.validate_license(key)

        assert result.valid is True
        assert result.license_type == LicenseType.FULL
        assert result.machine_id == machine_id

    def test_perpetual_license_no_expiry(self) -> None:
        """Süresiz lisans expires=None olmalı."""
        machine_id = "abc123def456abcd"
        lm = _make_manager_with_machine_id(machine_id)

        key = LicenseManager.generate_key(machine_id, LicenseType.FULL, expires=None)
        result = lm.validate_license(key)

        assert result.valid is True
        assert result.expires is None

    def test_license_with_future_expiry(self) -> None:
        """Gelecek tarihli lisans geçerli olmalı."""
        machine_id = "future123456abcd"
        future_date = date.today() + timedelta(days=365)
        lm = _make_manager_with_machine_id(machine_id)

        key = LicenseManager.generate_key(
            machine_id, LicenseType.FULL, expires=future_date
        )
        result = lm.validate_license(key)

        assert result.valid is True
        assert result.expires == future_date

    def test_trial_license_validates(self) -> None:
        """Trial tipi lisans da validate edilebilmeli."""
        machine_id = "trial12345678abc"
        lm = _make_manager_with_machine_id(machine_id)

        key = LicenseManager.generate_key(machine_id, LicenseType.TRIAL)
        result = lm.validate_license(key)

        assert result.valid is True
        assert result.license_type == LicenseType.TRIAL

    def test_dev_secret_is_default(self) -> None:
        """NEXUS_LICENSE_SECRET yokken dev secret kullanılmalı."""
        env = {k: v for k, v in os.environ.items() if k != "NEXUS_LICENSE_SECRET"}
        with patch.dict(os.environ, env, clear=True):
            machine_id = "dev_secret_test1"
            lm = _make_manager_with_machine_id(machine_id)
            key = LicenseManager.generate_key(machine_id, LicenseType.FULL)
            result = lm.validate_license(key)
        assert result.valid is True


class TestInvalidLicense:
    """Yanlış secret ile uygulama açılmamalı."""

    def test_wrong_secret_fails_validation(self) -> None:
        """Farklı HMAC secret ile imzalanmış anahtar reddedilmeli."""
        machine_id = "wrongsecret12345"
        lm = _make_manager_with_machine_id(machine_id)

        # Anahtarı DEV_SECRET ile oluştur
        key = LicenseManager.generate_key(machine_id, LicenseType.FULL)

        # Farklı secret ile doğrulamayı dene
        with patch.dict(os.environ, {"NEXUS_LICENSE_SECRET": "totally-wrong-secret"}):
            result = lm.validate_license(key)

        assert result.valid is False
        assert "imza" in result.message.lower() or "geçersiz" in result.message.lower()

    def test_wrong_machine_id_fails(self) -> None:
        """Başka makine için üretilmiş anahtar reddedilmeli."""
        lm = _make_manager_with_machine_id("my_machine_12345")

        # Başka machine_id için üretilmiş anahtar
        key = LicenseManager.generate_key("other_machine_123", LicenseType.FULL)
        result = lm.validate_license(key)

        assert result.valid is False
        assert "makine" in result.message.lower() or "değil" in result.message.lower()

    def test_expired_license_fails(self) -> None:
        """Süresi dolmuş lisans reddedilmeli."""
        machine_id = "expired_machine01"
        past_date = date.today() - timedelta(days=1)
        lm = _make_manager_with_machine_id(machine_id)

        key = LicenseManager.generate_key(
            machine_id, LicenseType.FULL, expires=past_date
        )
        result = lm.validate_license(key)

        assert result.valid is False
        assert "doldu" in result.message.lower()

    def test_tampered_key_fails(self) -> None:
        """Değiştirilmiş anahtar reddedilmeli."""
        machine_id = "tamper_machine001"
        lm = _make_manager_with_machine_id(machine_id)

        key = LicenseManager.generate_key(machine_id, LicenseType.FULL)
        # Anahtarı boz
        tampered = key[:-5] + "XXXXX"
        result = lm.validate_license(tampered)

        assert result.valid is False

    def test_malformed_key_fails(self) -> None:
        """Hatalı format nedeniyle reddedilmeli."""
        lm = _make_manager_with_machine_id("any_machine_1234")

        for bad_key in ["", "no_dots_here", "a.b.c", "x" * 100]:
            result = lm.validate_license(bad_key)
            assert result.valid is False


class TestMissingLicense:
    """NEXUS_LICENSE_SECRET yoksa çökmesin, net hata versin."""

    def test_no_license_key_falls_to_trial(self, tmp_path: Path) -> None:
        """Lisans anahtarı yoksa trial moduna düşmeli, çökmemeli."""
        trial_path = str(tmp_path / "trial.json")
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("NEXUS_LICENSE_KEY", "NEXUS_TRIAL_PATH")
        }
        env["NEXUS_TRIAL_PATH"] = trial_path

        with patch.dict(os.environ, env, clear=True):
            lm = LicenseManager()
            lm._machine_id = "no_key_machine01"

            # Trial başlangıç — 14 günlük pencere
            status = lm.trial_status()

        assert "started_on" in status
        assert isinstance(status["expired"], bool)

    def test_missing_secret_generates_key_with_dev_fallback(self) -> None:
        """NEXUS_LICENSE_SECRET yoksa dev secret fallback çalışmalı."""
        env = {k: v for k, v in os.environ.items() if k != "NEXUS_LICENSE_SECRET"}
        with patch.dict(os.environ, env, clear=True):
            machine_id = "dev_fallback_1234"
            key = LicenseManager.generate_key(machine_id, LicenseType.FULL)
            lm = _make_manager_with_machine_id(machine_id)
            result = lm.validate_license(key)

        # Dev secret ile doğrulama geçmeli
        assert result.valid is True

    def test_trial_status_no_crash(self, tmp_path: Path) -> None:
        """trial_status() herhangi bir durumda crash olmamalı."""
        trial_path = str(tmp_path / "trial2.json")
        with patch.dict(os.environ, {"NEXUS_TRIAL_PATH": trial_path}):
            lm = LicenseManager()
            lm._machine_id = "trial_no_crash01"
            status = lm.trial_status()

        assert "expired" in status
        assert "today_tasks" in status
        assert "daily_limit_reached" in status

    def test_is_licensed_no_crash_without_key(self, tmp_path: Path) -> None:
        """is_licensed() lisans anahtarı yoksa çökmemeli."""
        trial_path = str(tmp_path / "trial3.json")
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("NEXUS_LICENSE_KEY",)
        }
        env["NEXUS_TRIAL_PATH"] = trial_path

        with patch.dict(os.environ, env, clear=True):
            lm = LicenseManager()
            lm._machine_id = "no_crash_machine1"
            result = lm.is_licensed()

        assert isinstance(result, bool)


class TestSensitiveFieldsNotLogged:
    """Lisans anahtarı loglara yazılmamalı."""

    def test_license_key_not_in_logger_output(self) -> None:
        """License key log çıktısında açık halde bulunmamalı."""
        machine_id = "log_test_machine1"
        key = LicenseManager.generate_key(machine_id, LicenseType.FULL)

        buf = io.StringIO()
        from nexus.infra.logger import configure_logging

        configure_logging(level=logging.DEBUG, stream=buf)

        # Validate işlemi yapılırken key loglanmamalı
        lm = _make_manager_with_machine_id(machine_id)
        lm.validate_license(key)

        output = buf.getvalue()
        # Anahtarın payload kısmı log'da olmamalı (ham key formatında)
        # Genel kontrol: anahtar fragmanları log'a sızmamalı
        key_part = key.split(".")[0][:20] if "." in key else key[:20]
        # Çok kısa fragmentleri kontrol etme (false positive)
        if len(key_part) > 15:
            assert key_part not in output, (
                "License key fragment found in log output"
            )

    def test_trial_state_no_key_exposure(self, tmp_path: Path) -> None:
        """Trial durumu loglanırken anahtar görünmemeli."""
        trial_path = str(tmp_path / "trial_log.json")

        buf = io.StringIO()
        from nexus.infra.logger import configure_logging

        configure_logging(level=logging.DEBUG, stream=buf)

        with patch.dict(os.environ, {"NEXUS_TRIAL_PATH": trial_path}):
            lm = LicenseManager()
            lm._machine_id = "trial_log_machine"
            lm.trial_status()

        output = buf.getvalue()
        # API anahtarı formatındaki şeyler logda olmamalı
        import re
        assert not re.search(r"sk-[A-Za-z0-9]{10,}", output)

    def test_license_result_machine_id_not_secret(self) -> None:
        """LicenseResult machine_id içeriği HMAC secret değil."""
        machine_id = "safe_machine_1234"
        lm = _make_manager_with_machine_id(machine_id)
        key = LicenseManager.generate_key(machine_id, LicenseType.FULL)
        result = lm.validate_license(key)

        # machine_id ve HMAC secret birbirinden farklı
        assert result.machine_id != _DEV_SECRET
        # machine_id kısa bir hash prefix (16 char) veya test ortamında
        # directly set edilen değer olabilir
        assert len(result.machine_id) >= 1
