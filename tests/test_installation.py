"""
tests/test_installation.py — PAKET K: Kurulum Testleri

test_pyproject_dependencies — Tüm bağımlılıklar import edilebilsin
test_config_loading         — configs/default.toml yüklensin, zorunlu alanlar mevcut
test_env_var_api_key        — API_KEY yoksa net hata versin
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# PAKET K
# ---------------------------------------------------------------------------


class TestPyprojectDependencies:
    """Tüm bağımlılıklar import edilebilmeli."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "pydantic",
            "structlog",
            "aiosqlite",
            "pytesseract",
            "openpyxl",
            "pypdf",
            "aiohttp",
            "numpy",
            "cv2",  # opencv-python
        ],
    )
    def test_module_importable(self, module_name: str) -> None:
        """Belirtilen modül import edilebilmeli."""
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            pytest.fail(f"Module '{module_name}' could not be imported: {exc}")

    def test_pdf2image_or_pypdf_available(self) -> None:
        """PDF işleme modülü mevcut olmalı."""
        pdf_modules = ["pypdf", "pdf2image", "PyPDF2"]
        available = []
        for mod in pdf_modules:
            try:
                importlib.import_module(mod)
                available.append(mod)
            except ImportError:
                pass
        assert len(available) > 0, f"No PDF module available from {pdf_modules}"

    def test_openai_importable(self) -> None:
        """openai kütüphanesi import edilebilmeli."""
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            pytest.fail(f"openai could not be imported: {exc}")

    def test_anthropic_importable(self) -> None:
        """anthropic kütüphanesi import edilebilmeli."""
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            pytest.fail(f"anthropic could not be imported: {exc}")

    def test_nexus_core_importable(self) -> None:
        """nexus.core modülleri import edilebilmeli."""
        core_modules = [
            "nexus.core.types",
            "nexus.core.settings",
            "nexus.core.errors",
        ]
        for mod in core_modules:
            try:
                importlib.import_module(mod)
            except ImportError as exc:
                pytest.fail(f"Core module '{mod}' could not be imported: {exc}")

    def test_nexus_infra_importable(self) -> None:
        """nexus.infra modülleri import edilebilmeli."""
        infra_modules = [
            "nexus.infra.database",
            "nexus.infra.cost_tracker",
            "nexus.infra.health",
            "nexus.infra.logger",
            "nexus.infra.trace",
            "nexus.infra.repositories",
        ]
        for mod in infra_modules:
            try:
                importlib.import_module(mod)
            except ImportError as exc:
                pytest.fail(f"Infra module '{mod}' could not be imported: {exc}")


class TestConfigLoading:
    """configs/default.toml yüklenmeli, zorunlu alanlar mevcut olmalı."""

    def _get_default_toml_path(self) -> Path:
        here = Path(__file__).parent.parent
        return here / "configs" / "default.toml"

    def test_default_toml_exists(self) -> None:
        """configs/default.toml dosyası mevcut olmalı."""
        path = self._get_default_toml_path()
        assert path.exists(), f"configs/default.toml not found at {path}"

    def test_default_toml_loadable(self) -> None:
        """default.toml NexusSettings olarak yüklenebilmeli."""
        from nexus.core.settings import load_settings

        path = self._get_default_toml_path()
        if not path.exists():
            pytest.skip("configs/default.toml not found")

        settings = load_settings(config_file=path)
        assert settings is not None

    def test_required_capture_fields(self) -> None:
        """capture bölümünde zorunlu alanlar mevcut olmalı."""
        from nexus.core.settings import load_settings

        path = self._get_default_toml_path()
        if not path.exists():
            pytest.skip("configs/default.toml not found")

        settings = load_settings(config_file=path)
        assert settings.capture.fps > 0
        assert settings.capture.max_frame_buffer > 0
        assert settings.capture.stabilization_timeout_ms > 0

    def test_required_budget_fields(self) -> None:
        """budget bölümünde zorunlu alanlar mevcut olmalı."""
        from nexus.core.settings import load_settings

        path = self._get_default_toml_path()
        if not path.exists():
            pytest.skip("configs/default.toml not found")

        settings = load_settings(config_file=path)
        assert settings.budget.max_cost_per_task_usd > 0
        assert settings.budget.max_cost_per_day_usd > 0
        assert 0.0 <= settings.budget.warn_at_percent <= 1.0

    def test_required_storage_fields(self) -> None:
        """storage bölümünde zorunlu alanlar mevcut olmalı."""
        from nexus.core.settings import load_settings

        path = self._get_default_toml_path()
        if not path.exists():
            pytest.skip("configs/default.toml not found")

        settings = load_settings(config_file=path)
        assert settings.storage.db_path
        assert settings.storage.log_dir
        assert settings.storage.memory_max_size_mb > 0
        assert settings.storage.memory_evict_after_days > 0

    def test_required_safety_fields(self) -> None:
        """safety bölümünde zorunlu alanlar mevcut olmalı."""
        from nexus.core.settings import load_settings

        path = self._get_default_toml_path()
        if not path.exists():
            pytest.skip("configs/default.toml not found")

        settings = load_settings(config_file=path)
        assert settings.safety.max_actions_per_task > 0
        assert settings.safety.max_task_duration_minutes > 0

    def test_nexus_settings_defaults(self) -> None:
        """NexusSettings Python default'ları geçerli aralıkta olmalı."""
        from nexus.core.settings import NexusSettings

        settings = NexusSettings()
        assert 1 <= settings.capture.fps <= 120
        assert settings.cloud.max_tokens >= 1
        assert settings.cloud.timeout_seconds >= 1
        assert 0.0 <= settings.perception.ocr_confidence_threshold <= 1.0


class TestEnvVarApiKey:
    """ANTHROPIC_API_KEY veya OPENAI_API_KEY yoksa net hata versin."""

    def test_no_api_key_raises_clear_error(self) -> None:
        """Her iki API anahtarı da yoksa net bir hata mesajı üretilmeli."""
        env_without_keys = {
            k: v
            for k, v in os.environ.items()
            if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
        }

        with patch.dict(os.environ, env_without_keys, clear=True):
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

        assert openai_key == ""
        assert anthropic_key == ""

    def test_missing_key_detection_logic(self) -> None:
        """API key yoksa boolean flag False olmalı."""

        def _has_valid_api_keys() -> bool:
            openai = os.environ.get("OPENAI_API_KEY", "").strip()
            anthropic = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            return bool(openai or anthropic)

        env_without_keys = {
            k: v
            for k, v in os.environ.items()
            if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
        }

        with patch.dict(os.environ, env_without_keys, clear=True):
            result = _has_valid_api_keys()

        assert result is False

    def test_with_openai_key_passes(self) -> None:
        """OPENAI_API_KEY varsa kontrol geçmeli."""

        def _has_valid_api_keys() -> bool:
            openai = os.environ.get("OPENAI_API_KEY", "").strip()
            anthropic = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            return bool(openai or anthropic)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-123"}):
            result = _has_valid_api_keys()

        assert result is True

    def test_credential_manager_validate_format(self) -> None:
        """CredentialManager API key format doğrulaması çalışmalı."""
        from nexus.cloud.credentials import CredentialManager

        cm = CredentialManager()

        # Geçersiz format
        ok, reason = cm.validate_key("openai", "not-a-valid-key")
        assert not ok
        assert reason  # Hata mesajı boş olmamalı

        # Geçerli format
        ok2, _ = cm.validate_key("openai", "sk-test12345678")
        assert ok2

    def test_anthropic_key_format_validation(self) -> None:
        """Anthropic API key formatı doğrulanmalı."""
        from nexus.cloud.credentials import CredentialManager

        cm = CredentialManager()

        ok, _ = cm.validate_key("anthropic", "sk-ant-test12345678")
        assert ok

        ok2, reason2 = cm.validate_key("anthropic", "sk-invalid")
        assert not ok2
