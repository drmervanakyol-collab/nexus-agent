"""Unit tests for nexus/core/settings.py — targets 100 % coverage."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nexus.core.settings import (
    BudgetSettings,
    CaptureSettings,
    CloudSettings,
    ModelPricing,
    NexusSettings,
    PerceptionSettings,
    SafetySettings,
    SourceSettings,
    StorageSettings,
    TransportSettings,
    VerificationSettings,
    load_settings,
)

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"

# ---------------------------------------------------------------------------
# CaptureSettings
# ---------------------------------------------------------------------------


class TestCaptureSettings:
    def test_defaults(self) -> None:
        s = CaptureSettings()
        assert s.fps == 15
        assert s.max_frame_buffer == 10
        assert s.stabilization_timeout_ms == 3000
        assert s.stabilization_poll_ms == 100
        assert s.dirty_region_block_size == 32
        assert s.dirty_region_full_refresh_ratio == 0.7

    def test_fps_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            CaptureSettings(fps=0)

    def test_fps_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            CaptureSettings(fps=-5)

    def test_ratio_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            CaptureSettings(dirty_region_full_refresh_ratio=1.1)

    def test_ratio_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            CaptureSettings(dirty_region_full_refresh_ratio=-0.1)

    def test_valid_override(self) -> None:
        s = CaptureSettings(fps=30, dirty_region_full_refresh_ratio=0.5)
        assert s.fps == 30
        assert s.dirty_region_full_refresh_ratio == 0.5

    def test_ratio_boundary_zero(self) -> None:
        s = CaptureSettings(dirty_region_full_refresh_ratio=0.0)
        assert s.dirty_region_full_refresh_ratio == 0.0

    def test_ratio_boundary_one(self) -> None:
        s = CaptureSettings(dirty_region_full_refresh_ratio=1.0)
        assert s.dirty_region_full_refresh_ratio == 1.0


# ---------------------------------------------------------------------------
# SourceSettings
# ---------------------------------------------------------------------------


class TestSourceSettings:
    def test_defaults(self) -> None:
        s = SourceSettings()
        assert s.uia_enabled is True
        assert s.uia_timeout_ms == 2000
        assert s.dom_enabled is True
        assert s.dom_timeout_ms == 2000
        assert s.dom_debug_port == 9222
        assert s.file_direct_enabled is True
        assert s.visual_fallback_always is False

    def test_override(self) -> None:
        s = SourceSettings(visual_fallback_always=True, dom_debug_port=9333)
        assert s.visual_fallback_always is True
        assert s.dom_debug_port == 9333


# ---------------------------------------------------------------------------
# TransportSettings
# ---------------------------------------------------------------------------


class TestTransportSettings:
    def test_defaults(self) -> None:
        s = TransportSettings()
        assert s.prefer_native_action is True
        assert s.uia_action_timeout_ms == 2000
        assert s.dom_action_timeout_ms == 2000
        assert s.fallback_to_mouse is True

    def test_override(self) -> None:
        s = TransportSettings(fallback_to_mouse=False)
        assert s.fallback_to_mouse is False


# ---------------------------------------------------------------------------
# PerceptionSettings
# ---------------------------------------------------------------------------


class TestPerceptionSettings:
    def test_defaults(self) -> None:
        s = PerceptionSettings()
        assert s.ocr_confidence_threshold == 0.7
        assert s.locator_confidence_threshold == 0.6
        assert s.ambiguity_local_threshold == 0.4
        assert s.ambiguity_cloud_threshold == 0.7
        assert s.perception_cache_ttl_ms == 200
        assert s.max_graph_nodes == 500

    @pytest.mark.parametrize(
        "field",
        [
            "ocr_confidence_threshold",
            "locator_confidence_threshold",
            "ambiguity_local_threshold",
            "ambiguity_cloud_threshold",
        ],
    )
    def test_threshold_above_one_raises(self, field: str) -> None:
        with pytest.raises(ValidationError):
            PerceptionSettings(**{field: 1.1})

    @pytest.mark.parametrize(
        "field",
        [
            "ocr_confidence_threshold",
            "locator_confidence_threshold",
            "ambiguity_local_threshold",
            "ambiguity_cloud_threshold",
        ],
    )
    def test_threshold_below_zero_raises(self, field: str) -> None:
        with pytest.raises(ValidationError):
            PerceptionSettings(**{field: -0.1})

    def test_threshold_boundary_values(self) -> None:
        s = PerceptionSettings(
            ocr_confidence_threshold=0.0,
            locator_confidence_threshold=1.0,
        )
        assert s.ocr_confidence_threshold == 0.0
        assert s.locator_confidence_threshold == 1.0


# ---------------------------------------------------------------------------
# CloudSettings
# ---------------------------------------------------------------------------


class TestCloudSettings:
    def test_defaults(self) -> None:
        s = CloudSettings()
        assert s.primary_provider == "openai"
        assert s.openai_model == "gpt-4o"
        assert s.openai_fallback_model == "gpt-4o-mini"
        assert s.anthropic_model == "claude-3-5-sonnet-20241022"
        assert s.anthropic_fallback_model == "claude-3-haiku-20240307"
        assert s.max_tokens == 1000
        assert s.timeout_seconds == 30
        assert s.max_retries == 3

    def test_anthropic_provider(self) -> None:
        s = CloudSettings(primary_provider="anthropic")
        assert s.primary_provider == "anthropic"

    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(ValidationError):
            CloudSettings(primary_provider="google")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BudgetSettings
# ---------------------------------------------------------------------------


class TestBudgetSettings:
    def test_defaults(self) -> None:
        s = BudgetSettings()
        assert s.max_cost_per_task_usd == 1.0
        assert s.max_cost_per_day_usd == 10.0
        assert s.warn_at_percent == 0.8

    def test_warn_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            BudgetSettings(warn_at_percent=1.1)

    def test_warn_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            BudgetSettings(warn_at_percent=-0.1)

    def test_warn_boundary_zero(self) -> None:
        s = BudgetSettings(warn_at_percent=0.0)
        assert s.warn_at_percent == 0.0

    def test_warn_boundary_one(self) -> None:
        s = BudgetSettings(warn_at_percent=1.0)
        assert s.warn_at_percent == 1.0

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
        ],
    )
    def test_pricing_for_known_models(self, model: str) -> None:
        pricing = BudgetSettings().pricing_for(model)
        assert isinstance(pricing, ModelPricing)
        assert pricing.input_per_1k > 0
        assert pricing.output_per_1k > 0

    def test_pricing_exact_values(self) -> None:
        b = BudgetSettings()
        assert b.pricing_for("gpt-4o").input_per_1k == 0.005
        assert b.pricing_for("gpt-4o").output_per_1k == 0.015
        assert b.pricing_for("gpt-4o-mini").input_per_1k == 0.00015
        assert b.pricing_for("gpt-4o-mini").output_per_1k == 0.0006
        assert b.pricing_for("claude-3-5-sonnet-20241022").input_per_1k == 0.003
        assert b.pricing_for("claude-3-5-sonnet-20241022").output_per_1k == 0.015
        assert b.pricing_for("claude-3-haiku-20240307").input_per_1k == 0.00025
        assert b.pricing_for("claude-3-haiku-20240307").output_per_1k == 0.00125

    def test_pricing_for_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError, match="No pricing defined"):
            BudgetSettings().pricing_for("unknown-model-xyz")


# ---------------------------------------------------------------------------
# StorageSettings
# ---------------------------------------------------------------------------


class TestStorageSettings:
    def test_defaults(self) -> None:
        s = StorageSettings()
        assert s.db_path == "nexus.db"
        assert s.log_dir == "logs"
        assert s.max_db_size_mb == 500
        assert s.memory_max_size_mb == 256
        assert s.memory_evict_after_days == 30

    def test_override(self) -> None:
        s = StorageSettings(db_path=":memory:", log_dir="logs/test")
        assert s.db_path == ":memory:"
        assert s.log_dir == "logs/test"


# ---------------------------------------------------------------------------
# SafetySettings
# ---------------------------------------------------------------------------


class TestSafetySettings:
    def test_defaults(self) -> None:
        s = SafetySettings()
        assert s.max_actions_per_task == 100
        assert s.max_task_duration_minutes == 30
        assert s.sensitive_masking_enabled is True
        assert s.dry_run_mode is False

    def test_dry_run_override(self) -> None:
        s = SafetySettings(dry_run_mode=True)
        assert s.dry_run_mode is True


# ---------------------------------------------------------------------------
# VerificationSettings
# ---------------------------------------------------------------------------

ALL_ACTION_TYPES = [
    "click_method",
    "type_method",
    "select_method",
    "drag_method",
    "scroll_method",
    "hover_method",
    "keyboard_shortcut_method",
    "copy_paste_method",
    "field_replace_method",
    "form_submit_method",
    "row_write_method",
    "sheet_write_method",
    "cell_edit_method",
    "destructive_method",
    "delete_method",
    "overwrite_method",
    "file_open_method",
    "file_save_method",
    "navigate_method",
    "macroaction_method",
]


class TestVerificationSettings:
    @pytest.mark.parametrize("action_type", ALL_ACTION_TYPES)
    def test_action_type_present(self, action_type: str) -> None:
        s = VerificationSettings()
        assert hasattr(s, action_type), f"Missing action type: {action_type}"

    def test_spec_defaults(self) -> None:
        s = VerificationSettings()
        assert s.click_method == "visual"
        assert s.field_replace_method == "visual+semantic"
        assert s.row_write_method == "combined"
        assert s.destructive_method == "combined+preconfirm"
        assert s.sheet_write_method == "source_level"

    def test_all_defaults_are_valid_methods(self) -> None:
        s = VerificationSettings()
        valid = {
            "none", "visual", "semantic", "source",
            "visual+semantic", "source+visual",
            "combined", "combined+preconfirm", "source_level",
        }
        for action_type in ALL_ACTION_TYPES:
            value = getattr(s, action_type)
            assert value in valid, f"{action_type} has invalid default: {value!r}"

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValidationError):
            VerificationSettings(click_method="invalid")  # type: ignore[arg-type]

    def test_action_count(self) -> None:
        assert len(ALL_ACTION_TYPES) >= 20


# ---------------------------------------------------------------------------
# NexusSettings (root)
# ---------------------------------------------------------------------------


class TestNexusSettings:
    def test_all_sub_settings_present(self) -> None:
        s = NexusSettings()
        assert isinstance(s.capture, CaptureSettings)
        assert isinstance(s.source, SourceSettings)
        assert isinstance(s.transport, TransportSettings)
        assert isinstance(s.perception, PerceptionSettings)
        assert isinstance(s.cloud, CloudSettings)
        assert isinstance(s.budget, BudgetSettings)
        assert isinstance(s.storage, StorageSettings)
        assert isinstance(s.safety, SafetySettings)
        assert isinstance(s.verification, VerificationSettings)

    def test_nested_kwarg_override(self) -> None:
        s = NexusSettings(
            capture={"fps": 30},
            safety={"dry_run_mode": True},
        )
        assert s.capture.fps == 30
        assert s.safety.dry_run_mode is True

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEXUS_CAPTURE__FPS", "25")
        monkeypatch.setenv("NEXUS_SAFETY__DRY_RUN_MODE", "true")
        s = NexusSettings()
        assert s.capture.fps == 25
        assert s.safety.dry_run_mode is True

    def test_env_var_override_cloud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEXUS_CLOUD__PRIMARY_PROVIDER", "anthropic")
        s = NexusSettings()
        assert s.cloud.primary_provider == "anthropic"


# ---------------------------------------------------------------------------
# load_settings factory
# ---------------------------------------------------------------------------


class TestLoadSettings:
    def test_no_file_returns_defaults(self) -> None:
        s = load_settings()
        assert isinstance(s, NexusSettings)
        assert s.capture.fps == 15
        assert s.safety.dry_run_mode is False

    def test_toml_file_overrides_defaults(self, tmp_path: Path) -> None:
        toml = tmp_path / "cfg.toml"
        toml.write_text(
            "[capture]\nfps = 60\n\n[safety]\ndry_run_mode = true\n",
            encoding="utf-8",
        )
        s = load_settings(toml)
        assert s.capture.fps == 60
        assert s.safety.dry_run_mode is True

    def test_kwarg_overrides_toml(self, tmp_path: Path) -> None:
        toml = tmp_path / "cfg.toml"
        toml.write_text(
            "[safety]\ndry_run_mode = false\n",
            encoding="utf-8",
        )
        s = load_settings(toml, safety={"dry_run_mode": True})
        assert s.safety.dry_run_mode is True

    def test_kwarg_override_no_file(self) -> None:
        s = load_settings(safety={"dry_run_mode": True, "max_actions_per_task": 5})
        assert s.safety.dry_run_mode is True
        assert s.safety.max_actions_per_task == 5

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        toml = tmp_path / "cfg.toml"
        toml.write_text("[capture]\nfps = 5\n", encoding="utf-8")
        s = load_settings(str(toml))
        assert s.capture.fps == 5


# ---------------------------------------------------------------------------
# Shipped TOML config files
# ---------------------------------------------------------------------------


class TestConfigFiles:
    def test_default_toml_loads(self) -> None:
        s = load_settings(CONFIGS_DIR / "default.toml")
        assert isinstance(s, NexusSettings)
        assert s.capture.fps == 15
        assert s.safety.dry_run_mode is False

    def test_test_toml_loads(self) -> None:
        s = load_settings(CONFIGS_DIR / "test.toml")
        assert isinstance(s, NexusSettings)
        assert s.safety.dry_run_mode is True
        assert s.safety.max_actions_per_task == 10

    def test_default_toml_all_sections(self) -> None:
        s = load_settings(CONFIGS_DIR / "default.toml")
        assert s.cloud.openai_model == "gpt-4o"
        assert s.budget.warn_at_percent == 0.8
        assert s.storage.db_path == "nexus.db"
        assert s.verification.click_method == "visual"
