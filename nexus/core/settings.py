"""
nexus/core/settings.py
Single source of truth for all Nexus Agent configuration.

Environment variable overrides use the NEXUS_ prefix.
Nested fields use __ as delimiter:
    NEXUS_CAPTURE__FPS=30
    NEXUS_SAFETY__DRY_RUN_MODE=true

Use load_settings() to build a NexusSettings instance from a TOML file.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

VerificationMethod = Literal[
    "none",
    "visual",
    "semantic",
    "source",
    "visual+semantic",
    "source+visual",
    "combined",
    "combined+preconfirm",
    "source_level",
]

# ---------------------------------------------------------------------------
# Sub-settings
# ---------------------------------------------------------------------------


class CaptureSettings(BaseModel):
    fps: int = 15
    max_frame_buffer: int = 10
    stabilization_timeout_ms: int = 3000
    stabilization_poll_ms: int = 100
    dirty_region_block_size: int = 32
    dirty_region_full_refresh_ratio: float = 0.7

    @field_validator("fps")
    @classmethod
    def fps_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("fps must be a positive integer")
        return v

    @field_validator("dirty_region_full_refresh_ratio")
    @classmethod
    def ratio_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("dirty_region_full_refresh_ratio must be in [0, 1]")
        return v


class SourceSettings(BaseModel):
    uia_enabled: bool = True
    uia_timeout_ms: int = 2000
    dom_enabled: bool = True
    dom_timeout_ms: int = 2000
    dom_debug_port: int = 9222
    file_direct_enabled: bool = True
    visual_fallback_always: bool = False


class TransportSettings(BaseModel):
    prefer_native_action: bool = True
    uia_action_timeout_ms: int = 2000
    dom_action_timeout_ms: int = 2000
    fallback_to_mouse: bool = True
    key_press_delay_ms: int = 20          # inter-keystroke delay in ms
    type_verify_max_retries: int = 3      # OCR-verify retry attempts after mistype


class PerceptionSettings(BaseModel):
    ocr_confidence_threshold: float = 0.7
    locator_confidence_threshold: float = 0.6
    ambiguity_local_threshold: float = 0.4
    ambiguity_cloud_threshold: float = 0.7
    perception_cache_ttl_ms: int = 200
    max_graph_nodes: int = 500

    @field_validator(
        "ocr_confidence_threshold",
        "locator_confidence_threshold",
        "ambiguity_local_threshold",
        "ambiguity_cloud_threshold",
    )
    @classmethod
    def threshold_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return v


class CloudSettings(BaseModel):
    primary_provider: Literal["openai", "anthropic"] = "openai"
    openai_model: str = "gpt-4o"
    openai_fallback_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-sonnet-4-5"
    anthropic_fallback_model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 1000
    timeout_seconds: int = 30
    max_retries: int = 3


class ModelPricing(BaseModel):
    """USD cost per 1,000 tokens."""

    input_per_1k: float
    output_per_1k: float


class BudgetSettings(BaseModel):
    max_cost_per_task_usd: float = 1.0
    max_cost_per_day_usd: float = 10.0
    warn_at_percent: float = 0.8

    # Per-model pricing (USD / 1k tokens) — 2024-Q4 rates
    gpt4o_input_per_1k: float = 0.005
    gpt4o_output_per_1k: float = 0.015
    gpt4o_mini_input_per_1k: float = 0.00015
    gpt4o_mini_output_per_1k: float = 0.0006
    claude_sonnet_input_per_1k: float = 0.003
    claude_sonnet_output_per_1k: float = 0.015
    claude_haiku_input_per_1k: float = 0.00025
    claude_haiku_output_per_1k: float = 0.00125

    @field_validator("warn_at_percent")
    @classmethod
    def warn_unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("warn_at_percent must be in [0, 1]")
        return v

    def pricing_for(self, model: str) -> ModelPricing:
        """Return per-token pricing for a known model identifier."""
        pricing_map: dict[str, ModelPricing] = {
            "gpt-4o": ModelPricing(
                input_per_1k=self.gpt4o_input_per_1k,
                output_per_1k=self.gpt4o_output_per_1k,
            ),
            "gpt-4o-mini": ModelPricing(
                input_per_1k=self.gpt4o_mini_input_per_1k,
                output_per_1k=self.gpt4o_mini_output_per_1k,
            ),
            "claude-3-5-sonnet-20241022": ModelPricing(
                input_per_1k=self.claude_sonnet_input_per_1k,
                output_per_1k=self.claude_sonnet_output_per_1k,
            ),
            "claude-sonnet-4-5": ModelPricing(
                input_per_1k=self.claude_sonnet_input_per_1k,
                output_per_1k=self.claude_sonnet_output_per_1k,
            ),
            "claude-3-haiku-20240307": ModelPricing(
                input_per_1k=self.claude_haiku_input_per_1k,
                output_per_1k=self.claude_haiku_output_per_1k,
            ),
            "claude-haiku-4-5-20251001": ModelPricing(
                input_per_1k=self.claude_haiku_input_per_1k,
                output_per_1k=self.claude_haiku_output_per_1k,
            ),
        }
        if model not in pricing_map:
            raise KeyError(f"No pricing defined for model: {model!r}")
        return pricing_map[model]


class StorageSettings(BaseModel):
    db_path: str = "nexus.db"
    log_dir: str = "logs"
    max_db_size_mb: int = 500
    memory_max_size_mb: int = 256
    memory_evict_after_days: int = 30


class SafetySettings(BaseModel):
    max_actions_per_task: int = 100
    max_task_duration_minutes: int = 30
    sensitive_masking_enabled: bool = True
    dry_run_mode: bool = False


class VerificationSettings(BaseModel):
    # Core interaction
    click_method: VerificationMethod = "visual"
    type_method: VerificationMethod = "visual"
    select_method: VerificationMethod = "source"
    drag_method: VerificationMethod = "visual"
    scroll_method: VerificationMethod = "none"
    hover_method: VerificationMethod = "none"
    keyboard_shortcut_method: VerificationMethod = "visual"
    copy_paste_method: VerificationMethod = "visual"

    # Field / form
    field_replace_method: VerificationMethod = "visual+semantic"
    form_submit_method: VerificationMethod = "visual+semantic"

    # Spreadsheet
    row_write_method: VerificationMethod = "combined"
    sheet_write_method: VerificationMethod = "source_level"
    cell_edit_method: VerificationMethod = "source+visual"

    # Destructive
    destructive_method: VerificationMethod = "combined+preconfirm"
    delete_method: VerificationMethod = "combined+preconfirm"
    overwrite_method: VerificationMethod = "combined+preconfirm"

    # File operations
    file_open_method: VerificationMethod = "source"
    file_save_method: VerificationMethod = "source+visual"

    # Navigation
    navigate_method: VerificationMethod = "visual"

    # Macro
    macroaction_method: VerificationMethod = "combined"


# ---------------------------------------------------------------------------
# HITL settings
# ---------------------------------------------------------------------------


class HITLSettings(BaseModel):
    """Configuration for Human-in-the-Loop prompts (ADR-008)."""

    timeout_s: float = 300.0
    """Seconds to wait for a human response before using the default option."""

    headless: bool = False
    """When True skip all terminal prompts and return the default immediately."""

    default_action: str = "skip"
    """
    The default action string when a prompt times out or headless=True.
    Must match one of the options provided in the HITLRequest, or be a
    free-form string if no options list is given.
    """


# ---------------------------------------------------------------------------
# Root settings (env prefix: NEXUS_)
# ---------------------------------------------------------------------------


class NexusSettings(BaseSettings):
    """
    Root configuration object for Nexus Agent.

    Build with load_settings() or instantiate directly for tests.
    Env vars override TOML values; TOML values override Python defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEXUS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    source: SourceSettings = Field(default_factory=SourceSettings)
    transport: TransportSettings = Field(default_factory=TransportSettings)
    perception: PerceptionSettings = Field(default_factory=PerceptionSettings)
    cloud: CloudSettings = Field(default_factory=CloudSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    hitl: HITLSettings = Field(default_factory=lambda: HITLSettings())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_settings(
    config_file: str | Path | None = None,
    **overrides: Any,
) -> NexusSettings:
    """
    Build NexusSettings from an optional TOML file plus keyword overrides.

    Priority (highest → lowest):
      1. ``**overrides`` kwargs
      2. TOML file values
      3. NEXUS_* environment variables
      4. Python defaults
    """
    data: dict[str, Any] = {}
    if config_file is not None:
        with Path(config_file).open("rb") as fh:
            data = tomllib.load(fh)
    data.update(overrides)
    return NexusSettings(**data)
