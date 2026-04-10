"""
Unit tests for nexus/core/policy.py, sensitive_regions.py, screenshot_masker.py.

Coverage targets
----------------
- Every PolicyEngine rule (allow + trigger paths)
- SensitiveRegionDetector (point / rect detection)
- ScreenshotMasker (masking correctness, logging, edge cases)
- Bypass resistance (combinations that must never produce "allow")
"""
from __future__ import annotations

import numpy as np
import pytest

from nexus.core.policy import (
    RULE_DAILY_BUDGET,
    RULE_DRY_RUN,
    RULE_MAX_ACTIONS,
    RULE_MAX_DURATION,
    RULE_NATIVE_ACTION_SAFETY,
    RULE_SENSITIVE_COORDS,
    RULE_TASK_BUDGET,
    ActionContext,
    PolicyEngine,
    PolicyResult,
)
from nexus.core.screenshot_masker import MaskingResult, ScreenshotMasker
from nexus.core.sensitive_regions import SensitiveRegion, SensitiveRegionDetector
from nexus.core.settings import BudgetSettings, NexusSettings, SafetySettings
from nexus.core.types import Point, Rect

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(
    dry_run: bool = False,
    max_actions: int = 100,
    max_minutes: int = 30,
    max_task_usd: float = 1.0,
    max_daily_usd: float = 10.0,
) -> NexusSettings:
    return NexusSettings(
        safety=SafetySettings(
            dry_run_mode=dry_run,
            max_actions_per_task=max_actions,
            max_task_duration_minutes=max_minutes,
        ),
        budget=BudgetSettings(
            max_cost_per_task_usd=max_task_usd,
            max_cost_per_day_usd=max_daily_usd,
        ),
    )


def _ctx(**kwargs) -> ActionContext:  # type: ignore[type-arg]
    defaults = {
        "action_type": "click",
        "transport": "mouse",
        "is_destructive": False,
        "target_rect": None,
        "actions_so_far": 0,
        "elapsed_seconds": 0.0,
        "task_cost_usd": 0.0,
        "daily_cost_usd": 0.0,
    }
    defaults.update(kwargs)
    return ActionContext(**defaults)


def engine(
    dry_run: bool = False,
    max_actions: int = 100,
    max_minutes: int = 30,
    max_task_usd: float = 1.0,
    max_daily_usd: float = 10.0,
    detector: SensitiveRegionDetector | None = None,
) -> PolicyEngine:
    return PolicyEngine(
        _settings(dry_run, max_actions, max_minutes, max_task_usd, max_daily_usd),
        detector=detector,
    )


# ---------------------------------------------------------------------------
# PolicyResult dataclass
# ---------------------------------------------------------------------------


class TestPolicyResult:
    def test_fields(self) -> None:
        r = PolicyResult(verdict="block", rule="X", severity="block", message="m")
        assert r.verdict == "block"
        assert r.rule == "X"

    def test_allow_result_no_rule(self) -> None:
        r = PolicyResult(verdict="allow", rule=None, severity="info", message="ok")
        assert r.rule is None


# ---------------------------------------------------------------------------
# RULE_DRY_RUN
# ---------------------------------------------------------------------------


class TestRuleDryRun:
    def test_allow_when_dry_run_off(self) -> None:
        e = engine(dry_run=False)
        r = e.check_action(_ctx(is_destructive=True))
        assert r.verdict == "allow"

    def test_allow_non_destructive_in_dry_run(self) -> None:
        e = engine(dry_run=True)
        r = e.check_action(_ctx(is_destructive=False))
        assert r.verdict == "allow"

    def test_block_destructive_in_dry_run(self) -> None:
        e = engine(dry_run=True)
        r = e.check_action(_ctx(is_destructive=True))
        assert r.verdict == "block"
        assert r.rule == RULE_DRY_RUN

    def test_block_any_transport_in_dry_run(self) -> None:
        e = engine(dry_run=True)
        for transport in ("uia", "dom", "file", "mouse", "keyboard"):
            r = e.check_action(_ctx(transport=transport, is_destructive=True))
            assert r.verdict == "block", f"{transport} should be blocked"

    def test_message_contains_action_type(self) -> None:
        e = engine(dry_run=True)
        r = e.check_action(_ctx(action_type="delete_file", is_destructive=True))
        assert "delete_file" in r.message


# ---------------------------------------------------------------------------
# RULE_MAX_ACTIONS
# ---------------------------------------------------------------------------


class TestRuleMaxActions:
    def test_allow_below_cap(self) -> None:
        e = engine(max_actions=10)
        r = e.check_action(_ctx(actions_so_far=9))
        assert r.verdict == "allow"

    def test_block_at_cap(self) -> None:
        e = engine(max_actions=10)
        r = e.check_action(_ctx(actions_so_far=10))
        assert r.verdict == "block"
        assert r.rule == RULE_MAX_ACTIONS

    def test_block_above_cap(self) -> None:
        e = engine(max_actions=10)
        r = e.check_action(_ctx(actions_so_far=99))
        assert r.verdict == "block"
        assert r.rule == RULE_MAX_ACTIONS

    def test_zero_cap(self) -> None:
        e = engine(max_actions=0)
        r = e.check_action(_ctx(actions_so_far=0))
        assert r.verdict == "block"
        assert r.rule == RULE_MAX_ACTIONS

    def test_message_contains_counts(self) -> None:
        e = engine(max_actions=5)
        r = e.check_action(_ctx(actions_so_far=5))
        assert "5" in r.message


# ---------------------------------------------------------------------------
# RULE_MAX_DURATION
# ---------------------------------------------------------------------------


class TestRuleMaxDuration:
    def test_allow_within_limit(self) -> None:
        e = engine(max_minutes=30)
        r = e.check_action(_ctx(elapsed_seconds=1799.9))
        assert r.verdict == "allow"

    def test_block_at_limit(self) -> None:
        e = engine(max_minutes=30)
        r = e.check_action(_ctx(elapsed_seconds=1800.0))
        assert r.verdict == "block"
        assert r.rule == RULE_MAX_DURATION

    def test_block_over_limit(self) -> None:
        e = engine(max_minutes=1)
        r = e.check_action(_ctx(elapsed_seconds=61.0))
        assert r.verdict == "block"

    def test_zero_minutes_limit(self) -> None:
        e = engine(max_minutes=0)
        r = e.check_action(_ctx(elapsed_seconds=0.1))
        assert r.verdict == "block"
        assert r.rule == RULE_MAX_DURATION

    def test_message_contains_elapsed(self) -> None:
        e = engine(max_minutes=1)
        r = e.check_action(_ctx(elapsed_seconds=90.0))
        assert "1.5" in r.message  # 90s / 60 = 1.5 min


# ---------------------------------------------------------------------------
# RULE_TASK_BUDGET
# ---------------------------------------------------------------------------


class TestRuleTaskBudget:
    def test_allow_below_cap(self) -> None:
        e = engine(max_task_usd=1.0)
        r = e.check_action(_ctx(task_cost_usd=0.99))
        assert r.verdict == "allow"

    def test_block_at_cap(self) -> None:
        e = engine(max_task_usd=1.0)
        r = e.check_action(_ctx(task_cost_usd=1.0))
        assert r.verdict == "block"
        assert r.rule == RULE_TASK_BUDGET

    def test_block_over_cap(self) -> None:
        e = engine(max_task_usd=0.5)
        r = e.check_action(_ctx(task_cost_usd=0.51))
        assert r.verdict == "block"

    def test_zero_cap(self) -> None:
        e = engine(max_task_usd=0.0)
        r = e.check_action(_ctx(task_cost_usd=0.001))
        assert r.verdict == "block"


# ---------------------------------------------------------------------------
# RULE_DAILY_BUDGET
# ---------------------------------------------------------------------------


class TestRuleDailyBudget:
    def test_allow_below_cap(self) -> None:
        e = engine(max_daily_usd=10.0)
        r = e.check_action(_ctx(daily_cost_usd=9.99))
        assert r.verdict == "allow"

    def test_block_at_cap(self) -> None:
        e = engine(max_daily_usd=10.0)
        r = e.check_action(_ctx(daily_cost_usd=10.0))
        assert r.verdict == "block"
        assert r.rule == RULE_DAILY_BUDGET

    def test_block_over_cap(self) -> None:
        e = engine(max_daily_usd=5.0)
        r = e.check_action(_ctx(daily_cost_usd=5.01))
        assert r.verdict == "block"

    def test_task_budget_checked_before_daily(self) -> None:
        # Both exceeded — task budget rule must win (checked first)
        e = engine(max_task_usd=0.0, max_daily_usd=0.0)
        r = e.check_action(_ctx(task_cost_usd=0.01, daily_cost_usd=0.01))
        assert r.rule == RULE_TASK_BUDGET


# ---------------------------------------------------------------------------
# RULE_SENSITIVE_COORDS
# ---------------------------------------------------------------------------


class TestRuleSensitiveCoords:
    _WARN_REGION = SensitiveRegion(
        rect=Rect(100, 100, 200, 50), label="password_field", severity="warn"
    )
    _BLOCK_REGION = SensitiveRegion(
        rect=Rect(400, 200, 100, 100), label="payment_widget", severity="block"
    )

    def _det(self) -> SensitiveRegionDetector:
        return SensitiveRegionDetector([self._WARN_REGION, self._BLOCK_REGION])

    def test_allow_no_overlap(self) -> None:
        e = engine(detector=self._det())
        r = e.check_action(_ctx(target_rect=Rect(0, 0, 50, 50)))
        assert r.verdict == "allow"

    def test_warn_on_warn_region(self) -> None:
        e = engine(detector=self._det())
        # Overlaps password_field
        r = e.check_action(_ctx(target_rect=Rect(150, 110, 10, 10)))
        assert r.verdict == "warn"
        assert r.rule == RULE_SENSITIVE_COORDS
        assert "password_field" in r.message

    def test_block_on_block_region(self) -> None:
        e = engine(detector=self._det())
        # Overlaps payment_widget
        r = e.check_action(_ctx(target_rect=Rect(450, 250, 10, 10)))
        assert r.verdict == "block"
        assert r.rule == RULE_SENSITIVE_COORDS
        assert "payment_widget" in r.message

    def test_block_wins_over_warn_when_both_overlap(self) -> None:
        e = engine(detector=self._det())
        # Large rect overlapping both regions
        r = e.check_action(_ctx(target_rect=Rect(0, 0, 600, 400)))
        assert r.verdict == "block"

    def test_no_target_rect_skips_rule(self) -> None:
        e = engine(detector=self._det())
        r = e.check_action(_ctx(target_rect=None))
        assert r.verdict == "allow"

    def test_no_detector_skips_rule(self) -> None:
        e = engine(detector=None)
        r = e.check_action(_ctx(target_rect=Rect(0, 0, 100, 100)))
        assert r.verdict == "allow"


# ---------------------------------------------------------------------------
# RULE_NATIVE_ACTION_SAFETY
# ---------------------------------------------------------------------------


class TestRuleNativeActionSafety:
    def test_allow_non_native_non_destructive(self) -> None:
        e = engine()
        r = e.check_action(_ctx(transport="mouse", is_destructive=False))
        assert r.verdict == "allow"

    def test_allow_native_non_destructive(self) -> None:
        for transport in ("uia", "dom", "file"):
            e = engine()
            r = e.check_action(_ctx(transport=transport, is_destructive=False))
            assert r.verdict == "allow", f"{transport} non-destructive should allow"

    def test_warn_native_destructive_outside_dry_run(self) -> None:
        for transport in ("uia", "dom", "file"):
            e = engine(dry_run=False)
            r = e.check_action(_ctx(transport=transport, is_destructive=True))
            assert r.verdict == "warn", f"{transport} destructive should warn"
            assert r.rule == RULE_NATIVE_ACTION_SAFETY

    def test_block_native_destructive_in_dry_run(self) -> None:
        # RULE_DRY_RUN is checked first → block triggered by DRY_RUN
        # But the net effect must be block regardless of which rule fires.
        for transport in ("uia", "dom", "file"):
            e = engine(dry_run=True)
            r = e.check_action(_ctx(transport=transport, is_destructive=True))
            assert r.verdict == "block", (
                f"{transport} destructive in dry-run must be blocked"
            )

    def test_non_native_fallback_no_native_safety_warn(self) -> None:
        e = engine(dry_run=False)
        r = e.check_action(_ctx(transport="keyboard", is_destructive=True))
        # Not native → RULE_NATIVE_ACTION_SAFETY does not apply
        assert r.rule != RULE_NATIVE_ACTION_SAFETY

    def test_message_contains_transport_and_action(self) -> None:
        e = engine(dry_run=False)
        r = e.check_action(
            _ctx(transport="uia", is_destructive=True, action_type="delete_row")
        )
        assert "uia" in r.message
        assert "delete_row" in r.message


# ---------------------------------------------------------------------------
# Rule priority order
# ---------------------------------------------------------------------------


class TestRulePriority:
    def test_dry_run_before_max_actions(self) -> None:
        # Both dry-run + max_actions exceeded → dry_run must win
        e = engine(dry_run=True, max_actions=0)
        r = e.check_action(_ctx(is_destructive=True, actions_so_far=0))
        assert r.rule == RULE_DRY_RUN

    def test_max_actions_before_budget(self) -> None:
        e = engine(max_actions=0, max_task_usd=0.0)
        r = e.check_action(_ctx(actions_so_far=0, task_cost_usd=1.0))
        assert r.rule == RULE_MAX_ACTIONS

    def test_max_duration_before_budget(self) -> None:
        e = engine(max_minutes=0, max_task_usd=0.0)
        r = e.check_action(
            _ctx(elapsed_seconds=1.0, task_cost_usd=1.0)
        )
        assert r.rule == RULE_MAX_DURATION

    def test_clean_action_returns_allow(self) -> None:
        e = engine()
        r = e.check_action(_ctx())
        assert r.verdict == "allow"
        assert r.rule is None


# ---------------------------------------------------------------------------
# Bypass resistance
# ---------------------------------------------------------------------------


class TestBypassResistance:
    """Verify that dangerous combinations always result in a non-allow verdict."""

    def test_native_destructive_dry_run_always_blocked(self) -> None:
        e = engine(dry_run=True)
        for transport in ("uia", "dom", "file"):
            r = e.check_action(_ctx(transport=transport, is_destructive=True))
            assert r.verdict in ("block", "abort"), (
                f"transport={transport} must be blocked in dry-run"
            )

    def test_budget_exhausted_always_blocked(self) -> None:
        e = engine(max_task_usd=0.0)
        r = e.check_action(_ctx(task_cost_usd=0.001))
        assert r.verdict in ("block", "abort")

    def test_sensitive_block_region_always_blocked(self) -> None:
        det = SensitiveRegionDetector([
            SensitiveRegion(Rect(0, 0, 100, 100), "secret", severity="block")
        ])
        e = engine(detector=det)
        r = e.check_action(_ctx(target_rect=Rect(10, 10, 5, 5)))
        assert r.verdict in ("block", "abort")

    def test_daily_budget_exhausted_always_blocked(self) -> None:
        e = engine(max_daily_usd=0.0)
        r = e.check_action(_ctx(daily_cost_usd=0.001))
        assert r.verdict in ("block", "abort")

    def test_action_cap_always_blocked(self) -> None:
        e = engine(max_actions=0)
        r = e.check_action(_ctx(actions_so_far=0))
        assert r.verdict in ("block", "abort")


# ---------------------------------------------------------------------------
# SensitiveRegionDetector
# ---------------------------------------------------------------------------


class TestSensitiveRegionDetector:
    def test_detect_point_inside(self) -> None:
        det = SensitiveRegionDetector([
            SensitiveRegion(Rect(0, 0, 100, 100), "area")
        ])
        assert len(det.detect(Point(50, 50))) == 1

    def test_detect_point_outside(self) -> None:
        det = SensitiveRegionDetector([
            SensitiveRegion(Rect(0, 0, 100, 100), "area")
        ])
        assert det.detect(Point(200, 200)) == []

    def test_detect_point_boundary(self) -> None:
        det = SensitiveRegionDetector([
            SensitiveRegion(Rect(0, 0, 100, 100), "area")
        ])
        assert len(det.detect(Point(0, 0))) == 1
        assert len(det.detect(Point(100, 100))) == 1

    def test_detect_rect_overlap(self) -> None:
        det = SensitiveRegionDetector([
            SensitiveRegion(Rect(50, 50, 100, 100), "area")
        ])
        assert len(det.detect_rect(Rect(0, 0, 100, 100))) == 1

    def test_detect_rect_no_overlap(self) -> None:
        det = SensitiveRegionDetector([
            SensitiveRegion(Rect(200, 200, 50, 50), "area")
        ])
        assert det.detect_rect(Rect(0, 0, 100, 100)) == []

    def test_is_blocked_true(self) -> None:
        det = SensitiveRegionDetector([
            SensitiveRegion(Rect(0, 0, 100, 100), "secret", severity="block")
        ])
        assert det.is_blocked(Point(50, 50)) is True

    def test_is_blocked_false_for_warn(self) -> None:
        det = SensitiveRegionDetector([
            SensitiveRegion(Rect(0, 0, 100, 100), "hint", severity="warn")
        ])
        assert det.is_blocked(Point(50, 50)) is False

    def test_is_blocked_rect(self) -> None:
        det = SensitiveRegionDetector([
            SensitiveRegion(Rect(0, 0, 100, 100), "secret", severity="block")
        ])
        assert det.is_blocked_rect(Rect(50, 50, 10, 10)) is True

    def test_add_and_clear(self) -> None:
        det = SensitiveRegionDetector()
        assert det.regions == []
        det.add(SensitiveRegion(Rect(0, 0, 10, 10), "x"))
        assert len(det.regions) == 1
        det.clear()
        assert det.regions == []

    def test_multiple_regions_returned(self) -> None:
        det = SensitiveRegionDetector([
            SensitiveRegion(Rect(0, 0, 100, 100), "a"),
            SensitiveRegion(Rect(0, 0, 200, 200), "b"),
        ])
        hits = det.detect(Point(50, 50))
        assert len(hits) == 2

    def test_empty_detector_returns_empty(self) -> None:
        det = SensitiveRegionDetector()
        assert det.detect(Point(0, 0)) == []
        assert det.detect_rect(Rect(0, 0, 100, 100)) == []


# ---------------------------------------------------------------------------
# ScreenshotMasker
# ---------------------------------------------------------------------------


class TestScreenshotMasker:
    def _img(self, h: int = 100, w: int = 100, fill: int = 255) -> np.ndarray:
        return np.full((h, w, 3), fill, dtype=np.uint8)

    def test_mask_blacks_out_region(self) -> None:
        masker = ScreenshotMasker()
        img = self._img()
        out, _ = masker.mask(img, [Rect(0, 0, 50, 50)])
        assert np.all(out[0:50, 0:50] == 0)

    def test_unmasked_area_unchanged(self) -> None:
        masker = ScreenshotMasker()
        img = self._img()
        out, _ = masker.mask(img, [Rect(0, 0, 50, 50)])
        assert np.all(out[50:, 50:] == 255)

    def test_original_unchanged(self) -> None:
        masker = ScreenshotMasker()
        img = self._img()
        masker.mask(img, [Rect(0, 0, 50, 50)])
        assert np.all(img == 255)  # copy, not in-place

    def test_returns_masking_result(self) -> None:
        masker = ScreenshotMasker()
        img = self._img()
        _, result = masker.mask(img, [Rect(0, 0, 10, 10)])
        assert isinstance(result, MaskingResult)
        assert result.regions_masked == 1
        assert result.pixels_masked == 100

    def test_pixels_masked_counted_correctly(self) -> None:
        masker = ScreenshotMasker()
        img = self._img(200, 200)
        _, result = masker.mask(img, [Rect(0, 0, 20, 30)])
        assert result.pixels_masked == 600

    def test_multiple_regions(self) -> None:
        masker = ScreenshotMasker()
        img = self._img()
        out, result = masker.mask(img, [Rect(0, 0, 10, 10), Rect(50, 50, 10, 10)])
        assert result.regions_masked == 2
        assert np.all(out[0:10, 0:10] == 0)
        assert np.all(out[50:60, 50:60] == 0)

    def test_region_outside_image_clipped(self) -> None:
        masker = ScreenshotMasker()
        img = self._img(50, 50)
        out, result = masker.mask(img, [Rect(40, 40, 100, 100)])
        # Only 10×10 pixels are inside the 50×50 image
        assert result.pixels_masked == 100

    def test_region_fully_outside_image(self) -> None:
        masker = ScreenshotMasker()
        img = self._img(50, 50)
        out, result = masker.mask(img, [Rect(200, 200, 50, 50)])
        assert result.pixels_masked == 0
        assert np.all(out == 255)

    def test_custom_fill_value(self) -> None:
        masker = ScreenshotMasker(fill_value=128)
        img = self._img()
        out, _ = masker.mask(img, [Rect(0, 0, 10, 10)])
        assert np.all(out[0:10, 0:10] == 128)

    def test_wrong_dtype_raises(self) -> None:
        masker = ScreenshotMasker()
        img = np.zeros((50, 50, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            masker.mask(img, [])

    def test_empty_regions_returns_copy(self) -> None:
        masker = ScreenshotMasker()
        img = self._img()
        out, result = masker.mask(img, [])
        assert np.all(out == 255)
        assert result.pixels_masked == 0

    def test_grayscale_image(self) -> None:
        masker = ScreenshotMasker()
        img = np.full((50, 50), 200, dtype=np.uint8)
        out, _ = masker.mask(img, [Rect(10, 10, 20, 20)])
        assert np.all(out[10:30, 10:30] == 0)
        assert np.all(out[0:10, 0:10] == 200)

    def test_labels_stored_in_result(self) -> None:
        masker = ScreenshotMasker()
        img = self._img()
        _, result = masker.mask(
            img, [Rect(0, 0, 10, 10)], labels=["password_field"]
        )
        assert "password_field" in result.region_labels

    def test_mask_from_sensitive_regions(self) -> None:
        masker = ScreenshotMasker()
        img = self._img()
        regions = [SensitiveRegion(Rect(0, 0, 20, 20), "login_form", severity="block")]
        out, result = masker.mask_from_sensitive(img, regions)
        assert np.all(out[0:20, 0:20] == 0)
        assert "login_form" in result.region_labels
