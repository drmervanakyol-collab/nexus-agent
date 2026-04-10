"""
tests/adversarial/adv_007_sensitive_region.py
Adversarial Test 007 — Click on password field → PolicyEngine block + masking

Scenario:
  An action targets coordinates that fall inside a "block"-severity
  sensitive region (a simulated password field).

Success criteria:
  - PolicyEngine.check_action() returns verdict == "block",
    rule == RULE_SENSITIVE_COORDS.
  - SensitiveRegionDetector.is_blocked(point) returns True.
  - ScreenshotMasker.mask() applies masking to the sensitive region.
  - Warn-only sensitive region → verdict == "warn" (not "block").

All I/O is injected — no real OS calls.
"""
from __future__ import annotations

import numpy as np
import pytest

from nexus.core.policy import (
    RULE_SENSITIVE_COORDS,
    ActionContext,
    PolicyEngine,
)
from nexus.core.screenshot_masker import ScreenshotMasker
from nexus.core.sensitive_regions import SensitiveRegion, SensitiveRegionDetector
from nexus.core.settings import NexusSettings
from nexus.core.types import Point, Rect

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PASSWORD_RECT = Rect(x=100, y=300, width=300, height=40)
_PASSWORD_POINT = Point(x=200, y=320)   # inside the rect
_SAFE_POINT = Point(x=900, y=500)       # outside


def _settings() -> NexusSettings:
    return NexusSettings()


def _detector_with_block() -> SensitiveRegionDetector:
    return SensitiveRegionDetector(
        regions=[
            SensitiveRegion(
                rect=_PASSWORD_RECT,
                label="password_field",
                severity="block",
            )
        ]
    )


def _detector_with_warn() -> SensitiveRegionDetector:
    return SensitiveRegionDetector(
        regions=[
            SensitiveRegion(
                rect=_PASSWORD_RECT,
                label="sensitive_info",
                severity="warn",
            )
        ]
    )


def _ctx(target_rect: Rect) -> ActionContext:
    return ActionContext(
        action_type="click",
        transport="mouse",
        target_rect=target_rect,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.adversarial
class TestSensitiveRegion:
    """ADV-007: Actions targeting sensitive regions are blocked or warned."""

    def test_block_severity_blocks_action(self):
        """
        Password field marked 'block' → verdict 'block', rule RULE_SENSITIVE_COORDS.
        """
        engine = PolicyEngine(_settings(), detector=_detector_with_block())
        result = engine.check_action(_ctx(_PASSWORD_RECT))

        assert result.verdict == "block", (
            f"Expected 'block'; got {result.verdict!r}"
        )
        assert result.rule == RULE_SENSITIVE_COORDS, (
            f"Expected RULE_SENSITIVE_COORDS; got {result.rule!r}"
        )

    def test_warn_severity_warns_action(self):
        """Sensitive region marked 'warn' → verdict 'warn', not 'block'."""
        engine = PolicyEngine(_settings(), detector=_detector_with_warn())
        result = engine.check_action(_ctx(_PASSWORD_RECT))

        assert result.verdict == "warn", (
            f"Expected 'warn'; got {result.verdict!r}"
        )
        assert result.rule == RULE_SENSITIVE_COORDS

    def test_safe_coords_allowed(self):
        """Action targeting safe coordinates → verdict 'allow'."""
        engine = PolicyEngine(_settings(), detector=_detector_with_block())
        safe_rect = Rect(x=900, y=500, width=100, height=30)
        result = engine.check_action(_ctx(safe_rect))

        assert result.verdict == "allow", (
            f"Expected 'allow' for safe coords; got {result.verdict!r}"
        )

    def test_detector_is_blocked_point(self):
        """SensitiveRegionDetector.is_blocked() returns True for the password rect."""
        detector = _detector_with_block()
        assert detector.is_blocked(_PASSWORD_POINT) is True

    def test_detector_safe_point_not_blocked(self):
        """is_blocked() returns False for coordinates outside all regions."""
        detector = _detector_with_block()
        assert detector.is_blocked(_SAFE_POINT) is False

    def test_masker_masks_sensitive_region(self):
        """
        ScreenshotMasker.mask() with the password rect as a sensitive region
        should modify pixels in that area (mask == 0 for the region).
        """
        img = np.ones((600, 1200, 3), dtype=np.uint8) * 200

        masker = ScreenshotMasker()
        masked_img, mask_result = masker.mask(img, [_PASSWORD_RECT])

        # The masked region should be all-zero (blacked out)
        region = masked_img[
            _PASSWORD_RECT.y : _PASSWORD_RECT.y + _PASSWORD_RECT.height,
            _PASSWORD_RECT.x : _PASSWORD_RECT.x + _PASSWORD_RECT.width,
        ]
        assert region.max() == 0, (
            f"Masked region must be blacked out; max pixel value = {region.max()}"
        )
        assert mask_result.regions_masked >= 1

    def test_no_detector_skips_sensitive_check(self):
        """With no detector, RULE_SENSITIVE_COORDS is skipped — action allowed."""
        engine = PolicyEngine(_settings(), detector=None)
        result = engine.check_action(_ctx(_PASSWORD_RECT))
        assert result.verdict == "allow"
