"""
tests/adversarial/adv_005_anti_loop.py
Adversarial Test 005 — Anti-loop: 3 identical actions → cloud escalation flag

Scenario:
  The decision history contains _ANTI_LOOP_WINDOW (3) identical
  (action_type, target_description) records.

Success criteria:
  - _is_anti_loop(history) returns True.
  - With 2 identical records: _is_anti_loop returns False (below threshold).
  - Mixed records: _is_anti_loop returns False.

Tests exercise the internal helper directly (white-box) because the loop
detection is a pure function with no I/O.
"""
from __future__ import annotations

import pytest

from nexus.cloud.prompt_builder import ActionRecord
from nexus.decision.engine import _is_anti_loop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rec(action_type: str, target: str) -> ActionRecord:
    return ActionRecord(
        action_type=action_type,
        target_description=target,
        outcome="success",
        timestamp="2026-04-09T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.adversarial
class TestAntiLoop:
    """ADV-005: Anti-loop detection triggers cloud escalation after 3 identical actions."""

    def test_three_identical_records_triggers_loop(self):
        """3 identical (type, target) → _is_anti_loop returns True."""
        history = [
            _rec("click", "Kaydet butonu"),
            _rec("click", "Kaydet butonu"),
            _rec("click", "Kaydet butonu"),
        ]
        assert _is_anti_loop(history) is True, (
            "3 identical actions must trigger anti-loop detection"
        )

    def test_two_identical_below_threshold(self):
        """2 identical records → below _ANTI_LOOP_WINDOW (3) → False."""
        history = [
            _rec("click", "Kaydet butonu"),
            _rec("click", "Kaydet butonu"),
        ]
        assert _is_anti_loop(history) is False

    def test_mixed_records_no_loop(self):
        """Different action_types → no loop."""
        history = [
            _rec("click", "Kaydet"),
            _rec("type", "Kaydet"),
            _rec("scroll", "Kaydet"),
        ]
        assert _is_anti_loop(history) is False

    def test_different_targets_no_loop(self):
        """Same action_type but different targets → no loop."""
        history = [
            _rec("click", "Kaydet"),
            _rec("click", "İptal"),
            _rec("click", "Kapat"),
        ]
        assert _is_anti_loop(history) is False

    def test_four_identical_still_triggers(self):
        """4 identical records (> threshold) → True."""
        history = [_rec("click", "btn") for _ in range(4)]
        assert _is_anti_loop(history) is True

    def test_empty_history_no_loop(self):
        """Empty history → False (no actions yet)."""
        assert _is_anti_loop([]) is False

    def test_loop_in_tail_triggers(self):
        """
        History has diverse start but last 3 are identical →
        loop is detected in the tail.
        """
        history = [
            _rec("type", "İsim alanı"),
            _rec("scroll", "Liste"),
            _rec("click", "Kaydet butonu"),
            _rec("click", "Kaydet butonu"),
            _rec("click", "Kaydet butonu"),
        ]
        assert _is_anti_loop(history) is True
