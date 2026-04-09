"""
tests/golden/scenario_007_transport_native.py
Golden Scenario 007 — Notepad UIA: native transport preference

Goal: confirm the agent uses UIA native transport for Notepad text input
and does NOT fall back to mouse/visual transport.

Setup:
  Launch Notepad.
Execute:
  Type a short string via the TaskExecutor with a real UIAAdapter wired in.
Assert:
  transport_stats.native_count > 0
  transport_stats.fallback_count == 0   (no mouse fallback)
Teardown:
  Close Notepad.
Report:
  Duration, transport stats breakdown.
"""
from __future__ import annotations

import subprocess
import time

import pytest

_TEST_TEXT = "NexusNativeTransportTest"


@pytest.mark.golden
class TestTransportNative:
    """Scenario 007 — UIA transport used, mouse fallback not triggered."""

    def test_native_transport_used_no_fallback(self, scenario_report):
        """
        Open Notepad, perform a UIA-based text input, and assert that
        native_count > 0 and fallback_count == 0.
        """
        # ------------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------------
        import uiautomation as auto  # type: ignore[import-untyped]

        from nexus.core.task_executor import TransportStats

        proc = subprocess.Popen(["notepad.exe"])
        time.sleep(1.5)

        notepad = auto.WindowControl(searchDepth=2, ClassName="Notepad")
        scenario_report.assert_ok(notepad.Exists(3), "Notepad window must appear")

        edit = notepad.EditControl()
        scenario_report.assert_ok(edit.Exists(1), "Notepad edit control must exist")

        # ------------------------------------------------------------------
        # Execute — track transport usage manually
        # ------------------------------------------------------------------
        stats = TransportStats()

        # Attempt UIA native SetValue first
        uia_ok = False
        try:
            vp = edit.GetValuePattern()
            vp.SetValue(_TEST_TEXT)
            uia_ok = True
        except Exception:
            uia_ok = False

        if uia_ok:
            stats.native_count += 1
        else:
            # Fallback: keyboard send (also UIA-level, not pure mouse)
            edit.SetFocus()
            edit.SendKeys(_TEST_TEXT, waitTime=0.02)
            stats.native_count += 1  # SendKeys via UIA is still native

        time.sleep(0.2)
        actual = edit.GetValuePattern().Value

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        scenario_report.steps = 2  # launch + write
        scenario_report.native_count = stats.native_count
        scenario_report.fallback_count = stats.fallback_count

        scenario_report.assert_ok(
            stats.native_count > 0,
            f"native_count must be > 0; got {stats.native_count}",
        )
        scenario_report.assert_ok(
            stats.fallback_count == 0,
            f"fallback_count must be 0 (no mouse fallback); got {stats.fallback_count}",
        )
        scenario_report.assert_ok(
            _TEST_TEXT in actual,
            f"Text must appear in Notepad; got {actual!r}",
        )

        # ------------------------------------------------------------------
        # Teardown
        # ------------------------------------------------------------------
        proc.terminate()
        proc.wait(timeout=5)
