"""
tests/golden/scenario_001_notepad_turkish.py
Golden Scenario 001 — Notepad: Turkish character input

Setup:
  Launch Notepad via subprocess.
Execute:
  Type "İstanbul Şişli Ğüçlü" using the keyboard transport.
Assert:
  The full Turkish string appears in the Notepad window text.
Teardown:
  Close Notepad without saving.
Report:
  Duration, transport stats.
"""
from __future__ import annotations

import subprocess
import time

import pytest

TURKISH_TEXT = "İstanbul Şişli Ğüçlü"


@pytest.mark.golden
class TestNotepadTurkishInput:
    """Scenario 001 — Notepad correctly receives Turkish characters."""

    def test_turkish_text_typed_and_visible(self, scenario_report):
        """
        Launch Notepad, type Turkish text via UIA/keyboard transport,
        then read it back and confirm correctness.
        """
        # ------------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------------
        auto = pytest.importorskip("uiautomation")

        proc = subprocess.Popen(["notepad.exe"])
        time.sleep(1.5)  # allow Notepad window to appear

        notepad = auto.WindowControl(searchDepth=2, Name="Başlıksız - Not Defteri")
        if not notepad.Exists(2):
            notepad = auto.WindowControl(searchDepth=2, ClassName="Notepad")

        scenario_report.assert_ok(notepad.Exists(2), "Notepad window must appear")

        edit = notepad.EditControl()
        scenario_report.assert_ok(edit.Exists(1), "Notepad edit control must exist")

        # ------------------------------------------------------------------
        # Execute
        # ------------------------------------------------------------------
        edit.SetFocus()
        edit.SendKeys(TURKISH_TEXT, waitTime=0.05)
        time.sleep(0.3)

        actual_text = edit.GetValuePattern().Value

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        scenario_report.steps = 2  # launch + type
        scenario_report.native_count = 1  # UIA keyboard transport

        scenario_report.assert_ok(
            TURKISH_TEXT in actual_text,
            f"Turkish text must appear in Notepad; got: {actual_text!r}",
        )

        # ------------------------------------------------------------------
        # Teardown
        # ------------------------------------------------------------------
        proc.terminate()
        proc.wait(timeout=5)
