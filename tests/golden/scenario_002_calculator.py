"""
tests/golden/scenario_002_calculator.py
Golden Scenario 002 — Calculator: 123 + 456 = 579

Setup:
  Launch Windows Calculator in Standard mode.
Execute:
  Click 1, 2, 3, +, 4, 5, 6, = via UIA button controls.
Assert:
  Display shows "579".
Teardown:
  Close Calculator.
Report:
  Duration, transport stats (all native UIA clicks).
"""
from __future__ import annotations

import subprocess
import time

import pytest

_EXPECTED = "579"
_BUTTONS = ["1", "2", "3", "Plus", "4", "5", "6", "Equals"]


@pytest.mark.golden
class TestCalculator:
    """Scenario 002 — Calculator computes 123 + 456 = 579."""

    def test_addition_result(self, scenario_report):
        """
        Open Calculator, enter '123 + 456 =', and assert the result is 579.
        """
        # ------------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------------
        auto = pytest.importorskip("uiautomation")

        proc = subprocess.Popen(["calc.exe"])
        time.sleep(1.5)

        calc = auto.WindowControl(searchDepth=2, Name="Hesap Makinesi")
        if not calc.Exists(3):
            calc = auto.WindowControl(searchDepth=2, ClassName="ApplicationFrameWindow")

        scenario_report.assert_ok(calc.Exists(3), "Calculator window must appear")

        # ------------------------------------------------------------------
        # Execute — click each button by AutomationId or Name
        # ------------------------------------------------------------------
        native_clicks = 0
        for btn_name in _BUTTONS:
            btn = calc.ButtonControl(AutomationId=btn_name)
            if not btn.Exists(1):
                btn = calc.ButtonControl(Name=btn_name)
            scenario_report.assert_ok(
                btn.Exists(1), f"Button {btn_name!r} must be found"
            )
            btn.Click(waitTime=0.1)
            native_clicks += 1

        time.sleep(0.2)

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        display = calc.TextControl(AutomationId="CalculatorResults")
        if not display.Exists(1):
            display = calc.TextControl(AutomationId="Result")

        result_text = display.Name if display.Exists(1) else ""

        scenario_report.steps = len(_BUTTONS)
        scenario_report.native_count = native_clicks

        scenario_report.assert_ok(
            _EXPECTED in result_text,
            f"Calculator must show 579; got: {result_text!r}",
        )

        # ------------------------------------------------------------------
        # Teardown
        # ------------------------------------------------------------------
        proc.terminate()
        proc.wait(timeout=5)
