"""
tests/golden/scenario_003_chrome_search.py
Golden Scenario 003 — Chrome: google.com search

Setup:
  Open Chrome and navigate to https://www.google.com.
Execute:
  Type "Nexus Agent automation" into the search box and submit.
Assert:
  Page title is non-empty after results load.
Teardown:
  Close Chrome.
Report:
  Duration, transport stats.
"""
from __future__ import annotations

import subprocess
import time

import pytest

_SEARCH_QUERY = "Nexus Agent automation"
_URL = "https://www.google.com"


@pytest.mark.golden
class TestChromeSearch:
    """Scenario 003 — Chrome performs a Google search and shows results."""

    def test_search_result_title_not_empty(self, scenario_report):
        """
        Navigate to google.com, submit a search query, and assert that
        the resulting page title is non-empty.
        """
        # ------------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------------
        auto = pytest.importorskip("uiautomation")

        proc = subprocess.Popen(
            ["chrome.exe", "--new-window", _URL],
            shell=False,
        )
        time.sleep(3.0)  # allow Chrome + page to load

        chrome = auto.WindowControl(searchDepth=2, ClassName="Chrome_WidgetWin_1")
        scenario_report.assert_ok(chrome.Exists(5), "Chrome window must appear")

        # ------------------------------------------------------------------
        # Execute — locate the omnibox / search box and type query
        # ------------------------------------------------------------------
        omnibox = chrome.EditControl(Name="Adres ve arama çubuğu")
        if not omnibox.Exists(2):
            omnibox = chrome.EditControl(Name="Address and search bar")
        if not omnibox.Exists(2):
            omnibox = chrome.EditControl(AutomationId="omnibox")

        scenario_report.assert_ok(omnibox.Exists(2), "Chrome omnibox must be found")

        omnibox.SetFocus()
        time.sleep(0.3)
        omnibox.SendKeys(_SEARCH_QUERY + "{Enter}", waitTime=0.05)
        time.sleep(3.0)  # wait for results page

        # Read the window title after navigation
        title = chrome.Name

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        scenario_report.steps = 3  # open → navigate → search
        scenario_report.native_count = 2  # focus + SendKeys via UIA

        scenario_report.assert_ok(
            bool(title and title.strip()),
            f"Chrome page title must be non-empty; got: {title!r}",
        )

        # ------------------------------------------------------------------
        # Teardown
        # ------------------------------------------------------------------
        proc.terminate()
        proc.wait(timeout=5)
