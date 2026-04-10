"""
tests/golden/scenario_004_file_explorer.py
Golden Scenario 004 — File Explorer: create folder → rename → verify

Setup:
  Open File Explorer to a temp directory.
Execute:
  1. Create a new folder named "nexus_test_folder_<timestamp>".
  2. Rename it to "nexus_renamed_<timestamp>".
Assert:
  The renamed folder name is visible in the Explorer window.
Teardown:
  Delete the renamed folder; close Explorer.
Report:
  Duration, transport stats.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

_TIMESTAMP = str(int(time.time()))
_ORIGINAL_NAME = f"nexus_test_folder_{_TIMESTAMP}"
_RENAMED_NAME = f"nexus_renamed_{_TIMESTAMP}"


@pytest.mark.golden
class TestFileExplorer:
    """Scenario 004 — Explorer: create and rename a folder."""

    def test_folder_create_and_rename(self, scenario_report, tmp_path):
        """
        Open File Explorer in tmp_path, create a folder, rename it,
        and assert the new name is visible in the Explorer view.
        """
        # ------------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------------
        import uiautomation as auto  # type: ignore[import-untyped]

        work_dir = str(tmp_path)
        proc = subprocess.Popen(["explorer.exe", work_dir])
        time.sleep(2.0)

        explorer = auto.WindowControl(searchDepth=2, ClassName="CabinetWClass")
        scenario_report.assert_ok(explorer.Exists(5), "Explorer window must appear")

        # ------------------------------------------------------------------
        # Execute — Step 1: create new folder via Ctrl+Shift+N
        # ------------------------------------------------------------------
        explorer.SetFocus()
        time.sleep(0.3)
        explorer.SendKeys("{Ctrl}{Shift}n", waitTime=0.3)
        time.sleep(0.5)

        # Type the folder name in the inline rename box
        explorer.SendKeys(_ORIGINAL_NAME + "{Enter}", waitTime=0.05)
        time.sleep(0.5)

        original_path = os.path.join(work_dir, _ORIGINAL_NAME)
        scenario_report.assert_ok(
            os.path.isdir(original_path),
            f"Original folder must exist on disk: {original_path}",
        )

        # ------------------------------------------------------------------
        # Execute — Step 2: rename via F2
        # ------------------------------------------------------------------
        # Select the newly created folder item in Explorer
        item = explorer.ListItemControl(Name=_ORIGINAL_NAME)
        if not item.Exists(2):
            item = explorer.ListItemControl(Name=_ORIGINAL_NAME)

        if item.Exists(1):
            item.Click()
            time.sleep(0.2)

        explorer.SendKeys("{F2}", waitTime=0.3)
        time.sleep(0.3)
        explorer.SendKeys(
            "{Ctrl}a" + _RENAMED_NAME + "{Enter}",
            waitTime=0.05,
        )
        time.sleep(0.5)

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        renamed_path = os.path.join(work_dir, _RENAMED_NAME)
        scenario_report.steps = 4  # open, create, rename, verify
        scenario_report.native_count = 3  # focus + Ctrl+Shift+N + F2 via UIA

        scenario_report.assert_ok(
            os.path.isdir(renamed_path),
            f"Renamed folder must exist on disk: {renamed_path}",
        )

        renamed_item = explorer.ListItemControl(Name=_RENAMED_NAME)
        scenario_report.assert_ok(
            renamed_item.Exists(2),
            f"Renamed folder {_RENAMED_NAME!r} must be visible in Explorer",
        )

        # ------------------------------------------------------------------
        # Teardown
        # ------------------------------------------------------------------
        proc.terminate()
        proc.wait(timeout=5)
        for name in (_ORIGINAL_NAME, _RENAMED_NAME):
            p = os.path.join(work_dir, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
