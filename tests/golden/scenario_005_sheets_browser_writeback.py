"""
tests/golden/scenario_005_sheets_browser_writeback.py
Golden Scenario 005 — Google Sheets → Chrome → Sheets writeback

Real product-value scenario:
  1. Read an order list from Google Sheets (Sheet1!A2:B6).
  2. For each order, search Chrome for a tracking number.
  3. Write the tracking numbers back to Sheets (Sheet1!C2:C6).
  4. Verify via SOURCE_LEVEL check that Sheets was updated correctly.

Assert:
  - Sheets cell C2 is not empty after writeback.
  - SOURCE_LEVEL verification returns True.

Requires:
  NEXUS_SHEETS_SPREADSHEET_ID env var — the target Google Sheet.
  NEXUS_GOOGLE_TOKEN env var — valid OAuth access token.

Report:
  Duration, cost, transport stats, step count.
"""
from __future__ import annotations

import os

import pytest

_SPREADSHEET_ID = os.environ.get("NEXUS_SHEETS_SPREADSHEET_ID", "")
_GOOGLE_TOKEN = os.environ.get("NEXUS_GOOGLE_TOKEN", "")
_READ_RANGE = "Sheet1!A2:B6"
_WRITE_RANGE = "Sheet1!C2:C6"


def _requires_credentials() -> None:
    if not _SPREADSHEET_ID or not _GOOGLE_TOKEN:
        pytest.skip(
            "NEXUS_SHEETS_SPREADSHEET_ID and NEXUS_GOOGLE_TOKEN must be set"
        )


@pytest.mark.golden
class TestSheetsBrowserWriteback:
    """Scenario 005 — Read orders from Sheets, enrich via Chrome, write back."""

    def test_tracking_writeback_and_verification(self, scenario_report):
        """
        Full pipeline:
          1. GoogleSheetsClient.read_range → order rows
          2. Simulate Chrome tracking lookup (stub: append "-TRACKED")
          3. GoogleSheetsClient.write_range → write tracking numbers
          4. GoogleSheetsClient.read_range → verify C2 non-empty
        """
        _requires_credentials()

        from nexus.integrations.google.sheets import GoogleSheetsClient

        # ------------------------------------------------------------------
        # Setup — inject real HTTP fns; token from env
        # ------------------------------------------------------------------
        client = GoogleSheetsClient(
            get_token_fn=lambda: _GOOGLE_TOKEN,
        )

        # ------------------------------------------------------------------
        # Step 1: Read order list
        # ------------------------------------------------------------------
        orders = client.read_range(_SPREADSHEET_ID, _READ_RANGE)
        scenario_report.assert_ok(
            len(orders) > 0,
            f"Sheets must return at least one order row; got {orders!r}",
        )

        # ------------------------------------------------------------------
        # Step 2: Simulate per-order tracking lookup in Chrome
        # (In production: open Chrome, search each order ID, extract result.
        #  Here: deterministic stub — append "-TRACKED" to order ID.)
        # ------------------------------------------------------------------
        tracking_numbers = []
        for row in orders:
            order_id = row[0] if row else "UNKNOWN"
            tracking_numbers.append([f"{order_id}-TRACKED"])

        # ------------------------------------------------------------------
        # Step 3: Write tracking numbers back to Sheets
        # ------------------------------------------------------------------
        success = client.write_range(
            _SPREADSHEET_ID,
            _WRITE_RANGE,
            tracking_numbers,
            check_formulas=True,
        )
        scenario_report.assert_ok(
            success is True,
            "write_range must return True (SOURCE_LEVEL verification passed)",
        )

        # ------------------------------------------------------------------
        # Step 4: Read back and confirm C2 is non-empty
        # ------------------------------------------------------------------
        written = client.read_range(_SPREADSHEET_ID, "Sheet1!C2:C6")
        scenario_report.assert_ok(
            len(written) > 0 and written[0][0].endswith("-TRACKED"),
            f"Cell C2 must contain a tracking number; got {written!r}",
        )

        # ------------------------------------------------------------------
        # Report
        # ------------------------------------------------------------------
        scenario_report.steps = 4
        scenario_report.native_count = 0   # cloud HTTP transport
        scenario_report.cost_usd = 0.0     # Sheets API — no LLM cost
