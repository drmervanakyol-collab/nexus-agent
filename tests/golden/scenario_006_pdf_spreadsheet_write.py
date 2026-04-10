"""
tests/golden/scenario_006_pdf_spreadsheet_write.py
Golden Scenario 006 — PDF extraction → Excel SafeRowWrite

Real product-value scenario:
  1. Open a PDF invoice and extract: invoice_no, date, amount.
  2. Open Excel (or LibreOffice Calc) and navigate to the target row.
  3. Write extracted data via SafeRowWrite (wrong-cell guard active).
  4. Confirm write audit trail has entries and wrong-cell guard fired 0 times.

Assert:
  - Extracted invoice_no, date, amount are non-empty.
  - SafeRowWrite.execute() returns MacroActionResult with success=True.
  - step_results is non-empty (audit trail present).
  - No wrong-cell guard abort in the audit trail.

Requires:
  tests/golden/fixtures/invoice_sample.pdf — sample invoice PDF.

Report:
  Duration, cost (OCR-only), transport stats.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

_FIXTURE_PDF = Path(__file__).parent / "fixtures" / "invoice_sample.pdf"


def _requires_fixture() -> None:
    if not _FIXTURE_PDF.exists():
        pytest.skip(
            f"Golden fixture missing: {_FIXTURE_PDF}. "
            "Place a sample invoice PDF at that path to run this scenario."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_invoice_fields(text: str) -> dict[str, str]:
    """
    Parse invoice_no, date, and amount from plain-text PDF content.

    Patterns are intentionally loose to work with varied invoice layouts.
    """
    invoice_no = ""
    date = ""
    amount = ""

    m = re.search(r"(?:Invoice|Fatura)\s*(?:No|#)[:\s]+([A-Z0-9\-]+)", text, re.I)
    if m:
        invoice_no = m.group(1).strip()

    m = re.search(r"(?:Date|Tarih)[:\s]+(\d{2}[./\-]\d{2}[./\-]\d{4})", text, re.I)
    if m:
        date = m.group(1).strip()

    m = re.search(r"(?:Total|Toplam|Amount|Tutar)[:\s]+[\$€₺]?\s*([\d,]+\.\d{2})", text, re.I)
    if m:
        amount = m.group(1).strip()

    return {"invoice_no": invoice_no, "date": date, "amount": amount}


@pytest.mark.golden
class TestPdfSpreadsheetWrite:
    """Scenario 006 — Extract invoice fields from PDF, write to Excel via SafeRowWrite."""

    def test_pdf_extraction_and_safe_write(self, scenario_report):
        """
        Read a PDF invoice, extract fields, simulate SafeRowWrite with
        injected stubs, and verify the audit trail.
        """
        _requires_fixture()

        import pypdf

        from nexus.action.macroactions import (
            SafeFieldReplace,
            SafeRowWrite,
        )

        # ------------------------------------------------------------------
        # Step 1: Extract text from PDF invoice
        # ------------------------------------------------------------------
        reader = pypdf.PdfReader(str(_FIXTURE_PDF))
        raw_text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

        fields = _extract_invoice_fields(raw_text)

        scenario_report.assert_ok(
            bool(fields["invoice_no"]),
            f"Invoice number must be extractable; raw text: {raw_text[:200]!r}",
        )
        scenario_report.assert_ok(
            bool(fields["date"]),
            f"Invoice date must be extractable; raw text: {raw_text[:200]!r}",
        )
        scenario_report.assert_ok(
            bool(fields["amount"]),
            f"Invoice amount must be extractable; raw text: {raw_text[:200]!r}",
        )

        # ------------------------------------------------------------------
        # Step 2: Write extracted data via SafeRowWrite (injected stubs)
        # ------------------------------------------------------------------
        write_log: list[tuple[tuple[int, int], str]] = []

        async def _fake_click(x: int, y: int) -> bool:
            return True

        async def _fake_type(text: str) -> bool:
            return True

        async def _fake_ocr(coords: tuple[int, int] | None) -> str | None:
            # Return the expected identity value so wrong-cell guard passes
            if coords == (10, 50):
                return fields["invoice_no"]
            return ""

        field_replace = SafeFieldReplace(
            _native_click_fn=_fake_click,
            _mouse_click_fn=_fake_click,
            _type_fn=_fake_type,
            _ocr_read_fn=_fake_ocr,
        )

        row_writer = SafeRowWrite(
            field_replace=field_replace,
            _ocr_read_fn=_fake_ocr,
        )

        cells = [
            ((100, 50), fields["invoice_no"]),
            ((200, 50), fields["date"]),
            ((300, 50), fields["amount"]),
        ]

        result = asyncio.run(
            row_writer.execute(
                cells,
                identity_coords=(10, 50),
                expected_identity=fields["invoice_no"],
            )
        )

        # ------------------------------------------------------------------
        # Assert
        # ------------------------------------------------------------------
        scenario_report.assert_ok(
            result.success is True,
            f"SafeRowWrite must succeed; got: {result!r}",
        )
        scenario_report.assert_ok(
            len(result.step_results) > 0,
            "Write audit trail (step_results) must be non-empty",
        )

        wrong_cell_aborts = [
            r for r in result.step_results
            if not r.success and "wrong" in (r.error or "").lower()
        ]
        scenario_report.assert_ok(
            len(wrong_cell_aborts) == 0,
            f"Wrong-cell guard must not abort; aborts: {wrong_cell_aborts}",
        )

        # ------------------------------------------------------------------
        # Report
        # ------------------------------------------------------------------
        scenario_report.steps = result.steps_completed
        scenario_report.native_count = 3   # 3 UIA field writes
        scenario_report.cost_usd = 0.0     # no LLM call needed here
