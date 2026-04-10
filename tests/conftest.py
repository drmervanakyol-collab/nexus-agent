"""
tests/conftest.py
Shared pytest fixtures for the Nexus Agent test suite.

File fixtures
-------------
``test_pdf``         — Path to a single-page PDF with a readable text layer
``test_scanned_pdf`` — Path to an encrypted (password-protected) PDF
``test_xlsx``        — Path to an XLSX with two sheets and table data

All fixtures write to pytest's ``tmp_path``; no repo-level files are needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------


def _write_text_pdf(path: Path, text: str) -> None:
    """
    Write a minimal valid PDF-1.4 file whose content stream embeds *text*
    so that pypdf's ``extract_text()`` can recover it.

    Offsets in the xref table are computed dynamically.
    """
    content_stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET\n".encode()

    objects: list[bytes] = [
        # obj 1 — catalog
        b"<< /Type /Catalog /Pages 2 0 R >>",
        # obj 2 — pages node
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        # obj 3 — page (references content obj 4 and font F1)
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            b" /Contents 4 0 R"
            b" /Resources << /Font << /F1 << /Type /Font /Subtype /Type1"
            b" /BaseFont /Helvetica >> >> >> >>"
        ),
        # obj 4 — content stream
        (
            b"<< /Length "
            + str(len(content_stream)).encode()
            + b" >>\nstream\n"
            + content_stream
            + b"endstream"
        ),
    ]

    body = b"%PDF-1.4\n"
    offsets: list[int] = []
    for idx, obj_data in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{idx} 0 obj\n".encode() + obj_data + b"\nendobj\n"

    xref_start = len(body)
    n_objects = len(objects) + 1  # free object 0 + objs 1..n
    xref_lines = [
        b"xref\n",
        f"0 {n_objects}\n".encode(),
        b"0000000000 65535 f \n",
    ] + [f"{off:010d} 00000 n \n".encode() for off in offsets]

    trailer = (
        f"trailer\n<< /Size {n_objects} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    ).encode()

    path.write_bytes(body + b"".join(xref_lines) + trailer)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_pdf(tmp_path: Path) -> Path:
    """Single-page text PDF; pypdf extract_text() returns non-empty string."""
    out = tmp_path / "test.pdf"
    _write_text_pdf(out, "Hello World from Nexus")
    return out


@pytest.fixture()
def test_scanned_pdf(tmp_path: Path) -> Path:
    """Encrypted (AES-128) PDF; pypdf reports is_encrypted == True."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret")

    out = tmp_path / "test_scanned.pdf"
    with open(out, "wb") as fh:
        writer.write(fh)
    return out


@pytest.fixture()
def test_xlsx(tmp_path: Path) -> Path:
    """XLSX with two sheets (Sales, Costs) and simple table data."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Sales"
    ws1.append(["Product", "Q1", "Q2"])
    ws1.append(["Widget", 100, 200])
    ws1.append(["Gadget", 150, 175])

    ws2 = wb.create_sheet("Costs")
    ws2.append(["Item", "Amount"])
    ws2.append(["Rent", 5000])

    out = tmp_path / "test.xlsx"
    wb.save(out)
    return out
