"""
tests/unit/test_pdf_skills.py
Unit tests for nexus/skills/pdf/ — Faz 49.

TEST PLAN
---------
PDFReader:
  1.  Metin PDF → FileAdapter returns pdf_text → returned directly
  2.  Taranmış PDF → FileAdapter returns pdf_ocr → returned directly
  3.  Şifreli PDF → FileAdapter None + is_encrypted True → HITL requested
  4.  Extract failure (non-encrypted) → None, no HITL
  5.  Frame only (no file) → scroll-and-scan OCR path → pdf_ocr content
  6.  Both None → returns None

PDFNavigator:
  7.  scroll_to_page via injected _goto_page_fn
  8.  scroll_to_page keyboard fallback (Ctrl+G + type + Enter)
  9.  find_text in content → correct (page_idx, char_offset)
  10. find_text in content → not found → None
  11. find_text_in_viewer → Ctrl+F + type + Enter → (0, 0)
  12. find_text with no content → None

PDFExtractor:
  13. extract_field inline colon pattern
  14. extract_field next-line pattern
  15. extract_field not found → None
  16. extract_table from structured tables
  17. extract_table heuristic (tab/multi-space separated text)
  18. extract_table index out of range → []
  19. extract_table no text table content → []
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from nexus.capture.frame import Frame
from nexus.core.hitl_manager import HITLManager, HITLResponse
from nexus.skills.pdf.extractor import PDFExtractor
from nexus.skills.pdf.navigator import PDFNavigator
from nexus.skills.pdf.reader import PDFReader
from nexus.source.file.adapter import DocumentContent, FileAdapter

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _text_content(pages: list[str] | None = None) -> DocumentContent:
    return DocumentContent(
        source_type="pdf_text",
        pages=pages or ["Hello, world!"],
        tables=[],
        metadata={"page_count": 1},
        extraction_confidence=1.0,
    )


def _ocr_content(pages: list[str] | None = None) -> DocumentContent:
    return DocumentContent(
        source_type="pdf_ocr",
        pages=pages or ["Scanned text"],
        tables=[],
        metadata={"page_count": 1},
        extraction_confidence=0.85,
    )


def _mock_hitl(*, headless: bool = True) -> HITLManager:
    hitl = MagicMock(spec=HITLManager)
    hitl.request = AsyncMock(
        return_value=HITLResponse(
            task_id="test",
            chosen_option="Skip this file",
            chosen_index=1,
            timed_out=False,
            elapsed_s=0.0,
        )
    )
    return hitl


def _make_frame() -> Frame:
    data = np.zeros((100, 200, 3), dtype=np.uint8)
    return Frame(
        data=data,
        width=200,
        height=100,
        captured_at_monotonic=0.0,
        captured_at_utc="2026-04-08T00:00:00Z",
        sequence_number=1,
    )


# ---------------------------------------------------------------------------
# PDFReader tests
# ---------------------------------------------------------------------------


class TestPDFReaderFilePath:
    @pytest.mark.asyncio
    async def test_text_pdf_returned_directly(self):
        """FileAdapter returns pdf_text → content returned unchanged."""
        content = _text_content()
        adapter = MagicMock(spec=FileAdapter)
        adapter.extract.return_value = content

        reader = PDFReader(
            adapter, _mock_hitl(), _is_encrypted_fn=lambda p: False
        )
        result = await reader.read(file_path="doc.pdf")

        assert result is content
        assert result.source_type == "pdf_text"
        adapter.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_scanned_pdf_ocr_returned_directly(self):
        """FileAdapter returns pdf_ocr → content returned unchanged."""
        content = _ocr_content()
        adapter = MagicMock(spec=FileAdapter)
        adapter.extract.return_value = content

        reader = PDFReader(
            adapter, _mock_hitl(), _is_encrypted_fn=lambda p: False
        )
        result = await reader.read(file_path="scan.pdf")

        assert result is content
        assert result.source_type == "pdf_ocr"

    @pytest.mark.asyncio
    async def test_encrypted_pdf_triggers_hitl(self):
        """FileAdapter returns None + is_encrypted=True → HITL.request called."""
        adapter = MagicMock(spec=FileAdapter)
        adapter.extract.return_value = None

        hitl = _mock_hitl()

        reader = PDFReader(
            adapter,
            hitl,
            _is_encrypted_fn=lambda p: True,
        )
        result = await reader.read(file_path="secret.pdf", task_id="t1")

        assert result is None
        hitl.request.assert_awaited_once()
        req = hitl.request.call_args[0][0]
        assert req.task_id == "t1"

    @pytest.mark.asyncio
    async def test_non_encrypted_failure_no_hitl(self):
        """FileAdapter None + not encrypted → no HITL, returns None."""
        adapter = MagicMock(spec=FileAdapter)
        adapter.extract.return_value = None
        hitl = _mock_hitl()

        reader = PDFReader(
            adapter,
            hitl,
            _is_encrypted_fn=lambda p: False,
        )
        result = await reader.read(file_path="broken.pdf")

        assert result is None
        hitl.request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_frame_only_uses_ocr_path(self):
        """No file_path + current_frame → scroll-and-scan → pdf_ocr."""
        adapter = MagicMock(spec=FileAdapter)
        hitl = _mock_hitl()
        frame = _make_frame()

        reader = PDFReader(
            adapter,
            hitl,
            _ocr_frame_fn=lambda f: "frame text",
        )
        result = await reader.read(current_frame=frame)

        assert result is not None
        assert result.source_type == "pdf_ocr"
        assert result.pages == ["frame text"]
        adapter.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_both_none_returns_none(self):
        """No file_path and no frame → None."""
        adapter = MagicMock(spec=FileAdapter)
        hitl = _mock_hitl()
        reader = PDFReader(adapter, hitl)

        result = await reader.read()

        assert result is None


# ---------------------------------------------------------------------------
# PDFNavigator tests
# ---------------------------------------------------------------------------


class TestPDFNavigatorScrollToPage:
    @pytest.mark.asyncio
    async def test_injected_goto_page_fn_called(self):
        """_goto_page_fn replaces the keyboard sequence."""
        calls: list[int] = []

        async def fake_goto(page: int) -> bool:
            calls.append(page)
            return True

        nav = PDFNavigator(_goto_page_fn=fake_goto)
        result = await nav.scroll_to_page(5)

        assert result is True
        assert calls == [5]

    @pytest.mark.asyncio
    async def test_keyboard_fallback_sequence(self):
        """Default: Ctrl+G + type + Enter sent."""
        hotkey_calls: list[list[str]] = []
        type_calls: list[str] = []
        special_key_calls: list[str] = []

        async def fake_hotkey(keys: list[str]) -> bool:
            hotkey_calls.append(keys)
            return True

        async def fake_type(text: str) -> bool:
            type_calls.append(text)
            return True

        async def fake_special(key: str) -> bool:
            special_key_calls.append(key)
            return True

        nav = PDFNavigator(
            _hotkey_fn=fake_hotkey,
            _type_fn=fake_type,
            _special_key_fn=fake_special,
        )
        result = await nav.scroll_to_page(3)

        assert result is True
        assert ["ctrl", "g"] in hotkey_calls
        assert "3" in type_calls
        assert "enter" in special_key_calls


class TestPDFNavigatorFindText:
    def test_find_text_in_content_found(self):
        """find_text returns (page_idx, char_offset) on match."""
        content = _text_content(pages=["First page text", "Invoice Number: 12345"])
        nav = PDFNavigator()

        result = nav.find_text("Invoice Number", content=content)

        assert result == (1, 0)

    def test_find_text_in_content_not_found(self):
        """find_text returns None when text not in any page."""
        content = _text_content(pages=["Hello world"])
        nav = PDFNavigator()

        result = nav.find_text("missing text", content=content)

        assert result is None

    def test_find_text_mid_page(self):
        """Offset reflects position within the page string."""
        content = _text_content(pages=["prefix TARGET suffix"])
        nav = PDFNavigator()

        result = nav.find_text("TARGET", content=content)

        assert result == (0, 7)

    def test_find_text_no_content_returns_none(self):
        """find_text with no content and no viewer call → None."""
        nav = PDFNavigator()
        result = nav.find_text("anything")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_text_in_viewer_returns_position(self):
        """find_text_in_viewer delivers Ctrl+F + type + Enter."""
        hotkey_calls: list[list[str]] = []
        type_calls: list[str] = []
        special_calls: list[str] = []

        async def fake_hotkey(keys: list[str]) -> bool:
            hotkey_calls.append(keys)
            return True

        async def fake_type(text: str) -> bool:
            type_calls.append(text)
            return True

        async def fake_special(key: str) -> bool:
            special_calls.append(key)
            return True

        nav = PDFNavigator(
            _hotkey_fn=fake_hotkey,
            _type_fn=fake_type,
            _special_key_fn=fake_special,
        )

        result = await nav.find_text_in_viewer("search me")

        assert result == (0, 0)
        assert ["ctrl", "f"] in hotkey_calls
        assert "search me" in type_calls
        assert "enter" in special_calls


# ---------------------------------------------------------------------------
# PDFExtractor tests
# ---------------------------------------------------------------------------


class TestPDFExtractorField:
    def test_inline_colon_pattern(self):
        """'Field: value' on same line → value extracted."""
        content = _text_content(pages=["Invoice Number: INV-2026-001"])
        extractor = PDFExtractor()

        result = extractor.extract_field("Invoice Number", content)

        assert result == "INV-2026-001"

    def test_next_line_pattern(self):
        """Field name on one line, value on the next."""
        content = _text_content(pages=["Invoice Number\nINV-2026-001\n"])
        extractor = PDFExtractor()

        result = extractor.extract_field("Invoice Number", content)

        assert result == "INV-2026-001"

    def test_case_insensitive_match(self):
        """Field name matching is case-insensitive."""
        content = _text_content(pages=["TOTAL AMOUNT: 1,234.56"])
        extractor = PDFExtractor()

        result = extractor.extract_field("total amount", content)

        assert result == "1,234.56"

    def test_not_found_returns_none(self):
        """Field not in content → None."""
        content = _text_content(pages=["Hello world"])
        extractor = PDFExtractor()

        result = extractor.extract_field("Invoice Number", content)

        assert result is None

    def test_multipage_finds_second_page(self):
        """Field can be on any page."""
        content = _text_content(
            pages=["Page one text", "Due Date: 2026-05-01"]
        )
        extractor = PDFExtractor()

        result = extractor.extract_field("Due Date", content)

        assert result == "2026-05-01"


class TestPDFExtractorTable:
    def test_structured_table_returned(self):
        """content.tables has data → returned directly."""
        table = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        content = DocumentContent(
            source_type="xlsx",
            pages=["Sheet1"],
            tables=[table],
            metadata={},
            extraction_confidence=1.0,
        )
        extractor = PDFExtractor()

        result = extractor.extract_table(content, table_index=0)

        assert result == table

    def test_table_index_out_of_range(self):
        """Requesting an index beyond available tables → []."""
        content = DocumentContent(
            source_type="xlsx",
            pages=["Sheet1"],
            tables=[[["a", "b"]]],
            metadata={},
            extraction_confidence=1.0,
        )
        extractor = PDFExtractor()

        result = extractor.extract_table(content, table_index=5)

        assert result == []

    def test_heuristic_table_from_text(self):
        """Multi-space-separated columns parsed from page text."""
        text = "Name    Age    City\nAlice   30     London\nBob     25     Paris"
        content = _text_content(pages=[text])
        extractor = PDFExtractor()

        result = extractor.extract_table(content)

        assert len(result) == 3
        assert result[0][0] == "Name"
        assert result[1][0] == "Alice"
        assert result[2][1] == "25"

    def test_heuristic_non_zero_index_returns_empty(self):
        """Heuristic mode only supports index 0."""
        text = "Col1  Col2\nA     B"
        content = _text_content(pages=[text])
        extractor = PDFExtractor()

        result = extractor.extract_table(content, table_index=1)

        assert result == []

    def test_no_table_content_returns_empty(self):
        """Plain prose text with no columns → []."""
        content = _text_content(pages=["This is a normal sentence."])
        extractor = PDFExtractor()

        result = extractor.extract_table(content)

        assert result == []
