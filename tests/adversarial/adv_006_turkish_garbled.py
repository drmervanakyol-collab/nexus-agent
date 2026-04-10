"""
tests/adversarial/adv_006_turkish_garbled.py
Adversarial Test 006 — Garbled Turkish font → low OCR confidence → graceful degradation

Scenario:
  TesseractOCREngine receives a TSV response where Turkish characters have
  confidence < 0.30 (the _LOW_CONF_THRESHOLD for "garbled" text).

Success criteria:
  - OCRResult objects are still returned (no crash, no empty list drop).
  - Each result has confidence < 0.30 (confirming low-conf handling).
  - No OCRError raised.
  - A WARNING log event "ocr_low_confidence" is emitted.

All I/O is injected — no real Tesseract subprocess.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nexus.perception.reader.ocr_engine import TesseractOCREngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# TSV row format (tab-separated):
# level page_num block_num par_num line_num word_num left top width height conf text
_TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num"
    "\tleft\ttop\twidth\theight\tconf\ttext"
)


def _tsv_row(text: str, conf: float) -> str:
    """Build one Tesseract TSV word row with given confidence (0–100)."""
    conf_int = int(conf * 100)
    return f"5\t1\t1\t1\t1\t1\t10\t20\t60\t18\t{conf_int}\t{text}"


def _garbled_tsv() -> str:
    """TSV with three garbled Turkish words, all confidence < 0.30."""
    rows = [
        _tsv_row("�stanbul", 0.15),   # garbled İ
        _tsv_row("�i�li",    0.22),   # garbled Şişli
        _tsv_row("g�çl�",   0.28),   # garbled ğüçlü
    ]
    return _TSV_HEADER + "\n" + "\n".join(rows)


class _CapLog:
    """Captures structlog events."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def warning(self, event: str, **kw: Any) -> None:
        self.events.append({"event": event, **kw})

    def debug(self, *a: Any, **kw: Any) -> None:
        pass

    def info(self, *a: Any, **kw: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.adversarial
class TestTurkishGarbled:
    """ADV-006: Low-confidence Turkish OCR — graceful degradation, no crash."""

    def test_garbled_text_still_returns_results(self):
        """
        Even with very low confidence, OCR results are returned (not dropped).
        No OCRError raised.
        """
        engine = TesseractOCREngine(
            confidence_threshold=0.5,
            _run_fn=lambda img, lang, timeout: _garbled_tsv(),
            _detect_lang_fn=lambda text: "tur",
        )

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        results = engine.extract(img)

        assert len(results) == 3, (
            f"All 3 garbled words must be in results; got {len(results)}"
        )

    def test_all_results_have_low_confidence(self):
        """Each result confidence < 0.30 (threshold for 'garbled' text)."""
        engine = TesseractOCREngine(
            confidence_threshold=0.5,
            _run_fn=lambda img, lang, timeout: _garbled_tsv(),
            _detect_lang_fn=lambda text: "tur",
        )

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        results = engine.extract(img)

        for r in results:
            assert r.confidence < 0.30, (
                f"Expected confidence < 0.30 for garbled text; "
                f"got {r.confidence:.3f} for {r.text!r}"
            )

    def test_low_confidence_warning_logged(self):
        """ocr_low_confidence WARNING is emitted for each garbled word."""
        import nexus.perception.reader.ocr_engine as _mod

        original_log = _mod._log
        cap = _CapLog()
        _mod._log = cap  # type: ignore[assignment]
        try:
            engine = TesseractOCREngine(
                confidence_threshold=0.5,
                _run_fn=lambda img, lang, timeout: _garbled_tsv(),
                _detect_lang_fn=lambda text: "tur",
            )
            img = np.zeros((100, 200, 3), dtype=np.uint8)
            engine.extract(img)
        finally:
            _mod._log = original_log

        low_conf_events = [e for e in cap.events if "low_confidence" in e.get("event", "")]
        assert len(low_conf_events) >= 1, (
            f"At least one 'ocr_low_confidence' warning expected; got {cap.events}"
        )

    def test_high_confidence_text_not_warned(self):
        """Words with confidence >= 0.80 must NOT emit low_confidence warnings."""
        import nexus.perception.reader.ocr_engine as _mod

        good_tsv = _TSV_HEADER + "\n" + _tsv_row("Merhaba", 0.92)

        original_log = _mod._log
        cap = _CapLog()
        _mod._log = cap  # type: ignore[assignment]
        try:
            engine = TesseractOCREngine(
                confidence_threshold=0.5,
                _run_fn=lambda img, lang, timeout: good_tsv,
                _detect_lang_fn=lambda text: "tur",
            )
            img = np.zeros((100, 200, 3), dtype=np.uint8)
            engine.extract(img)
        finally:
            _mod._log = original_log

        low_conf_events = [e for e in cap.events if "low_confidence" in e.get("event", "")]
        assert len(low_conf_events) == 0, (
            f"No low_confidence warnings for high-confidence text; got {cap.events}"
        )
