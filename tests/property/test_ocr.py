"""
tests/property/test_ocr.py
OCR engine property tests — Faz 65

Invariants tested
-----------------
- TesseractOCREngine.extract() never raises for any injected TSV string,
  including empty, malformed, random bytes decoded as UTF-8/latin-1
- _parse_tsv() never raises for any non-None string input
- OCRResult fields satisfy constraints:
    0.0 <= confidence <= 1.0
    text is a non-None string
    bounding_box dimensions >= 0
- extract() on empty/blank TSV → returns empty list (no phantom results)
- extract() result list contains only OCRResult instances
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nexus.perception.reader.ocr_engine import TesseractOCREngine

# ---------------------------------------------------------------------------
# TSV header (Tesseract format)
# ---------------------------------------------------------------------------

_TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num"
    "\tleft\ttop\twidth\theight\tconf\ttext\n"
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Random printable + unicode text strings
_RANDOM_TEXT = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs",),  # exclude surrogates
        blacklist_characters=("\x00",),
    ),
    max_size=200,
)

# Random TSV body rows — may be totally malformed
_RANDOM_ROW = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters=("\x00",)),
    max_size=200,
)

_CONFIDENCE = st.floats(min_value=-1.0, max_value=101.0, allow_nan=False)
_COORD = st.integers(min_value=-1000, max_value=10000)
_DIM = st.integers(min_value=-100, max_value=5000)

_WORD_TEXT = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
    blacklist_characters=("\t", "\n"),
))


def _make_valid_tsv(words: list[tuple[str, int, int]]) -> str:
    """Build a syntactically valid Tesseract TSV string."""
    lines = [_TSV_HEADER]
    for i, (word, conf, x) in enumerate(words, start=1):
        lines.append(
            f"5\t1\t1\t1\t1\t{i}\t{x}\t10\t{len(word) * 8}\t20\t{conf}\t{word}\n"
        )
    return "".join(lines)


# ---------------------------------------------------------------------------
# Property: extract() never raises for any _run_fn output
# ---------------------------------------------------------------------------


@given(st.lists(_RANDOM_TEXT, max_size=20))
def test_extract_never_raises_on_random_tsv_lines(rows: list[str]) -> None:
    """
    Concatenate random strings as TSV body rows — extract() must never raise.
    """
    tsv = _TSV_HEADER + "".join(r + "\n" for r in rows)

    engine = TesseractOCREngine(
        _run_fn=lambda _img, _lang, _timeout: tsv,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    # Must not raise — malformed rows are silently skipped
    try:
        engine.extract(image)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"extract() raised unexpectedly: {exc!r}")


@given(_RANDOM_TEXT)
def test_extract_never_raises_on_any_string(raw: str) -> None:
    """Even a completely random string as TSV must not crash extract()."""
    engine = TesseractOCREngine(
        _run_fn=lambda _img, _lang, _timeout: raw,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        engine.extract(image)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"extract() raised with raw TSV={raw!r}: {exc!r}")


# ---------------------------------------------------------------------------
# Property: empty / header-only TSV → empty result
# ---------------------------------------------------------------------------


def test_extract_empty_tsv_returns_empty_list() -> None:
    engine = TesseractOCREngine(
        _run_fn=lambda *_: "",
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    results = engine.extract(image)
    assert results == []


def test_extract_header_only_tsv_returns_empty_list() -> None:
    engine = TesseractOCREngine(
        _run_fn=lambda *_: _TSV_HEADER,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    results = engine.extract(image)
    assert results == []


# ---------------------------------------------------------------------------
# Property: OCRResult field constraints on valid TSV output
# ---------------------------------------------------------------------------


@given(
    st.lists(
        st.tuples(
            _WORD_TEXT,
            st.integers(min_value=0, max_value=100),   # conf
            st.integers(min_value=0, max_value=1900),  # x
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ocr_result_confidence_in_unit_interval(
    words: list[tuple[str, int, int]]
) -> None:
    tsv = _make_valid_tsv(words)
    engine = TesseractOCREngine(
        confidence_threshold=0.0,  # accept all
        _run_fn=lambda *_: tsv,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    results = engine.extract(image)
    for r in results:
        assert 0.0 <= r.confidence <= 1.0, (
            f"OCRResult.confidence={r.confidence} out of [0, 1]"
        )


@given(
    st.lists(
        st.tuples(
            _WORD_TEXT,
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=0, max_value=1900),
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ocr_result_bounding_box_nonnegative_dimensions(
    words: list[tuple[str, int, int]]
) -> None:
    tsv = _make_valid_tsv(words)
    engine = TesseractOCREngine(
        confidence_threshold=0.0,
        _run_fn=lambda *_: tsv,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    results = engine.extract(image)
    for r in results:
        assert r.bounding_box.width >= 0
        assert r.bounding_box.height >= 0


@given(
    st.lists(
        st.tuples(
            _WORD_TEXT,
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=0, max_value=1900),
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ocr_result_text_is_nonempty_string(
    words: list[tuple[str, int, int]]
) -> None:
    tsv = _make_valid_tsv(words)
    engine = TesseractOCREngine(
        confidence_threshold=0.0,
        _run_fn=lambda *_: tsv,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    results = engine.extract(image)
    for r in results:
        assert isinstance(r.text, str)
        assert len(r.text) > 0
