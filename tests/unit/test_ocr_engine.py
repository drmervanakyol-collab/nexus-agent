"""
tests/unit/test_ocr_engine.py
Unit tests for nexus/perception/reader/ocr_engine.py — Faz 23.

Sections:
  1.  OCRResult value object
  2.  OCREngine Protocol compliance
  3.  TesseractOCREngine — normal extraction (mock subprocess)
  4.  TesseractOCREngine — region cropping
  5.  TesseractOCREngine — language selection
  6.  TesseractOCREngine — low-confidence handling
  7.  TesseractOCREngine — timeout → OCRError
  8.  TesseractOCREngine — crash & restart mechanics
  9.  TesseractOCREngine — Turkish post-processing
  10. TesseractOCREngine — preprocessing robustness
  11. Hypothesis — arbitrary array → no unexpected exception
  12. Real Tesseract accuracy (skipped when binary unavailable)
"""
from __future__ import annotations

import subprocess
from collections.abc import Callable
from itertools import count

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nexus.capture.frame import Frame
from nexus.core.errors import OCRError
from nexus.core.types import Rect
from nexus.perception.reader.ocr_engine import (
    OCREngine,
    OCRResult,
    TesseractOCREngine,
    _TURKISH_OCR_FIXES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num"
    "\tleft\ttop\twidth\theight\tconf\ttext"
)


def _tsv(*words: tuple[str, int, int, int, int, float]) -> str:
    """
    Build a Tesseract TSV string.

    Each *word* is (text, left, top, width, height, conf_0_100).
    """
    lines = [_TSV_HEADER]
    for i, (text, left, top, w, h, conf) in enumerate(words, start=1):
        lines.append(
            f"5\t1\t1\t1\t1\t{i}"
            f"\t{left}\t{top}\t{w}\t{h}"
            f"\t{conf:.1f}\t{text}"
        )
    return "\n".join(lines)


def _make_image(
    width: int = 64,
    height: int = 32,
    color: tuple[int, int, int] = (200, 200, 200),
) -> np.ndarray:
    return np.full((height, width, 3), color, dtype=np.uint8)


def _make_engine(
    *,
    run_fn: Callable[[np.ndarray, str, float], str] | None = None,
    detect_lang_fn: Callable[[str], str] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    confidence_threshold: float = 0.5,
    dpi: int = 96,
) -> TesseractOCREngine:
    """Return a TesseractOCREngine with a no-op sleep and injectable runner."""
    return TesseractOCREngine(
        confidence_threshold=confidence_threshold,
        dpi=dpi,
        _run_fn=run_fn or (lambda img, lang, t: ""),
        _detect_lang_fn=detect_lang_fn or (lambda text: "en"),
        _sleep_fn=sleep_fn or (lambda s: None),
    )


# ---------------------------------------------------------------------------
# 1. OCRResult value object
# ---------------------------------------------------------------------------


class TestOCRResult:
    def test_frozen(self) -> None:
        r = OCRResult("hello", 0.9, Rect(0, 0, 10, 10), "eng")
        with pytest.raises((AttributeError, TypeError)):
            r.text = "world"  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        bbox = Rect(5, 10, 20, 15)
        r = OCRResult(text="test", confidence=0.85, bounding_box=bbox, language="eng")
        assert r.text == "test"
        assert r.confidence == pytest.approx(0.85)
        assert r.bounding_box == bbox
        assert r.language == "eng"

    def test_confidence_zero(self) -> None:
        r = OCRResult("?", 0.0, Rect(0, 0, 1, 1), "unk")
        assert r.confidence == pytest.approx(0.0)

    def test_confidence_one(self) -> None:
        r = OCRResult("A", 1.0, Rect(0, 0, 5, 5), "eng")
        assert r.confidence == pytest.approx(1.0)

    def test_equality(self) -> None:
        r1 = OCRResult("hi", 0.9, Rect(0, 0, 10, 10), "eng")
        r2 = OCRResult("hi", 0.9, Rect(0, 0, 10, 10), "eng")
        assert r1 == r2

    def test_inequality_text(self) -> None:
        r1 = OCRResult("hi", 0.9, Rect(0, 0, 10, 10), "eng")
        r2 = OCRResult("bye", 0.9, Rect(0, 0, 10, 10), "eng")
        assert r1 != r2


# ---------------------------------------------------------------------------
# 2. OCREngine Protocol compliance
# ---------------------------------------------------------------------------


class TestOCREngineProtocol:
    def test_engine_implements_protocol(self) -> None:
        engine = _make_engine()
        assert isinstance(engine, OCREngine)

    def test_protocol_is_runtime_checkable(self) -> None:
        """OCREngine must be decorated with @runtime_checkable."""

        class _Fake:
            def extract(
                self,
                image: np.ndarray,
                region: Rect | None = None,
                languages: list[str] | None = None,
            ) -> list[OCRResult]:
                return []

        assert isinstance(_Fake(), OCREngine)

    def test_non_compliant_not_instance(self) -> None:
        class _Bad:
            pass

        assert not isinstance(_Bad(), OCREngine)


# ---------------------------------------------------------------------------
# 3. Normal extraction (mock subprocess)
# ---------------------------------------------------------------------------


class TestNormalExtraction:
    def test_returns_list(self) -> None:
        engine = _make_engine(
            run_fn=lambda img, lang, t: _tsv(("Hello", 10, 5, 40, 12, 96.0))
        )
        results = engine.extract(_make_image())
        assert isinstance(results, list)

    def test_single_word_result(self) -> None:
        engine = _make_engine(
            run_fn=lambda img, lang, t: _tsv(("Hello", 10, 5, 40, 12, 96.0))
        )
        results = engine.extract(_make_image())
        assert len(results) == 1
        assert results[0].text == "Hello"

    def test_confidence_converted_to_fraction(self) -> None:
        engine = _make_engine(
            run_fn=lambda img, lang, t: _tsv(("OK", 0, 0, 10, 10, 80.0))
        )
        results = engine.extract(_make_image())
        assert results[0].confidence == pytest.approx(0.80)

    def test_multiple_words(self) -> None:
        tsv = _tsv(
            ("Hello", 0, 0, 30, 12, 95.0),
            ("World", 35, 0, 30, 12, 90.0),
        )
        engine = _make_engine(run_fn=lambda img, lang, t: tsv)
        results = engine.extract(_make_image())
        assert len(results) == 2
        texts = {r.text for r in results}
        assert texts == {"Hello", "World"}

    def test_empty_tsv_returns_empty_list(self) -> None:
        engine = _make_engine(run_fn=lambda img, lang, t: "")
        results = engine.extract(_make_image())
        assert results == []

    def test_header_only_tsv_returns_empty(self) -> None:
        engine = _make_engine(run_fn=lambda img, lang, t: _TSV_HEADER)
        results = engine.extract(_make_image())
        assert results == []

    def test_negative_conf_rows_skipped(self) -> None:
        """Rows with conf=-1 (non-word elements) must be excluded."""
        lines = [
            _TSV_HEADER,
            "1\t1\t1\t1\t1\t0\t0\t0\t640\t480\t-1\t",  # block row
            f"5\t1\t1\t1\t1\t1\t5\t5\t30\t12\t92.0\tWord",
        ]
        engine = _make_engine(run_fn=lambda img, lang, t: "\n".join(lines))
        results = engine.extract(_make_image())
        assert len(results) == 1
        assert results[0].text == "Word"

    def test_result_language_set(self) -> None:
        engine = _make_engine(
            run_fn=lambda img, lang, t: _tsv(("A", 0, 0, 10, 10, 90.0)),
            detect_lang_fn=lambda text: "tr",
        )
        results = engine.extract(_make_image())
        assert results[0].language == "tr"


# ---------------------------------------------------------------------------
# 4. Region cropping
# ---------------------------------------------------------------------------


class TestRegionCropping:
    def test_region_offsets_bounding_box(self) -> None:
        """Bounding boxes must be offset to original image coordinates."""
        # Run with DPI=300 to skip scaling (scale=1)
        engine = _make_engine(
            run_fn=lambda img, lang, t: _tsv(("Hi", 0, 0, 20, 10, 95.0)),
            dpi=300,
        )
        region = Rect(50, 30, 200, 100)
        results = engine.extract(_make_image(400, 300), region=region)
        assert len(results) == 1
        # x offset: 0 (in crop) + 50 (region.x) = 50
        assert results[0].bounding_box.x == 50
        assert results[0].bounding_box.y == 30

    def test_out_of_bounds_region_does_not_crash(self) -> None:
        """A region that exceeds image bounds must not raise."""
        engine = _make_engine(run_fn=lambda img, lang, t: "")
        region = Rect(9000, 9000, 100, 100)  # way outside
        results = engine.extract(_make_image(32, 32), region=region)
        assert isinstance(results, list)

    def test_zero_size_region_returns_empty(self) -> None:
        engine = _make_engine(run_fn=lambda img, lang, t: "")
        region = Rect(5, 5, 0, 0)
        results = engine.extract(_make_image(), region=region)
        assert results == []

    def test_no_region_uses_full_image(self) -> None:
        calls: list[tuple[int, int]] = []

        def capture_shape(img: np.ndarray, lang: str, t: float) -> str:
            calls.append((img.shape[0], img.shape[1]))
            return ""

        img = _make_image(100, 80)
        engine = _make_engine(run_fn=capture_shape, dpi=300)
        engine.extract(img)
        h, w = calls[0]
        # No scaling at 300 DPI; image passed through unchanged
        assert h == 80 or w == 100


# ---------------------------------------------------------------------------
# 5. Language selection
# ---------------------------------------------------------------------------


class TestLanguageSelection:
    def test_default_language_is_eng(self) -> None:
        captured: list[str] = []

        def capture_lang(img: np.ndarray, lang: str, t: float) -> str:
            captured.append(lang)
            return ""

        _make_engine(run_fn=capture_lang).extract(_make_image())
        assert captured[0] == "eng"

    def test_explicit_single_language(self) -> None:
        captured: list[str] = []

        def capture_lang(img: np.ndarray, lang: str, t: float) -> str:
            captured.append(lang)
            return ""

        _make_engine(run_fn=capture_lang).extract(
            _make_image(), languages=["tur"]
        )
        assert captured[0] == "tur"

    def test_multiple_languages_joined(self) -> None:
        captured: list[str] = []

        def capture_lang(img: np.ndarray, lang: str, t: float) -> str:
            captured.append(lang)
            return ""

        _make_engine(run_fn=capture_lang).extract(
            _make_image(), languages=["eng", "tur"]
        )
        assert captured[0] == "eng+tur"


# ---------------------------------------------------------------------------
# 6. Low-confidence handling
# ---------------------------------------------------------------------------


class TestLowConfidenceHandling:
    def test_low_confidence_does_not_raise(self) -> None:
        """Confidence below threshold → warning, NOT an exception."""
        engine = _make_engine(
            run_fn=lambda img, lang, t: _tsv(("blurry", 0, 0, 10, 10, 10.0)),
            confidence_threshold=0.9,
        )
        results = engine.extract(_make_image())  # must not raise
        assert isinstance(results, list)

    def test_low_confidence_word_still_returned(self) -> None:
        engine = _make_engine(
            run_fn=lambda img, lang, t: _tsv(("blurry", 0, 0, 10, 10, 20.0)),
            confidence_threshold=0.9,
        )
        results = engine.extract(_make_image())
        assert len(results) == 1
        assert results[0].text == "blurry"
        assert results[0].confidence == pytest.approx(0.20)

    def test_high_confidence_word_returned(self) -> None:
        engine = _make_engine(
            run_fn=lambda img, lang, t: _tsv(("Clear", 0, 0, 30, 12, 99.0)),
            confidence_threshold=0.5,
        )
        results = engine.extract(_make_image())
        assert results[0].confidence == pytest.approx(0.99)

    def test_mixed_confidence_all_returned(self) -> None:
        tsv = _tsv(
            ("Good", 0, 0, 30, 12, 95.0),
            ("Bad", 35, 0, 20, 12, 5.0),
        )
        engine = _make_engine(
            run_fn=lambda img, lang, t: tsv,
            confidence_threshold=0.7,
        )
        results = engine.extract(_make_image())
        assert len(results) == 2


# ---------------------------------------------------------------------------
# 7. Timeout → OCRError
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_timeout_raises_ocr_error(self) -> None:
        def timeout_fn(img: np.ndarray, lang: str, t: float) -> str:
            raise subprocess.TimeoutExpired(cmd="tesseract", timeout=t)

        engine = _make_engine(run_fn=timeout_fn)
        with pytest.raises(OCRError, match="[Tt]imed? ?[Oo]ut"):
            engine.extract(_make_image())

    def test_timeout_error_has_context(self) -> None:
        def timeout_fn(img: np.ndarray, lang: str, t: float) -> str:
            raise subprocess.TimeoutExpired(cmd="tesseract", timeout=t)

        engine = _make_engine(run_fn=timeout_fn)
        with pytest.raises(OCRError) as exc_info:
            engine.extract(_make_image())
        assert "timeout_s" in exc_info.value.context

    def test_timeout_not_retried(self) -> None:
        """TimeoutExpired must propagate immediately — no retries."""
        call_count = 0

        def timeout_fn(img: np.ndarray, lang: str, t: float) -> str:
            nonlocal call_count
            call_count += 1
            raise subprocess.TimeoutExpired(cmd="tesseract", timeout=t)

        engine = _make_engine(run_fn=timeout_fn)
        with pytest.raises(OCRError):
            engine.extract(_make_image())
        assert call_count == 1

    def test_timeout_is_recoverable(self) -> None:
        def timeout_fn(img: np.ndarray, lang: str, t: float) -> str:
            raise subprocess.TimeoutExpired(cmd="tesseract", timeout=t)

        engine = _make_engine(run_fn=timeout_fn)
        with pytest.raises(OCRError) as exc_info:
            engine.extract(_make_image())
        assert exc_info.value.recoverable is True


# ---------------------------------------------------------------------------
# 8. Crash & restart mechanics
# ---------------------------------------------------------------------------


class TestCrashAndRestart:
    def test_crash_raises_ocr_error_after_max_restarts(self) -> None:
        def crash_fn(img: np.ndarray, lang: str, t: float) -> str:
            raise RuntimeError("process died")

        engine = _make_engine(run_fn=crash_fn)
        with pytest.raises(OCRError):
            engine.extract(_make_image())

    def test_crash_retries_exactly_max_restarts_plus_one(self) -> None:
        call_count = 0

        def crash_fn(img: np.ndarray, lang: str, t: float) -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        engine = _make_engine(run_fn=crash_fn)
        with pytest.raises(OCRError):
            engine.extract(_make_image())

        assert call_count == TesseractOCREngine._MAX_RESTARTS + 1

    def test_recovery_after_transient_crash(self) -> None:
        """Engine should succeed if it recovers within _MAX_RESTARTS."""
        attempts = count()

        def flaky_fn(img: np.ndarray, lang: str, t: float) -> str:
            n = next(attempts)
            if n == 0:
                raise RuntimeError("first attempt fails")
            return _tsv(("Recovered", 0, 0, 50, 12, 90.0))

        engine = _make_engine(run_fn=flaky_fn)
        results = engine.extract(_make_image())
        assert len(results) == 1
        assert results[0].text == "Recovered"

    def test_crash_ocr_error_context_has_restarts(self) -> None:
        def crash_fn(img: np.ndarray, lang: str, t: float) -> str:
            raise RuntimeError("crashed")

        engine = _make_engine(run_fn=crash_fn)
        with pytest.raises(OCRError) as exc_info:
            engine.extract(_make_image())
        assert "restarts" in exc_info.value.context

    def test_sleep_called_between_restarts(self) -> None:
        sleep_calls: list[float] = []
        crash_counter = 0

        def crash_fn(img: np.ndarray, lang: str, t: float) -> str:
            nonlocal crash_counter
            crash_counter += 1
            raise RuntimeError("crash")

        engine = _make_engine(
            run_fn=crash_fn,
            sleep_fn=sleep_calls.append,
        )
        with pytest.raises(OCRError):
            engine.extract(_make_image())

        # sleep called between restarts: once per intermediate failure
        assert len(sleep_calls) == TesseractOCREngine._MAX_RESTARTS

    def test_crash_then_success_no_error(self) -> None:
        attempt = [0]

        def sometimes_crash(img: np.ndarray, lang: str, t: float) -> str:
            attempt[0] += 1
            if attempt[0] <= 2:
                raise RuntimeError("not yet")
            return ""

        engine = _make_engine(run_fn=sometimes_crash)
        results = engine.extract(_make_image())  # must not raise
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 9. Turkish post-processing
# ---------------------------------------------------------------------------


class TestTurkishPostprocessing:
    def test_latin1_capital_i_with_dot_fixed(self) -> None:
        assert TesseractOCREngine._postprocess_turkish("\u00dd") == "\u0130"

    def test_latin1_lowercase_dotless_i_fixed(self) -> None:
        assert TesseractOCREngine._postprocess_turkish("\u00fd") == "\u0131"

    def test_latin1_S_cedilla_capital_fixed(self) -> None:
        assert TesseractOCREngine._postprocess_turkish("\u00de") == "\u015e"

    def test_latin1_s_cedilla_lower_fixed(self) -> None:
        assert TesseractOCREngine._postprocess_turkish("\u00fe") == "\u015f"

    def test_null_bytes_removed(self) -> None:
        assert TesseractOCREngine._postprocess_turkish("ab\x00cd") == "abcd"

    def test_replacement_chars_removed(self) -> None:
        assert TesseractOCREngine._postprocess_turkish("x\ufffdy") == "xy"

    def test_clean_text_unchanged(self) -> None:
        text = "Merhaba Dünya"
        assert TesseractOCREngine._postprocess_turkish(text) == text

    def test_turkish_post_applied_when_tur_in_lang(self) -> None:
        """extract(languages=['tur']) must apply Turkish post-processing."""
        artifact = "\u00dd"  # Ý → İ
        engine = _make_engine(
            run_fn=lambda img, lang, t: _tsv((artifact, 0, 0, 10, 10, 90.0)),
            detect_lang_fn=lambda text: "tr",
            dpi=300,
        )
        results = engine.extract(_make_image(), languages=["tur"])
        assert results[0].text == "\u0130"

    def test_turkish_post_not_applied_for_eng(self) -> None:
        """extract(languages=['eng']) must NOT apply Turkish post-processing."""
        artifact = "\u00dd"
        engine = _make_engine(
            run_fn=lambda img, lang, t: _tsv((artifact, 0, 0, 10, 10, 90.0)),
            detect_lang_fn=lambda text: "en",
            dpi=300,
        )
        results = engine.extract(_make_image(), languages=["eng"])
        assert results[0].text == artifact  # unchanged


# ---------------------------------------------------------------------------
# 10. Preprocessing robustness
# ---------------------------------------------------------------------------


class TestPreprocessingRobustness:
    def test_grayscale_2d_input(self) -> None:
        engine = _make_engine()
        gray = np.zeros((32, 64), dtype=np.uint8)
        result = engine._preprocess(gray)
        assert result.ndim == 2

    def test_bgr_3_channel_input(self) -> None:
        engine = _make_engine()
        bgr = np.zeros((32, 64, 3), dtype=np.uint8)
        result = engine._preprocess(bgr)
        assert result.ndim == 2

    def test_bgra_4_channel_input(self) -> None:
        engine = _make_engine()
        bgra = np.zeros((32, 64, 4), dtype=np.uint8)
        result = engine._preprocess(bgra)
        assert result.ndim == 2

    def test_dpi_below_300_upscales(self) -> None:
        engine = _make_engine(dpi=96)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = engine._preprocess(img)
        # 300/96 ≈ 3.125 → should be larger than original
        assert result.shape[0] > 10 or result.shape[1] > 10

    def test_dpi_300_no_upscale(self) -> None:
        engine = _make_engine(dpi=300)
        img = np.zeros((20, 40, 3), dtype=np.uint8)
        result = engine._preprocess(img)
        assert result.shape == (20, 40)

    def test_bad_image_in_extract_returns_empty(self) -> None:
        """If preprocessing raises, extract() returns [] without propagating."""
        engine = _make_engine()
        # Very weird 5D array should fail gracefully
        weird = np.zeros((2, 2, 2, 2, 3), dtype=np.uint8)
        # No OCRError; just empty list
        results = engine.extract(weird)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 11. Hypothesis — arbitrary array → no unexpected exception
# ---------------------------------------------------------------------------


class TestHypothesisNoException:
    """
    Property: for any uint8 array of any reasonable shape, extract() must
    not raise anything other than OCRError (which is allowed, e.g. timeout).
    For these tests _run_fn always returns '' so OCRError is never raised;
    we only verify that preprocessing + parsing do not crash.
    """

    _engine = _make_engine(
        run_fn=lambda img, lang, t: "",
        detect_lang_fn=lambda text: "en",
        sleep_fn=lambda s: None,
    )

    @given(
        arrays(
            dtype=np.uint8,
            shape=st.one_of(
                st.tuples(st.integers(1, 64), st.integers(1, 64)),
                st.tuples(st.integers(1, 64), st.integers(1, 64), st.just(3)),
                st.tuples(st.integers(1, 64), st.integers(1, 64), st.just(4)),
                st.tuples(st.integers(1, 64), st.integers(1, 64), st.just(1)),
            ),
        )
    )
    @settings(max_examples=300)
    def test_no_exception_on_arbitrary_image(self, arr: np.ndarray) -> None:
        results = self._engine.extract(arr)
        assert isinstance(results, list)

    @given(
        arrays(dtype=np.uint8, shape=st.tuples(
            st.integers(1, 32), st.integers(1, 32), st.just(3)
        )),
        st.one_of(st.none(), st.just(["eng"]), st.just(["tur"]), st.just(["eng", "tur"])),
    )
    @settings(max_examples=150)
    def test_no_exception_various_languages(
        self, arr: np.ndarray, languages: list[str] | None
    ) -> None:
        results = self._engine.extract(arr, languages=languages)
        assert isinstance(results, list)

    @given(
        arrays(dtype=np.uint8, shape=st.tuples(
            st.integers(1, 64), st.integers(1, 64), st.just(3)
        )),
        st.integers(0, 100).map(lambda x: x + 1),   # dpi 1..101
    )
    @settings(max_examples=100)
    def test_no_exception_various_dpi(
        self, arr: np.ndarray, dpi: int
    ) -> None:
        engine = _make_engine(
            run_fn=lambda img, lang, t: "",
            dpi=dpi,
        )
        results = engine.extract(arr)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 12. Real Tesseract accuracy (skipped when binary unavailable)
# ---------------------------------------------------------------------------

_TESSERACT_AVAILABLE = False
try:
    import pytesseract as _pyt

    _pyt.get_tesseract_version()
    _TESSERACT_AVAILABLE = True
except Exception:
    pass

_skip_no_tess = pytest.mark.skipif(
    not _TESSERACT_AVAILABLE,
    reason="Tesseract binary not installed",
)


def _render_text_image(
    text: str,
    width: int = 640,
    height: int = 120,
    font_size: int = 40,
) -> np.ndarray:
    """Render *text* onto a white PIL image and return as BGR ndarray."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 30), text, fill=(0, 0, 0), font=font)
    return np.array(img)


@pytest.fixture()
def ocr_english() -> np.ndarray:
    return _render_text_image("Hello World Testing 123")


@pytest.fixture()
def ocr_turkish() -> np.ndarray:
    return _render_text_image("İstanbul Şehri Güzel")


@pytest.fixture()
def ocr_blurry() -> np.ndarray:
    import cv2

    base = _render_text_image("Blurry Text Example")
    return cv2.GaussianBlur(base, (15, 15), 5)


def _char_accuracy(expected: str, actual: str) -> float:
    """Naive character-level accuracy (Jaccard of character multisets)."""
    from collections import Counter

    ec = Counter(expected.replace(" ", "").lower())
    ac = Counter(actual.replace(" ", "").lower())
    intersection = sum((ec & ac).values())
    union = sum((ec | ac).values())
    return intersection / union if union else 1.0


class TestRealTesseract:
    @_skip_no_tess
    def test_english_extraction_returns_results(self, ocr_english: np.ndarray) -> None:
        engine = TesseractOCREngine(dpi=96)
        results = engine.extract(ocr_english, languages=["eng"])
        assert len(results) > 0

    @_skip_no_tess
    def test_turkish_character_accuracy(self, ocr_turkish: np.ndarray) -> None:
        """Türkçe karakter doğruluğu >%95 olmalı."""
        expected = "İstanbul Şehri Güzel"
        engine = TesseractOCREngine(dpi=96)
        results = engine.extract(ocr_turkish, languages=["tur"])
        actual = " ".join(r.text for r in results)
        accuracy = _char_accuracy(expected, actual)
        assert accuracy >= 0.95, (
            f"Turkish accuracy {accuracy:.1%} < 95%. "
            f"Expected: {expected!r}, Got: {actual!r}"
        )

    @_skip_no_tess
    def test_blurry_no_exception(self, ocr_blurry: np.ndarray) -> None:
        """Düşük kalite görüntü → exception yok."""
        engine = TesseractOCREngine(dpi=96)
        results = engine.extract(ocr_blurry, languages=["eng"])
        assert isinstance(results, list)

    @_skip_no_tess
    def test_confidence_in_valid_range(self, ocr_english: np.ndarray) -> None:
        engine = TesseractOCREngine(dpi=96)
        results = engine.extract(ocr_english, languages=["eng"])
        for r in results:
            assert 0.0 <= r.confidence <= 1.0
