"""
nexus/perception/reader/ocr_engine.py
OCR Engine — extracts text from screen regions.

OCREngine (Protocol)
--------------------
  extract(image, region, languages) -> list[OCRResult]

TesseractOCREngine
------------------
Pipeline for each extract() call:
  1. Crop to region (if specified)
  2. Preprocess — grayscale, CLAHE contrast, Gaussian denoise, DPI-aware scale
  3. Language selection — explicit list or auto-detect via langdetect
  4. Run Tesseract subprocess (timeout=10 s, crash → restart up to _MAX_RESTARTS)
  5. Parse TSV output → list[OCRResult]; low confidence → warning, partial result
  6. Turkish post-processing when "tur" in languages
  7. Re-offset bounding boxes to original image coordinates

Errors
------
OCRError (from nexus.core.errors):
  - raised on timeout
  - raised after _MAX_RESTARTS subprocess crashes

Low confidence results are NEVER silently dropped or exception-raised;
they are returned with a structured-log warning at WARNING level.
"""
from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from nexus.core.errors import OCRError
from nexus.core.types import Rect
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DPI: int = 96
_TARGET_DPI: int = 300
_MAX_SCALE: float = 4.0          # cap on DPI-aware upscaling
_CLAHE_CLIP_LIMIT: float = 2.0
_CLAHE_TILE: tuple[int, int] = (8, 8)
_GAUSSIAN_KERNEL: tuple[int, int] = (3, 3)
_LOW_CONF_THRESHOLD: float = 0.3  # below this → WARNING logged

# Common OCR misreadings for Turkish characters (encoding artifacts)
_TURKISH_OCR_FIXES: tuple[tuple[str, str], ...] = (
    ("\u00dd", "\u0130"),   # Ý  → İ  (Latin-1 artifact)
    ("\u00fd", "\u0131"),   # ý  → ı  (Latin-1 artifact)
    ("\u00de", "\u015e"),   # Þ  → Ş  (Latin-1 artifact)
    ("\u00fe", "\u015f"),   # þ  → ş  (Latin-1 artifact)
    ("\u00f0", "\u011f"),   # ð  → ğ  (Latin-1 artifact)
    ("\u00d0", "\u011e"),   # Ð  → Ğ  (Latin-1 artifact)
    ("\u00fc\u0308", "\u00fc"),  # ü + combining umlaut → ü
    ("\u006f\u0308", "\u00f6"),  # o + combining umlaut → ö
    ("\u0063\u0327", "\u00e7"),  # c + cedilla → ç
    ("\u0043\u0327", "\u00c7"),  # C + cedilla → Ç
    ("\x00", ""),           # null bytes
    ("\ufffd", ""),         # Unicode replacement characters
)


# ---------------------------------------------------------------------------
# OCRResult value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OCRResult:
    """
    A single word-level OCR recognition result.

    Attributes
    ----------
    text:
        Recognised text string.
    confidence:
        Confidence score in [0.0, 1.0].  Values below the engine threshold
        are included but logged at WARNING level.
    bounding_box:
        Position of this word in the *original* (un-cropped, un-scaled)
        image coordinates.
    language:
        ISO 639-1/3 language code; detected by langdetect or inferred from
        the ``languages`` parameter passed to ``extract()``.
    """

    text: str
    confidence: float
    bounding_box: Rect
    language: str


# ---------------------------------------------------------------------------
# OCREngine Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OCREngine(Protocol):
    """
    Minimal interface for OCR engines.

    Parameters
    ----------
    image:
        Raw uint8 numpy array — BGR, BGRA, RGB, RGBA, or grayscale.
    region:
        Optional crop rectangle in *image* pixel coordinates.  When provided
        the bounding boxes in the returned results are offset so they remain
        in original image space.
    languages:
        List of BCP-47 / Tesseract language codes (e.g. ``["eng", "tur"]``).
        When ``None`` the engine picks the best available default.

    Returns
    -------
    list[OCRResult]
        Possibly empty; never raises for low-confidence results.
    """

    def extract(
        self,
        image: np.ndarray,
        region: Rect | None = None,
        languages: list[str] | None = None,
    ) -> list[OCRResult]: ...


# ---------------------------------------------------------------------------
# Default subprocess runner
# ---------------------------------------------------------------------------


def _default_run_tesseract(
    image: np.ndarray,
    lang: str,
    timeout_s: float,
) -> str:
    """
    Write *image* to a temp file, invoke the ``tesseract`` binary, and
    return its TSV stdout.  Raises ``subprocess.TimeoutExpired`` on timeout.
    """
    import os
    import tempfile

    from PIL import Image as _PILImage

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pil_img = _PILImage.fromarray(image)
        pil_img.save(tmp_path)

        result = subprocess.run(
            [
                "tesseract",
                tmp_path,
                "stdout",
                "-l",
                lang,
                "--psm",
                "3",
                "tsv",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        if result.returncode != 0:
            raise subprocess.SubprocessError(
                f"tesseract exited {result.returncode}: {result.stderr[:200]}"
            )

        return result.stdout

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Default language detector
# ---------------------------------------------------------------------------


def _detect_language(text: str) -> str:
    """Detect the primary language of *text* using langdetect."""
    try:
        from langdetect import detect  # type: ignore[import-untyped]

        return detect(text)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# TesseractOCREngine
# ---------------------------------------------------------------------------


class TesseractOCREngine:
    """
    OCR engine that drives Tesseract as an external subprocess.

    Parameters
    ----------
    confidence_threshold:
        Results with confidence < this value are included but logged at
        WARNING level.  Default: 0.5.
    dpi:
        Capture DPI of the source screen.  Used to scale the image up to
        ``_TARGET_DPI`` (300) before sending to Tesseract.  Default: 96.
    _run_fn:
        ``(image: np.ndarray, lang: str, timeout_s: float) -> str``
        Injectable replacement for the real tesseract subprocess call.
        Must return a Tesseract TSV string.
    _detect_lang_fn:
        ``(text: str) -> str``
        Injectable language detector.  Defaults to langdetect.
    _sleep_fn:
        ``(seconds: float) -> None``
        Injectable sleep for back-off between restarts.
    """

    _MAX_RESTARTS: int = 3
    _TIMEOUT_S: float = 10.0

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        dpi: int = _DEFAULT_DPI,
        *,
        _run_fn: Callable[[np.ndarray, str, float], str] | None = None,
        _detect_lang_fn: Callable[[str], str] | None = None,
        _sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._dpi = dpi
        self._run_fn: Callable[[np.ndarray, str, float], str] = (
            _run_fn or _default_run_tesseract
        )
        self._detect_lang: Callable[[str], str] = (
            _detect_lang_fn or _detect_language
        )
        self._sleep: Callable[[float], None] = _sleep_fn or time.sleep

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        image: np.ndarray,
        region: Rect | None = None,
        languages: list[str] | None = None,
    ) -> list[OCRResult]:
        """
        Extract text from *image*, optionally restricted to *region*.

        Never raises for preprocessing errors or low-confidence results.
        Raises ``OCRError`` only on subprocess timeout or exhausted restarts.
        """
        # 1. Crop to region
        crop_offset: tuple[int, int] = (0, 0)
        work_image = image
        if region is not None:
            work_image, crop_offset = self._crop(image, region)

        # 2. Preprocess
        try:
            processed = self._preprocess(work_image)
        except Exception as exc:
            _log.warning("ocr_preprocess_failed", error=str(exc))
            return []

        # 3. Language selection
        lang = "+".join(languages) if languages else "eng"

        # 4. Run tesseract with restart on crash
        tsv = self._run_with_restart(processed, lang)

        # 5. Parse TSV → OCRResult list
        results = self._parse_tsv(tsv, lang, offset=crop_offset)

        # 6. Turkish post-processing
        if "tur" in lang:
            results = [
                OCRResult(
                    text=self._postprocess_turkish(r.text),
                    confidence=r.confidence,
                    bounding_box=r.bounding_box,
                    language=r.language,
                )
                for r in results
            ]

        # 7. Detect language for metadata (best-effort)
        all_text = " ".join(r.text for r in results if r.text.strip())
        if all_text:
            try:
                detected = self._detect_lang(all_text)
            except Exception:
                detected = lang
            results = [
                OCRResult(
                    text=r.text,
                    confidence=r.confidence,
                    bounding_box=r.bounding_box,
                    language=detected,
                )
                for r in results
            ]

        return results

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale → CLAHE contrast → Gaussian denoise →
        DPI-aware upscale.  Returns a uint8 grayscale ndarray.
        """
        # Ensure uint8
        img = np.asarray(image, dtype=np.uint8)

        # Convert to grayscale
        if img.ndim == 2:
            gray = img
        elif img.ndim == 3 and img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3 and img.shape[2] == 1:
            gray = img[:, :, 0]
        else:
            # Flatten to 2D best-effort
            gray = img.reshape(img.shape[0], -1).astype(np.uint8)

        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=_CLAHE_CLIP_LIMIT,
            tileGridSize=_CLAHE_TILE,
        )
        gray = clahe.apply(gray)

        # Gaussian denoise
        gray = cv2.GaussianBlur(gray, _GAUSSIAN_KERNEL, 0)

        # DPI-aware upscale (target: 300 DPI)
        if self._dpi < _TARGET_DPI:
            scale = min(_TARGET_DPI / self._dpi, _MAX_SCALE)
            h, w = gray.shape[:2]
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            gray = cv2.resize(
                gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC
            )

        return gray

    # ------------------------------------------------------------------
    # Subprocess with restart
    # ------------------------------------------------------------------

    def _run_with_restart(self, image: np.ndarray, lang: str) -> str:
        """
        Call ``_run_fn`` and retry on crash up to ``_MAX_RESTARTS`` times.
        Timeout propagates immediately as ``OCRError``.
        """
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RESTARTS + 1):
            try:
                return self._run_fn(image, lang, self._TIMEOUT_S)
            except subprocess.TimeoutExpired as exc:
                raise OCRError(
                    "Tesseract timed out",
                    context={"timeout_s": self._TIMEOUT_S},
                ) from exc
            except Exception as exc:
                last_exc = exc
                _log.warning(
                    "tesseract_crash",
                    attempt=attempt,
                    max_restarts=self._MAX_RESTARTS,
                    error=str(exc),
                )
                if attempt < self._MAX_RESTARTS:
                    self._sleep(0.05 * (attempt + 1))

        raise OCRError(
            f"Tesseract crashed after {self._MAX_RESTARTS} restarts",
            context={
                "restarts": self._MAX_RESTARTS,
                "error": str(last_exc),
            },
        ) from last_exc

    # ------------------------------------------------------------------
    # TSV parser
    # ------------------------------------------------------------------

    def _parse_tsv(
        self,
        tsv: str,
        lang: str,
        offset: tuple[int, int] = (0, 0),
    ) -> list[OCRResult]:
        """
        Parse Tesseract TSV output into ``OCRResult`` objects.

        Low-confidence words (below ``_confidence_threshold``) are included
        but trigger a structured-log WARNING.
        """
        results: list[OCRResult] = []
        lines = tsv.strip().splitlines()
        if len(lines) < 2:
            return results

        ox, oy = offset

        for line in lines[1:]:  # skip header row
            parts = line.split("\t")
            if len(parts) < 12:
                continue

            conf_str = parts[10].strip()
            text = parts[11].strip() if len(parts) > 11 else ""

            if not text or conf_str in ("-1", ""):
                continue

            try:
                conf_raw = float(conf_str)
                x = int(parts[6])
                y = int(parts[7])
                w = int(parts[8])
                h = int(parts[9])
            except (ValueError, IndexError):
                continue

            if conf_raw < 0:
                continue

            confidence = conf_raw / 100.0

            if confidence < _LOW_CONF_THRESHOLD:
                _log.warning(
                    "ocr_low_confidence",
                    confidence=round(confidence, 3),
                    text_preview=text[:30],
                )
            elif confidence < self._confidence_threshold:
                _log.warning(
                    "ocr_below_threshold",
                    confidence=round(confidence, 3),
                    threshold=self._confidence_threshold,
                    text_preview=text[:30],
                )

            # Reverse DPI scaling on bounding box coordinates
            scale = (
                min(_TARGET_DPI / self._dpi, _MAX_SCALE)
                if self._dpi < _TARGET_DPI
                else 1.0
            )
            bx = max(0, int(x / scale)) + ox
            by = max(0, int(y / scale)) + oy
            bw = max(0, int(w / scale))
            bh = max(0, int(h / scale))

            results.append(
                OCRResult(
                    text=text,
                    confidence=confidence,
                    bounding_box=Rect(bx, by, bw, bh),
                    language=lang,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Turkish post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess_turkish(text: str) -> str:
        """Fix common Latin-1 encoding artifacts that masquerade as Turkish chars."""
        for wrong, correct in _TURKISH_OCR_FIXES:
            text = text.replace(wrong, correct)
        return text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _crop(
        image: np.ndarray,
        region: Rect,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Crop *image* to *region*, clamping to image bounds.

        Returns (cropped_image, (offset_x, offset_y)).
        """
        h, w = image.shape[:2]
        x1 = max(0, region.x)
        y1 = max(0, region.y)
        x2 = min(w, region.x + region.width)
        y2 = min(h, region.y + region.height)
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = np.zeros((1, 1, *image.shape[2:]), dtype=np.uint8)
        return cropped, (x1, y1)
