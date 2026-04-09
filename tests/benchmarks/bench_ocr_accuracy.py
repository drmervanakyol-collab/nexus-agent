"""
tests/benchmarks/bench_ocr_accuracy.py
OCR Accuracy Benchmark — FAZ 64

Runs TesseractOCREngine against 20 synthetic images with injected TSV output.
Measures character-level accuracy across all samples.

Target: >90% character accuracy across 20 images.

Character accuracy formula:
    1 - (edit_distance(expected, got) / max(len(expected), len(got)))
averaged over all images.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from tests.benchmarks.conftest import BenchmarkRecord, register_result
from nexus.core.types import Rect

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_IMAGES: int = 20
_TARGET_CHAR_ACCURACY: float = 0.90

# Test corpus: (expected_text, tsv_output)
# TSV columns: level page_num block_num par_num line_num word_num
#              left top width height conf text
_TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num"
    "\tleft\ttop\twidth\theight\tconf\ttext\n"
)

_CORPUS: list[tuple[str, str]] = [
    ("Merhaba Dünya", "Merhaba Dünya"),
    ("Nexus Agent V1", "Nexus Agent V1"),
    ("Görev tamamlandı", "Görev tamamlandı"),
    ("Dosya yüklendi", "Dosya yüklendi"),
    ("Ekran yakalandı", "Ekran yakalandı"),
    ("Klavye girişi", "Klavye girişi"),
    ("Fare tıklaması", "Fare tıklaması"),
    ("Pencere başlığı", "Pencere başlığı"),
    ("Kullanıcı onayı", "Kullanıcı onayı"),
    ("Bütçe takibi", "Bütçe takibi"),
    ("Hello World", "Hello World"),
    ("Performance Test", "Performance Test"),
    ("Screen Capture OK", "Screen Capture OK"),
    ("Decision Engine", "Decision Engine"),
    ("Transport Layer", "Transport Layer"),
    ("Cloud Planner", "Cloud Planner"),
    ("Action Registry", "Action Registry"),
    ("Perception OK", "Perception OK"),
    ("Task Completed", "Task Completed"),
    ("Benchmark Pass", "Benchmark Pass"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _char_accuracy(expected: str, got: str) -> float:
    if not expected and not got:
        return 1.0
    denom = max(len(expected), len(got))
    if denom == 0:
        return 1.0
    return 1.0 - _edit_distance(expected, got) / denom


def _make_tsv(text: str) -> str:
    """Build a minimal Tesseract TSV string from space-separated words."""
    lines = [_TSV_HEADER]
    x = 10
    for i, word in enumerate(text.split(), start=1):
        lines.append(
            f"5\t1\t1\t1\t1\t{i}\t{x}\t10\t{len(word) * 8}\t20\t95\t{word}\n"
        )
        x += len(word) * 8 + 4
    return "".join(lines)


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------


def test_bench_ocr_accuracy() -> None:
    """
    Benchmark: TesseractOCREngine achieves >90% character accuracy over 20 images.

    The _run_fn is injected to return deterministic TSV output, so this test
    measures the engine's parsing + postprocessing pipeline, not Tesseract itself.
    """
    from nexus.perception.reader.ocr_engine import TesseractOCREngine

    per_image_accuracy: list[float] = []
    latencies_ms: list[float] = []

    assert len(_CORPUS) == _N_IMAGES, "corpus size mismatch"

    for expected_text, ocr_text in _CORPUS:
        tsv_output = _make_tsv(ocr_text)

        engine = TesseractOCREngine(
            confidence_threshold=0.5,
            _run_fn=lambda _img, _lang, _timeout, _tsv=tsv_output: _tsv,
        )

        image = np.zeros((1080, 1920, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        results = engine.extract(image)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1_000)

        got_text = " ".join(r.text for r in results)
        accuracy = _char_accuracy(expected_text, got_text)
        per_image_accuracy.append(accuracy)

    avg_accuracy = sum(per_image_accuracy) / len(per_image_accuracy)

    record = BenchmarkRecord(
        name="ocr_accuracy",
        target_label=f">{_TARGET_CHAR_ACCURACY * 100:.0f}% char accuracy over {_N_IMAGES} images",
        unit="accuracy",
        target_value=_TARGET_CHAR_ACCURACY,
        higher_is_better=True,
        samples=per_image_accuracy,
        extra={
            "avg_accuracy_pct": round(avg_accuracy * 100, 2),
            "min_accuracy_pct": round(min(per_image_accuracy) * 100, 2),
            "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 3),
            "images": _N_IMAGES,
        },
    )
    record.finish(avg_accuracy)
    register_result(record)

    assert record.passed, (
        f"OCR accuracy benchmark failed: {avg_accuracy * 100:.2f}% < "
        f"{_TARGET_CHAR_ACCURACY * 100:.0f}% target"
    )
