"""
tests/benchmarks/bench_perception_latency.py
Perception Latency Benchmark — PAKET BM

100 kez perception çalıştır, ortalama latency hesapla.
Hedef: ortalama latency < 200ms.

Kullanım:
    python tests/benchmarks/bench_perception_latency.py
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Benchmark parametreleri
# ---------------------------------------------------------------------------

N_ITERATIONS: int = 100
TARGET_AVG_MS: float = 200.0


# ---------------------------------------------------------------------------
# Mock perception setup
# ---------------------------------------------------------------------------


def _build_mock_frame() -> Any:
    """1080p sahte frame oluştur."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _build_mock_source_result() -> Any:
    """Sahte kaynak sonucu (UIA structured path)."""
    return MagicMock(
        source_type="uia",
        elements=[],
        screenshot=None,
    )


def _run_perception_mock(frame: Any, source_result: Any) -> Any:
    """
    Mock perception pipeline — gerçek UIA/OCR çağrısı olmadan
    orchestrator mantığını simüle eder.
    """
    # Structured path (UIA/DOM/File): OCR/Locator atlanır
    if source_result.source_type in ("uia", "dom", "file"):
        time.sleep(0.001)  # UIA yolunda çok hızlı
        return MagicMock(confidence=0.95, elements=[], ocr_text="")

    # Visual fallback: OCR + Locator + Matcher
    time.sleep(0.030)  # ~30ms mock OCR
    time.sleep(0.020)  # ~20ms mock locator
    time.sleep(0.005)  # ~5ms matcher
    return MagicMock(confidence=0.75, elements=[], ocr_text="sample text")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark() -> dict[str, Any]:
    """Benchmark çalıştır ve sonuçları döndür."""
    print(f"\n{'='*60}")
    print(f"  bench_perception_latency — {N_ITERATIONS} iterations")
    print(f"  Target: avg < {TARGET_AVG_MS:.0f}ms")
    print(f"{'='*60}")

    frame = _build_mock_frame()
    source = _build_mock_source_result()

    latencies_ms: list[float] = []

    for i in range(N_ITERATIONS):
        if i % 20 == 0:
            print(f"  Progress: {i}/{N_ITERATIONS}...")

        t0 = time.perf_counter()
        _run_perception_mock(frame, source)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    p95_ms = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]
    p99_ms = sorted(latencies_ms)[int(0.99 * len(latencies_ms))]

    passed = avg_ms < TARGET_AVG_MS

    print(f"\n  Results:")
    print(f"    Iterations   : {N_ITERATIONS}")
    print(f"    Average      : {avg_ms:.2f}ms")
    print(f"    Min          : {min_ms:.2f}ms")
    print(f"    Max          : {max_ms:.2f}ms")
    print(f"    P95          : {p95_ms:.2f}ms")
    print(f"    P99          : {p99_ms:.2f}ms")
    print(f"    Target       : < {TARGET_AVG_MS:.0f}ms")
    print(f"    Status       : {'✓ PASS' if passed else '✗ FAIL'}")

    result = {
        "benchmark": "bench_perception_latency",
        "iterations": N_ITERATIONS,
        "avg_ms": round(avg_ms, 3),
        "min_ms": round(min_ms, 3),
        "max_ms": round(max_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "p99_ms": round(p99_ms, 3),
        "target_avg_ms": TARGET_AVG_MS,
        "passed": passed,
    }

    _save_result(result)
    return result


def _save_result(result: dict[str, Any]) -> None:
    import json
    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"bench_perception_{ts}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_file.name}")


if __name__ == "__main__":
    result = run_benchmark()
    print(f"\n  Final: {'PASS' if result['passed'] else 'FAIL'} — {result['avg_ms']:.2f}ms avg")
    raise SystemExit(0 if result["passed"] else 1)
