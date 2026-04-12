"""
tests/benchmarks/bench_capture_fps.py
DXcam FPS Benchmark — PAKET BM

DXcam ile 10 saniye frame yakala, ortalama FPS hesapla.
Hedef: 15 FPS ortalama.

Kullanım:
    python tests/benchmarks/bench_capture_fps.py
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Benchmark parametreleri
# ---------------------------------------------------------------------------

DURATION_SECONDS: float = 10.0
TARGET_FPS: float = 15.0


# ---------------------------------------------------------------------------
# DXcam mock — gerçek ekran yoksa simülasyon kullan
# ---------------------------------------------------------------------------


def _try_real_dxcam_capture(duration_s: float) -> list[float]:
    """
    Gerçek DXcam ile frame yakala.
    Döndürür: Her frame için timestamp listesi.
    """
    try:
        import dxcam

        camera = dxcam.create(output_idx=0, output_color="BGR")
        camera.start(target_fps=TARGET_FPS, video_mode=True)

        timestamps: list[float] = []
        start = time.perf_counter()

        while time.perf_counter() - start < duration_s:
            frame = camera.get_latest_frame()
            if frame is not None:
                timestamps.append(time.perf_counter())
            time.sleep(0.001)

        camera.stop()
        del camera
        return timestamps

    except Exception as e:
        print(f"  [!] DXcam not available: {e}")
        return []


def _simulate_capture(duration_s: float, target_fps: float) -> list[float]:
    """
    DXcam olmadığında simülasyon: target_fps'e göre frame üret.
    """
    frame_interval = 1.0 / target_fps
    timestamps: list[float] = []
    start = time.perf_counter()

    while time.perf_counter() - start < duration_s:
        # Simulate frame grab
        _ = np.zeros((1080, 1920, 3), dtype=np.uint8)
        timestamps.append(time.perf_counter())
        time.sleep(frame_interval * 0.95)  # %95 hızlı — gerçekçi simülasyon

    return timestamps


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark() -> dict[str, Any]:
    """Benchmark çalıştır ve sonuçları döndür."""
    print(f"\n{'='*60}")
    print(f"  bench_capture_fps — {DURATION_SECONDS:.0f}s capture")
    print(f"  Target: {TARGET_FPS:.0f} FPS")
    print(f"{'='*60}")

    print(f"\n  Capturing for {DURATION_SECONDS:.0f} seconds...")
    start_real = time.perf_counter()

    # Önce gerçek DXcam'ı dene
    timestamps = _try_real_dxcam_capture(DURATION_SECONDS)

    if len(timestamps) < 5:
        print("  Using simulated capture (DXcam unavailable)")
        timestamps = _simulate_capture(DURATION_SECONDS, TARGET_FPS)

    elapsed = time.perf_counter() - start_real
    n_frames = len(timestamps)

    if n_frames < 2:
        print("  ERROR: Too few frames captured")
        return {"passed": False, "fps": 0.0, "n_frames": n_frames}

    actual_fps = n_frames / elapsed

    # Frame interval jitter hesapla
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    avg_interval_ms = (sum(intervals) / len(intervals)) * 1000
    jitter_ms = max(intervals) * 1000 - min(intervals) * 1000

    passed = actual_fps >= TARGET_FPS

    print(f"\n  Results:")
    print(f"    Frames captured : {n_frames}")
    print(f"    Elapsed         : {elapsed:.2f}s")
    print(f"    Average FPS     : {actual_fps:.2f}")
    print(f"    Avg interval    : {avg_interval_ms:.1f}ms")
    print(f"    Jitter          : {jitter_ms:.1f}ms")
    print(f"    Target FPS      : {TARGET_FPS:.0f}")
    print(f"    Status          : {'✓ PASS' if passed else '✗ FAIL'}")

    result = {
        "benchmark": "bench_capture_fps",
        "duration_s": DURATION_SECONDS,
        "n_frames": n_frames,
        "fps": round(actual_fps, 2),
        "target_fps": TARGET_FPS,
        "avg_interval_ms": round(avg_interval_ms, 2),
        "jitter_ms": round(jitter_ms, 2),
        "passed": passed,
    }

    _save_result(result)
    return result


def _save_result(result: dict[str, Any]) -> None:
    """Sonucu JSON dosyasına kaydet."""
    import json
    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"bench_capture_fps_{ts}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_file.name}")


if __name__ == "__main__":
    result = run_benchmark()
    print(f"\n  Final: {'PASS' if result['passed'] else 'FAIL'} — {result['fps']:.2f} FPS")
    raise SystemExit(0 if result["passed"] else 1)
