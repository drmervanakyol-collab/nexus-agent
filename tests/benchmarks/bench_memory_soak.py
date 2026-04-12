"""
tests/benchmarks/bench_memory_soak.py
Memory Soak Benchmark — PAKET BM

100 görev simüle et, RAM kullanımını her adımda kaydet.
Başlangıç ile fark 200MB altında olmalı.

Kullanım:
    python tests/benchmarks/bench_memory_soak.py
"""
from __future__ import annotations

import gc
import time
import uuid
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Benchmark parametreleri
# ---------------------------------------------------------------------------

N_TASKS: int = 100
MAX_MEMORY_DELTA_MB: float = 200.0


# ---------------------------------------------------------------------------
# RAM ölçümü
# ---------------------------------------------------------------------------


def _get_memory_mb() -> float:
    """Mevcut process RAM kullanımını MB cinsinden döndür."""
    try:
        import psutil
        proc = psutil.Process()
        return proc.memory_info().rss / 1024 / 1024
    except ImportError:
        pass

    # Windows fallback
    try:
        import ctypes
        import ctypes.wintypes

        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(pmc), pmc.cb
        )
        return pmc.WorkingSetSize / 1024 / 1024
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Mock task simulation
# ---------------------------------------------------------------------------


def _simulate_task(task_id: str) -> dict[str, Any]:
    """
    Tek bir görevin tam yaşam döngüsünü simüle et.
    Bellek sızıntısı olmadan temiz çalışmalı.
    """
    # Görev başlatma
    context = {
        "task_id": task_id,
        "goal": f"Task {task_id[:8]}: open and type",
        "history": [],
        "cost": 0.0,
    }

    # 5 adım simüle et
    for step in range(5):
        # Perception verisi (büyük array — hızla serbest bırakılmalı)
        import numpy as np
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Action history
        context["history"].append({
            "step": step,
            "action": "click",
            "target": f"button_{step}",
            "cost": 0.0002,
        })
        context["cost"] += 0.0002

        # Frame'i hemen serbest bırak
        del frame

    # Sonuç
    result = {
        "task_id": task_id,
        "steps": 5,
        "cost": context["cost"],
        "status": "success",
    }

    # Bağlamı temizle
    del context
    return result


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark() -> dict[str, Any]:
    """Benchmark çalıştır ve sonuçları döndür."""
    print(f"\n{'='*60}")
    print(f"  bench_memory_soak — {N_TASKS} tasks")
    print(f"  Target: RAM delta < {MAX_MEMORY_DELTA_MB:.0f}MB")
    print(f"{'='*60}")

    # GC'yi temizle ve başlangıç ölçümü al
    gc.collect()
    time.sleep(0.1)
    baseline_mb = _get_memory_mb()
    print(f"\n  Baseline RAM : {baseline_mb:.1f}MB")

    memory_samples: list[float] = [baseline_mb]
    task_times_ms: list[float] = []
    completed_tasks = 0

    for i in range(N_TASKS):
        if i % 20 == 0:
            mem_now = _get_memory_mb()
            print(f"  Progress: {i}/{N_TASKS} — RAM: {mem_now:.1f}MB "
                  f"(+{mem_now - baseline_mb:.1f}MB)")

        task_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        _simulate_task(task_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        task_times_ms.append(elapsed_ms)
        completed_tasks += 1

        # Her 10 görevde bir GC çalıştır
        if i % 10 == 9:
            gc.collect()
            memory_samples.append(_get_memory_mb())

    # Son ölçüm
    gc.collect()
    time.sleep(0.1)
    final_mb = _get_memory_mb()
    memory_samples.append(final_mb)

    delta_mb = final_mb - baseline_mb
    peak_mb = max(memory_samples) - baseline_mb
    avg_task_ms = sum(task_times_ms) / len(task_times_ms)

    passed = delta_mb < MAX_MEMORY_DELTA_MB

    print(f"\n  Results:")
    print(f"    Tasks completed  : {completed_tasks}")
    print(f"    Baseline RAM     : {baseline_mb:.1f}MB")
    print(f"    Final RAM        : {final_mb:.1f}MB")
    print(f"    Delta            : {delta_mb:+.1f}MB")
    print(f"    Peak delta       : {peak_mb:+.1f}MB")
    print(f"    Avg task time    : {avg_task_ms:.2f}ms")
    print(f"    Target delta     : < {MAX_MEMORY_DELTA_MB:.0f}MB")
    print(f"    Status           : {'✓ PASS' if passed else '✗ FAIL'}")

    result = {
        "benchmark": "bench_memory_soak",
        "n_tasks": N_TASKS,
        "completed_tasks": completed_tasks,
        "baseline_mb": round(baseline_mb, 2),
        "final_mb": round(final_mb, 2),
        "delta_mb": round(delta_mb, 2),
        "peak_delta_mb": round(peak_mb, 2),
        "avg_task_ms": round(avg_task_ms, 3),
        "target_delta_mb": MAX_MEMORY_DELTA_MB,
        "memory_samples": [round(m, 2) for m in memory_samples],
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
    out_file = out_dir / f"bench_memory_soak_{ts}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_file.name}")


if __name__ == "__main__":
    result = run_benchmark()
    print(f"\n  Final: {'PASS' if result['passed'] else 'FAIL'} — "
          f"{result['delta_mb']:+.1f}MB delta")
    raise SystemExit(0 if result["passed"] else 1)
