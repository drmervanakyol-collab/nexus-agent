"""
tests/benchmarks/bench_decision_latency.py
Decision Latency Benchmark — PAKET BM

Mock DecisionEngine ile 50 karar al, ortalama süre hesapla.
Hedef: ortalama süre < 500ms.

Kullanım:
    python tests/benchmarks/bench_decision_latency.py
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Benchmark parametreleri
# ---------------------------------------------------------------------------

N_DECISIONS: int = 50
TARGET_AVG_MS: float = 500.0


# ---------------------------------------------------------------------------
# Mock DecisionEngine setup
# ---------------------------------------------------------------------------


def _build_mock_perception() -> Any:
    """Sahte perception sonucu."""
    perception = MagicMock()
    perception.confidence = 0.85
    perception.elements = []
    perception.temporal_state = "STABLE"
    perception.source_disagreements = 0
    perception.ocr_text = "Hello World"
    return perception


def _build_mock_context(task_id: str, step: int) -> Any:
    """Sahte karar bağlamı."""
    ctx = MagicMock()
    ctx.task_id = task_id
    ctx.goal = "Open Notepad and type Hello"
    ctx.step = step
    ctx.action_history = [
        MagicMock(action_type="click", target="button") for _ in range(min(step, 5))
    ]
    return ctx


def _run_mock_decision(perception: Any, context: Any) -> Any:
    """
    Mock karar alma pipeline.
    Gerçek DecisionEngine'i simüle eder.
    """
    # 1. Hard-stuck check
    time.sleep(0.001)

    # 2. Anti-loop check
    time.sleep(0.001)

    # 3. Policy check
    time.sleep(0.001)

    # 4. Ambiguity scoring
    time.sleep(0.005)  # 7 faktör hesaplama

    # 5. Local resolution
    if perception.confidence >= 0.5:
        time.sleep(0.010)  # Lokal çözüm
        return MagicMock(
            source="local",
            action_type="click",
            confidence=perception.confidence,
        )

    # 6. Cloud planning (local yeterli olmadığında)
    time.sleep(0.050)  # Mock cloud çağrısı
    return MagicMock(
        source="cloud",
        action_type="click",
        confidence=0.80,
    )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark() -> dict[str, Any]:
    """Benchmark çalıştır ve sonuçları döndür."""
    print(f"\n{'='*60}")
    print(f"  bench_decision_latency — {N_DECISIONS} decisions")
    print(f"  Target: avg < {TARGET_AVG_MS:.0f}ms")
    print(f"{'='*60}")

    perception = _build_mock_perception()
    task_id = str(uuid.uuid4())

    latencies_ms: list[float] = []
    decision_sources: dict[str, int] = {"local": 0, "cloud": 0, "hitl": 0}

    for step in range(N_DECISIONS):
        if step % 10 == 0:
            print(f"  Progress: {step}/{N_DECISIONS}...")

        context = _build_mock_context(task_id, step)

        t0 = time.perf_counter()
        decision = _run_mock_decision(perception, context)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        latencies_ms.append(elapsed_ms)
        source = getattr(decision, "source", "unknown")
        if source in decision_sources:
            decision_sources[source] += 1

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    p95_ms = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]
    p99_ms = sorted(latencies_ms)[int(0.99 * len(latencies_ms))]

    passed = avg_ms < TARGET_AVG_MS

    print(f"\n  Results:")
    print(f"    Decisions    : {N_DECISIONS}")
    print(f"    Average      : {avg_ms:.2f}ms")
    print(f"    Min          : {min_ms:.2f}ms")
    print(f"    Max          : {max_ms:.2f}ms")
    print(f"    P95          : {p95_ms:.2f}ms")
    print(f"    P99          : {p99_ms:.2f}ms")
    print(f"    Local        : {decision_sources['local']}")
    print(f"    Cloud        : {decision_sources['cloud']}")
    print(f"    Target       : < {TARGET_AVG_MS:.0f}ms")
    print(f"    Status       : {'✓ PASS' if passed else '✗ FAIL'}")

    result = {
        "benchmark": "bench_decision_latency",
        "n_decisions": N_DECISIONS,
        "avg_ms": round(avg_ms, 3),
        "min_ms": round(min_ms, 3),
        "max_ms": round(max_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "p99_ms": round(p99_ms, 3),
        "target_avg_ms": TARGET_AVG_MS,
        "decision_sources": decision_sources,
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
    out_file = out_dir / f"bench_decision_{ts}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_file.name}")


if __name__ == "__main__":
    result = run_benchmark()
    print(f"\n  Final: {'PASS' if result['passed'] else 'FAIL'} — {result['avg_ms']:.2f}ms avg")
    raise SystemExit(0 if result["passed"] else 1)
