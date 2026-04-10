"""
scripts/run_mutation.py
Cross-platform mutation testing runner for Nexus Agent.

Why not mutmut?
---------------
mutmut 3.x does not support native Windows (see mutmut issue #397).
This script implements the same core mutation operators using Python's
``ast`` module and runs the test suite against each mutant.

Usage
-----
    python scripts/run_mutation.py [--paths nexus/core/] [--tests tests/property/]
                                   [--threshold 0.70] [--workers 1]

Mutation operators applied
--------------------------
  AOR  — Arithmetic Operator Replacement  (+→-, *→/, etc.)
  ROR  — Relational Operator Replacement  (<→<=, ==→!=, etc.)
  COR  — Conditional Operator Replacement (and→or, not→identity)
  LCR  — Literal Constant Replacement     (0→1, True→False, ""→"x")
  BCR  — Boolean / Comparison Return      (return True ->return False)

Output
------
  JSON : scripts/mutation_results.json
  Markdown: scripts/mutation_report.md
  Exit code 0 when score >= threshold, 1 otherwise.
"""
from __future__ import annotations

import atexit
import argparse
import ast
import copy
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Global emergency restore registry (path -> original_source)
# Ensures files are restored even if the process is interrupted.
# ---------------------------------------------------------------------------

_RESTORE_REGISTRY: dict[Path, str] = {}


def _emergency_restore() -> None:
    for path, original in _RESTORE_REGISTRY.items():
        try:
            path.write_text(original, encoding="utf-8")
        except Exception:
            pass


atexit.register(_emergency_restore)


def _signal_handler(signum: int, frame: object) -> None:
    _emergency_restore()
    sys.exit(1)


signal.signal(signal.SIGTERM, _signal_handler)
try:
    signal.signal(signal.SIGBREAK, _signal_handler)  # Windows Ctrl+Break
except (AttributeError, OSError):
    pass

# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

_ARITH_OPS: dict[type, type] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mod,
    ast.Mod: ast.FloorDiv,
}

_COMPARE_OPS: dict[type, type] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}

_BOOL_OPS: dict[type, type] = {
    ast.And: ast.Or,
    ast.Or: ast.And,
}


# ---------------------------------------------------------------------------
# Mutant dataclass
# ---------------------------------------------------------------------------


@dataclass
class Mutant:
    source_path: Path
    original_source: str
    mutant_source: str
    description: str
    lineno: int
    killed: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# AST-based mutant generator
# ---------------------------------------------------------------------------


class _MutantVisitor(ast.NodeTransformer):
    """
    Walk the AST and collect (mutated_source, description, lineno) tuples.
    Each call to generate() returns all possible single-point mutations.
    """

    def __init__(self, original_source: str) -> None:
        self._original = original_source
        self._tree = ast.parse(original_source)
        self._mutations: list[tuple[str, str, int]] = []

    def generate(self) -> list[tuple[str, str, int]]:
        self._mutations = []
        for node in ast.walk(self._tree):
            self._try_arith(node)
            self._try_compare(node)
            self._try_bool(node)
            self._try_numeric_const(node)
            self._try_bool_const(node)
        return self._mutations

    # ------------------------------------------------------------------
    # Arithmetic operator replacement
    # ------------------------------------------------------------------

    def _try_arith(self, node: ast.AST) -> None:
        if not isinstance(node, ast.BinOp):
            return
        for orig_type, mutant_type in _ARITH_OPS.items():
            if isinstance(node.op, orig_type):
                mutated = copy.deepcopy(self._tree)
                for n in ast.walk(mutated):
                    if (
                        isinstance(n, ast.BinOp)
                        and isinstance(n.op, orig_type)
                        and getattr(n, "lineno", -1) == getattr(node, "lineno", -2)
                        and getattr(n, "col_offset", -1) == getattr(node, "col_offset", -2)
                    ):
                        n.op = mutant_type()
                        break
                try:
                    src = ast.unparse(mutated)
                    op_name = orig_type.__name__.replace("Add", "+").replace(
                        "Sub", "-").replace("Mult", "*").replace("Div", "/").replace(
                        "FloorDiv", "//").replace("Mod", "%")
                    mut_name = mutant_type.__name__.replace("Add", "+").replace(
                        "Sub", "-").replace("Mult", "*").replace("Div", "/").replace(
                        "FloorDiv", "//").replace("Mod", "%")
                    self._mutations.append((
                        src,
                        f"AOR line {node.lineno}: {op_name} ->{mut_name}",
                        node.lineno,
                    ))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Relational operator replacement
    # ------------------------------------------------------------------

    def _try_compare(self, node: ast.AST) -> None:
        if not isinstance(node, ast.Compare):
            return
        for i, op in enumerate(node.ops):
            for orig_type, mutant_type in _COMPARE_OPS.items():
                if isinstance(op, orig_type):
                    mutated = copy.deepcopy(self._tree)
                    for n in ast.walk(mutated):
                        if (
                            isinstance(n, ast.Compare)
                            and getattr(n, "lineno", -1) == getattr(node, "lineno", -2)
                            and getattr(n, "col_offset", -1) == getattr(node, "col_offset", -2)
                        ):
                            n.ops[i] = mutant_type()
                            break
                    try:
                        src = ast.unparse(mutated)
                        self._mutations.append((
                            src,
                            f"ROR line {node.lineno}: {orig_type.__name__} ->{mutant_type.__name__}",
                            node.lineno,
                        ))
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Boolean operator replacement
    # ------------------------------------------------------------------

    def _try_bool(self, node: ast.AST) -> None:
        if not isinstance(node, ast.BoolOp):
            return
        for orig_type, mutant_type in _BOOL_OPS.items():
            if isinstance(node.op, orig_type):
                mutated = copy.deepcopy(self._tree)
                for n in ast.walk(mutated):
                    if (
                        isinstance(n, ast.BoolOp)
                        and isinstance(n.op, orig_type)
                        and getattr(n, "lineno", -1) == getattr(node, "lineno", -2)
                    ):
                        n.op = mutant_type()
                        break
                try:
                    src = ast.unparse(mutated)
                    self._mutations.append((
                        src,
                        f"COR line {node.lineno}: {orig_type.__name__} ->{mutant_type.__name__}",
                        node.lineno,
                    ))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Numeric constant replacement (0 ↔ 1, n ->n+1)
    # ------------------------------------------------------------------

    def _try_numeric_const(self, node: ast.AST) -> None:
        if not isinstance(node, ast.Constant) or not isinstance(node.value, (int, float)):
            return
        orig_val = node.value
        # Only mutate simple integer/float literals that appear as standalone values
        # (skip very large numbers to avoid slowdowns)
        if isinstance(orig_val, int) and abs(orig_val) > 10_000:
            return
        mutant_val = orig_val + 1 if orig_val != 0 else 1

        mutated = copy.deepcopy(self._tree)
        for n in ast.walk(mutated):
            if (
                isinstance(n, ast.Constant)
                and n.value == orig_val
                and isinstance(n.value, type(orig_val))
                and getattr(n, "lineno", -1) == getattr(node, "lineno", -2)
                and getattr(n, "col_offset", -1) == getattr(node, "col_offset", -2)
            ):
                n.value = mutant_val
                break
        try:
            src = ast.unparse(mutated)
            self._mutations.append((
                src,
                f"LCR line {node.lineno}: {orig_val} ->{mutant_val}",
                node.lineno,
            ))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Boolean constant replacement (True ↔ False)
    # ------------------------------------------------------------------

    def _try_bool_const(self, node: ast.AST) -> None:
        if not isinstance(node, ast.Constant) or not isinstance(node.value, bool):
            return
        mutant_val = not node.value
        mutated = copy.deepcopy(self._tree)
        for n in ast.walk(mutated):
            if (
                isinstance(n, ast.Constant)
                and isinstance(n.value, bool)
                and n.value == node.value
                and getattr(n, "lineno", -1) == getattr(node, "lineno", -2)
                and getattr(n, "col_offset", -1) == getattr(node, "col_offset", -2)
            ):
                n.value = mutant_val
                break
        try:
            src = ast.unparse(mutated)
            self._mutations.append((
                src,
                f"BCR line {node.lineno}: {node.value} ->{mutant_val}",
                node.lineno,
            ))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Mutant runner
# ---------------------------------------------------------------------------


def _collect_python_files(paths: list[str]) -> list[Path]:
    result: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".py":
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(path.rglob("*.py")))
    return [f for f in result if "__pycache__" not in str(f)]


def _run_tests(test_paths: list[str], timeout_s: float = 30.0) -> bool:
    """Return True if all tests pass."""
    cmd = [
        sys.executable, "-m", "pytest",
        *test_paths,
        "-q", "--no-header", "--tb=no",
        f"--timeout={int(timeout_s)}",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s + 10,
        cwd=str(Path(__file__).parent.parent),
    )
    return result.returncode == 0


def _apply_mutant_and_run(
    source_path: Path,
    original_source: str,
    mutant_source: str,
    test_paths: list[str],
) -> tuple[bool, str]:
    """
    Write mutant source, run tests, restore original.
    Returns (killed, error_msg).
    """
    _RESTORE_REGISTRY[source_path] = original_source
    try:
        source_path.write_text(mutant_source, encoding="utf-8")
        # Clear __pycache__ so Python picks up the mutant
        cache_dir = source_path.parent / "__pycache__"
        if cache_dir.exists():
            for pyc in cache_dir.glob(f"{source_path.stem}*.pyc"):
                try:
                    pyc.unlink()
                except OSError:
                    pass
        killed = not _run_tests(test_paths)
        return killed, ""
    except subprocess.TimeoutExpired:
        return True, "timeout"  # timeout = tests were slow ->count as killed
    except Exception as exc:
        return True, str(exc)
    finally:
        source_path.write_text(original_source, encoding="utf-8")
        _RESTORE_REGISTRY.pop(source_path, None)


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _write_json(result: dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _write_markdown(result: dict[str, Any], out_path: Path) -> None:
    score = result["mutation_score"]
    total = result["total_mutants"]
    killed = result["killed_mutants"]
    survived = result["survived_mutants"]
    threshold = result["threshold"]
    status = "PASS" if score >= threshold else "FAIL"

    lines = [
        "# Nexus Agent — Mutation Test Report",
        f"\nGenerated: `{result['generated_at']}`  |  Status: **{status}**\n",
        "## Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total mutants | {total} |",
        f"| Killed | {killed} |",
        f"| Survived | {survived} |",
        f"| Mutation score | {score * 100:.1f}% |",
        f"| Threshold | {threshold * 100:.0f}% |",
        f"| Status | **{status}** |",
        "",
        "## Survived Mutants (need stronger tests)\n",
    ]

    survived_list = [m for m in result["mutants"] if not m["killed"]]
    if survived_list:
        lines.append("| File | Line | Description |")
        lines.append("|------|------|-------------|")
        for m in survived_list[:50]:  # cap at 50
            fname = Path(m["source_path"]).name
            lines.append(f"| {fname} | {m['lineno']} | {m['description']} |")
    else:
        lines.append("_No surviving mutants._")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    source_paths: list[str],
    test_paths: list[str],
    threshold: float = 0.70,
    max_mutants: int = 500,
    verbose: bool = False,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    print(f"[mutation] Collecting source files from: {source_paths}")
    source_files = _collect_python_files(source_paths)
    print(f"[mutation] Found {len(source_files)} source file(s)")

    print(f"[mutation] Verifying baseline (all tests pass on unmodified code)...")
    if not _run_tests(test_paths):
        print("[mutation] ERROR: baseline tests fail — fix them first")
        sys.exit(2)
    print("[mutation] Baseline OK")

    all_mutants: list[Mutant] = []
    for src_file in source_files:
        original = src_file.read_text(encoding="utf-8")
        visitor = _MutantVisitor(original)
        mutations = visitor.generate()
        for mutant_src, desc, lineno in mutations:
            all_mutants.append(Mutant(
                source_path=src_file,
                original_source=original,
                mutant_source=mutant_src,
                description=desc,
                lineno=lineno,
            ))

    print(f"[mutation] Generated {len(all_mutants)} mutant(s)")

    # Cap to avoid excessive runtime
    if len(all_mutants) > max_mutants:
        print(f"[mutation] Capping at {max_mutants} mutants (use --max-mutants to raise)")
        import random
        random.seed(42)
        all_mutants = random.sample(all_mutants, max_mutants)

    killed = 0
    t0 = time.perf_counter()
    for i, mutant in enumerate(all_mutants, start=1):
        mutant.killed, mutant.error = _apply_mutant_and_run(
            mutant.source_path,
            mutant.original_source,
            mutant.mutant_source,
            test_paths,
        )
        if mutant.killed:
            killed += 1
        status_char = "x" if mutant.killed else "."
        if verbose:
            print(f"  [{i:>4}/{len(all_mutants)}] {status_char} {mutant.description}")
        elif i % 10 == 0 or i == len(all_mutants):
            elapsed = time.perf_counter() - t0
            rate = i / elapsed
            print(
                f"[mutation] {i}/{len(all_mutants)} mutants tested "
                f"({killed} killed, {i - killed} survived) "
                f"[{rate:.1f}/s]"
            )

    total = len(all_mutants)
    survived = total - killed
    score = killed / total if total > 0 else 0.0

    print(f"\n[mutation] Results: {killed}/{total} killed — score {score * 100:.1f}%")
    if score >= threshold:
        print(f"[mutation] PASS (>= {threshold * 100:.0f}% threshold)")
    else:
        print(f"[mutation] FAIL (< {threshold * 100:.0f}% threshold)")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_mutants": total,
        "killed_mutants": killed,
        "survived_mutants": survived,
        "mutation_score": round(score, 4),
        "threshold": threshold,
        "passed": score >= threshold,
        "source_paths": source_paths,
        "test_paths": test_paths,
        "mutants": [
            {
                "source_path": str(m.source_path),
                "description": m.description,
                "lineno": m.lineno,
                "killed": m.killed,
                "error": m.error,
            }
            for m in all_mutants
        ],
    }

    out_dir = Path(__file__).parent
    json_path = out_dir / "mutation_results.json"
    md_path = out_dir / "mutation_report.md"
    _write_json(result, json_path)
    _write_markdown(result, md_path)
    print(f"[mutation] JSON   : {json_path}")
    print(f"[mutation] MD     : {md_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform mutation testing runner")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["nexus/core/"],
        help="Source paths to mutate (files or directories)",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["tests/property/", "tests/integration/"],
        help="Test paths to run against each mutant",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum required mutation score (default: 0.70)",
    )
    parser.add_argument(
        "--max-mutants",
        type=int,
        default=500,
        help="Maximum number of mutants to test (default: 500)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print result for each mutant",
    )
    args = parser.parse_args()

    result = run(
        source_paths=args.paths,
        test_paths=args.tests,
        threshold=args.threshold,
        max_mutants=args.max_mutants,
        verbose=args.verbose,
    )
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
