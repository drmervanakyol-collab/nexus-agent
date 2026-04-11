"""
nexus/__main__.py
Entry point for Nexus Agent.

Supports two invocation modes:
  1. Interactive REPL   — no args, prompts for tasks in a loop
  2. Single-shot        — `--task "..."` runs one task and exits

Usage
-----
    python -m nexus
    python -m nexus --task "Open Excel and calculate the sum of column B"
    python -m nexus --task "..." --config /path/to/config.toml
    python -m nexus --version
    python -m nexus --machine-id        # print machine ID and exit (for licensing)

Environment variables
---------------------
NEXUS_LICENSE_KEY       — license key (full or trial)
NEXUS_LICENSE_SECRET    — HMAC secret (injected by build pipeline)
NEXUS_TRIAL_PATH        — override trial state file path
NEXUS_DB_PATH           — override SQLite DB path (default: nexus.db)
OPENAI_API_KEY          — OpenAI API key (primary provider)
ANTHROPIC_API_KEY       — Anthropic API key (fallback provider)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import uuid
from datetime import UTC
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nexus-agent",
        description="Nexus Agent — AI-powered desktop automation",
    )
    parser.add_argument("--version", action="version", version=f"nexus-agent {__version__}")
    parser.add_argument(
        "--task",
        metavar="GOAL",
        help="Run a single task and exit",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Path to TOML config file (default: nexus.toml if present)",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help="Path to SQLite database file (default: nexus.db or NEXUS_DB_PATH)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enable dry-run mode — plan actions but do not execute them",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--machine-id",
        action="store_true",
        help="Print the machine ID used for license binding and exit",
    )
    return parser


# ---------------------------------------------------------------------------
# License check
# ---------------------------------------------------------------------------


def _check_license() -> bool:
    """Return True when the agent is licensed to run (full or active trial)."""
    try:
        from nexus.release.license_manager import LicenseManager

        lm = LicenseManager()
        if lm.is_licensed():
            return True
        msg = lm.trial_message()
        print(f"[nexus] {msg}", file=sys.stderr)
        print("[nexus] Lisans almak icin: https://nexus-agent.io/buy", file=sys.stderr)
        return False
    except Exception:  # noqa: BLE001
        # License module unavailable — allow execution (dev environment)
        return True


# ---------------------------------------------------------------------------
# Component wiring
# ---------------------------------------------------------------------------


def _build_provider(settings: Any) -> Any:
    """
    Build a cloud LLM provider from available API keys.

    Priority: OpenAI (primary) → Anthropic (fallback).
    If only one key is set, that provider is used directly.
    If neither key is set, raises RuntimeError.
    """
    from nexus.cloud.providers import AnthropicProvider, FallbackProvider, OpenAIProvider

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not openai_key and not anthropic_key:
        raise RuntimeError(
            "No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )

    if openai_key and anthropic_key:
        primary = OpenAIProvider(api_key=openai_key)
        secondary = AnthropicProvider(api_key=anthropic_key)
        return FallbackProvider(primary=primary, secondary=secondary)

    if openai_key:
        return OpenAIProvider(api_key=openai_key)

    return AnthropicProvider(api_key=anthropic_key)


def _pick_model(settings: Any) -> str:
    """Select the primary model based on available API keys and settings."""
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if openai_key:
        return str(settings.cloud.openai_model)
    if anthropic_key:
        return str(settings.cloud.anthropic_model)
    return str(settings.cloud.openai_model)  # fallback — error will be raised by provider


def _build_decision_engine(settings: Any, cost_tracker: Any, task_id: str) -> Any:
    """Wire up DecisionEngine with all its dependencies."""
    from nexus.cloud.planner import CloudPlanner
    from nexus.cloud.prompt_builder import PromptBuilder
    from nexus.core.policy import PolicyEngine
    from nexus.decision.ambiguity_scorer import AmbiguityScorer
    from nexus.decision.engine import DecisionEngine, LocalResolver

    provider = _build_provider(settings)
    model = _pick_model(settings)

    policy = PolicyEngine(settings)
    scorer = AmbiguityScorer()
    resolver = LocalResolver()
    prompt_builder = PromptBuilder()
    planner = CloudPlanner(
        provider=provider,
        cost_tracker=cost_tracker,
        prompt_builder=prompt_builder,
        task_id=task_id,
        model=model,
        max_tokens=settings.cloud.max_tokens,
        timeout=float(settings.cloud.timeout_seconds),
    )

    return DecisionEngine(
        policy=policy,
        scorer=scorer,
        resolver=resolver,
        planner=planner,
        cost_before_fn=lambda tid: cost_tracker.get_task_cost(tid),
    )


def _build_transport_resolver(settings: Any, uia_adapter: Any = None) -> Any:
    """Build TransportResolver with real UIA/OS transports where available."""
    from nexus.source.transport.resolver import TransportResolver

    uia_invoker = None
    uia_value_setter = None
    uia_selector = None

    _uia = uia_adapter
    if _uia is not None:
        def uia_invoker(element: Any) -> bool:
            return _uia.invoke(element)  # type: ignore[no-any-return]

        def uia_value_setter(element: Any, text: str) -> bool:
            return _uia.set_value(element, text)  # type: ignore[no-any-return]

        def uia_selector(element: Any) -> bool:
            return _uia.select(element)  # type: ignore[no-any-return]

    return TransportResolver(
        settings,
        _uia_invoker=uia_invoker,
        _uia_value_setter=uia_value_setter,
        _uia_selector=uia_selector,
    )


def _try_create_uia_adapter(settings: Any) -> Any:
    """Create a UIAAdapter if comtypes/UIA is available; return None otherwise."""
    try:
        from nexus.source.uia.adapter import UIAAdapter  # noqa: PLC0415
        adapter = UIAAdapter(settings)
        if adapter.is_available():
            return adapter
    except Exception:  # noqa: BLE001
        pass
    return None


def _build_source_fn(settings: Any, uia_adapter: Any) -> Any:
    """Return an async source function that reuses the given UIAAdapter."""
    import ctypes

    from nexus.source.resolver import SourcePriorityResolver

    uia_probe = None
    if uia_adapter is not None:
        def uia_probe(ctx: dict[str, Any]) -> Any:
            hwnd = ctx.get("window_handle") or ctypes.windll.user32.GetForegroundWindow()
            if not hwnd:
                return None
            elements = uia_adapter.get_elements(hwnd)
            # Return None (not empty list) when no elements found so
            # the resolver falls through to the visual fallback.
            return elements if elements else None

    async def _source_fn() -> Any:
        resolver = SourcePriorityResolver(settings, _uia_probe=uia_probe)
        return resolver.resolve({})

    return _source_fn


# ---------------------------------------------------------------------------
# Source / capture / perceive stubs (real screen integration)
# ---------------------------------------------------------------------------


async def _default_source_fn():  # type: ignore[no-untyped-def]
    """Fallback source function (used when no UIA adapter is available)."""
    from nexus.core.settings import NexusSettings
    from nexus.source.resolver import SourcePriorityResolver

    settings = NexusSettings()
    resolver = SourcePriorityResolver(settings)
    return resolver.resolve({})


async def _default_capture_fn():  # type: ignore[no-untyped-def]
    """Capture the current screen frame."""
    import time
    from datetime import datetime

    import numpy as np

    from nexus.capture.frame import Frame

    now_monotonic = time.monotonic()
    now_utc = datetime.now(UTC).isoformat()

    try:
        import dxcam  # noqa: PLC0415

        camera = dxcam.create()
        raw = camera.grab()
        if raw is not None:
            return Frame(
                data=raw,
                width=raw.shape[1],
                height=raw.shape[0],
                captured_at_monotonic=now_monotonic,
                captured_at_utc=now_utc,
                sequence_number=1,
            )
    except Exception:  # noqa: BLE001
        pass

    # Fallback: blank frame
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    return Frame(
        data=blank,
        width=1920,
        height=1080,
        captured_at_monotonic=now_monotonic,
        captured_at_utc=now_utc,
        sequence_number=1,
    )


async def _default_perceive_fn(frame, source_result):  # type: ignore[no-untyped-def]
    """Build a PerceptionResult from a frame and source data."""
    from nexus.perception.arbitration.arbitrator import PerceptionArbitrator
    from nexus.perception.locator.locator import Locator
    from nexus.perception.matcher.matcher import Matcher
    from nexus.perception.orchestrator import PerceptionOrchestrator
    from nexus.perception.reader.ocr_engine import TesseractOCREngine
    from nexus.perception.temporal.temporal_expert import TemporalExpert

    def _null_ocr(image, lang, timeout):  # type: ignore[no-untyped-def]
        """No-op OCR — returns empty TSV when Tesseract is not installed."""
        _hdr = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num"
        return f"{_hdr}\tleft\ttop\twidth\theight\tconf\ttext\n"

    try:
        import subprocess
        subprocess.run(["tesseract", "--version"], capture_output=True, timeout=5)
        ocr_engine = TesseractOCREngine()
    except (FileNotFoundError, Exception):
        ocr_engine = TesseractOCREngine(_run_fn=_null_ocr)

    orchestrator = PerceptionOrchestrator(
        temporal_expert=TemporalExpert(),
        locator=Locator(),
        matcher=Matcher(),
        arbitrator=PerceptionArbitrator(),
        ocr_engine=ocr_engine,
    )
    return await orchestrator.perceive(frame, source_result)


# ---------------------------------------------------------------------------
# Optional component builders
# ---------------------------------------------------------------------------


def _build_health_checker(db_path: str) -> Any:
    """Create HealthChecker; returns None on import error."""
    try:
        from nexus.infra.health import HealthChecker  # noqa: PLC0415
        return HealthChecker(db_path=db_path)
    except Exception:  # noqa: BLE001
        return None


def _build_fingerprint_store(db: Any, settings: Any) -> Any:
    """Create FingerprintStore; returns None on import or init error."""
    try:
        from nexus.core.settings import NexusSettings  # noqa: PLC0415
        from nexus.infra.database import Database  # noqa: PLC0415
        from nexus.memory.fingerprint_store import FingerprintStore  # noqa: PLC0415
        if isinstance(db, Database) and isinstance(settings, NexusSettings):
            return FingerprintStore(db, settings)
    except Exception:  # noqa: BLE001
        pass
    return None


def _build_hitl_manager(db: Any, settings: Any) -> Any:
    """Create HITLManager for interactive human-in-the-loop prompts."""
    try:
        from nexus.core.hitl_manager import HITLManager  # noqa: PLC0415
        from nexus.core.settings import NexusSettings  # noqa: PLC0415
        from nexus.infra.database import Database  # noqa: PLC0415
        if isinstance(db, Database) and isinstance(settings, NexusSettings):
            return HITLManager(db, settings.hitl)
    except Exception:  # noqa: BLE001
        pass
    return None


def _build_suspend_manager(db: Any) -> Any:
    """Create SuspendManager for task suspension persistence."""
    try:
        from nexus.core.suspend_manager import SuspendManager  # noqa: PLC0415
        from nexus.infra.database import Database  # noqa: PLC0415
        if isinstance(db, Database):
            return SuspendManager(db)
    except Exception:  # noqa: BLE001
        pass
    return None


def _build_verifier_fn() -> Any:
    """
    Return an async verifier_fn wrapping VisualVerifier.

    Signature: (before_frame, after_frame, action_type) -> VerificationResult
    Returns None when the verification module is unavailable.
    """
    try:
        from nexus.verification import VerificationPolicy  # noqa: PLC0415
        from nexus.verification.visual_verification import VisualVerifier  # noqa: PLC0415

        verifier = VisualVerifier()
        policy = VerificationPolicy()

        from nexus.capture.frame import Frame  # noqa: PLC0415
        from nexus.verification import VerificationMode, VerificationResult  # noqa: PLC0415

        async def _verifier_fn(
            before_frame: Any, after_frame: Any, action_type: str
        ) -> Any:
            if not isinstance(before_frame, Frame) or not isinstance(after_frame, Frame):
                return VerificationResult(
                    success=True, confidence=1.0, mode_used=VerificationMode.SKIP
                )
            return verifier.verify(before_frame, after_frame, policy)

        return _verifier_fn
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


async def _run_task(goal: str, args: argparse.Namespace) -> int:
    """Wire all subsystems and execute a single task. Returns exit code."""
    from nexus.core.settings import load_settings
    from nexus.core.task_executor import TaskExecutor
    from nexus.infra.cost_tracker import CostTracker
    from nexus.infra.database import Database
    from nexus.infra.logger import get_logger

    log = get_logger(__name__)

    # Settings
    config_path = args.config or ("nexus.toml" if Path("nexus.toml").exists() else None)
    overrides: dict[str, Any] = {}
    if args.dry_run:
        overrides["safety"] = {"dry_run_mode": True}
    settings = load_settings(config_path, **overrides)

    # Database
    db_path = args.db or os.environ.get("NEXUS_DB_PATH", settings.storage.db_path)
    db = Database(db_path)
    await db.init()

    # Cost tracker (shared between CloudPlanner and TaskExecutor)
    cost_tracker = CostTracker(settings)

    # Generate task_id upfront so CloudPlanner can track costs per task
    task_id = str(uuid.uuid4())

    # Build components
    try:
        decision_engine = _build_decision_engine(settings, cost_tracker, task_id)
    except RuntimeError as exc:
        print(f"[nexus] Configuration error: {exc}", file=sys.stderr)
        return 2

    # Single shared UIAAdapter — reused by both source and transport
    uia_adapter = _try_create_uia_adapter(settings)
    source_fn = _build_source_fn(settings, uia_adapter) if uia_adapter else _default_source_fn
    transport_resolver = _build_transport_resolver(settings, uia_adapter)

    # Optional components — wired when available, None otherwise
    health_checker = _build_health_checker(db_path)
    fingerprint_store = _build_fingerprint_store(db, settings)
    hitl_manager = _build_hitl_manager(db, settings)
    suspend_manager = _build_suspend_manager(db)
    verifier_fn = _build_verifier_fn()

    def _progress_fn(msg: str) -> None:
        print(f"[nexus] {msg}", flush=True)

    executor = TaskExecutor(
        db=db,
        settings=settings,
        source_fn=source_fn,
        capture_fn=_default_capture_fn,
        perceive_fn=_default_perceive_fn,
        decision_engine=decision_engine,
        transport_resolver=transport_resolver,
        cost_tracker=cost_tracker,
        health_checker=health_checker,
        fingerprint_store=fingerprint_store,
        hitl_manager=hitl_manager,
        suspend_manager=suspend_manager,
        verifier_fn=verifier_fn,
        progress_fn=_progress_fn,
    )

    # Graceful SIGINT / SIGTERM → cancel current task
    loop = asyncio.get_running_loop()

    def _on_signal() -> None:
        log.info("interrupt_received", goal=goal)
        executor.cancel()

    import contextlib
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, OSError):
            loop.add_signal_handler(sig, _on_signal)

    log.info("task_starting", goal=goal, task_id=task_id)
    result = await executor.execute(goal, task_id=task_id)

    if result.success:
        native_pct = int(result.transport_stats.native_ratio * 100)
        print(
            f"[nexus] Tamamlandi — {result.duration_ms:.0f} ms, "
            f"{result.steps_completed} adim, "
            f"${result.total_cost_usd:.4f}, "
            f"native transport %{native_pct}"
        )
        return 0

    print(f"[nexus] Basarisiz: {result.error}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------


async def _interactive_loop(args: argparse.Namespace) -> int:
    """Prompt for goals in a loop until the user quits."""
    print(f"Nexus Agent {__version__}  (Ctrl-C veya 'quit' ile cikis)")
    print()

    exit_code = 0
    while True:
        try:
            goal = input("nexus> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not goal:
            continue
        if goal.lower() in {"quit", "exit", "q", "cikis"}:
            break

        code = await _run_task(goal, args)
        if code != 0:
            exit_code = code

    return exit_code


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Primary entry point — called by ``python -m nexus`` and the EXE stub."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    # --machine-id: quick exit, no license check needed
    if args.machine_id:
        from nexus.release.license_manager import LicenseManager

        lm = LicenseManager()
        print(lm.generate_machine_id())
        sys.exit(0)

    # Configure logging first so all startup messages are structured
    from nexus.infra.logger import configure_logging

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    configure_logging(level=level)

    # License gate
    if not _check_license():
        sys.exit(2)

    # Run
    if args.task:
        exit_code = asyncio.run(_run_task(args.task, args))
    else:
        exit_code = asyncio.run(_interactive_loop(args))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
