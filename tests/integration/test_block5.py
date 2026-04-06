"""
tests/integration/test_block5.py
Blok 5 Integration Tests — Faz 43

Full pipeline: Action layer (Preflight → TransportResolver → Verification)
wired with Suspend/Resume/HITL layer.  No real UIA, DOM, or mouse hardware
is exercised; all OS-level callables are replaced with lightweight stubs.

TEST 1 — Full action pipeline (transport-aware)
  Decision transport_hint="uia" → Preflight passes → TransportResolver uses
  UIA native path → VerificationResult SOURCE success.

TEST 2 — Native fail → OS fallback
  UIA invoker returns False → MouseTransport fallback → fallback_used=True
  → audit record captured.

TEST 3 — Verification policy: sheet_write uses SOURCE_LEVEL
  NexusSettings.verification.sheet_write_method == "source_level"
  → SourceVerifier with SOURCE policy returns success.

TEST 4 — False positive detection
  Before frame == after frame + require_change=True
  → VerificationResult.false_positive=True, success=False.

TEST 5 — Suspend + resume with drift detection
  SuspendManager.suspend() → row in suspended_tasks
  → SuspendManager.resume() with matching fingerprint → drift_detected=False.

TEST 6 — MacroAction transport selection
  SafeFieldReplace: UIA native available → native called, fallback skipped.
  SafeFieldReplace: native returns False → fallback (mouse) called.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
import pytest_asyncio

from nexus.action.macroactions import SafeFieldReplace
from nexus.action.preflight import PreflightChecker, PreflightContext
from nexus.action.registry import ActionSpec as RegistryActionSpec
from nexus.capture.frame import Frame
from nexus.core.settings import NexusSettings
from nexus.core.suspend_manager import SuspendManager
from nexus.infra.database import Database
from nexus.perception.arbitration.arbitrator import ArbitrationResult
from nexus.perception.orchestrator import PerceptionResult
from nexus.perception.spatial_graph import SpatialGraph
from nexus.perception.temporal.temporal_expert import ScreenState, StateType
from nexus.source.resolver import SourceResult
from nexus.source.transport.fallback import MouseTransport
from nexus.source.transport.resolver import (
    ActionSpec as TransportActionSpec,
)
from nexus.source.transport.resolver import (
    TransportResolver,
    TransportResult,
)
from nexus.verification import (
    SourceVerifier,
    VerificationMode,
    VerificationPolicy,
    VisualVerifier,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UTC = "2026-04-07T00:00:00+00:00"
_TASK_ID = "blok5-task"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _frame(pixel_value: int = 128, width: int = 8, height: int = 8) -> Frame:
    data = np.full((height, width, 3), pixel_value, dtype=np.uint8)
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=time.monotonic(),
        captured_at_utc=_UTC,
        sequence_number=1,
    )


def _uia_source() -> SourceResult:
    return SourceResult(
        source_type="uia",
        data={"elements": []},
        confidence=1.0,
        latency_ms=0.0,
    )


def _visual_source() -> SourceResult:
    return SourceResult(
        source_type="visual",
        data={"visual_pending": True},
        confidence=0.70,
        latency_ms=0.0,
    )


def _minimal_perception(source: SourceResult | None = None) -> PerceptionResult:
    """Minimal PerceptionResult with empty spatial graph."""
    stable_state = ScreenState(
        state_type=StateType.STABLE,
        confidence=1.0,
        blocks_perception=False,
        reason="stable",
        retry_after_ms=0,
    )
    arbitration = ArbitrationResult(
        resolved_elements=(),
        resolved_labels=(),
        conflicts_detected=0,
        conflicts_resolved=0,
        temporal_blocked=False,
        overall_confidence=1.0,
    )
    return PerceptionResult(
        spatial_graph=SpatialGraph([], [], {}),
        screen_state=stable_state,
        arbitration=arbitration,
        source_result=source or _uia_source(),
        perception_ms=0.0,
        frame_sequence=1,
        timestamp=_UTC,
    )


def _settings(**overrides: Any) -> NexusSettings:
    return NexusSettings(**overrides)


# ---------------------------------------------------------------------------
# Fixture: Database
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "blok5.db"))
    await database.init()
    return database


# ---------------------------------------------------------------------------
# TEST 1 — Full action pipeline (transport-aware)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """
    Decision transport_hint="uia" → Preflight → TransportResolver (UIA)
    → VerificationResult SOURCE success.
    """

    @pytest.mark.asyncio
    async def test_uia_native_path_end_to_end(self):
        settings = _settings()

        # 1. Preflight: uia transport available → passes
        preflight = PreflightChecker()
        registry_spec = RegistryActionSpec(
            action_type="click",
            preferred_transport="uia",
        )
        perception = _minimal_perception(_uia_source())
        ctx = PreflightContext(
            uia_available=True,
            dom_available=False,
            allow_transport_fallback=True,
        )
        pf_result = preflight.check(registry_spec, perception, ctx)
        assert pf_result.passed, f"Preflight failed: {pf_result.check_name}"

        # 2. Transport: UIA invoker succeeds → method_used="uia"
        audit_records: list[TransportResult] = []

        async def _audit(result: TransportResult, spec: TransportActionSpec) -> None:
            audit_records.append(result)

        resolver = TransportResolver(
            settings,
            _uia_invoker=lambda el: True,
            _audit_writer=_audit,
        )
        transport_spec = TransportActionSpec(
            action_type="click",
            task_id=_TASK_ID,
        )
        tr = await resolver.execute(
            transport_spec, _uia_source(), target_element=object()
        )
        assert tr.method_used == "uia"
        assert tr.success is True
        assert tr.fallback_used is False

        # 3. Verification SOURCE: probe returns expected value
        probe = lambda ctx: "clicked"  # noqa: E731
        verifier = SourceVerifier(source_probe=probe)
        v_result = verifier.verify("clicked", VerificationPolicy.source())
        assert v_result.success is True
        assert v_result.mode_used == VerificationMode.SOURCE
        assert v_result.confidence == 1.0


# ---------------------------------------------------------------------------
# TEST 2 — Native fail → OS fallback
# ---------------------------------------------------------------------------


class TestNativeFailFallback:
    """
    UIA invoker returns False → MouseTransport fallback → fallback_used=True
    + audit record captured with fallback_used=True.
    """

    @pytest.mark.asyncio
    async def test_fallback_used_on_uia_failure(self):
        settings = _settings()
        audit_records: list[TransportResult] = []

        async def _audit(result: TransportResult, spec: TransportActionSpec) -> None:
            audit_records.append(result)

        # UIA invoker always fails
        mouse_clicks: list[tuple[int, int]] = []

        def _mouse_click_fn(x: int, y: int) -> None:
            mouse_clicks.append((x, y))

        mouse = MouseTransport(_click_fn=_mouse_click_fn)

        resolver = TransportResolver(
            settings,
            _uia_invoker=lambda el: False,  # always fail
            _mouse_transport=mouse,
            _audit_writer=_audit,
        )

        # Provide a target_element with a bounding_rect so the resolver can
        # derive click coordinates from it.
        from nexus.core.types import Rect

        class _FakeElement:
            bounding_rect = Rect(x=100, y=200, width=60, height=20)

        transport_spec = TransportActionSpec(
            action_type="click",
            task_id=_TASK_ID,
        )
        tr = await resolver.execute(
            transport_spec,
            _uia_source(),
            target_element=_FakeElement(),
        )

        # Fallback used
        assert tr.fallback_used is True
        assert tr.method_used == "mouse"
        assert tr.success is True

        # Mouse actually called
        assert len(mouse_clicks) == 1

        # Audit record captured
        assert len(audit_records) == 1
        assert audit_records[0].fallback_used is True


# ---------------------------------------------------------------------------
# TEST 3 — Verification policy: sheet_write uses SOURCE_LEVEL
# ---------------------------------------------------------------------------


class TestVerificationPolicySheetWrite:
    """
    NexusSettings.verification.sheet_write_method == "source_level"
    Wiring that setting to SourceVerifier yields VerificationMode.SOURCE result.
    """

    def test_sheet_write_method_is_source_level(self):
        settings = _settings()
        assert settings.verification.sheet_write_method == "source_level"

    @pytest.mark.asyncio
    async def test_source_verifier_used_for_sheet_write(self):
        # Simulate what the pipeline would do for sheet_write:
        # use SourceVerifier with SOURCE policy.
        probe = lambda ctx: "42.00"  # noqa: E731
        verifier = SourceVerifier(source_probe=probe)
        policy = VerificationPolicy.source(confidence_threshold=0.90)
        result = verifier.verify("42.00", policy)

        assert result.success is True
        assert result.mode_used == VerificationMode.SOURCE
        assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# TEST 4 — False positive detection
# ---------------------------------------------------------------------------


class TestFalsePositiveDetection:
    """
    Before frame == after frame with require_change=True policy
    → VerificationResult.false_positive=True, success=False.
    """

    def test_identical_frames_false_positive(self):
        verifier = VisualVerifier()
        before = _frame(100)
        after = _frame(100)  # identical

        policy = VerificationPolicy(
            mode=VerificationMode.VISUAL,
            require_change=True,
            confidence_threshold=0.5,
        )
        result = verifier.verify(before, after, policy)

        assert result.false_positive is True
        assert result.success is False
        assert result.confidence == 0.0

    def test_changed_frames_not_false_positive(self):
        verifier = VisualVerifier()
        before = _frame(0)
        after_data = np.full((8, 8, 3), 200, dtype=np.uint8)
        after = Frame(
            data=after_data,
            width=8,
            height=8,
            captured_at_monotonic=time.monotonic(),
            captured_at_utc=_UTC,
            sequence_number=2,
        )
        policy = VerificationPolicy(
            mode=VerificationMode.VISUAL,
            require_change=True,
            confidence_threshold=0.5,
        )
        result = verifier.verify(before, after, policy)

        assert result.false_positive is False
        assert result.success is True


# ---------------------------------------------------------------------------
# TEST 5 — Suspend + resume with drift detection
# ---------------------------------------------------------------------------


class TestSuspendResume:
    """
    Full suspend/resume cycle with real DB:
    - Row inserted into suspended_tasks on suspend
    - Row removed on resume
    - Matching fingerprints → drift_detected=False
    - Different fingerprints → drift_detected=True
    """

    @pytest.mark.asyncio
    async def test_suspend_inserts_row_resume_removes_it(self, db):
        mgr = SuspendManager(db)
        await mgr.suspend(_TASK_ID, "policy block", {"step": 3})

        suspended = await mgr.list_suspended()
        task_ids = {t.task_id for t in suspended}
        assert _TASK_ID in task_ids

        result = await mgr.resume(_TASK_ID)
        assert result.success is True

        suspended_after = await mgr.list_suspended()
        assert all(t.task_id != _TASK_ID for t in suspended_after)

    @pytest.mark.asyncio
    async def test_resume_reality_check_no_drift(self, db):
        fp_iter = iter(["fp-stable", "fp-stable"])
        mgr = SuspendManager(db, fingerprint_fn=lambda: next(fp_iter))
        await mgr.suspend(_TASK_ID + "-a", "test")
        result = await mgr.resume(_TASK_ID + "-a")
        assert result.success is True
        assert result.drift_detected is False

    @pytest.mark.asyncio
    async def test_resume_reality_check_drift_detected(self, db):
        fp_iter = iter(["aaaaaaaaaa", "zzzzzzzzzz"])
        mgr = SuspendManager(db, fingerprint_fn=lambda: next(fp_iter))
        await mgr.suspend(_TASK_ID + "-b", "test")
        result = await mgr.resume(_TASK_ID + "-b")
        assert result.success is True
        assert result.drift_detected is True
        assert result.drift_score > 0.10


# ---------------------------------------------------------------------------
# TEST 6 — MacroAction transport selection
# ---------------------------------------------------------------------------


class TestMacroActionTransport:
    """
    SafeFieldReplace:
    - UIA native available → native_click_fn called, fallback not called.
    - native returns False → fallback (mouse) called.
    """

    @pytest.mark.asyncio
    async def test_native_path_used_when_available(self):
        native_calls: list[str | None] = []
        fallback_calls: list[tuple[int, int]] = []

        async def native_click(element_id: str | None) -> bool:
            native_calls.append(element_id)
            return True

        async def fallback_click(coords: tuple[int, int]) -> bool:
            fallback_calls.append(coords)
            return True

        async def type_fn(text: str) -> bool:
            return True

        async def hotkey_fn(keys: list[str]) -> bool:
            return True

        macro = SafeFieldReplace(
            _native_click_fn=native_click,
            _fallback_click_fn=fallback_click,
            _type_fn=type_fn,
            _hotkey_fn=hotkey_fn,
            preferred_transport="uia",
            max_retries=1,
        )
        result = await macro.execute(coordinates=(100, 200), new_value="hello")

        assert result.success is True
        assert len(native_calls) >= 1, "native click was not called"
        assert len(fallback_calls) == 0, "fallback should not be called when native succeeds"  # noqa: E501

    @pytest.mark.asyncio
    async def test_fallback_used_when_native_fails(self):
        native_calls: list[str | None] = []
        fallback_calls: list[tuple[int, int]] = []

        async def native_click_fail(element_id: str | None) -> bool:
            native_calls.append(element_id)
            return False  # always fail

        async def fallback_click(coords: tuple[int, int]) -> bool:
            fallback_calls.append(coords)
            return True

        async def type_fn(text: str) -> bool:
            return True

        async def hotkey_fn(keys: list[str]) -> bool:
            return True

        macro = SafeFieldReplace(
            _native_click_fn=native_click_fail,
            _fallback_click_fn=fallback_click,
            _type_fn=type_fn,
            _hotkey_fn=hotkey_fn,
            preferred_transport="uia",
            max_retries=1,
        )
        result = await macro.execute(coordinates=(100, 200), new_value="hello")

        assert result.success is True
        assert len(native_calls) >= 1, "native click should have been attempted"
        assert len(fallback_calls) >= 1, "fallback click should have been used"
