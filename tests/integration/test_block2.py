"""
tests/integration/test_block2.py
Blok 2 Integration Tests — Faz 22

TEST 1 — Full capture pipeline
TEST 2 — Stabilization timeout
TEST 3 — Lock screen suspend
TEST 4 — Dirty region + cache
TEST 5 — Memory budget
TEST 6 — Dedicated subprocess restart
  3 crash → restart
  4. crash → CaptureError

Her test gerçek sınıfların birlikte çalışmasını doğrular.  Platform/OS
bağımlılıkları inject edilir; gerçek donanım veya dxcam gerekmez.
"""
from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator
from itertools import cycle
from typing import Any

import numpy as np
import pytest

from nexus.capture.frame import Frame
from nexus.capture.orchestrator import CaptureOrchestrator, StableFrame
from nexus.capture.session_detector import (
    FrozenFrameWatchdog,
    SessionDetector,
    SessionInfo,
    SessionType,
)
from nexus.capture.stabilization import StabilizationGate, StabilizationResult
from nexus.core.errors import CaptureError, PolicyBlockedError
from nexus.core.settings import CaptureSettings, NexusSettings

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_UTC = "2026-04-04T00:00:00+00:00"


def _make_frame(
    color: tuple[int, int, int] = (100, 100, 100),
    width: int = 32,
    height: int = 32,
    seq: int = 1,
) -> Frame:
    data = np.full((height, width, 3), color, dtype=np.uint8)
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=0.0,
        captured_at_utc=_UTC,
        sequence_number=seq,
    )


def _frame_array(data: np.ndarray, seq: int = 1) -> Frame:
    h, w = data.shape[:2]
    return Frame(
        data=data.copy(),
        width=w,
        height=h,
        captured_at_monotonic=0.0,
        captured_at_utc=_UTC,
        sequence_number=seq,
    )


def _normal_detector() -> SessionDetector:
    return SessionDetector(
        _get_title_fn=lambda: "TestApp",
        _is_locked_fn=lambda: False,
        _is_secure_desktop_fn=lambda: False,
        _is_minimized_fn=lambda: False,
    )


def _locked_detector() -> SessionDetector:
    return SessionDetector(
        _get_title_fn=lambda: "LockApp",
        _is_locked_fn=lambda: True,
        _is_secure_desktop_fn=lambda: False,
        _is_minimized_fn=lambda: False,
    )


def _secure_detector() -> SessionDetector:
    return SessionDetector(
        _get_title_fn=lambda: "",
        _is_locked_fn=lambda: False,
        _is_secure_desktop_fn=lambda: True,
        _is_minimized_fn=lambda: False,
    )


class _StableGate:
    """Immediately returns stable=True."""

    def wait_for_stable(self, **_: object) -> StabilizationResult:
        return StabilizationResult(
            stable=True, waited_ms=10.0, reason="stable", change_ratio_final=0.0
        )


class _TimeoutGate:
    """Immediately returns stable=False / reason='timeout'."""

    def wait_for_stable(self, **_: object) -> StabilizationResult:
        return StabilizationResult(
            stable=False, waited_ms=3000.0, reason="timeout", change_ratio_final=0.5
        )


class _PassingWatchdog:
    def check(self, frame: Frame, session: SessionInfo) -> None:  # noqa: ARG002
        pass


def _cycling_get_frame(frames: list[Frame]) -> Callable[[], Frame]:
    cyc = cycle(frames)
    return lambda: next(cyc)


def _make_orchestrator(
    *,
    frames: list[Frame],
    session_detector: SessionDetector | None = None,
    gate: object | None = None,
    watchdog: object | None = None,
    memory_bytes: int = 0,
    max_memory_bytes: int = 512 * 1024 * 1024,
    on_pressure: Callable[[], None] | None = None,
    settings: CaptureSettings | None = None,
) -> CaptureOrchestrator:
    get_frame = _cycling_get_frame(frames)
    return CaptureOrchestrator(
        settings=settings or CaptureSettings(),
        _get_frame_fn=get_frame,
        _session_detector=session_detector or _normal_detector(),
        _stabilization_gate=gate or _StableGate(),  # type: ignore[arg-type]
        _frozen_watchdog=watchdog or _PassingWatchdog(),  # type: ignore[arg-type]
        _memory_fn=lambda: memory_bytes,
        _max_memory_bytes=max_memory_bytes,
        _on_pressure_fn=on_pressure,
        _utc_now_fn=lambda: _UTC,
    )


@contextlib.contextmanager
def _managed_client(client: Any) -> Iterator[Any]:
    """
    Yield *client* and guarantee shared-memory cleanup on exit.

    ``CaptureWorkerClient.stop()`` returns early when ``_running`` is
    already False (supervisor has given up).  This context manager
    forces cleanup regardless.
    """
    try:
        yield client
    finally:
        client._intentional_stop = True
        client._running = False
        # Give background threads a moment to notice _running=False.
        for thread in (
            getattr(client, "_supervisor_thread", None),
            getattr(client, "_frame_reader_thread", None),
        ):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)
        shm = getattr(client, "_shm", None)
        if shm is not None:
            with contextlib.suppress(Exception):
                shm.close()
            with contextlib.suppress(Exception):
                shm.unlink()
            client._shm = None


# ---------------------------------------------------------------------------
# TEST 1 — Full capture pipeline
# ---------------------------------------------------------------------------


class TestFullCapturePipeline:
    """
    CaptureOrchestrator avec real StabilizationGate, real DirtyRegionDetector,
    real FrozenFrameWatchdog, real CapturePolicy, real SessionDetector
    (injected platform calls).  Static frame → stabilises in 4 polls;
    StableFrame returned with all fields populated.
    """

    def _real_orchestrator(self, frame: Frame) -> CaptureOrchestrator:
        get_frame = lambda: frame  # noqa: E731
        gate = StabilizationGate(
            _get_frame_fn=get_frame,
            _ocr_fn=lambda f: "",
            _sleep_fn=lambda s: None,  # instant
        )
        frozen_t = 0.0
        watchdog = FrozenFrameWatchdog(
            threshold_ms=5000.0,
            _time_fn=lambda: frozen_t,  # time never advances → never frozen
        )
        return CaptureOrchestrator(
            settings=CaptureSettings(
                stabilization_timeout_ms=5000,
                stabilization_poll_ms=1,
            ),
            _get_frame_fn=get_frame,
            _session_detector=_normal_detector(),
            _stabilization_gate=gate,
            _frozen_watchdog=watchdog,
            _memory_fn=lambda: 0,
            _utc_now_fn=lambda: _UTC,
        )

    def test_returns_stable_frame(self) -> None:
        orch = self._real_orchestrator(_make_frame(color=(10, 20, 30)))
        result = orch.get_stable_frame()
        assert isinstance(result, StableFrame)

    def test_frame_pixel_data_preserved(self) -> None:
        f = _make_frame(color=(11, 22, 33))
        orch = self._real_orchestrator(f)
        result = orch.get_stable_frame()
        assert np.array_equal(result.frame.data, f.data)

    def test_stabilization_result_is_stable(self) -> None:
        orch = self._real_orchestrator(_make_frame())
        result = orch.get_stable_frame()
        assert result.stabilization_result.stable is True
        assert result.stabilization_result.reason == "stable"

    def test_change_ratio_is_zero_for_static_frame(self) -> None:
        orch = self._real_orchestrator(_make_frame())
        result = orch.get_stable_frame()
        assert result.stabilization_result.change_ratio_final == pytest.approx(0.0)

    def test_session_info_is_normal(self) -> None:
        orch = self._real_orchestrator(_make_frame())
        result = orch.get_stable_frame()
        assert result.session.session_type is SessionType.NORMAL
        assert result.session.is_locked is False

    def test_captured_at_populated(self) -> None:
        orch = self._real_orchestrator(_make_frame())
        result = orch.get_stable_frame()
        assert result.captured_at == _UTC

    def test_first_call_no_prev_frame_no_dirty(self) -> None:
        orch = self._real_orchestrator(_make_frame())
        result = orch.get_stable_frame()
        assert result.prev_frame is None
        assert result.dirty_regions is None

    def test_metrics_increment_after_success(self) -> None:
        orch = self._real_orchestrator(_make_frame())
        orch.get_stable_frame()
        m = orch.get_metrics()
        assert m["total_calls"] == 1
        assert m["stable_frames"] == 1


# ---------------------------------------------------------------------------
# TEST 2 — Stabilization timeout
# ---------------------------------------------------------------------------


class TestStabilizationTimeout:
    """
    Timeout scenario: gate returns stable=False.  Orchestrator returns a
    StableFrame (graceful degradation) with the timeout result embedded.
    Also covers the real gate with a 1 ms deadline on ever-changing frames.
    """

    def test_timeout_gate_returns_stable_frame(self) -> None:
        orch = _make_orchestrator(frames=[_make_frame()], gate=_TimeoutGate())
        result = orch.get_stable_frame()
        assert isinstance(result, StableFrame)

    def test_timeout_result_stable_false(self) -> None:
        orch = _make_orchestrator(frames=[_make_frame()], gate=_TimeoutGate())
        result = orch.get_stable_frame()
        assert result.stabilization_result.stable is False

    def test_timeout_reason_embedded(self) -> None:
        orch = _make_orchestrator(frames=[_make_frame()], gate=_TimeoutGate())
        result = orch.get_stable_frame()
        assert result.stabilization_result.reason == "timeout"

    def test_timeout_waited_ms_forwarded(self) -> None:
        orch = _make_orchestrator(frames=[_make_frame()], gate=_TimeoutGate())
        result = orch.get_stable_frame()
        assert result.stabilization_result.waited_ms == pytest.approx(3000.0)

    def test_real_gate_timeout_on_ever_changing_frames(self) -> None:
        """Real StabilizationGate with 1 ms timeout + always-changing frames."""
        color_seq = cycle(range(0, 256, 16))

        def get_frame() -> Frame:
            c = next(color_seq)
            return _make_frame(color=(c, 255 - c, c // 2))

        gate = StabilizationGate(
            _get_frame_fn=get_frame,
            _ocr_fn=lambda f: "",
            _sleep_fn=lambda s: None,
        )
        orch = CaptureOrchestrator(
            settings=CaptureSettings(stabilization_timeout_ms=1, stabilization_poll_ms=1),
            _get_frame_fn=get_frame,
            _session_detector=_normal_detector(),
            _stabilization_gate=gate,
            _frozen_watchdog=_PassingWatchdog(),
            _memory_fn=lambda: 0,
            _utc_now_fn=lambda: _UTC,
        )
        result = orch.get_stable_frame()
        assert result.stabilization_result.stable is False


# ---------------------------------------------------------------------------
# TEST 3 — Lock screen suspend
# ---------------------------------------------------------------------------


class TestLockScreenSuspend:
    """
    LOCKED session → CapturePolicy → SUSPEND_NOTIFY → PolicyBlockedError.
    SECURE_DESKTOP session → SUSPEND → PolicyBlockedError.
    Normal / RDP_MINIMIZED sessions must NOT raise.
    """

    def test_locked_raises_policy_blocked(self) -> None:
        orch = _make_orchestrator(
            frames=[_make_frame()],
            session_detector=_locked_detector(),
        )
        with pytest.raises(PolicyBlockedError):
            orch.get_stable_frame()

    def test_secure_desktop_raises_policy_blocked(self) -> None:
        orch = _make_orchestrator(
            frames=[_make_frame()],
            session_detector=_secure_detector(),
        )
        with pytest.raises(PolicyBlockedError):
            orch.get_stable_frame()

    def test_locked_rule_contains_locked(self) -> None:
        orch = _make_orchestrator(
            frames=[_make_frame()],
            session_detector=_locked_detector(),
        )
        with pytest.raises(PolicyBlockedError) as exc:
            orch.get_stable_frame()
        assert "locked" in exc.value.rule

    def test_secure_rule_contains_secure(self) -> None:
        orch = _make_orchestrator(
            frames=[_make_frame()],
            session_detector=_secure_detector(),
        )
        with pytest.raises(PolicyBlockedError) as exc:
            orch.get_stable_frame()
        assert "secure" in exc.value.rule

    def test_policy_blocked_not_recoverable(self) -> None:
        orch = _make_orchestrator(
            frames=[_make_frame()],
            session_detector=_locked_detector(),
        )
        with pytest.raises(PolicyBlockedError) as exc:
            orch.get_stable_frame()
        assert exc.value.recoverable is False

    def test_policy_blocked_severity_is_block(self) -> None:
        orch = _make_orchestrator(
            frames=[_make_frame()],
            session_detector=_locked_detector(),
        )
        with pytest.raises(PolicyBlockedError) as exc:
            orch.get_stable_frame()
        assert exc.value.severity == "block"

    def test_normal_session_no_raise(self) -> None:
        orch = _make_orchestrator(
            frames=[_make_frame()],
            session_detector=_normal_detector(),
        )
        assert isinstance(orch.get_stable_frame(), StableFrame)

    def test_rdp_minimized_no_raise(self) -> None:
        rdp_detector = SessionDetector(
            _get_title_fn=lambda: "RDP",
            _is_locked_fn=lambda: False,
            _is_secure_desktop_fn=lambda: False,
            _is_minimized_fn=lambda: True,
        )
        orch = _make_orchestrator(
            frames=[_make_frame()],
            session_detector=rdp_detector,
        )
        assert isinstance(orch.get_stable_frame(), StableFrame)

    def test_repeated_locked_increments_policy_blocked_metric(self) -> None:
        orch = _make_orchestrator(
            frames=[_make_frame()],
            session_detector=_locked_detector(),
        )
        for _ in range(4):
            with pytest.raises(PolicyBlockedError):
                orch.get_stable_frame()
        assert orch.get_metrics()["policy_blocked"] == 4


# ---------------------------------------------------------------------------
# TEST 4 — Dirty region + cache
# ---------------------------------------------------------------------------


class TestDirtyRegionAndCache:
    """
    prev_frame cache: first call → no dirty regions.  Second call with
    different content → real DirtyRegionDetector computes dirty blocks.
    force_full_refresh bypasses the cache.
    """

    def test_first_call_dirty_regions_none(self) -> None:
        orch = _make_orchestrator(frames=[_make_frame()])
        result = orch.get_stable_frame()
        assert result.dirty_regions is None

    def test_second_call_with_same_frame_zero_dirty_blocks(self) -> None:
        f = _make_frame(color=(77, 77, 77))
        orch = _make_orchestrator(frames=[f])
        orch.get_stable_frame()
        result = orch.get_stable_frame()
        assert result.dirty_regions is not None
        assert result.dirty_regions.blocks == ()
        assert result.dirty_regions.change_ratio == pytest.approx(0.0)

    def test_changed_frame_produces_dirty_regions(self) -> None:
        f1 = _make_frame(color=(0, 0, 0), seq=1)
        f2 = _make_frame(color=(200, 100, 50), seq=2)
        orch = _make_orchestrator(frames=[f1, f2])
        orch.get_stable_frame()
        result = orch.get_stable_frame()
        assert result.dirty_regions is not None
        assert result.dirty_regions.change_ratio > 0.0

    def test_fully_changed_frames_full_refresh(self) -> None:
        f1 = _make_frame(color=(0, 0, 0), seq=1)
        f2 = _make_frame(color=(255, 255, 255), seq=2)
        orch = _make_orchestrator(frames=[f1, f2])
        orch.get_stable_frame()
        result = orch.get_stable_frame()
        assert result.dirty_regions is not None
        assert result.dirty_regions.full_refresh is True
        assert result.dirty_regions.change_ratio == pytest.approx(1.0)

    def test_force_full_refresh_skips_dirty_computation(self) -> None:
        f1 = _make_frame(color=(0, 0, 0), seq=1)
        f2 = _make_frame(color=(255, 0, 0), seq=2)
        orch = _make_orchestrator(frames=[f1, f2])
        orch.get_stable_frame()
        result = orch.get_stable_frame(force_full_refresh=True)
        assert result.dirty_regions is None

    def test_partial_change_dirty_blocks_are_rects(self) -> None:
        """Only the right half of the frame changes → partial dirty region."""
        from nexus.core.types import Rect

        w, h = 64, 32
        data1 = np.zeros((h, w, 3), dtype=np.uint8)
        data2 = data1.copy()
        data2[:, w // 2 :] = 200  # right half dirty

        get_frames = _cycling_get_frame([_frame_array(data1, 1), _frame_array(data2, 2)])

        orch = CaptureOrchestrator(
            settings=CaptureSettings(dirty_region_block_size=8),
            _get_frame_fn=get_frames,
            _session_detector=_normal_detector(),
            _stabilization_gate=_StableGate(),
            _frozen_watchdog=_PassingWatchdog(),
            _memory_fn=lambda: 0,
            _utc_now_fn=lambda: _UTC,
        )
        orch.get_stable_frame()
        result = orch.get_stable_frame()
        assert result.dirty_regions is not None
        assert 0.0 < result.dirty_regions.change_ratio < 1.0
        assert all(isinstance(b, Rect) for b in result.dirty_regions.blocks)

    def test_prev_frame_stored_after_each_call(self) -> None:
        f1 = _make_frame(seq=1)
        f2 = _make_frame(seq=2)
        orch = _make_orchestrator(frames=[f1, f2])
        r1 = orch.get_stable_frame()
        r2 = orch.get_stable_frame()
        assert r1.prev_frame is None
        assert r2.prev_frame is not None


# ---------------------------------------------------------------------------
# TEST 5 — Memory budget
# ---------------------------------------------------------------------------


class TestMemoryBudget:
    """
    Memory probe returns > max → _on_pressure_fn is called.
    Orchestrator must still return StableFrame (no exception raised).
    """

    _LIMIT = 256 * 1024 * 1024  # 256 MB

    def test_under_budget_no_callback(self) -> None:
        called: list[bool] = []
        orch = _make_orchestrator(
            frames=[_make_frame()],
            memory_bytes=100 * 1024 * 1024,
            max_memory_bytes=self._LIMIT,
            on_pressure=lambda: called.append(True),
        )
        orch.get_stable_frame()
        assert called == []

    def test_over_budget_triggers_callback(self) -> None:
        called: list[bool] = []
        orch = _make_orchestrator(
            frames=[_make_frame()],
            memory_bytes=400 * 1024 * 1024,
            max_memory_bytes=self._LIMIT,
            on_pressure=lambda: called.append(True),
        )
        orch.get_stable_frame()
        assert called == [True]

    def test_over_budget_still_returns_stable_frame(self) -> None:
        orch = _make_orchestrator(
            frames=[_make_frame()],
            memory_bytes=400 * 1024 * 1024,
            max_memory_bytes=self._LIMIT,
        )
        result = orch.get_stable_frame()
        assert isinstance(result, StableFrame)

    def test_no_callback_no_exception(self) -> None:
        """Over budget with no callback must not raise."""
        orch = _make_orchestrator(
            frames=[_make_frame()],
            memory_bytes=600 * 1024 * 1024,
            max_memory_bytes=self._LIMIT,
            on_pressure=None,
        )
        assert isinstance(orch.get_stable_frame(), StableFrame)

    def test_memory_bytes_tracked_in_metrics(self) -> None:
        mb = 150 * 1024 * 1024
        orch = _make_orchestrator(frames=[_make_frame()], memory_bytes=mb)
        orch.get_stable_frame()
        assert orch.get_metrics()["memory_bytes"] == mb

    def test_callback_called_on_every_over_budget_call(self) -> None:
        count: list[int] = [0]
        orch = _make_orchestrator(
            frames=[_make_frame()],
            memory_bytes=400 * 1024 * 1024,
            max_memory_bytes=self._LIMIT,
            on_pressure=lambda: count.__setitem__(0, count[0] + 1),
        )
        orch.get_stable_frame()
        orch.get_stable_frame()
        assert count[0] == 2


# ---------------------------------------------------------------------------
# TEST 6 — Dedicated subprocess restart
# ---------------------------------------------------------------------------


class _CrashProcess:
    """
    Mock multiprocessing.Process that 'crashes' immediately after start.
    is_alive() always returns False so the supervisor sees an immediate exit.
    """

    pid: int = 99999
    exitcode: int = -1

    def start(self) -> None:
        pass

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        pass


def _wait_for_error(client: Any, timeout: float = 3.0) -> None:
    """Block until ``client._error`` is set or *timeout* expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client._error is not None:
            return
        time.sleep(0.01)


class TestSubprocessRestart:
    """
    CaptureWorkerClient supervisor: 3 crashes → restart (restart_count ≤ 3).
    4th crash (restart_count=4 > MAX_RESTARTS=3) → _error set.
    Subsequent get_latest_frame() / get_frame_history() raise CaptureError.
    """

    def _make_client(self) -> Any:
        from nexus.capture.capture_worker import CaptureWorkerClient

        return CaptureWorkerClient(
            NexusSettings(),
            _process_factory=lambda: _CrashProcess(),
            _restart_delay_s=0.0,
            _supervisor_poll_s=0.005,
        )

    def test_get_latest_frame_raises_after_max_crashes(self) -> None:
        client = self._make_client()
        with _managed_client(client):
            client.start()
            _wait_for_error(client)
            assert client._error is not None, "Supervisor must have set _error"
            with pytest.raises(CaptureError):
                client.get_latest_frame()

    def test_get_frame_history_raises_after_max_crashes(self) -> None:
        client = self._make_client()
        with _managed_client(client):
            client.start()
            _wait_for_error(client)
            with pytest.raises(CaptureError):
                client.get_frame_history(5)

    def test_is_healthy_false_after_max_crashes(self) -> None:
        client = self._make_client()
        with _managed_client(client):
            client.start()
            _wait_for_error(client)
            assert not client.is_healthy()

    def test_restart_count_exceeds_max_restarts(self) -> None:
        from nexus.capture.capture_worker import CaptureWorkerClient

        client = self._make_client()
        with _managed_client(client):
            client.start()
            _wait_for_error(client)
            assert client._restart_count > CaptureWorkerClient._MAX_RESTARTS

    def test_process_factory_called_initial_plus_max_restarts_times(self) -> None:
        """
        1 initial start + _MAX_RESTARTS restarts = _MAX_RESTARTS + 1 total calls
        to the process factory.
        """
        from nexus.capture.capture_worker import CaptureWorkerClient

        call_count: list[int] = [0]

        def factory() -> _CrashProcess:
            call_count[0] += 1
            return _CrashProcess()

        client = CaptureWorkerClient(
            NexusSettings(),
            _process_factory=factory,
            _restart_delay_s=0.0,
            _supervisor_poll_s=0.005,
        )
        with _managed_client(client):
            client.start()
            _wait_for_error(client)
            expected = CaptureWorkerClient._MAX_RESTARTS + 1
            assert call_count[0] == expected

    def test_error_is_capture_error_instance(self) -> None:
        client = self._make_client()
        with _managed_client(client):
            client.start()
            _wait_for_error(client)
            assert isinstance(client._error, CaptureError)

    def test_error_context_contains_restart_count(self) -> None:
        client = self._make_client()
        with _managed_client(client):
            client.start()
            _wait_for_error(client)
            assert client._error is not None
            assert "restart_count" in client._error.context
