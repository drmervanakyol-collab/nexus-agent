"""
tests/unit/test_capture_worker.py
Unit tests for Faz 18 — Capture Worker V1-core.

Tests are organized into four sections:
  1. Frame value object (crop / resize / png)
  2. RingBuffer (overflow, concurrent access)
  3. Shared-memory transport (write_shm_frame / read_shm_frame)
  4. CaptureWorkerClient supervisor (restart / 4th crash)

All tests run without a real display, dxcam, or subprocess spawn.
"""
from __future__ import annotations

import multiprocessing
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from nexus.capture.capture_worker import (
    CaptureWorkerClient,
    CaptureWorkerProcess,
    _SHM_HEADER_SIZE,
    _SHM_TOTAL_SIZE,
    read_shm_frame,
    write_shm_frame,
)
from nexus.capture.frame import Frame
from nexus.capture.ring_buffer import RingBuffer
from nexus.core.errors import CaptureError
from nexus.core.settings import NexusSettings
from nexus.core.types import Rect

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(
    fps: int = 15,
    max_frame_buffer: int = 8,
) -> NexusSettings:
    return NexusSettings.model_validate(
        {"capture": {"fps": fps, "max_frame_buffer": max_frame_buffer}}
    )


def _make_frame(
    width: int = 4,
    height: int = 4,
    seq: int = 1,
    color: tuple[int, int, int] = (128, 64, 32),
) -> Frame:
    """Create a minimal test frame filled with a solid color."""
    data = np.full((height, width, 3), color, dtype=np.uint8)
    return Frame(
        data=data,
        width=width,
        height=height,
        captured_at_monotonic=time.monotonic(),
        captured_at_utc="2026-04-03T00:00:00+00:00",
        sequence_number=seq,
    )


def _crashing_process_factory(
    n_alive_checks: int = 0,
) -> tuple[list[Any], Any]:
    """
    Returns (processes_list, factory_fn).

    Each call to factory_fn() returns a new mock Process that reports
    is_alive() == False after *n_alive_checks* checks have been made.
    processes_list collects all created mocks for inspection.
    """
    created: list[Any] = []

    def factory() -> Any:
        proc = MagicMock()
        proc.pid = len(created) + 1
        proc.exitcode = -1

        # is_alive() returns True for the first n_alive_checks calls, then False
        alive_state = {"remaining": n_alive_checks}

        def is_alive() -> bool:
            if alive_state["remaining"] > 0:
                alive_state["remaining"] -= 1
                return True
            return False

        proc.is_alive.side_effect = is_alive
        created.append(proc)
        return proc

    return created, factory


# ---------------------------------------------------------------------------
# 1. Frame value object
# ---------------------------------------------------------------------------


class TestFrame:
    def test_attributes(self) -> None:
        f = _make_frame(width=10, height=8, seq=42)
        assert f.width == 10
        assert f.height == 8
        assert f.sequence_number == 42
        assert f.data.shape == (8, 10, 3)
        assert f.data.dtype == np.uint8

    def test_crop_dimensions(self) -> None:
        f = _make_frame(width=20, height=16)
        cropped = f.crop(Rect(2, 3, 10, 8))
        assert cropped.width == 10
        assert cropped.height == 8

    def test_crop_pixel_values(self) -> None:
        f = _make_frame(width=10, height=10, color=(255, 0, 0))
        cropped = f.crop(Rect(0, 0, 5, 5))
        assert np.all(cropped.data == 255) or cropped.data[0, 0, 0] == 255

    def test_crop_clamps_to_frame_bounds(self) -> None:
        f = _make_frame(width=10, height=10)
        # Rect extends beyond frame — result is clipped
        cropped = f.crop(Rect(8, 8, 100, 100))
        assert cropped.width == 2
        assert cropped.height == 2

    def test_crop_inherits_timestamps(self) -> None:
        f = _make_frame()
        cropped = f.crop(Rect(0, 0, 2, 2))
        assert cropped.captured_at_utc == f.captured_at_utc
        assert cropped.sequence_number == f.sequence_number

    def test_crop_returns_independent_copy(self) -> None:
        f = _make_frame(width=10, height=10)
        cropped = f.crop(Rect(0, 0, 5, 5))
        # Mutating crop should not affect original
        cropped.data[0, 0, 0] = 0
        assert f.data[0, 0, 0] == 128  # original color unchanged

    def test_resize_half(self) -> None:
        f = _make_frame(width=20, height=20)
        r = f.resize(0.5)
        assert r.width == 10
        assert r.height == 10

    def test_resize_double(self) -> None:
        f = _make_frame(width=10, height=8)
        r = f.resize(2.0)
        assert r.width == 20
        assert r.height == 16

    def test_resize_identity(self) -> None:
        f = _make_frame(width=10, height=10)
        r = f.resize(1.0)
        assert r.width == 10
        assert r.height == 10

    def test_resize_zero_raises(self) -> None:
        f = _make_frame()
        with pytest.raises(ValueError, match="scale"):
            f.resize(0.0)

    def test_resize_negative_raises(self) -> None:
        f = _make_frame()
        with pytest.raises(ValueError, match="scale"):
            f.resize(-1.0)

    def test_to_png_bytes_valid_png(self) -> None:
        f = _make_frame(width=8, height=8)
        png = f.to_png_bytes()
        # PNG magic bytes
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_to_png_bytes_non_empty(self) -> None:
        f = _make_frame(width=16, height=16)
        assert len(f.to_png_bytes()) > 0


# ---------------------------------------------------------------------------
# 2. RingBuffer
# ---------------------------------------------------------------------------


class TestRingBuffer:
    def test_push_and_latest(self) -> None:
        rb: RingBuffer[int] = RingBuffer(4)
        rb.push(10)
        assert rb.latest() == 10

    def test_latest_empty_is_none(self) -> None:
        rb: RingBuffer[int] = RingBuffer(4)
        assert rb.latest() is None

    def test_len_after_push(self) -> None:
        rb: RingBuffer[int] = RingBuffer(4)
        for i in range(3):
            rb.push(i)
        assert len(rb) == 3

    def test_capacity(self) -> None:
        rb: RingBuffer[int] = RingBuffer(5)
        assert rb.capacity == 5

    def test_overflow_drops_oldest(self) -> None:
        """
        Pushing beyond capacity must silently drop the oldest item.
        """
        rb: RingBuffer[int] = RingBuffer(3)
        for i in range(6):  # push 0..5
            rb.push(i)
        # Buffer should hold the 3 most recent: 3, 4, 5
        assert len(rb) == 3
        assert rb.last_n(3) == [3, 4, 5]

    def test_overflow_latest_is_newest(self) -> None:
        rb: RingBuffer[int] = RingBuffer(2)
        rb.push(1)
        rb.push(2)
        rb.push(3)  # 1 is dropped
        assert rb.latest() == 3

    def test_last_n_all(self) -> None:
        rb: RingBuffer[int] = RingBuffer(5)
        for i in range(5):
            rb.push(i)
        assert rb.last_n(5) == [0, 1, 2, 3, 4]

    def test_last_n_partial(self) -> None:
        rb: RingBuffer[int] = RingBuffer(5)
        for i in range(5):
            rb.push(i)
        assert rb.last_n(3) == [2, 3, 4]

    def test_last_n_more_than_available(self) -> None:
        rb: RingBuffer[int] = RingBuffer(5)
        rb.push(1)
        rb.push(2)
        result = rb.last_n(10)
        assert result == [1, 2]

    def test_last_n_empty_buffer(self) -> None:
        rb: RingBuffer[int] = RingBuffer(4)
        assert rb.last_n(4) == []

    def test_clear_empties_buffer(self) -> None:
        rb: RingBuffer[int] = RingBuffer(4)
        for i in range(4):
            rb.push(i)
        rb.clear()
        assert len(rb) == 0
        assert rb.latest() is None

    def test_is_full(self) -> None:
        rb: RingBuffer[int] = RingBuffer(3)
        rb.push(1)
        rb.push(2)
        rb.push(3)
        assert rb.is_full is True

    def test_is_empty(self) -> None:
        rb: RingBuffer[int] = RingBuffer(3)
        assert rb.is_empty is True

    def test_capacity_one(self) -> None:
        rb: RingBuffer[int] = RingBuffer(1)
        rb.push(7)
        rb.push(8)
        assert rb.latest() == 8
        assert len(rb) == 1

    def test_capacity_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            RingBuffer(0)

    # ------------------------------------------------------------------
    # Concurrent read / write safety
    # ------------------------------------------------------------------

    def test_concurrent_push_and_read_no_corruption(self) -> None:
        """
        100 threads push frames simultaneously; no exception must occur
        and len() must never exceed capacity.
        """
        capacity = 20
        rb: RingBuffer[int] = RingBuffer(capacity)
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(start, start + 50):
                    rb.push(i)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i * 50,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Exceptions during concurrent push: {errors}"
        assert len(rb) <= capacity

    def test_concurrent_push_and_latest_no_exception(self) -> None:
        """
        Concurrent producers and consumers must not raise any exceptions.
        """
        rb: RingBuffer[Frame] = RingBuffer(10)
        errors: list[Exception] = []
        stop = threading.Event()

        def producer() -> None:
            try:
                for i in range(200):
                    rb.push(_make_frame(seq=i))
            except Exception as exc:
                errors.append(exc)

        def consumer() -> None:
            try:
                while not stop.is_set():
                    _ = rb.latest()
                    _ = rb.last_n(5)
            except Exception as exc:
                errors.append(exc)

        consumers = [threading.Thread(target=consumer) for _ in range(3)]
        producers = [threading.Thread(target=producer) for _ in range(3)]

        for t in consumers + producers:
            t.start()
        for t in producers:
            t.join()
        stop.set()
        for t in consumers:
            t.join()

        assert errors == [], f"Exceptions during concurrent access: {errors}"

    def test_overflow_during_concurrent_push(self) -> None:
        """
        Pushing 10x capacity across threads must not corrupt the buffer.
        """
        rb: RingBuffer[int] = RingBuffer(5)
        errors: list[Exception] = []

        def push_many(start: int) -> None:
            try:
                for i in range(start, start + 10):
                    rb.push(i)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=push_many, args=(i * 10,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(rb) == 5  # never exceeds capacity


# ---------------------------------------------------------------------------
# 3. Shared-memory transport
# ---------------------------------------------------------------------------


class TestShmTransport:
    @pytest.fixture()
    def shm(self):  # type: ignore[type-arg]
        """Allocate a SharedMemory block and clean it up after the test."""
        shm = SharedMemory(create=True, size=_SHM_TOTAL_SIZE)
        yield shm
        shm.close()
        shm.unlink()

    def test_read_empty_shm_returns_none(self, shm: SharedMemory) -> None:
        assert read_shm_frame(shm.buf) is None

    def test_write_then_read_roundtrip(self, shm: SharedMemory) -> None:
        width, height = 8, 6
        original = np.arange(width * height * 3, dtype=np.uint8).reshape(
            (height, width, 3)
        )
        mono = 123.456
        utc = "2026-04-03T00:00:00+00:00"

        write_shm_frame(shm.buf, original, sequence_number=1, captured_at_monotonic=mono, captured_at_utc=utc)
        frame = read_shm_frame(shm.buf)

        assert frame is not None
        assert frame.width == width
        assert frame.height == height
        assert frame.sequence_number == 1
        assert frame.captured_at_monotonic == pytest.approx(mono)
        assert utc[:31] in frame.captured_at_utc or frame.captured_at_utc[:20] == utc[:20]
        assert np.array_equal(frame.data, original)

    def test_pixel_values_preserved(self, shm: SharedMemory) -> None:
        """Every pixel value is preserved across the write/read cycle."""
        width, height = 16, 12
        data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        write_shm_frame(shm.buf, data, 42, 0.0, "2026-01-01T00:00:00+00:00")
        frame = read_shm_frame(shm.buf)
        assert frame is not None
        assert np.array_equal(frame.data, data)

    def test_returned_data_is_copy(self, shm: SharedMemory) -> None:
        """
        Mutating the frame data after reading must not corrupt shm.
        """
        data = np.full((4, 4, 3), 200, dtype=np.uint8)
        write_shm_frame(shm.buf, data, 1, 0.0, "")
        frame = read_shm_frame(shm.buf)
        assert frame is not None
        frame.data[0, 0, 0] = 0  # mutate copy
        # Re-read from shm — original value must still be there
        frame2 = read_shm_frame(shm.buf)
        assert frame2 is not None
        assert frame2.data[0, 0, 0] == 200

    def test_sequence_number_stored(self, shm: SharedMemory) -> None:
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        write_shm_frame(shm.buf, data, sequence_number=999, captured_at_monotonic=0.0, captured_at_utc="")
        frame = read_shm_frame(shm.buf)
        assert frame is not None
        assert frame.sequence_number == 999

    def test_overwrite_updates_frame(self, shm: SharedMemory) -> None:
        first = np.full((4, 4, 3), 10, dtype=np.uint8)
        second = np.full((4, 4, 3), 20, dtype=np.uint8)
        write_shm_frame(shm.buf, first, 1, 0.0, "")
        write_shm_frame(shm.buf, second, 2, 1.0, "")
        frame = read_shm_frame(shm.buf)
        assert frame is not None
        assert frame.sequence_number == 2
        assert frame.data[0, 0, 0] == 20

    def test_different_resolutions(self, shm: SharedMemory) -> None:
        for w, h in [(100, 50), (320, 240), (1920, 1080)]:
            data = np.zeros((h, w, 3), dtype=np.uint8)
            write_shm_frame(shm.buf, data, 1, 0.0, "")
            frame = read_shm_frame(shm.buf)
            assert frame is not None
            assert frame.width == w
            assert frame.height == h

    def test_header_size_is_128(self) -> None:
        assert _SHM_HEADER_SIZE == 128


# ---------------------------------------------------------------------------
# 4. CaptureWorkerClient supervisor
# ---------------------------------------------------------------------------


class TestSupervisor:
    """
    Tests use mock process factories with no real subprocess spawning.
    restart_delay and poll interval are set to 0 for fast execution.
    """

    def _fast_client(
        self,
        process_factory: Any,
        max_frame_buffer: int = 4,
    ) -> CaptureWorkerClient:
        """Build a client wired with a mock factory and no delays."""
        return CaptureWorkerClient(
            _settings(max_frame_buffer=max_frame_buffer),
            _process_factory=process_factory,
            _restart_delay_s=0.0,
            _supervisor_poll_s=0.01,  # 10 ms poll for fast tests
        )

    def test_start_creates_process(self) -> None:
        processes, factory = _crashing_process_factory(n_alive_checks=100)
        client = self._fast_client(factory)
        try:
            client.start()
            assert len(processes) >= 1
            assert processes[0].start.called
        finally:
            client._intentional_stop = True
            client._running = False

    def test_is_healthy_when_process_alive(self) -> None:
        _, factory = _crashing_process_factory(n_alive_checks=1000)
        client = self._fast_client(factory)
        try:
            client.start()
            time.sleep(0.05)
            assert client.is_healthy() is True
        finally:
            client._intentional_stop = True
            client._running = False

    def test_is_healthy_false_before_start(self) -> None:
        _, factory = _crashing_process_factory()
        client = self._fast_client(factory)
        assert client.is_healthy() is False

    def test_supervisor_restarts_after_crash(self) -> None:
        """
        A crashing process must be restarted by the supervisor.
        After the first crash the factory must have been called at least twice.
        """
        processes, factory = _crashing_process_factory(n_alive_checks=0)
        client = self._fast_client(factory)
        try:
            client.start()
            # Give supervisor time to detect crash and restart
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if len(processes) >= 2:
                    break
                time.sleep(0.02)
            assert len(processes) >= 2, (
                f"Expected >=2 process creations (restart), got {len(processes)}"
            )
        finally:
            client._intentional_stop = True
            client._running = False

    def test_supervisor_restart_count_increments(self) -> None:
        processes, factory = _crashing_process_factory(n_alive_checks=0)
        client = self._fast_client(factory)
        try:
            client.start()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if client._restart_count >= 1:
                    break
                time.sleep(0.02)
            assert client._restart_count >= 1
        finally:
            client._intentional_stop = True
            client._running = False

    def test_fourth_crash_sets_capture_error(self) -> None:
        """
        After _MAX_RESTARTS (3) restarts the supervisor must set _error
        to a CaptureError instance.
        """
        _, factory = _crashing_process_factory(n_alive_checks=0)
        client = self._fast_client(factory)
        # _MAX_RESTARTS == 3, so 4th crash triggers error
        try:
            client.start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if client._error is not None:
                    break
                time.sleep(0.02)
            assert client._error is not None
            assert isinstance(client._error, CaptureError)
        finally:
            client._intentional_stop = True
            client._running = False

    def test_get_latest_frame_raises_after_max_restarts(self) -> None:
        """
        get_latest_frame() must raise CaptureError once the supervisor
        has given up.
        """
        _, factory = _crashing_process_factory(n_alive_checks=0)
        client = self._fast_client(factory)
        try:
            client.start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if client._error is not None:
                    break
                time.sleep(0.02)

            with pytest.raises(CaptureError):
                client.get_latest_frame()
        finally:
            client._intentional_stop = True
            client._running = False

    def test_get_frame_history_raises_after_max_restarts(self) -> None:
        _, factory = _crashing_process_factory(n_alive_checks=0)
        client = self._fast_client(factory)
        try:
            client.start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if client._error is not None:
                    break
                time.sleep(0.02)

            with pytest.raises(CaptureError):
                client.get_frame_history(5)
        finally:
            client._intentional_stop = True
            client._running = False

    def test_max_restarts_constant_is_3(self) -> None:
        assert CaptureWorkerClient._MAX_RESTARTS == 3

    def test_stop_sets_intentional_flag(self) -> None:
        _, factory = _crashing_process_factory(n_alive_checks=1000)
        client = self._fast_client(factory)
        client.start()
        client.stop()
        assert client._intentional_stop is True
        assert client._running is False


# ---------------------------------------------------------------------------
# 5. CaptureWorkerProcess (unit, no real subprocess)
# ---------------------------------------------------------------------------


class TestCaptureWorkerProcessUnit:
    """
    Drive CaptureWorkerProcess.run() directly in-process using an
    injectable _capture_fn that returns synthetic frames.
    """

    def test_writes_frames_to_shm(self) -> None:
        """Worker writes captured frames to shared memory."""
        shm = SharedMemory(create=True, size=_SHM_TOTAL_SIZE)
        stop_ev = MagicMock()
        fq: Any = MagicMock()
        fq.put_nowait = MagicMock()

        call_count = 0

        def capture_fn() -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return np.full((4, 4, 3), call_count, dtype=np.uint8)

        real_stop = multiprocessing.Event()

        worker = CaptureWorkerProcess(
            shm_name=shm.name,
            stop_event=real_stop,
            frame_queue=fq,
            fps=100,
            _capture_fn=capture_fn,
        )

        def run_and_stop() -> None:
            time.sleep(0.08)
            real_stop.set()

        t = threading.Thread(target=run_and_stop)
        t.start()
        worker.run()
        t.join()

        # At least one frame must have been written
        assert call_count >= 1
        frame = read_shm_frame(shm.buf)
        assert frame is not None
        assert frame.width == 4
        assert frame.height == 4

        shm.close()
        shm.unlink()

    def test_none_capture_fn_skips_write(self) -> None:
        """If capture_fn returns None no write occurs."""
        shm = SharedMemory(create=True, size=_SHM_TOTAL_SIZE)
        fq: Any = MagicMock()
        fq.put_nowait = MagicMock()

        real_stop = multiprocessing.Event()
        frames_skipped: list[int] = []

        def capture_fn() -> None:
            frames_skipped.append(1)
            return None

        worker = CaptureWorkerProcess(
            shm_name=shm.name,
            stop_event=real_stop,
            frame_queue=fq,
            fps=100,
            _capture_fn=capture_fn,
        )

        def stop_soon() -> None:
            time.sleep(0.06)
            real_stop.set()

        t = threading.Thread(target=stop_soon)
        t.start()
        worker.run()
        t.join()

        assert len(frames_skipped) >= 1
        assert read_shm_frame(shm.buf) is None  # nothing written

        shm.close()
        shm.unlink()

    def test_stop_method_sets_event(self) -> None:
        real_stop = multiprocessing.Event()
        fq: Any = MagicMock()
        fq.put_nowait = MagicMock()
        shm_mock = MagicMock()

        worker = CaptureWorkerProcess(
            shm_name="dummy",
            stop_event=real_stop,
            frame_queue=fq,
            fps=15,
            _capture_fn=lambda: None,
        )
        worker.stop()
        assert real_stop.is_set()


# ---------------------------------------------------------------------------
# 6. FPS tracker
# ---------------------------------------------------------------------------


class TestFpsTracker:
    def test_fps_zero_with_single_sample(self) -> None:
        from nexus.capture.capture_worker import _FpsTracker

        t = _FpsTracker()
        t.record(1.0)
        assert t.fps() == pytest.approx(0.0)

    def test_fps_zero_empty(self) -> None:
        from nexus.capture.capture_worker import _FpsTracker

        t = _FpsTracker()
        assert t.fps() == pytest.approx(0.0)

    def test_fps_computed_correctly(self) -> None:
        from nexus.capture.capture_worker import _FpsTracker

        t = _FpsTracker(window_s=10.0)
        base = 1000.0
        # Simulate 10 frames at exactly 10 fps (0.1 s apart)
        for i in range(11):
            t.record(base + i * 0.1)
        fps = t.fps()
        assert fps == pytest.approx(10.0, rel=0.05)

    def test_fps_old_samples_expire(self) -> None:
        from nexus.capture.capture_worker import _FpsTracker

        t = _FpsTracker(window_s=1.0)
        # Record old samples well outside the window
        for i in range(5):
            t.record(0.0 + i * 0.1)
        # Record 10 fps samples in recent window
        base = 100.0
        for i in range(11):
            t.record(base + i * 0.1)
        fps = t.fps()
        assert fps == pytest.approx(10.0, rel=0.1)
