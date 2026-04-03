"""
nexus/capture/capture_worker.py
Dedicated screen-capture subprocess with shared-memory frame transport.

Architecture
------------

                ┌─────────────────────────────┐
  main process  │  CaptureWorkerClient        │
                │  ─────────────────────────  │
                │  RingBuffer[Frame]  (local) │
                │  SharedMemory (read side)   │
                │  Supervisor thread          │
                │  FrameReader thread         │
                └────────┬────────────────────┘
                         │  multiprocessing.shared_memory (zero-copy read)
                         │  multiprocessing.Queue (frame notifications)
                         │  multiprocessing.Event (stop signal)
                ┌────────▼────────────────────┐
  subprocess    │  CaptureWorkerProcess       │
                │  ─────────────────────────  │
                │  dxcam capture loop         │
                │  SharedMemory (write side)  │
                └─────────────────────────────┘

Shared Memory Layout
--------------------
Offset 0: header (_SHM_HEADER_SIZE bytes, see _SHM_HEADER_STRUCT)
Offset _SHM_HEADER_SIZE: raw RGB pixel data (width × height × 3 bytes)

The header carries metadata (magic, sequence, dimensions, timestamps,
flags).  Frame data is written first; then the header with flags=READY,
so a reader that checks magic + flags + sequence gets a consistent view.

Supervisor
----------
A background thread in CaptureWorkerClient watches the subprocess.
On unexpected exit it waits _restart_delay_s seconds and restarts.
After _MAX_RESTARTS consecutive crashes CaptureError is stored in
``_error`` and  subsequent ``get_latest_frame()`` / ``get_frame_history()``
calls raise it.

Testability
-----------
- CaptureWorkerProcess accepts an injectable ``_capture_fn`` so tests
  can drive it without dxcam.
- CaptureWorkerClient accepts an injectable ``_process_factory``
  (Callable[[], Process]) so the supervisor logic can be tested with
  mock processes that "crash" on demand.
- ``_restart_delay_s`` and ``_supervisor_poll_s`` are injectable for
  fast test execution.
"""
from __future__ import annotations

import contextlib
import datetime
import multiprocessing
import multiprocessing.queues
import multiprocessing.synchronize
import queue
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from nexus.capture.frame import Frame
from nexus.capture.ring_buffer import RingBuffer
from nexus.core.errors import CaptureError
from nexus.core.settings import NexusSettings
from nexus.infra.logger import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared memory constants
# ---------------------------------------------------------------------------

# Header struct layout (big-endian, 128 bytes total):
#   I   magic              4 bytes
#   Q   sequence_number    8 bytes
#   I   width              4 bytes
#   I   height             4 bytes
#   d   captured_at_mono   8 bytes  (float64)
#   32s captured_at_utc   32 bytes  (null-padded UTF-8)
#   B   flags              1 byte   (bit 0 = FRAME_READY)
#   67x padding           67 bytes
_SHM_HEADER_STRUCT: struct.Struct = struct.Struct(">IQIId32sB67x")
_SHM_HEADER_SIZE: int = 128  # == _SHM_HEADER_STRUCT.size
assert _SHM_HEADER_STRUCT.size == _SHM_HEADER_SIZE  # noqa: S101

_SHM_MAGIC: int = 0xCAFEBABE
_SHM_FLAG_READY: int = 0x01

# Pre-allocate shm for frames up to 4 K resolution.
# 3840 × 2160 × 3 = 24 883 200 bytes + 128 header ≈ 25 MB.
_SHM_MAX_WIDTH: int = 3840
_SHM_MAX_HEIGHT: int = 2160
_SHM_FRAME_MAX_BYTES: int = _SHM_MAX_WIDTH * _SHM_MAX_HEIGHT * 3
_SHM_TOTAL_SIZE: int = _SHM_HEADER_SIZE + _SHM_FRAME_MAX_BYTES


# ---------------------------------------------------------------------------
# Shared memory read / write helpers (module-level → testable in isolation)
# ---------------------------------------------------------------------------


def write_shm_frame(
    buf: memoryview,
    frame_data: np.ndarray,
    sequence_number: int,
    captured_at_monotonic: float,
    captured_at_utc: str,
) -> None:
    """
    Write a frame into the shared-memory buffer.

    Data is written first, then the header with FRAME_READY set, so
    a reader that verifies the magic + flag gets a consistent view.

    Parameters
    ----------
    buf:
        ``memoryview`` of the SharedMemory (must be >= _SHM_TOTAL_SIZE).
    frame_data:
        HxWx3 uint8 numpy array (RGB).
    sequence_number:
        Monotonically increasing frame counter.
    captured_at_monotonic, captured_at_utc:
        Timestamps recorded at capture time.
    """
    h, w = frame_data.shape[:2]
    flat = frame_data.flatten()
    nbytes = flat.nbytes

    # 1. Write pixel data first (before setting READY flag)
    buf[_SHM_HEADER_SIZE : _SHM_HEADER_SIZE + nbytes] = flat.tobytes()

    # 2. Write header with READY flag
    utc_bytes = (
        captured_at_utc.encode("utf-8", errors="replace")[:32].ljust(32, b"\x00")
    )
    header = _SHM_HEADER_STRUCT.pack(
        _SHM_MAGIC,
        sequence_number,
        w,
        h,
        captured_at_monotonic,
        utc_bytes,
        _SHM_FLAG_READY,
    )
    buf[:_SHM_HEADER_SIZE] = header


def read_shm_frame(buf: memoryview) -> Frame | None:
    """
    Read the latest frame from the shared-memory buffer.

    Returns ``None`` when no valid frame has been written yet
    (magic mismatch or READY flag unset).

    The returned ``Frame.data`` array is a *copy* of the shared memory
    so it remains stable after the next write.
    """
    header_bytes = bytes(buf[:_SHM_HEADER_SIZE])
    if len(header_bytes) < _SHM_HEADER_SIZE:
        return None

    try:
        (
            magic,
            seq,
            width,
            height,
            mono,
            utc_raw,
            flags,
        ) = _SHM_HEADER_STRUCT.unpack(header_bytes)
    except struct.error:
        return None

    if magic != _SHM_MAGIC:
        return None
    if not (flags & _SHM_FLAG_READY):
        return None

    nbytes = width * height * 3
    if nbytes <= 0 or nbytes > _SHM_FRAME_MAX_BYTES:
        return None

    pixel_bytes = bytes(buf[_SHM_HEADER_SIZE : _SHM_HEADER_SIZE + nbytes])
    arr = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape((height, width, 3)).copy()

    utc_str = utc_raw.rstrip(b"\x00").decode("utf-8", errors="replace")

    return Frame(
        data=arr,
        width=width,
        height=height,
        captured_at_monotonic=mono,
        captured_at_utc=utc_str,
        sequence_number=seq,
    )


# ---------------------------------------------------------------------------
# FPS tracker (sliding window)
# ---------------------------------------------------------------------------


class _FpsTracker:
    """Computes frames-per-second over a sliding time window."""

    def __init__(self, window_s: float = 2.0) -> None:
        self._window_s = window_s
        self._ts: deque[float] = deque()
        self._lock = threading.Lock()

    def record(self, monotonic: float) -> None:
        cutoff = monotonic - self._window_s
        with self._lock:
            self._ts.append(monotonic)
            while self._ts and self._ts[0] < cutoff:
                self._ts.popleft()

    def fps(self) -> float:
        with self._lock:
            n = len(self._ts)
            if n < 2:
                return 0.0
            elapsed = self._ts[-1] - self._ts[0]
            return (n - 1) / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# CaptureWorkerProcess — runs inside the subprocess
# ---------------------------------------------------------------------------


class CaptureWorkerProcess:
    """
    Capture loop that runs inside the dedicated subprocess.

    Parameters
    ----------
    shm_name:
        Name of the pre-existing SharedMemory created by
        ``CaptureWorkerClient``.
    stop_event:
        ``multiprocessing.Event`` that signals shutdown.
    frame_queue:
        ``multiprocessing.Queue`` used to notify the client of new frames
        (puts the sequence number; no frame data is sent through the queue).
    fps:
        Target capture rate.
    _capture_fn:
        Injectable capture callable ``() -> np.ndarray | None``.
        When ``None``, ``dxcam`` is used.  Inject a function in tests to
        avoid needing a real display.
    """

    def __init__(
        self,
        shm_name: str,
        stop_event: multiprocessing.synchronize.Event,
        frame_queue: multiprocessing.queues.Queue,  # type: ignore[type-arg]
        fps: int = 15,
        _capture_fn: Callable[[], np.ndarray | None] | None = None,
    ) -> None:
        self._shm_name = shm_name
        self._stop_event = stop_event
        self._frame_queue = frame_queue
        self._fps = fps
        self._capture_fn = _capture_fn

    def run(self) -> None:
        """
        Capture loop entry point.  Blocks until ``stop_event`` is set.
        """
        shm = SharedMemory(name=self._shm_name, create=False)
        camera: Any = None

        if self._capture_fn is None:
            import dxcam

            camera = dxcam.create(output_color="RGB")

        seq = 0
        interval = 1.0 / max(1, self._fps)

        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()

                if self._capture_fn is not None:
                    frame_data = self._capture_fn()
                else:
                    frame_data = camera.grab()

                if frame_data is not None and shm.buf is not None:
                    seq += 1
                    mono = time.monotonic()
                    utc = datetime.datetime.now(datetime.UTC).isoformat()
                    write_shm_frame(shm.buf, frame_data, seq, mono, utc)
                    with contextlib.suppress(queue.Full):
                        self._frame_queue.put_nowait(seq)

                elapsed = time.monotonic() - t0
                remaining = interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            if camera is not None:
                with contextlib.suppress(Exception):
                    camera.release()
            shm.close()

    def stop(self) -> None:
        """Signal the run loop to exit."""
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Module-level subprocess entry point (must be picklable on Windows spawn)
# ---------------------------------------------------------------------------


def _worker_main(
    shm_name: str,
    stop_event: multiprocessing.synchronize.Event,
    frame_queue: multiprocessing.queues.Queue,  # type: ignore[type-arg]
    fps: int,
) -> None:
    """
    Subprocess entry point.

    Called by ``multiprocessing.Process``; must be a module-level
    function to be picklable on Windows (spawn start method).
    """
    worker = CaptureWorkerProcess(shm_name, stop_event, frame_queue, fps)
    worker.run()


# ---------------------------------------------------------------------------
# CaptureWorkerClient — lives in the main process
# ---------------------------------------------------------------------------


class CaptureWorkerClient:
    """
    Manages the capture subprocess, shared memory, and frame history.

    Parameters
    ----------
    settings:
        Reads ``capture.fps`` and ``capture.max_frame_buffer``.
    _process_factory:
        ``Callable[[], multiprocessing.Process]``.  When omitted, a
        real process targeting ``_worker_main`` is created.  Inject a
        factory that returns mock processes to test the supervisor.
    _restart_delay_s:
        Seconds to wait between a crash and a restart attempt.
        Default 3.0; set to 0 in tests for speed.
    _supervisor_poll_s:
        Supervisor thread poll interval.  Default 0.5 s; set small in
        tests for speed.
    """

    _MAX_RESTARTS: int = 3

    def __init__(
        self,
        settings: NexusSettings,
        *,
        _process_factory: Callable[[], multiprocessing.Process] | None = None,
        _restart_delay_s: float = 3.0,
        _supervisor_poll_s: float = 0.5,
    ) -> None:
        self._settings = settings
        self._external_process_factory = _process_factory
        self._restart_delay_s = _restart_delay_s
        self._supervisor_poll_s = _supervisor_poll_s

        # Shared memory (created in start(), cleaned up in stop())
        self._shm: SharedMemory | None = None

        # IPC channels (created once; survive restarts)
        self._stop_event: multiprocessing.synchronize.Event = (
            multiprocessing.Event()
        )
        self._frame_queue: multiprocessing.queues.Queue[int] = (
            multiprocessing.Queue(maxsize=64)
        )

        # Subprocess handle
        self._process: multiprocessing.Process | None = None
        self._process_factory: Callable[[], multiprocessing.Process] | None = None
        self._restart_count: int = 0
        self._intentional_stop: bool = False

        # Error state (set by supervisor on max restarts)
        self._error: CaptureError | None = None

        # Frame storage
        self._ring_buffer: RingBuffer[Frame] = RingBuffer(
            settings.capture.max_frame_buffer
        )

        # Background threads
        self._running: bool = False
        self._supervisor_thread: threading.Thread | None = None
        self._frame_reader_thread: threading.Thread | None = None

        # FPS tracking
        self._fps_tracker = _FpsTracker()

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the capture subprocess and background threads.

        Creates shared memory, launches the subprocess, starts the
        supervisor and frame-reader threads.  Returns immediately;
        actual frame delivery is asynchronous.
        """
        if self._running:
            return

        # Create shared memory for frame transport
        self._shm = SharedMemory(create=True, size=_SHM_TOTAL_SIZE)

        # Build the real process factory if none was injected
        shm_name = self._shm.name
        stop_event = self._stop_event
        frame_queue = self._frame_queue
        fps = self._settings.capture.fps

        if self._external_process_factory is not None:
            self._process_factory = self._external_process_factory
        else:
            def _default_factory() -> multiprocessing.Process:
                return multiprocessing.Process(
                    target=_worker_main,
                    args=(shm_name, stop_event, frame_queue, fps),
                    daemon=True,
                )
            self._process_factory = _default_factory

        self._running = True
        self._intentional_stop = False
        self._restart_count = 0
        self._error = None

        self._start_process()

        self._supervisor_thread = threading.Thread(
            target=self._supervisor_loop, daemon=True, name="capture-supervisor"
        )
        self._supervisor_thread.start()

        self._frame_reader_thread = threading.Thread(
            target=self._frame_reader_loop, daemon=True, name="capture-frame-reader"
        )
        self._frame_reader_thread.start()

        _log.info("capture_worker_started", shm_name=shm_name, fps=fps)

    def stop(self) -> None:
        """
        Stop the capture subprocess and clean up resources.

        Idempotent: safe to call multiple times.
        """
        if not self._running:
            return

        self._intentional_stop = True
        self._running = False
        self._stop_event.set()

        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=3.0)

        if self._supervisor_thread is not None:
            self._supervisor_thread.join(timeout=2.0)
        if self._frame_reader_thread is not None:
            self._frame_reader_thread.join(timeout=2.0)

        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception as exc:
                _log.debug("shm_cleanup_failed", error=str(exc))
            self._shm = None

        _log.info("capture_worker_stopped")

    # ------------------------------------------------------------------
    # Frame access API
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> Frame | None:
        """
        Return the most recent frame via a zero-copy shared-memory read.

        Raises
        ------
        CaptureError
            When the supervisor has given up after max restarts.
        """
        self._raise_if_error()
        if self._shm is None or self._shm.buf is None:
            return None
        return read_shm_frame(self._shm.buf)

    def get_frame_history(self, n: int) -> list[Frame]:
        """
        Return up to *n* most recent frames (oldest first) from the
        local ring buffer.

        Raises
        ------
        CaptureError
            When the supervisor has given up after max restarts.
        """
        self._raise_if_error()
        return self._ring_buffer.last_n(n)

    # ------------------------------------------------------------------
    # Health / metrics API
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """
        Return True when the subprocess is running and no error is set.
        """
        if self._error is not None:
            return False
        if not self._running:
            return False
        if self._process is None:
            return False
        return self._process.is_alive()

    def get_fps(self) -> float:
        """Return measured capture rate (frames per second)."""
        return self._fps_tracker.fps()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error

    def _start_process(self) -> None:
        """Create and start a new capture subprocess."""
        assert self._process_factory is not None  # noqa: S101
        self._process = self._process_factory()
        self._process.start()
        _log.info(
            "capture_subprocess_started",
            pid=getattr(self._process, "pid", None),
            restart_count=self._restart_count,
        )

    def _supervisor_loop(self) -> None:
        """
        Background thread that monitors the subprocess and restarts it
        on unexpected exit.
        """
        while self._running:
            time.sleep(self._supervisor_poll_s)

            if self._intentional_stop:
                break

            if self._process is None or self._process.is_alive():
                continue

            # Subprocess died unexpectedly
            self._restart_count += 1
            _log.warning(
                "capture_subprocess_crashed",
                restart_count=self._restart_count,
                max_restarts=self._MAX_RESTARTS,
                exitcode=getattr(self._process, "exitcode", None),
            )

            if self._restart_count > self._MAX_RESTARTS:
                self._error = CaptureError(
                    f"Capture worker crashed {self._restart_count} times "
                    f"(max={self._MAX_RESTARTS}); giving up.",
                    context={
                        "restart_count": self._restart_count,
                        "exitcode": getattr(self._process, "exitcode", None),
                    },
                )
                self._running = False
                _log.error(
                    "capture_supervisor_giving_up",
                    restart_count=self._restart_count,
                )
                break

            time.sleep(self._restart_delay_s)
            self._start_process()

    def _frame_reader_loop(self) -> None:
        """
        Background thread that drains the notification queue and
        populates the local ring buffer from shared memory.
        """
        while self._running:
            try:
                seq = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame = self._read_latest_into_buffer(seq)
            if frame is not None:
                self._fps_tracker.record(frame.captured_at_monotonic)

    def _read_latest_into_buffer(self, expected_seq: int) -> Frame | None:
        """
        Read the frame from shared memory and push it to the ring buffer.

        Skips the push if the sequence number in shm doesn't match
        *expected_seq* (a newer write has already superseded it).
        """
        if self._shm is None or self._shm.buf is None:
            return None
        frame = read_shm_frame(self._shm.buf)
        if frame is None:
            return None
        if frame.sequence_number != expected_seq:
            return None  # superseded by a newer write
        self._ring_buffer.push(frame)
        return frame
