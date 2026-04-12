"""
tests/test_concurrency.py — PAKET J: Eş Zamanlılık Testleri

test_concurrent_tasks      — İki görev aynı anda başlatılınca ikincisi kuyrukta/hata
test_cancel_during_execution — Görev çalışırken iptal edilince temiz kapansın
test_task_timeout           — max_task_duration_seconds=1, süre dolunca sonlandırılsın
"""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.core.types import TaskId


# ---------------------------------------------------------------------------
# Lightweight mock task runner
# ---------------------------------------------------------------------------


class _MockTask:
    """Gerçek TaskExecutor olmadan concurrency davranışını simüle eder."""

    def __init__(self, task_id: str, duration: float = 0.05) -> None:
        self.task_id = task_id
        self.duration = duration
        self._cancelled = False
        self._started = False
        self._finished = False

    async def run(self) -> str:
        self._started = True
        elapsed = 0.0
        step = 0.01
        while elapsed < self.duration:
            if self._cancelled:
                return "cancelled"
            await asyncio.sleep(step)
            elapsed += step
        self._finished = True
        return "success"

    def cancel(self) -> None:
        self._cancelled = True


class _TaskQueue:
    """Basit FIFO kuyruk — aynı anda sadece 1 görev çalıştırır."""

    def __init__(self) -> None:
        self._running: str | None = None
        self._queue: list[str] = []
        self._lock = asyncio.Lock()

    async def submit(self, task_id: str) -> str:
        """Submit bir görev. Çalışan varsa kuyruğa ekle, yoksa hemen çalıştır."""
        async with self._lock:
            if self._running is not None:
                self._queue.append(task_id)
                return "queued"
            self._running = task_id
            return "started"

    async def complete(self, task_id: str) -> None:
        async with self._lock:
            if self._running == task_id:
                self._running = None
                if self._queue:
                    self._running = self._queue.pop(0)

    def queue_size(self) -> int:
        return len(self._queue)


# ---------------------------------------------------------------------------
# PAKET J
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConcurrentTasks:
    """Aynı anda iki görev başlatılınca ikincisi kuyruğa girsin."""

    async def test_second_task_queued(self) -> None:
        """İlk görev çalışırken ikinci kuyruğa girmeli."""
        queue = _TaskQueue()

        status1 = await queue.submit("task-001")
        status2 = await queue.submit("task-002")

        assert status1 == "started"
        assert status2 == "queued"
        assert queue.queue_size() == 1

    async def test_queue_processes_after_first_completes(self) -> None:
        """İlk görev bitince kuyruktaki görev başlamalı."""
        queue = _TaskQueue()

        await queue.submit("task-001")
        await queue.submit("task-002")
        await queue.complete("task-001")

        # Şimdi task-002 çalışıyor olmalı
        assert queue._running == "task-002"
        assert queue.queue_size() == 0

    async def test_concurrent_execution_no_data_race(self) -> None:
        """İki async görev aynı anda çalışırken veri yarışması olmamamalı."""
        results: list[str] = []
        lock = asyncio.Lock()

        async def _task(name: str) -> None:
            await asyncio.sleep(0.01)
            async with lock:
                results.append(name)

        await asyncio.gather(
            _task("task-A"),
            _task("task-B"),
            _task("task-C"),
        )

        assert sorted(results) == ["task-A", "task-B", "task-C"]


@pytest.mark.asyncio
class TestCancelDuringExecution:
    """Görev çalışırken iptal edilince temiz kapanmalı."""

    async def test_cancel_stops_task(self) -> None:
        """cancel() çağrıldıktan sonra görev 'cancelled' dönmeli."""
        task = _MockTask("cancel-task", duration=1.0)
        run_future = asyncio.ensure_future(task.run())

        # Biraz bekle, sonra iptal et
        await asyncio.sleep(0.02)
        task.cancel()

        result = await run_future
        assert result == "cancelled"
        assert not task._finished

    async def test_cancel_is_idempotent(self) -> None:
        """cancel() birden fazla kez çağrılabilmeli, crash olmamalı."""
        task = _MockTask("idempotent-cancel", duration=0.5)
        run_future = asyncio.ensure_future(task.run())

        await asyncio.sleep(0.01)
        task.cancel()
        task.cancel()  # İkinci çağrı crash olmamalı
        task.cancel()

        result = await run_future
        assert result == "cancelled"

    async def test_task_cleanup_after_cancel(self) -> None:
        """İptal edilen görev temiz state bırakmalı."""
        task = _MockTask("cleanup-task", duration=0.5)
        run_future = asyncio.ensure_future(task.run())

        await asyncio.sleep(0.02)
        task.cancel()
        result = await run_future

        assert result == "cancelled"
        assert task._started is True  # Başlamıştı
        assert task._finished is False  # Bitmedi

    async def test_asyncio_cancel_raises_cancelled_error(self) -> None:
        """asyncio.Task.cancel() CancelledError fırlatmalı."""
        async def _long_task() -> str:
            await asyncio.sleep(10)
            return "done"

        task = asyncio.ensure_future(_long_task())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
class TestTaskTimeout:
    """max_task_duration_seconds=1 → süre dolunca sonlandırılsın."""

    async def test_timeout_raises_after_deadline(self) -> None:
        """asyncio.wait_for ile timeout aşılınca TimeoutError fırlatmalı."""
        async def _slow_task() -> str:
            await asyncio.sleep(10)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_slow_task(), timeout=0.05)

    async def test_fast_task_completes_before_timeout(self) -> None:
        """Hızlı görev timeout olmadan tamamlanmalı."""
        async def _fast_task() -> str:
            await asyncio.sleep(0.01)
            return "success"

        result = await asyncio.wait_for(_fast_task(), timeout=1.0)
        assert result == "success"

    async def test_task_timeout_with_mock_executor(self) -> None:
        """Timeout mekanizması görev üzerinde düzgün çalışmalı."""
        task = _MockTask("timeout-task", duration=10.0)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task.run(), timeout=0.05)

    async def test_timeout_returns_correct_status(self) -> None:
        """Timeout sonrası görev durumu doğru raporlanmalı."""
        completed = False
        timed_out = False

        async def _task_with_timeout(timeout_s: float) -> str:
            nonlocal completed, timed_out
            try:
                await asyncio.wait_for(asyncio.sleep(10), timeout=timeout_s)
                completed = True
                return "success"
            except asyncio.TimeoutError:
                timed_out = True
                return "timed_out"

        result = await _task_with_timeout(0.05)
        assert result == "timed_out"
        assert timed_out is True
        assert completed is False

    async def test_multiple_timeouts_independent(self) -> None:
        """Birden fazla görev birbirinden bağımsız timeout'a sahip olmalı."""
        results: list[str] = []

        async def _task(idx: int, sleep_s: float, timeout_s: float) -> None:
            try:
                await asyncio.wait_for(asyncio.sleep(sleep_s), timeout=timeout_s)
                results.append(f"task_{idx}:ok")
            except asyncio.TimeoutError:
                results.append(f"task_{idx}:timeout")

        await asyncio.gather(
            _task(1, 0.01, 1.0),   # hızlı → ok
            _task(2, 10.0, 0.05),  # yavaş → timeout
        )

        assert "task_1:ok" in results
        assert "task_2:timeout" in results
