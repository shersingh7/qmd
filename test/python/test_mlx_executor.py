"""
test_mlx_executor.py — Tests for GPUExecutor single execution owner, priority queue,
monotonic deadlines, fair interleaving, starvation prevention, safe callback locking, and shutdown.
"""

import threading
import time
import pytest
from scripts.qmd_mlx.executor import GPUExecutor
from scripts.qmd_mlx.protocol import (
    DeadlineExceededError,
    OverloadedError,
    RequestCancelledError,
    WorkerUnavailableError,
)


def test_executor_single_thread_serialization():
    """Verify that all tasks execute strictly sequentially on a single worker thread."""
    executor = GPUExecutor(max_queue_size=10)
    execution_order = []
    active_threads = set()
    lock = threading.Lock()

    def worker_job(item_id: int):
        with lock:
            active_threads.add(threading.current_thread().name)
        time.sleep(0.05)
        execution_order.append(item_id)
        return item_id * 10

    # Submit 3 jobs concurrently from 3 client threads
    def submitter(item_id):
        return executor.submit(lambda: worker_job(item_id), priority=1, timeout_s=5.0)

    threads = []
    results = [None, None, None]

    def run_sub(idx, item_id):
        results[idx] = submitter(item_id)

    for i in range(3):
        t = threading.Thread(target=run_sub, args=(i, i + 1))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    executor.shutdown()

    assert results == [10, 20, 30]
    assert len(active_threads) == 1
    assert "MLX-GPU-Executor" in list(active_threads)[0]


def test_executor_priority_scheduling():
    """Verify that priority 0 (interactive) jumps ahead of priority 1 (bulk)."""
    executor = GPUExecutor(max_queue_size=20)
    order = []

    # Pause worker temporarily to load the queue
    blocker_event = threading.Event()
    executor.submit_async(lambda: blocker_event.wait(timeout=2.0), priority=0)

    # Submit bulk jobs (p1)
    f_bulk1, _, _ = executor.submit_async(lambda: order.append("bulk1"), priority=1)
    f_bulk2, _, _ = executor.submit_async(lambda: order.append("bulk2"), priority=1)

    # Submit interactive job (p0)
    f_interactive, _, _ = executor.submit_async(lambda: order.append("interactive"), priority=0)

    # Unblock worker
    blocker_event.set()

    f_bulk1.result(timeout=2.0)
    f_bulk2.result(timeout=2.0)
    f_interactive.result(timeout=2.0)

    executor.shutdown()

    # Interactive must have executed before the bulk jobs that were waiting in queue
    assert order == ["interactive", "bulk1", "bulk2"]


def test_executor_deadline_exceeded():
    """Verify that expired jobs fail with DeadlineExceededError and do not block subsequent jobs."""
    executor = GPUExecutor(max_queue_size=10)

    # Job with negative/instant deadline
    with pytest.raises(DeadlineExceededError):
        executor.submit(lambda: time.sleep(0.1), timeout_s=0.01)

    # Subsequent job must succeed normally
    res = executor.submit(lambda: 42, timeout_s=2.0)
    assert res == 42

    executor.shutdown()


def test_executor_request_cancellation():
    """Verify that cancelled jobs are aborted without crashing the executor."""
    executor = GPUExecutor(max_queue_size=10)
    cancel_event = threading.Event()
    cancel_event.set()  # Cancelled before start

    with pytest.raises(RequestCancelledError):
        executor.submit(lambda: 100, cancel_event=cancel_event, timeout_s=2.0)

    # Executor remains healthy
    assert executor.submit(lambda: 200, timeout_s=2.0) == 200

    executor.shutdown()


def test_executor_queue_full_raises_overloaded():
    """Verify that exceeding max_queue_size raises OverloadedError (HTTP 429)."""
    executor = GPUExecutor(max_queue_size=2)
    started = threading.Event()
    blocker = threading.Event()

    def running_job():
        started.set()
        blocker.wait(timeout=2.0)

    # Job 1 running
    executor.submit_async(running_job, priority=0)
    assert started.wait(timeout=1.0)

    # Jobs 2 and 3 fill the queue (capacity 2)
    executor.submit_async(lambda: 1, priority=1)
    executor.submit_async(lambda: 2, priority=1)

    # Job 4 must be rejected with OverloadedError
    with pytest.raises(OverloadedError):
        executor.submit(lambda: 3, priority=1)

    blocker.set()
    executor.shutdown()


def test_executor_worker_fault_recovery():
    """Verify that if a job raises an exception, the worker catches it and continues."""
    executor = GPUExecutor(max_queue_size=10)

    def failing_job():
        raise ValueError("Simulated computation crash")

    with pytest.raises(ValueError, match="Simulated computation crash"):
        executor.submit(failing_job, timeout_s=2.0)

    # Worker thread is still alive and processes next job
    assert executor.is_alive()
    assert executor.submit(lambda: "recovered", timeout_s=2.0) == "recovered"

    executor.shutdown()


def test_executor_idle_callback_outside_lock():
    """Verify that idle callbacks run outside _cv lock and do not block submit() or shutdown()."""
    executor = GPUExecutor(max_queue_size=10)
    cb_entered = threading.Event()
    cb_unblock = threading.Event()

    def slow_idle_callback():
        cb_entered.set()
        cb_unblock.wait(timeout=2.0)

    executor.register_idle_callback(slow_idle_callback)

    # Wait for idle callback to be triggered
    assert cb_entered.wait(timeout=3.0)

    # Submit a job while callback is currently executing
    # If callback held _cv lock, submit would deadlock!
    res_fut, _, _ = executor.submit_async(lambda: 99, priority=0, timeout_s=2.0)

    # Release callback
    cb_unblock.set()

    # Job must execute and complete
    assert res_fut.result(timeout=2.0) == 99
    executor.shutdown()


def test_executor_shutdown_rejects_admission():
    """Verify that after shutdown(), submit() and submit_async() are atomically rejected."""
    executor = GPUExecutor(max_queue_size=10)
    executor.shutdown(timeout=2.0)

    assert not executor.is_alive()

    with pytest.raises(WorkerUnavailableError):
        executor.submit(lambda: 123)

    with pytest.raises(WorkerUnavailableError):
        executor.submit_async(lambda: 456)


def test_executor_fair_interleaving_and_starvation_prevention():
    """
    Verify bounded interleaving: under sustained interactive (p0) load,
    bulk (p1) work is interleaved and makes progress without starvation.
    """
    executor = GPUExecutor(max_queue_size=50, max_consecutive_interactive=3)
    order = []

    blocker = threading.Event()
    executor.submit_async(lambda: blocker.wait(timeout=2.0), priority=0)

    # Queue 2 bulk jobs (p1)
    f_b1, _, _ = executor.submit_async(lambda: order.append("bulk_1"), priority=1)
    f_b2, _, _ = executor.submit_async(lambda: order.append("bulk_2"), priority=1)

    # Queue 6 interactive jobs (p0)
    f_i = []
    for i in range(6):
        f, _, _ = executor.submit_async(lambda idx=i: order.append(f"interactive_{idx}"), priority=0)
        f_i.append(f)

    # Unblock
    blocker.set()

    for f in f_i:
        f.result(timeout=3.0)
    f_b1.result(timeout=3.0)
    f_b2.result(timeout=3.0)

    executor.shutdown()

    # With max_consecutive_interactive = 3:
    # First 3 interactive jobs run -> then bulk_1 MUST run -> then next interactive jobs -> bulk_2
    assert "bulk_1" in order
    assert "bulk_2" in order

    # Verify bulk_1 ran before interactive_4
    idx_bulk_1 = order.index("bulk_1")
    assert idx_bulk_1 <= 3  # Bulk 1 was interleaved after at most 3 interactive jobs!


def test_executor_worker_loop_guard_keeps_serving():
    """Verify that unhandled exceptions in worker loop bookkeeping do not crash the worker thread."""
    executor = GPUExecutor(max_queue_size=10)

    # Register an idle callback that raises an error
    def crashing_callback():
        raise RuntimeError("Simulated unhandled callback error")

    executor.register_idle_callback(crashing_callback)
    time.sleep(0.05)

    # Submit job and ensure it succeeds
    res = executor.submit(lambda: "still alive", timeout_s=2.0)
    assert res == "still alive"
    assert executor.is_alive()
    executor.shutdown()
