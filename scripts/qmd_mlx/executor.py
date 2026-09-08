"""
executor.py — Single GPU Execution Owner with Priority Scheduling, Fair Interleaving, and Monotonic Deadlines
"""

from __future__ import annotations

import heapq
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .protocol import (
    DeadlineExceededError,
    OverloadedError,
    RequestCancelledError,
    WorkerUnavailableError,
)


@dataclass(order=True)
class ExecutionJob:
    priority: int  # 0 = interactive (rerank, generate, query embed), 1 = bulk embed
    enqueue_time: float = field(compare=False)
    seq: int = field(compare=True)  # FIFO tie-breaker for same priority
    deadline: float = field(compare=False)
    fn: Callable[..., Any] = field(compare=False)
    future: Future = field(compare=False)
    cancel_event: threading.Event = field(compare=False)
    description: str = field(compare=False, default="")


class GPUExecutor:
    """
    Single GPU execution owner thread.
    Serializes all Metal/MLX model evaluation, loading, and unloading operations.
    Enforces strict monotonic deadlines, priority scheduling, and fair interleaving.
    """

    def __init__(self, max_queue_size: int = 64, max_consecutive_interactive: int = 4):
        self.max_queue_size = max_queue_size
        self.max_consecutive_interactive = max_consecutive_interactive
        self._queue: list[ExecutionJob] = []
        self._cv = threading.Condition()
        self._seq = 0
        self._running = True
        self._current_job: Optional[ExecutionJob] = None
        self._idle_callbacks: list[Callable[[], None]] = []
        self._consecutive_interactive = 0

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="MLX-GPU-Executor")
        self._worker_thread.start()

    def register_idle_callback(self, cb: Callable[[], None]):
        """Registers a callback to be called on the owner thread when executor is idle."""
        with self._cv:
            if cb not in self._idle_callbacks and len(self._idle_callbacks) < 32:
                self._idle_callbacks.append(cb)

    def is_alive(self) -> bool:
        with self._cv:
            return self._running and self._worker_thread.is_alive()

    def is_accepting(self) -> bool:
        """Returns True if the executor is active and accepting new submissions."""
        with self._cv:
            return self._running

    def is_worker_alive(self) -> bool:
        """Returns True if the underlying OS worker thread is currently running."""
        return self._worker_thread.is_alive()

    def is_overloaded(self) -> bool:
        """Returns True if the internal priority queue is at maximum capacity."""
        with self._cv:
            return len(self._queue) >= self.max_queue_size

    def get_queue_depth(self) -> int:
        """Returns current pending queue length under lock."""
        with self._cv:
            return len(self._queue)

    def join_worker(self, timeout: float = 5.0):
        """Waits for the background worker thread to join."""
        if self._worker_thread.is_alive() and threading.current_thread() != self._worker_thread:
            self._worker_thread.join(timeout=timeout)

    def is_owner_thread(self) -> bool:
        """Returns True if the current thread is the dedicated GPU worker thread."""
        return threading.current_thread() == self._worker_thread

    def submit(
        self,
        fn: Callable[..., Any],
        priority: int = 0,
        timeout_s: float = 300.0,
        cancel_event: Optional[threading.Event] = None,
        description: str = "",
    ) -> Any:
        """
        Submits a callable to run on the GPU execution owner thread.
        Blocks until completion or deadline/cancellation.
        """
        now = time.monotonic()
        deadline = now + timeout_s
        event = cancel_event or threading.Event()
        fut: Future = Future()

        with self._cv:
            if not self._running:
                raise WorkerUnavailableError("GPU Executor worker thread is not running")

            if len(self._queue) >= self.max_queue_size:
                raise OverloadedError(
                    f"MLX GPU Executor queue is full ({len(self._queue)}/{self.max_queue_size} jobs)"
                )

            self._seq += 1
            job = ExecutionJob(
                priority=priority,
                enqueue_time=now,
                seq=self._seq,
                deadline=deadline,
                fn=fn,
                future=fut,
                cancel_event=event,
                description=description,
            )
            heapq.heappush(self._queue, job)
            self._cv.notify()

        try:
            # Wait for future with remaining timeout
            rem = max(0.01, deadline - time.monotonic())
            return fut.result(timeout=rem)
        except TimeoutError:
            event.set()
            raise DeadlineExceededError(
                f"Request '{description}' timed out after {timeout_s:.1f}s (deadline exceeded)"
            )
        except BaseException:
            event.set()
            raise

    def submit_async(
        self,
        fn: Callable[..., Any],
        priority: int = 0,
        timeout_s: float = 300.0,
        cancel_event: Optional[threading.Event] = None,
        description: str = "",
    ) -> tuple[Future, threading.Event, float]:
        """Submits a job and returns (future, cancel_event, deadline)."""
        now = time.monotonic()
        deadline = now + timeout_s
        event = cancel_event or threading.Event()
        fut: Future = Future()

        with self._cv:
            if not self._running:
                raise WorkerUnavailableError("GPU Executor worker thread is not running")

            if len(self._queue) >= self.max_queue_size:
                raise OverloadedError(
                    f"MLX GPU Executor queue is full ({len(self._queue)}/{self.max_queue_size} jobs)"
                )

            self._seq += 1
            job = ExecutionJob(
                priority=priority,
                enqueue_time=now,
                seq=self._seq,
                deadline=deadline,
                fn=fn,
                future=fut,
                cancel_event=event,
                description=description,
            )
            heapq.heappush(self._queue, job)
            self._cv.notify()

        return fut, event, deadline

    def _worker_loop(self):
        """Main execution loop running exclusively on the dedicated GPU thread."""
        while True:
            try:
                job: Optional[ExecutionJob] = None
                callbacks_to_run: list[Callable[[], None]] = []

                with self._cv:
                    while self._running and not self._queue:
                        self._cv.wait(timeout=1.0)
                        if not self._queue and self._running:
                            callbacks_to_run = list(self._idle_callbacks)
                            break

                    if not self._running and not self._queue:
                        break

                    if self._queue:
                        # Fair interleaving: avoid starvation of bulk jobs under sustained interactive load
                        if self._consecutive_interactive >= self.max_consecutive_interactive:
                            bulk_idx = -1
                            for idx, j in enumerate(self._queue):
                                if j.priority > 0:
                                    bulk_idx = idx
                                    break
                            if bulk_idx != -1:
                                job = self._queue.pop(bulk_idx)
                                heapq.heapify(self._queue)
                                self._consecutive_interactive = 0
                            else:
                                job = heapq.heappop(self._queue)
                                if job.priority == 0:
                                    self._consecutive_interactive += 1
                                else:
                                    self._consecutive_interactive = 0
                        else:
                            job = heapq.heappop(self._queue)
                            if job.priority == 0:
                                self._consecutive_interactive += 1
                            else:
                                self._consecutive_interactive = 0

                        self._current_job = job

                # Execute idle callbacks strictly OUTSIDE the condition variable lock
                if callbacks_to_run:
                    for cb in callbacks_to_run:
                        try:
                            cb()
                        except Exception as e:
                            print(f"[mlx-executor] Idle callback error: {e}", file=sys.stderr)

                if job is None:
                    continue

                # Check if job was cancelled or expired while waiting in queue
                if job.cancel_event.is_set():
                    if not job.future.done():
                        job.future.set_exception(RequestCancelledError(f"Job '{job.description}' was cancelled"))
                    self._current_job = None
                    continue

                if time.monotonic() > job.deadline:
                    if not job.future.done():
                        job.future.set_exception(DeadlineExceededError(f"Job '{job.description}' expired in queue"))
                    self._current_job = None
                    continue

                # Execute job on the GPU owner thread
                try:
                    result = job.fn()
                    if not job.future.done():
                        job.future.set_result(result)
                except Exception as exc:
                    if not job.future.done():
                        job.future.set_exception(exc)
                finally:
                    self._current_job = None
            except Exception as e:
                # Top-level guard prevents unhandled bookkeeping errors from silently terminating the worker
                print(f"[mlx-executor] Worker loop unhandled error: {e}", file=sys.stderr)
                self._current_job = None
                time.sleep(0.01)

    def shutdown(self, timeout: float = 5.0):
        """Cancels all pending jobs and terminates the worker thread."""
        with self._cv:
            self._running = False
            if self._current_job is not None and hasattr(self._current_job, "cancel_event"):
                self._current_job.cancel_event.set()
            # Settle all pending futures
            while self._queue:
                job = heapq.heappop(self._queue)
                if not job.future.done():
                    job.future.set_exception(WorkerUnavailableError("Executor is shutting down"))
            self._cv.notify_all()

        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
