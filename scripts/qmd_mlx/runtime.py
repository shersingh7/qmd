"""
runtime.py — Model-aware MLX embedding runtime behind single GPU execution owner
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Optional
import numpy as np

from .adapters import (
    BaseEmbeddingAdapter,
    TokenizedBatch,
    resolve_embedding_adapter,
)
from .batching import BatchPlanner
from .executor import GPUExecutor
from .model_manager import ModelResidencyManager
from .protocol import (
    DeadlineExceededError,
    InvalidInputError,
    MLXRuntimeError,
    MLXServerError,
    ModelUnavailableError,
    OutOfMemoryError,
    OverloadedError,
    ProtocolError,
    RequestCancelledError,
    UnsupportedModelError,
    WorkerUnavailableError,
    WorkClass,
)

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


from dataclasses import dataclass


@dataclass
class AdmissionLease:
    text_count: int
    total_bytes: int
    acquired_time: float


class _DummyWorkQueue:
    def __init__(self, executor: GPUExecutor):
        self._executor = executor

    def join(self):
        time.sleep(0.05)


class MLXEmbeddingRuntime:
    """
    Manages MLX embedding model lifecycle and execution behind a dedicated
    GPUExecutor owner thread and ModelResidencyManager.
    """

    def __init__(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        dtype_str: str = "float32",
        max_length: int = 2048,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        max_batch_tokens: int = 0,
        executor: Optional[GPUExecutor] = None,
        model_manager: Optional[ModelResidencyManager] = None,
        lazy_load: bool = False,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.dtype_str = dtype_str
        self.max_length = max_length
        self.revision = revision
        self.trust_remote_code = trust_remote_code

        # Shared or private executor & residency manager
        self.executor = executor or GPUExecutor()
        self._owns_executor = executor is None

        self.model_manager = model_manager or ModelResidencyManager(self.executor)

        # Instantiate explicit model adapter
        self.adapter: BaseEmbeddingAdapter = resolve_embedding_adapter(
            model_name=model_name,
            quantization=quantization,
            dtype_str=dtype_str,
            max_length=max_length,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        if not getattr(self.adapter, "synthetic", False) and not _MLX_AVAILABLE:
            raise MLXRuntimeError("MLX is not installed. Please install mlx and mlx-lm.")

        self.model_manager.register_adapter("embed", self.adapter)

        # Batch planner
        self.batch_planner = BatchPlanner(
            max_batch_tokens=max_batch_tokens if max_batch_tokens > 0 else None
        )
        self.batch_planner.tune_for_model(getattr(self.adapter, "model_params_b", 0.6))

        # Bounded admission leasing controls
        self.max_concurrent_admissions: int = 32
        self.max_in_flight_admission_bytes: int = 64 * 1024 * 1024  # 64 MB
        self._admission_lock = threading.Condition()
        self._current_admitted_requests: int = 0
        self._current_admitted_bytes: int = 0
        self._shutting_down: bool = False

        self.start_time: float = time.time()
        self.total_requests: int = 0
        self.total_latency_ms: float = 0.0
        self.compiled_shapes: set[tuple[int, ...]] = set()

        if not lazy_load:
            self.model_manager.ensure_loaded("embed")
            self.batch_planner.tune_for_model(getattr(self.adapter, "model_params_b", 0.6))

    @property
    def in_flight_requests(self) -> int:
        with self._admission_lock:
            return self._current_admitted_requests

    @property
    def in_flight_bytes(self) -> int:
        with self._admission_lock:
            return self._current_admitted_bytes

    def acquire_admission_lease(
        self,
        text_count: int = 0,
        total_bytes: int = 0,
        deadline: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
        num_texts: Optional[int] = None,
        num_bytes: Optional[int] = None,
    ) -> AdmissionLease:
        """
        Acquires an in-flight admission lease before performing CPU tokenization or large allocations.
        Guarantees bounded request concurrency, byte caps, and executor queue capacity.
        """
        if num_texts is not None:
            text_count = num_texts
        if num_bytes is not None:
            total_bytes = num_bytes

        if text_count < 0 or total_bytes < 0:
            raise InvalidInputError("Invalid admission lease request: count and bytes must be non-negative")
        if text_count > 512:
            raise InvalidInputError(f"Request item count ({text_count}) exceeds maximum allowed limit (512)")
        if total_bytes > self.max_in_flight_admission_bytes:
            raise InvalidInputError(
                f"Request size ({total_bytes} bytes) exceeds maximum in-flight capacity ({self.max_in_flight_admission_bytes} bytes)"
            )

        if self._shutting_down or not self.executor.is_accepting() or not self.executor.is_worker_alive():
            raise WorkerUnavailableError("GPU Executor worker thread is not running")

        now = time.monotonic()
        dl = deadline if deadline is not None else (now + 300.0)

        with self._admission_lock:
            if self.executor.is_overloaded():
                raise OverloadedError(
                    f"MLX GPU Executor queue is full ({self.executor.get_queue_depth()}/{self.executor.max_queue_size} jobs)"
                )

            while (
                self._current_admitted_requests >= self.max_concurrent_admissions
                or (self._current_admitted_bytes + total_bytes > self.max_in_flight_admission_bytes)
            ):
                if self._shutting_down or not self.executor.is_accepting() or not self.executor.is_worker_alive():
                    raise WorkerUnavailableError("GPU Executor is shutting down")
                if cancel_event and cancel_event.is_set():
                    raise RequestCancelledError("Request cancelled while waiting for admission")
                rem = dl - time.monotonic()
                if rem <= 0:
                    raise OverloadedError("Server admission capacity saturated; request timed out waiting for admission slot")
                self._admission_lock.wait(timeout=min(0.2, max(0.01, rem)))

            if self._shutting_down or not self.executor.is_accepting() or not self.executor.is_worker_alive():
                raise WorkerUnavailableError("GPU Executor is shutting down")
            if cancel_event and cancel_event.is_set():
                raise RequestCancelledError("Request cancelled before admission")
            if time.monotonic() > dl:
                raise DeadlineExceededError("Request deadline exceeded before admission")

            self._current_admitted_requests += 1
            self._current_admitted_bytes += total_bytes
            return AdmissionLease(text_count=text_count, total_bytes=total_bytes, acquired_time=time.monotonic())

    def release_admission_lease(self, lease: AdmissionLease):
        """Releases an in-flight admission lease."""
        with self._admission_lock:
            self._current_admitted_requests = max(0, self._current_admitted_requests - 1)
            self._current_admitted_bytes = max(0, self._current_admitted_bytes - lease.total_bytes)
            self._admission_lock.notify_all()

    @property
    def native_dims(self) -> int:
        return self.adapter.native_dims

    @property
    def pooling_strategy(self) -> str:
        return self.adapter.pooling_strategy

    @property
    def idle_unload_s(self) -> float:
        return self.model_manager.idle_unload_s

    @idle_unload_s.setter
    def idle_unload_s(self, value: float):
        self.model_manager.idle_unload_s = float(value)

    @property
    def _weights_loaded(self) -> bool:
        return self.adapter.is_loaded()

    @property
    def _work_queue(self) -> _DummyWorkQueue:
        return _DummyWorkQueue(self.executor)

    def is_ready(self) -> bool:
        return self.adapter.is_loaded() and self.executor.is_alive()

    def tokenize(
        self,
        texts: list[str],
        deadline: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> list[list[int]]:
        """
        Tokenizes texts into token IDs on CPU without requiring model weights loaded on GPU.
        Protected by bounded admission controls.
        """
        if not texts:
            return []

        if len(texts) > 512:
            raise InvalidInputError(f"Request batch size ({len(texts)}) exceeds maximum allowed texts (512)")

        total_bytes = 0
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                raise InvalidInputError(f"Text at index {i} is empty or whitespace")
            tb = len(t.encode("utf-8"))
            if tb > 256 * 1024:
                raise InvalidInputError(f"Text at index {i} exceeds max length of 256KB ({tb} bytes)")
            total_bytes += tb

        if total_bytes > 10 * 1024 * 1024:
            raise InvalidInputError(f"Total request text size ({total_bytes} bytes) exceeds 10MB limit")

        now_mono = time.monotonic()
        req_deadline = deadline if deadline is not None else (now_mono + 120.0)
        event = cancel_event or threading.Event()

        if event.is_set():
            raise RequestCancelledError("Tokenize request cancelled before start")
        if time.monotonic() > req_deadline:
            raise DeadlineExceededError("Tokenize deadline exceeded before start")

        lease = self.acquire_admission_lease(len(texts), total_bytes, deadline=req_deadline, cancel_event=event)
        try:
            if getattr(self.adapter, "raw_hf_tokenizer", None) is None:
                self.model_manager.acquire_lease("embed")
                try:
                    if not self.adapter.is_loaded():
                        self.model_manager.ensure_loaded("embed", deadline=req_deadline, cancel_event=event)
                        self.batch_planner.tune_for_model(getattr(self.adapter, "model_params_b", 0.6))
                finally:
                    self.model_manager.release_lease("embed")

            batch = self.adapter.tokenize_texts(texts)

            if event.is_set():
                raise RequestCancelledError("Tokenize request cancelled after tokenization")
            if time.monotonic() > req_deadline:
                raise DeadlineExceededError("Tokenize deadline exceeded after tokenization")

            if isinstance(batch, TokenizedBatch):
                return batch.token_ids
            elif hasattr(batch, "token_ids"):
                return batch.token_ids
            elif isinstance(batch, list):
                return batch
            return list(batch)
        finally:
            self.release_admission_lease(lease)

    def warmup(self):
        """Runs warmup inference passes on GPU."""
        print("[mlx-runtime] Running GPU warmup...")
        t0 = time.time()
        for count in (1, 4):
            sample_texts = ["Warmup embedding Metal compiler verification."] * count
            self.submit_embed(sample_texts, timeout=30.0, is_query=True)
        elapsed = time.time() - t0
        print(f"[mlx-runtime] GPU warmup complete in {elapsed:.2f}s")

    def embed_direct(
        self,
        texts: list[str],
        requested_dims: Optional[int] = None,
        is_query: bool = False,
    ) -> np.ndarray:
        return self.submit_embed(texts, requested_dims=requested_dims, is_query=is_query)

    def submit_embed(
        self,
        texts: list[str],
        requested_dims: Optional[int] = None,
        is_query: bool = False,
        timeout: float = 300.0,
        cancel_event: Optional[threading.Event] = None,
        deadline: Optional[float] = None,
    ) -> np.ndarray:
        """
        Submits an embedding request through the GPU execution owner.
        Enforces a single monotonic deadline budget across admission lease, tokenize, queue, and micro-batches.
        Supports micro-batch interleaving so interactive requests yield between bulk steps.
        """
        if not texts:
            return np.empty((0, self.adapter.native_dims or 0), dtype=np.float32)

        # 1. Early bounded input validation
        if len(texts) > 512:
            raise InvalidInputError(f"Request exceeds maximum allowed texts ({len(texts)} > 512)")

        total_bytes = 0
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise InvalidInputError(f"Item at index {i} is not a string")
            if not t.strip():
                raise InvalidInputError(f"Item at index {i} is empty or whitespace")
            if len(t) > 256 * 1024:
                raise InvalidInputError(f"Item at index {i} exceeds max length 256KB")
            total_bytes += len(t.encode("utf-8", errors="ignore"))

        if total_bytes > 10 * 1024 * 1024:
            raise InvalidInputError(f"Total request text size ({total_bytes} bytes) exceeds 10MB limit")

        now_mono = time.monotonic()
        req_deadline = deadline if deadline is not None else (now_mono + timeout)
        event = cancel_event or threading.Event()

        if event.is_set():
            raise RequestCancelledError("Request cancelled before start")
        if time.monotonic() > req_deadline:
            raise DeadlineExceededError("Request deadline exceeded before start")

        t0 = time.time()

        # 2. Acquire bounded admission lease before heavy CPU tokenization
        lease = self.acquire_admission_lease(len(texts), total_bytes, deadline=req_deadline, cancel_event=event)
        try:
            # 3. Tokenize (CPU side). Ensure tokenizer is loaded under lease if needed.
            if getattr(self.adapter, "raw_hf_tokenizer", None) is None:
                self.model_manager.acquire_lease("embed")
                try:
                    if not self.adapter.is_loaded():
                        self.model_manager.ensure_loaded("embed", deadline=req_deadline, cancel_event=event)
                        self.batch_planner.tune_for_model(getattr(self.adapter, "model_params_b", 0.6))
                finally:
                    self.model_manager.release_lease("embed")

            tokenized_batch = self.adapter.tokenize_texts(texts)

            # Retune batch planner with measured model params BEFORE planning micro-batches!
            if self.adapter.is_loaded() or getattr(self.adapter, "model_params_b", None):
                self.batch_planner.tune_for_model(getattr(self.adapter, "model_params_b", 0.6))

            if event.is_set():
                raise RequestCancelledError("Request cancelled after tokenization")
            if time.monotonic() > req_deadline:
                raise DeadlineExceededError("Request deadline exceeded after tokenization")

            # Interactive queries get priority 0, bulk document embeddings get priority 1
            work_class = WorkClass.INTERACTIVE if (is_query or len(texts) <= 2) else WorkClass.BULK
            priority = 0 if work_class == WorkClass.INTERACTIVE else 1

            # 4. Plan micro-batches for fair interleaving at micro-batch boundaries
            sub_batches, micro_batches_indices = self.batch_planner.plan_micro_batches(
                tokenized_batch,
                is_query=is_query,
                work_class=work_class,
            )

            results: list[np.ndarray] = []
            for batch_idx, sub_batch in enumerate(sub_batches):
                if event.is_set():
                    raise RequestCancelledError("Embedding request cancelled")
                if time.monotonic() > req_deadline:
                    raise DeadlineExceededError(
                        f"Deadline exceeded after {batch_idx}/{len(sub_batches)} micro-batches"
                    )

                rem_timeout = max(0.01, req_deadline - time.monotonic())

                def _forward_sub_batch(sb=sub_batch) -> np.ndarray:
                    if event.is_set():
                        raise RequestCancelledError("Embedding request cancelled")
                    if time.monotonic() > req_deadline:
                        raise DeadlineExceededError("Micro-batch deadline exceeded")

                    if not self.adapter.is_loaded():
                        self.model_manager.ensure_loaded("embed", deadline=req_deadline, cancel_event=event)
                    self.model_manager.touch("embed")

                    def _run_forward(b: TokenizedBatch) -> np.ndarray:
                        if len(b) > 0:
                            padded_shape = (len(b), max(b.lengths) if b.lengths else 1)
                            if len(self.compiled_shapes) >= 256:
                                self.compiled_shapes.clear()
                            self.compiled_shapes.add(padded_shape)
                        return self.adapter.forward_batch(b, requested_dims=requested_dims)

                    return self.batch_planner.execute_sub_batch_with_retry(sb, _run_forward)

                batch_res = self.executor.submit(
                    _forward_sub_batch,
                    priority=priority,
                    timeout_s=rem_timeout,
                    cancel_event=event,
                    description=f"Embed batch {batch_idx+1}/{len(sub_batches)} ({'query' if is_query else 'doc'})",
                )
                results.append(batch_res)

            # 5. Restore original input ordering
            res = self.batch_planner.restore_ordering(results, micro_batches_indices, len(tokenized_batch))

            latency_ms = (time.time() - t0) * 1000
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            self.model_manager.touch("embed")
            self.model_manager.update_peak_memory()
            return res
        except MLXServerError:
            raise
        except Exception as exc:
            raise MLXRuntimeError(f"Embedding execution failed: {exc}")
        finally:
            self.release_admission_lease(lease)

    def get_descriptor(self, requested_dims: Optional[int] = None) -> dict[str, Any]:
        return self.adapter.get_descriptor(requested_dims=requested_dims)

    def get_memory_info(self) -> dict[str, Any]:
        return self.model_manager.get_memory_info()

    def get_stats_info(self) -> dict[str, Any]:
        uptime = time.time() - self.start_time
        avg_ms = round(self.total_latency_ms / self.total_requests, 2) if self.total_requests > 0 else 0.0
        return {
            "total_requests": self.total_requests,
            "avg_ms": avg_ms,
            "compiled_shapes": len(self.compiled_shapes),
            "uptime_sec": round(uptime, 1),
            "weights_loaded": self.adapter.is_loaded(),
            "idle_unload_s": self.model_manager.idle_unload_s,
        }

    def stop_admission(self):
        """Stops accepting new admission leases and wakes waiting threads."""
        with self._admission_lock:
            self._shutting_down = True
            self._admission_lock.notify_all()

    def shutdown(self, timeout: float = 5.0):
        """Cleanly terminates the runtime and executor (if owned) under owner-thread serialization."""
        self.stop_admission()

        if self._owns_executor:
            self.executor.shutdown(timeout=timeout)
        if self.model_manager:
            try:
                self.model_manager.unload("embed")
            except Exception:
                if not self.executor.is_worker_alive():
                    self.adapter.unload()
        else:
            if not self.executor.is_worker_alive():
                self.adapter.unload()
