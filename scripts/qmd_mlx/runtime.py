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
    MLXRuntimeError,
    MLXServerError,
    ModelUnavailableError,
    OverloadedError,
    ProtocolError,
    RequestCancelledError,
    UnsupportedModelError,
)

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


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
        if not _MLX_AVAILABLE:
            raise MLXRuntimeError("MLX is not installed. Please install mlx and mlx-lm.")

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

        self.model_manager.register_adapter("embed", self.adapter)

        # Batch planner
        self.batch_planner = BatchPlanner(
            max_batch_tokens=max_batch_tokens if max_batch_tokens > 0 else None
        )
        self.batch_planner.tune_for_model(getattr(self.adapter, "model_params_b", 0.6))

        self.start_time: float = time.time()
        self.total_requests: int = 0
        self.total_latency_ms: float = 0.0
        self.compiled_shapes: set[tuple[int, ...]] = set()

        if not lazy_load:
            self.model_manager.ensure_loaded("embed")

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

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        """Tokenizes texts into token ID sequences."""
        if not self.adapter.is_loaded():
            self.model_manager.ensure_loaded("embed")
        batch = self.adapter.tokenize_texts(texts)
        return batch.token_ids

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
        Enforces a single monotonic deadline budget across load, tokenize, queue, and micro-batches.
        Supports micro-batch interleaving so interactive requests yield between bulk steps.
        """
        if not texts:
            return np.empty((0, self.adapter.native_dims or 0), dtype=np.float32)

        # 1. Monotonic end-to-end deadline calculated ONCE at entry
        now_mono = time.monotonic()
        req_deadline = deadline if deadline is not None else (now_mono + timeout)
        event = cancel_event or threading.Event()

        if event.is_set():
            raise RequestCancelledError("Request cancelled before start")
        if time.monotonic() > req_deadline:
            raise DeadlineExceededError("Request deadline exceeded before start")

        t0 = time.time()

        # 2. Ensure model loaded using the single end-to-end deadline budget
        if not self.adapter.is_loaded():
            self.model_manager.ensure_loaded("embed", deadline=req_deadline, cancel_event=event)

        if event.is_set():
            raise RequestCancelledError("Request cancelled after model load")
        if time.monotonic() > req_deadline:
            raise DeadlineExceededError("Request deadline exceeded after model load")

        # 3. Single-pass tokenization (CPU side)
        tokenized_batch = self.adapter.tokenize_texts(texts)

        if event.is_set():
            raise RequestCancelledError("Request cancelled after tokenization")
        if time.monotonic() > req_deadline:
            raise DeadlineExceededError("Request deadline exceeded after tokenization")

        # Interactive queries get priority 0, bulk document embeddings get priority 1
        priority = 0 if is_query or len(texts) <= 2 else 1

        # 4. Plan micro-batches for fair interleaving
        sub_batches, micro_batches_indices = self.batch_planner.plan_micro_batches(tokenized_batch)

        results: list[np.ndarray] = []
        try:
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

    def shutdown(self):
        """Cleanly terminates the runtime and executor under owner-thread serialization."""
        if self.model_manager:
            try:
                self.model_manager.unload("embed")
            except Exception:
                self.adapter.unload()
        else:
            self.adapter.unload()
        if self._owns_executor:
            self.executor.shutdown()
