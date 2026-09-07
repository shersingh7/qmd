"""
runtime.py — Model-aware MLX embedding runtime behind a single GPU execution owner
"""

import os
import sys
import time
import threading
import queue
import numpy as np
from typing import Any, Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


class MLXRuntimeError(RuntimeError):
    """Raised when model loading, compilation, or execution fails."""
    pass


class MLXEmbeddingRuntime:
    """
    Manages MLX model lifecycle, Metal execution synchronization,
    and embedding generation behind a dedicated execution owner thread.
    """

    def __init__(
        self,
        model_name: str,
        quantization: str = "bf16",
        dtype_str: str = "float32",
        max_length: int = 2048,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        max_batch_tokens: int = 0,
    ):
        if not _MLX_AVAILABLE:
            raise MLXRuntimeError("MLX is not installed. Please install mlx and mlx-lm.")

        self.model_name = model_name
        self.quantization = quantization
        self.dtype_str = dtype_str
        self.max_length = max_length
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        # Length-aware micro-batching: bounds per-forward-pass padded tokens
        # so production batches (32 x ~900-token chunks) split into safe
        # micro-batches instead of one giant OOM-prone forward pass.
        from .batching import BatchPlanner
        self.batch_planner = BatchPlanner(
            max_batch_tokens=max_batch_tokens if max_batch_tokens > 0 else None
        )

        # Idle unload: same policy as the GGUF in-process path — drop model
        # weights after N seconds with no work, reload transparently on the
        # next request (pays a one-time reload cost, frees GPU memory).
        # 0 disables (always-on).
        self.idle_unload_s = float(os.getenv("MLX_IDLE_UNLOAD_S", "300"))
        self._last_active = time.time()
        self._weights_loaded = True

        self.model: Any = None
        self.tokenizer: Any = None
        self.raw_hf_tokenizer: Any = None
        self.native_dims: int = 0
        self.pooling_strategy: str = "mean"
        self.is_causal_lm: bool = False
        self.model_type: str = "generic"

        self.peak_memory_mb: float = 0.0
        self.model_memory_mb: float = 0.0
        self.start_time: float = time.time()
        self.total_requests: int = 0
        self.total_latency_ms: float = 0.0
        self.compiled_shapes: set[tuple[int, ...]] = set()

        # Initialization synchronization
        self._ready_event = threading.Event()
        self._init_error: Optional[Exception] = None

        # Single execution owner queue & thread
        self._work_queue: queue.Queue = queue.Queue(maxsize=32)
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        # Wait for worker thread to complete loading
        self._ready_event.wait(timeout=120.0)
        if self._init_error:
            raise self._init_error
        if not self._ready_event.is_set():
            raise MLXRuntimeError(f"Timed out loading model '{model_name}'")

    def _get_active_memory_mb(self) -> float:
        try:
            if hasattr(mx, "get_active_memory"):
                return mx.get_active_memory() / (1024 * 1024)
            if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def _get_peak_memory_mb(self) -> float:
        try:
            if hasattr(mx, "get_peak_memory"):
                return mx.get_peak_memory() / (1024 * 1024)
            if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
                return mx.metal.get_peak_memory() / (1024 * 1024)
        except Exception:
            pass
        return self.peak_memory_mb

    def _update_peak_memory(self):
        cur = self._get_peak_memory_mb()
        if cur > self.peak_memory_mb:
            self.peak_memory_mb = cur

    def _load_on_worker(self):
        """Loads model and tokenizer directly on the execution owner thread."""
        mem_before = self._get_active_memory_mb()
        t0 = time.time()

        # 1. Try mlx_lm for causal models (Qwen, Gemma, Llama, etc.)
        try:
            import mlx_lm
            model, tokenizer_wrap = mlx_lm.load(
                self.model_name,
                revision=self.revision,
            )
            self.model = model
            self.tokenizer = tokenizer_wrap
            self.raw_hf_tokenizer = getattr(tokenizer_wrap, "_tokenizer", tokenizer_wrap)
            self.is_causal_lm = True

            config = getattr(model, "args", getattr(model, "config", None))
            if config and hasattr(config, "hidden_size"):
                self.native_dims = config.hidden_size
            else:
                dummy = mx.zeros((1, 4), dtype=mx.int32)
                backbone = getattr(model, "model", getattr(model, "transformer", model))
                out = backbone(dummy)
                self.native_dims = out.shape[-1]
                del out, dummy

            m_lower = self.model_name.lower()
            if "qwen" in m_lower or "gemma" in m_lower:
                self.pooling_strategy = "last_token"
            else:
                self.pooling_strategy = "mean"

            self.model_type = "causal"
        except Exception as e_mlx_lm:
            # 2. Try encoder architectures (BERT, NomicBERT, etc.) via mlx_embedding_models or transformers
            try:
                from transformers import AutoTokenizer
                from mlx_embedding_models.model import Bert
                from mlx_embedding_models.nomic_model import NomicBert

                self.raw_hf_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    revision=self.revision,
                )
                self.tokenizer = self.raw_hf_tokenizer

                m_lower = self.model_name.lower()
                if "nomic" in m_lower:
                    self.model = NomicBert.from_pretrained(self.model_name)
                    self.pooling_strategy = "mean"
                    self.model_type = "nomic_bert"
                else:
                    self.model = Bert.from_pretrained(self.model_name)
                    self.pooling_strategy = "mean"
                    self.model_type = "bert"

                dummy_tok = self.raw_hf_tokenizer(["test"], return_tensors="np", padding=True, truncation=True)
                dummy_ids = mx.array(dummy_tok["input_ids"])
                hidden_states, _ = self.model(dummy_ids)
                self.native_dims = hidden_states.shape[-1]
                del dummy_ids, dummy_tok, hidden_states
                self.is_causal_lm = False
            except Exception as e_enc:
                raise MLXRuntimeError(
                    f"Failed to load model '{self.model_name}' as MLX causal or encoder model:\n"
                    f"  mlx_lm error: {e_mlx_lm}\n"
                    f"  encoder error: {e_enc}"
                )

        # Tune the micro-batch budget to the loaded model's actual size
        # (prevents quadratic-attention timeouts/OOM on large models).
        try:
            cfg = getattr(self.model, "args", getattr(self.model, "config", None))
            hidden = int(getattr(cfg, "hidden_size", 0) or 0)
            layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
            vocab = int(getattr(cfg, "vocab_size", 0) or 0)
            if hidden > 0 and layers > 0:
                params_b = (layers * 12 * hidden * hidden + vocab * hidden) / 1e9
                budget = self.batch_planner.tune_for_model(params_b)
                print(f"[mlx-runtime] Model ~{params_b:.1f}B params → micro-batch budget {budget} tokens")
        except Exception as exc:
            print(f"[mlx-runtime] Model-size auto-tune skipped: {exc}")

        self.model_memory_mb = max(0.0, self._get_active_memory_mb() - mem_before)
        self._update_peak_memory()
        elapsed = time.time() - t0
        print(
            f"[mlx-runtime] Loaded '{self.model_name}' ✓ ({self.native_dims}d, {self.model_type}, "
            f"pooling={self.pooling_strategy}, {self.model_memory_mb:.1f}MB Metal) in {elapsed:.2f}s"
        )

    def _worker_loop(self):
        """Single execution owner thread loop."""
        try:
            self._load_on_worker()
        except Exception as exc:
            self._init_error = exc
            self._ready_event.set()
            return

        self._ready_event.set()

        while True:
            # Idle unload: when the queue is empty past the deadline, drop the
            # weights (Unified memory is reclaimed); reload on the next job.
            try:
                job = self._work_queue.get(timeout=1.0)
            except queue.Empty:
                if (self._weights_loaded and self.idle_unload_s > 0
                        and time.time() - self._last_active > self.idle_unload_s):
                    self._unload_weights()
                continue
            if job is None:
                break
            texts, requested_dims, is_query, response_future, cancel_event = job

            if cancel_event.is_set():
                response_future.set_exception(RuntimeError("Request canceled by client"))
                self._work_queue.task_done()
                continue

            if not self._weights_loaded:
                self._reload_weights()

            self._last_active = time.time()
            t0 = time.time()
            try:
                # Length-aware micro-batching (never one giant forward pass).
                res = self.batch_planner.plan_and_execute(
                    texts,
                    self.raw_hf_tokenizer,
                    lambda batch: self._embed_sync(batch, requested_dims, is_query),
                    self.max_length,
                )
                latency_ms = (time.time() - t0) * 1000
                self.total_requests += 1
                self.total_latency_ms += latency_ms
                self._update_peak_memory()
                response_future.set_result(res)
            except Exception as exc:
                response_future.set_exception(exc)
            finally:
                self._work_queue.task_done()

    def _embed_sync(
        self,
        texts: list[str],
        requested_dims: Optional[int] = None,
        is_query: bool = False,
    ) -> np.ndarray:
        """Internal synchronous forward pass executed exclusively on the owner thread."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        # Single-pass tokenization
        tok_out = self.raw_hf_tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = mx.array(tok_out["input_ids"], dtype=mx.int32)
        attention_mask = mx.array(tok_out["attention_mask"], dtype=mx.int32)
        token_type_ids = mx.array(tok_out["token_type_ids"]) if "token_type_ids" in tok_out else None

        self.compiled_shapes.add(input_ids.shape)

        # Forward pass through model backbone
        if self.is_causal_lm:
            backbone = getattr(self.model, "model", getattr(self.model, "transformer", self.model))
            hidden_states = backbone(input_ids)
        else:
            hidden_states, _ = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        # Pooling
        if self.pooling_strategy == "last_token":
            last_idx = mx.sum(attention_mask, axis=1) - 1
            batch_size = input_ids.shape[0]
            pooled = hidden_states[mx.arange(batch_size), last_idx]
        elif self.pooling_strategy == "cls":
            pooled = hidden_states[:, 0, :]
        else:
            # Mean pooling
            expanded_mask = mx.expand_dims(attention_mask.astype(hidden_states.dtype), -1)
            sum_hidden = mx.sum(hidden_states * expanded_mask, axis=1)
            sum_mask = mx.maximum(mx.sum(expanded_mask, axis=1), 1e-9)
            pooled = sum_hidden / sum_mask

        # Cast to float32 before normalization
        pooled_f32 = pooled.astype(mx.float32)

        # L2 Normalization
        norm = mx.sqrt(mx.sum(pooled_f32 ** 2, axis=-1, keepdims=True))
        normalized = pooled_f32 / mx.maximum(norm, 1e-12)

        # Matryoshka reduction if requested
        actual_dims = requested_dims if requested_dims and requested_dims < self.native_dims else self.native_dims
        if actual_dims < normalized.shape[-1]:
            normalized = normalized[:, :actual_dims]
            norm = mx.sqrt(mx.sum(normalized ** 2, axis=-1, keepdims=True))
            normalized = normalized / mx.maximum(norm, 1e-12)

        # Force Metal evaluation
        mx.eval(normalized)

        # Convert to numpy array
        result_np = np.array(normalized, copy=False).astype(np.float32)
        del normalized, pooled, pooled_f32, hidden_states, input_ids, attention_mask
        return result_np

    def _unload_weights(self):
        """Drops model weights from GPU/unified memory (idle policy).

        Runs ON the owner thread (safe — no concurrent forward pass possible).
        Keeps tokenizer state in RAM (a few hundred MB saved vs the weights).
        """
        if self.model is None:
            return
        try:
            del self.model
            self.model = None
            mx.clear_cache()
            self._weights_loaded = False
            print(
                f"[mlx-runtime] Idle {self.idle_unload_s:.0f}s — weights unloaded "
                f"({self._get_active_memory_mb():.0f}MB active Metal)"
            )
        except Exception as exc:
            print(f"[mlx-runtime] Idle unload failed (will retry): {exc}", file=sys.stderr)

    def _reload_weights(self):
        """Reloads weights on the owner thread after an idle unload."""
        print("[mlx-runtime] Reloading weights after idle unload...")
        t0 = time.time()
        try:
            self._load_on_worker()
            self._weights_loaded = True
            print(f"[mlx-runtime] Reloaded in {time.time() - t0:.2f}s")
        except Exception as exc:
            self._init_error = exc
            raise MLXRuntimeError(f"Failed to reload model after idle unload: {exc}")

    def warmup(self):
        """Runs warmup inference passes via the execution owner thread."""
        print("[mlx-runtime] Running GPU warmup...")
        t0 = time.time()
        for count in (1, 4, 16):
            sample_texts = ["Warmup embedding Metal compiler verification."] * count
            self.submit_embed(sample_texts, timeout=30.0)
        elapsed = time.time() - t0
        print(f"[mlx-runtime] GPU warmup complete in {elapsed:.2f}s")

    def embed_direct(
        self,
        texts: list[str],
        requested_dims: Optional[int] = None,
        is_query: bool = False,
    ) -> np.ndarray:
        """Executes embedding via the execution owner queue."""
        return self.submit_embed(texts, requested_dims=requested_dims, is_query=is_query)

    def submit_embed(
        self,
        texts: list[str],
        requested_dims: Optional[int] = None,
        is_query: bool = False,
        # 300s to match the TS client ceiling (bulk batches on large models
        # legitimately take minutes; see DEFAULT_TIMEOUT_MS in src/mlx.ts).
        timeout: float = 300.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> np.ndarray:
        """Submits an embedding task to the execution owner thread."""
        from concurrent.futures import Future

        fut: Future[np.ndarray] = Future()
        event = cancel_event or threading.Event()

        try:
            self._work_queue.put_nowait((texts, requested_dims, is_query, fut, event))
        except queue.Full:
            raise MLXRuntimeError("MLX server is overloaded: execution queue is full")

        return fut.result(timeout=timeout)

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        """Tokenizes texts and returns token IDs."""
        return [self.raw_hf_tokenizer.encode(t) for t in texts]

    def get_descriptor(self, requested_dims: Optional[int] = None) -> dict[str, Any]:
        """Returns the canonical descriptor for contract negotiation."""
        out_dims = requested_dims if requested_dims and requested_dims < self.native_dims else self.native_dims
        return {
            "version": 1,
            "backend": "mlx",
            "model": self.model_name,
            "revision": self.revision or "",
            "pooling": self.pooling_strategy,
            "nativeDimensions": self.native_dims,
            "outputDimensions": out_dims,
            "maxTokens": self.max_length,
            "normalized": True,
            "dtype": self.dtype_str,
            "quantization": self.quantization,
        }

    def get_memory_info(self) -> dict[str, float]:
        return {
            "active_mb": round(self._get_active_memory_mb(), 1),
            "peak_mb": round(self._get_peak_memory_mb(), 1),
            "model_mb": round(self.model_memory_mb, 1),
        }

    def get_stats_info(self) -> dict[str, Any]:
        uptime = time.time() - self.start_time
        avg_ms = round(self.total_latency_ms / self.total_requests, 2) if self.total_requests > 0 else 0.0
        return {
            "total_requests": self.total_requests,
            "avg_ms": avg_ms,
            "compiled_shapes": len(self.compiled_shapes),
            "uptime_sec": round(uptime, 1),
            "weights_loaded": self._weights_loaded,
            "idle_unload_s": self.idle_unload_s,
        }

    def shutdown(self):
        """Cleanly terminates the worker thread."""
        self._work_queue.put(None)
        self._worker_thread.join(timeout=2.0)
