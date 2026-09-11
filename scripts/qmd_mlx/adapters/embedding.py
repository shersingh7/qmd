"""
embedding.py — Explicit Model Adapters for MLX Embedding Models
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from typing import Any, Optional
import numpy as np

from ..protocol import (
    MLXServerError,
    ModelUnavailableError,
    OutOfMemoryError,
    UnsupportedModelError,
)
from .tokenization import TokenizedBatch, TokenizerHelper

try:
    import mlx.core as mx
    import mlx.nn as nn
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


def infer_model_params_b(model_name: str, config: Optional[dict] = None) -> float:
    """
    Derives model parameter count in billions from model name or configuration.
    Distinguishes parameter count (e.g. 0.6B, 1.7B, 4B, 7B) from quantization bits (e.g. 4bit, 8bit).
    """
    cfg = config or {}
    if "num_parameters" in cfg and cfg["num_parameters"]:
        return float(cfg["num_parameters"]) / 1e9

    hidden = cfg.get("hidden_size") or cfg.get("d_model")
    layers = cfg.get("num_hidden_layers") or cfg.get("num_layers")
    if hidden and layers:
        if hidden >= 3500:
            return 7.0
        elif hidden >= 2000:
            return 4.0
        elif hidden >= 1500:
            return 1.7
        elif hidden >= 1000:
            return 0.6

    # Regex: match e.g. 0.5b, 0.6b, 1.5b, 1.7b, 4b, 7b, 8b, 14b, 32b, 70b
    # Crucially do NOT match 4bit, 8bit, 16bit!
    m = re.search(r'(?:^|[_\-/])(\d+(?:\.\d+)?)[bB](?:[_\-/]|$)', model_name)
    if m:
        try:
            val = float(m.group(1))
            if 0.01 <= val <= 1000.0:
                return val
        except ValueError:
            pass

    name_l = model_name.lower()
    if "minilm" in name_l:
        return 0.033
    if "nomic" in name_l:
        return 0.137
    if "bert" in name_l:
        return 0.110

    return 0.6


def estimate_model_memory_mb(model_name: str, quantization: Optional[str] = None, params_b: Optional[float] = None) -> float:
    """
    Computes a consistent conservative pre-load memory estimate in MB.
    """
    pb = params_b if params_b is not None else infer_model_params_b(model_name)
    q = (quantization or "").lower()
    name_l = model_name.lower()

    if "4bit" in q or "4bit" in name_l or "q4" in q or "q4" in name_l or "int4" in q or "dwq" in name_l:
        bits = 4.5
    elif "8bit" in q or "8bit" in name_l or "q8" in q or "q8" in name_l or "int8" in q or "mxfp8" in name_l:
        bits = 8.5
    elif "16" in q or "fp16" in name_l or "bf16" in name_l or "float16" in q or "bfloat16" in q:
        bits = 16.0
    elif "32" in q or "fp32" in name_l or "float32" in q:
        bits = 32.0
    else:
        bits = 4.5 if ("4b" in name_l and "4bit" in name_l) else 16.0

    bytes_per_param = bits / 8.0
    est_mb = (pb * 1e9 * bytes_per_param / (1024 * 1024)) * 1.25
    return max(100.0, est_mb)


class BaseEmbeddingAdapter:
    """Base class for all MLX embedding model adapters."""

    def __init__(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        dtype_str: str = "float32",
        max_length: int = 2048,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.dtype_str = dtype_str
        self.max_length = max_length
        self.revision = revision
        self.trust_remote_code = trust_remote_code

        self.measured_quantization: Optional[str] = None
        self.measured_revision: Optional[str] = None

        self._init_lock = threading.RLock()

        self.model: Any = None
        self.tokenizer: Any = None
        self.raw_hf_tokenizer: Any = None

        self.native_dims: int = 0
        self.pooling_strategy: str = "mean"
        self.padding_side: str = "right"
        self.is_causal: bool = False
        self.model_type: str = "base"
        self.model_params_b: float = infer_model_params_b(model_name)
        self.estimated_memory_mb: float = estimate_model_memory_mb(model_name, quantization, self.model_params_b)
        self.model_memory_mb: float = 0.0

    def _get_active_memory_mb(self) -> float:
        try:
            if _MLX_AVAILABLE:
                if hasattr(mx, "get_active_memory"):
                    return mx.get_active_memory() / (1024 * 1024)
                if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                    return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def is_loaded(self) -> bool:
        return self.model is not None and self.raw_hf_tokenizer is not None

    def load(self):
        raise NotImplementedError

    def unload(self):
        with self._init_lock:
            if self.model is not None:
                del self.model
                self.model = None
            self.model_memory_mb = 0.0

    def tokenize_texts(self, texts: list[str]) -> TokenizedBatch:
        """Tokenizes texts into TokenizedBatch on CPU using tokenizer."""
        if self.raw_hf_tokenizer is None:
            with self._init_lock:
                if self.raw_hf_tokenizer is None:
                    self.load()
        return TokenizerHelper.tokenize_once(texts, self.raw_hf_tokenizer, max_length=self.max_length)

    def forward_batch(
        self,
        tokenized_batch: TokenizedBatch,
        requested_dims: Optional[int] = None,
    ) -> np.ndarray:
        """Executes forward pass directly on TokenizedBatch without re-tokenizing."""
        raise NotImplementedError

    def get_descriptor(self, requested_dims: Optional[int] = None) -> dict[str, Any]:
        out_dims = requested_dims if requested_dims and requested_dims < self.native_dims else self.native_dims
        quant = self.measured_quantization or self.quantization or "unknown"
        rev = self.measured_revision or self.revision or "unknown"
        return {
            "version": 1,
            "backend": "mlx",
            "model": self.model_name,
            "revision": rev,
            "pooling": self.pooling_strategy,
            "nativeDimensions": self.native_dims,
            "outputDimensions": out_dims,
            "maxTokens": self.max_length,
            "normalized": True,
            "dtype": self.dtype_str,
            "quantization": quant,
        }


class QwenEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Adapter for Qwen / Qwen2 / Qwen3 causal embedding architectures on MLX.
    Enforces right-padding and exact last-token non-pad pooling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooling_strategy = "last_token"
        self.padding_side = "right"
        self.is_causal = True
        self.model_type = "qwen_causal"

    def load(self):
        with self._init_lock:
            if self.model is not None:
                return

            if not _MLX_AVAILABLE:
                raise ModelUnavailableError("MLX is not installed.")

            t0 = time.time()
            mem_before = self._get_active_memory_mb()
            import mlx_lm

            try:
                res = mlx_lm.load(
                    self.model_name,
                    revision=self.revision,
                    return_config=True,
                )
                if isinstance(res, tuple) and len(res) == 3:
                    model, tokenizer_wrap, config = res
                elif isinstance(res, tuple) and len(res) == 2:
                    model, tokenizer_wrap = res
                    config = getattr(model, "config", {}) or {}
                else:
                    model, tokenizer_wrap = res, None
                    config = {}
            except TypeError:
                try:
                    model, tokenizer_wrap = mlx_lm.load(
                        self.model_name,
                        revision=self.revision,
                    )
                    config = getattr(model, "config", {}) or {}
                except Exception as exc:
                    raise ModelUnavailableError(f"Failed to load Qwen embedding model '{self.model_name}': {exc}")
            except Exception as exc:
                raise ModelUnavailableError(f"Failed to load Qwen embedding model '{self.model_name}': {exc}")

            # Honor the advertised compute dtype, including quantization scales.
            # Casting only the pooled output cannot recover precision lost in
            # BF16 batch-dependent kernels. Packed integer weights stay quantized.
            compute_dtype = {"float32": mx.float32, "float16": mx.float16,
                             "bfloat16": mx.bfloat16}.get(self.dtype_str)
            if compute_dtype is None:
                raise ModelUnavailableError(f"Unsupported compute dtype: {self.dtype_str}")
            model.set_dtype(compute_dtype)

            self.model = model
            self.tokenizer = tokenizer_wrap
            self.raw_hf_tokenizer = getattr(tokenizer_wrap, "_tokenizer", tokenizer_wrap)

            # Extract measured quantization & revision from config
            if isinstance(config, dict):
                q_info = config.get("quantization")
                if isinstance(q_info, dict):
                    if "quant_type" in q_info:
                        self.measured_quantization = str(q_info["quant_type"])
                    elif "bits" in q_info:
                        self.measured_quantization = f"{q_info['bits']}bit"
                    else:
                        self.measured_quantization = "quantized"
                elif isinstance(q_info, str):
                    self.measured_quantization = q_info
                elif "torch_dtype" in config:
                    td = str(config["torch_dtype"])
                    if td in ("bfloat16", "bf16"):
                        self.measured_quantization = "bf16"
                    elif td in ("float16", "fp16"):
                        self.measured_quantization = "fp16"
                    elif td in ("float32", "fp32"):
                        self.measured_quantization = "fp32"
                    else:
                        self.measured_quantization = td

                if config.get("_commit_hash"):
                    self.measured_revision = str(config["_commit_hash"])
                elif config.get("revision"):
                    self.measured_revision = str(config["revision"])

            # Enforce right padding
            if hasattr(self.raw_hf_tokenizer, "padding_side"):
                self.raw_hf_tokenizer.padding_side = "right"

            # Determine native dimensions
            config_obj = getattr(model, "args", getattr(model, "config", None))
            if config_obj and hasattr(config_obj, "hidden_size"):
                self.native_dims = int(config_obj.hidden_size)
            else:
                dummy = mx.zeros((1, 4), dtype=mx.int32)
                backbone = getattr(model, "model", getattr(model, "transformer", model))
                out = backbone(dummy)
                self.native_dims = int(out.shape[-1])
                del out, dummy

            # Update model_params_b and measured memory
            cfg_dict = config if isinstance(config, dict) else (config_obj.__dict__ if hasattr(config_obj, "__dict__") else {})
            self.model_params_b = infer_model_params_b(self.model_name, cfg_dict)
            mem_delta = self._get_active_memory_mb() - mem_before
            self.model_memory_mb = max(0.0, mem_delta) if mem_delta > 10.0 else self.estimated_memory_mb

            print(f"[mlx-qwen] Loaded '{self.model_name}' ✓ ({self.native_dims}d, {self.model_params_b}B params, {self.model_memory_mb:.1f}MB, last_token pooling) in {time.time() - t0:.2f}s")

    def forward_batch(
        self,
        tokenized_batch: TokenizedBatch,
        requested_dims: Optional[int] = None,
    ) -> np.ndarray:
        if len(tokenized_batch) == 0:
            return np.empty((0, self.native_dims), dtype=np.float32)

        padded_ids, attn_mask, lengths = tokenized_batch.pad_micro_batch(padding_side="right")
        batch_size = padded_ids.shape[0]

        input_ids = mx.array(padded_ids, dtype=mx.int32)
        backbone = getattr(self.model, "model", getattr(self.model, "transformer", self.model))

        # Forward pass through backbone
        hidden_states = backbone(input_ids)

        # Last-token pooling: extract logits at the exact last non-pad token per row
        last_indices = mx.array([max(0, l - 1) for l in lengths], dtype=mx.int32)
        pooled = hidden_states[mx.arange(batch_size), last_indices]

        # Cast to float32 before normalization
        pooled_f32 = pooled.astype(mx.float32)

        # L2 Normalization
        norm = mx.sqrt(mx.sum(pooled_f32 ** 2, axis=-1, keepdims=True))
        normalized = pooled_f32 / mx.maximum(norm, 1e-12)

        # Matryoshka dimension truncation if requested
        actual_dims = requested_dims if requested_dims and requested_dims < self.native_dims else self.native_dims
        if actual_dims < normalized.shape[-1]:
            normalized = normalized[:, :actual_dims]
            norm = mx.sqrt(mx.sum(normalized ** 2, axis=-1, keepdims=True))
            normalized = normalized / mx.maximum(norm, 1e-12)

        mx.eval(normalized)
        result_np = np.array(normalized, copy=False).astype(np.float32)
        del normalized, pooled, pooled_f32, hidden_states, input_ids
        return result_np


class NomicEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for NomicBERT architectures with mean pooling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooling_strategy = "mean"
        self.padding_side = "right"
        self.is_causal = False
        self.model_type = "nomic_bert"

    def load(self):
        with self._init_lock:
            if self.model is not None:
                return

            if not _MLX_AVAILABLE:
                raise ModelUnavailableError("MLX is not installed.")

            t0 = time.time()
            mem_before = self._get_active_memory_mb()
            try:
                from transformers import AutoTokenizer
                from mlx_embedding_models.nomic_model import NomicBert

                self.raw_hf_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    revision=self.revision,
                )
                self.tokenizer = self.raw_hf_tokenizer
                self.model = NomicBert.from_pretrained(self.model_name)

                dummy_tok = self.raw_hf_tokenizer(["test"], return_tensors="np", padding=True, truncation=True)
                dummy_ids = mx.array(dummy_tok["input_ids"])
                hidden_states, _ = self.model(dummy_ids)
                self.native_dims = int(hidden_states.shape[-1])
                del dummy_ids, dummy_tok, hidden_states
            except Exception as exc:
                raise ModelUnavailableError(f"Failed to load NomicBERT model '{self.model_name}': {exc}")

            self.model_params_b = infer_model_params_b(self.model_name)
            mem_delta = self._get_active_memory_mb() - mem_before
            self.model_memory_mb = max(0.0, mem_delta) if mem_delta > 10.0 else self.estimated_memory_mb

            print(f"[mlx-nomic] Loaded '{self.model_name}' ✓ ({self.native_dims}d, {self.model_params_b}B params, {self.model_memory_mb:.1f}MB, mean pooling) in {time.time() - t0:.2f}s")

    def forward_batch(
        self,
        tokenized_batch: TokenizedBatch,
        requested_dims: Optional[int] = None,
    ) -> np.ndarray:
        if len(tokenized_batch) == 0:
            return np.empty((0, self.native_dims), dtype=np.float32)

        padded_ids, attn_mask, lengths = tokenized_batch.pad_micro_batch(padding_side="right")
        input_ids = mx.array(padded_ids, dtype=mx.int32)
        attention_mask = mx.array(attn_mask, dtype=mx.int32)

        hidden_states, _ = self.model(input_ids, attention_mask=attention_mask)

        # Mean pooling
        expanded_mask = mx.expand_dims(attention_mask.astype(hidden_states.dtype), -1)
        sum_hidden = mx.sum(hidden_states * expanded_mask, axis=1)
        sum_mask = mx.maximum(mx.sum(expanded_mask, axis=1), 1e-9)
        pooled = sum_hidden / sum_mask

        pooled_f32 = pooled.astype(mx.float32)
        norm = mx.sqrt(mx.sum(pooled_f32 ** 2, axis=-1, keepdims=True))
        normalized = pooled_f32 / mx.maximum(norm, 1e-12)

        actual_dims = requested_dims if requested_dims and requested_dims < self.native_dims else self.native_dims
        if actual_dims < normalized.shape[-1]:
            normalized = normalized[:, :actual_dims]
            norm = mx.sqrt(mx.sum(normalized ** 2, axis=-1, keepdims=True))
            normalized = normalized / mx.maximum(norm, 1e-12)

        mx.eval(normalized)
        result_np = np.array(normalized, copy=False).astype(np.float32)
        del normalized, pooled, pooled_f32, hidden_states, input_ids, attention_mask
        return result_np


class BertEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for standard BERT encoder architectures with mean pooling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooling_strategy = "mean"
        self.padding_side = "right"
        self.is_causal = False
        self.model_type = "bert"

    def load(self):
        with self._init_lock:
            if self.model is not None:
                return

            if not _MLX_AVAILABLE:
                raise ModelUnavailableError("MLX is not installed.")

            t0 = time.time()
            mem_before = self._get_active_memory_mb()
            try:
                from transformers import AutoTokenizer
                from mlx_embedding_models.model import Bert

                self.raw_hf_tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    revision=self.revision,
                )
                self.tokenizer = self.raw_hf_tokenizer
                self.model = Bert.from_pretrained(self.model_name)

                dummy_tok = self.raw_hf_tokenizer(["test"], return_tensors="np", padding=True, truncation=True)
                dummy_ids = mx.array(dummy_tok["input_ids"])
                hidden_states, _ = self.model(dummy_ids)
                self.native_dims = int(hidden_states.shape[-1])
                del dummy_ids, dummy_tok, hidden_states
            except Exception as exc:
                raise ModelUnavailableError(f"Failed to load BERT model '{self.model_name}': {exc}")

            self.model_params_b = infer_model_params_b(self.model_name)
            mem_delta = self._get_active_memory_mb() - mem_before
            self.model_memory_mb = max(0.0, mem_delta) if mem_delta > 10.0 else self.estimated_memory_mb

            print(f"[mlx-bert] Loaded '{self.model_name}' ✓ ({self.native_dims}d, {self.model_params_b}B params, {self.model_memory_mb:.1f}MB, mean pooling) in {time.time() - t0:.2f}s")

    def forward_batch(
        self,
        tokenized_batch: TokenizedBatch,
        requested_dims: Optional[int] = None,
    ) -> np.ndarray:
        if len(tokenized_batch) == 0:
            return np.empty((0, self.native_dims), dtype=np.float32)

        padded_ids, attn_mask, lengths = tokenized_batch.pad_micro_batch(padding_side="right")
        input_ids = mx.array(padded_ids, dtype=mx.int32)
        attention_mask = mx.array(attn_mask, dtype=mx.int32)

        hidden_states, _ = self.model(input_ids, attention_mask=attention_mask)

        # Mean pooling
        expanded_mask = mx.expand_dims(attention_mask.astype(hidden_states.dtype), -1)
        sum_hidden = mx.sum(hidden_states * expanded_mask, axis=1)
        sum_mask = mx.maximum(mx.sum(expanded_mask, axis=1), 1e-9)
        pooled = sum_hidden / sum_mask

        pooled_f32 = pooled.astype(mx.float32)
        norm = mx.sqrt(mx.sum(pooled_f32 ** 2, axis=-1, keepdims=True))
        normalized = pooled_f32 / mx.maximum(norm, 1e-12)

        actual_dims = requested_dims if requested_dims and requested_dims < self.native_dims else self.native_dims
        if actual_dims < normalized.shape[-1]:
            normalized = normalized[:, :actual_dims]
            norm = mx.sqrt(mx.sum(normalized ** 2, axis=-1, keepdims=True))
            normalized = normalized / mx.maximum(norm, 1e-12)

        mx.eval(normalized)
        result_np = np.array(normalized, copy=False).astype(np.float32)
        del normalized, pooled, pooled_f32, hidden_states, input_ids, attention_mask
        return result_np


class SyntheticTokenizer:
    """Lightweight deterministic CPU tokenizer for synthetic fixture rehearsals."""

    def __init__(self, max_length: int = 2048):
        self.max_length = max_length
        self.pad_token_id = 0
        self.eos_token_id = 151643
        self.padding_side = "right"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if not text:
            return []
        words = text.split()
        tokens: list[int] = []
        for w in words:
            h = int.from_bytes(hashlib.md5(w.encode("utf-8")).digest()[:4], byteorder="big") % 100000 + 1
            tokens.append(h)
        if not tokens:
            tokens = [1]
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        words = [f"tok_{t}" for t in token_ids if t != self.eos_token_id and t != self.pad_token_id]
        return " ".join(words)


class SyntheticEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Test-only synthetic embedding adapter for offline qualification and rehearsals.
    Exercises the full real server, executor, residency manager, and runtime pipeline
    without loading real weights or requiring MLX.
    Outputs are explicitly labeled synthetic.
    """

    def __init__(
        self,
        model_name: str = "synthetic-qwen3-4b",
        quantization: Optional[str] = None,
        dtype_str: str = "float32",
        max_length: int = 2048,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        dims: int = 2560,
    ):
        super().__init__(
            model_name=model_name,
            quantization=quantization,
            dtype_str=dtype_str,
            max_length=max_length,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        self.synthetic = True
        self.native_dims = dims
        self.pooling_strategy = "last_token"
        self.padding_side = "right"
        self.is_causal = True
        self.model_type = "synthetic_embedding"
        self.model_params_b = infer_model_params_b(model_name)
        if self.model_params_b == 0.6 and ("4b" in model_name.lower() or "4B" in model_name):
            self.model_params_b = 4.0
        self.estimated_memory_mb = 128.0
        self.model_memory_mb = 128.0
        self._loaded = False
        self.hang_on_embed = os.environ.get("MLX_HANG_ON_EMBED") == "1" or ("hang" in model_name.lower())
        self.hang_after_requests = int(os.environ.get("MLX_HANG_AFTER_REQUESTS", "0"))
        self._request_count = 0

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def load(self):
        with self._init_lock:
            if self._loaded:
                return
            self._loaded = True
            self.raw_hf_tokenizer = SyntheticTokenizer(max_length=self.max_length)
            self.tokenizer = self.raw_hf_tokenizer
            self.model = object()  # non-None sentinel

    def unload(self):
        with self._init_lock:
            self._loaded = False
            self.model = None

    def tokenize_texts(self, texts: list[str]) -> TokenizedBatch:
        if not self.is_loaded():
            with self._init_lock:
                if not self.is_loaded():
                    self.load()
        return TokenizerHelper.tokenize_once(texts, self.raw_hf_tokenizer, max_length=self.max_length)

    def forward_batch(
        self,
        tokenized_batch: TokenizedBatch,
        requested_dims: Optional[int] = None,
    ) -> np.ndarray:
        if self.hang_on_embed:
            while True:
                time.sleep(0.5)

        if self.hang_after_requests > 0 and self._request_count >= self.hang_after_requests:
            while True:
                time.sleep(0.5)

        self._request_count += 1

        if len(tokenized_batch) == 0:
            return np.empty((0, self.native_dims), dtype=np.float32)

        actual_dims = requested_dims if (requested_dims and requested_dims < self.native_dims) else self.native_dims
        count = len(tokenized_batch)
        arr = np.empty((count, actual_dims), dtype=np.float32)

        fail_consistency = os.environ.get("MLX_FAKE_FAIL_CONSISTENCY") == "1"

        for i, seq in enumerate(tokenized_batch.token_ids):
            seed_bytes = hashlib.sha256(bytes(str(seq), "utf-8")).digest()
            seed = int.from_bytes(seed_bytes[:8], byteorder="big")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.native_dims).astype(np.float32)
            if actual_dims < self.native_dims:
                vec = vec[:actual_dims]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            orig_idx = (
                tokenized_batch.original_indices[i]
                if tokenized_batch.original_indices and i < len(tokenized_batch.original_indices)
                else i
            )
            if fail_consistency and count > 1 and orig_idx == 0:
                # Deliberately perturb original item 0 when embedded in a batch so cosine similarity drops to ~0.999570
                perturb = np.zeros_like(vec)
                perturb[0] = 0.0293
                vec = vec + perturb
                vec = vec / np.linalg.norm(vec)
            arr[i] = vec

        return arr

    def get_descriptor(self, requested_dims: Optional[int] = None) -> dict[str, Any]:
        out_dims = requested_dims if (requested_dims and requested_dims < self.native_dims) else self.native_dims
        return {
            "version": 1,
            "backend": "mlx_synthetic",
            "model": self.model_name,
            "revision": "synthetic-rehearsal",
            "pooling": self.pooling_strategy,
            "nativeDimensions": self.native_dims,
            "outputDimensions": out_dims,
            "maxTokens": self.max_length,
            "normalized": True,
            "dtype": self.dtype_str,
            "quantization": "synthetic-4bit",
            "synthetic": True,
        }


def resolve_embedding_adapter(
    model_name: str,
    quantization: Optional[str] = None,
    dtype_str: str = "float32",
    max_length: int = 2048,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
) -> BaseEmbeddingAdapter:
    """Resolves and instantiates the correct embedding adapter for a model name."""
    m_lower = model_name.lower()

    if "synthetic" in m_lower or "fake" in m_lower or m_lower.startswith("test-"):
        m_dims = re.search(r'(\d+)d\b', m_lower)
        dims = int(m_dims.group(1)) if m_dims else (384 if "minilm" in m_lower or "384" in m_lower else 2560)
        return SyntheticEmbeddingAdapter(
            model_name=model_name,
            quantization=quantization,
            dtype_str=dtype_str,
            max_length=max_length,
            revision=revision,
            trust_remote_code=trust_remote_code,
            dims=dims,
        )
    elif "qwen" in m_lower or "gemma" in m_lower or "embeddinggemma" in m_lower:
        return QwenEmbeddingAdapter(
            model_name=model_name,
            quantization=quantization,
            dtype_str=dtype_str,
            max_length=max_length,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
    elif "nomic" in m_lower:
        return NomicEmbeddingAdapter(
            model_name=model_name,
            quantization=quantization,
            dtype_str=dtype_str,
            max_length=max_length,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
    elif "minilm" in m_lower or "bert" in m_lower:
        return BertEmbeddingAdapter(
            model_name=model_name,
            quantization=quantization,
            dtype_str=dtype_str,
            max_length=max_length,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
    else:
        raise UnsupportedModelError(
            f"Unsupported MLX embedding model: '{model_name}'. "
            f"Registered adapters: Synthetic/Fake, Qwen/Gemma, Nomic-BERT, MiniLM/BERT."
        )
