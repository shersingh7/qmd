"""
embedding.py — Explicit Model Adapters for MLX Embedding Models
"""

from __future__ import annotations

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

        self._init_lock = threading.Lock()

        self.model: Any = None
        self.tokenizer: Any = None
        self.raw_hf_tokenizer: Any = None

        self.native_dims: int = 0
        self.pooling_strategy: str = "mean"
        self.padding_side: str = "right"
        self.is_causal: bool = False
        self.model_type: str = "base"
        self.model_memory_mb: float = 0.0
        self.model_params_b: float = 0.6

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
        """Tokenizes texts into TokenizedBatch."""
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
        config = getattr(model, "args", getattr(model, "config", None))
        if config and hasattr(config, "hidden_size"):
            self.native_dims = int(config.hidden_size)
        else:
            dummy = mx.zeros((1, 4), dtype=mx.int32)
            backbone = getattr(model, "model", getattr(model, "transformer", model))
            out = backbone(dummy)
            self.native_dims = int(out.shape[-1])
            del out, dummy

        print(f"[mlx-qwen] Loaded '{self.model_name}' ✓ ({self.native_dims}d, last_token pooling) in {time.time() - t0:.2f}s")

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

            print(f"[mlx-nomic] Loaded '{self.model_name}' ✓ ({self.native_dims}d, mean pooling) in {time.time() - t0:.2f}s")

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

        print(f"[mlx-bert] Loaded '{self.model_name}' ✓ ({self.native_dims}d, mean pooling) in {time.time() - t0:.2f}s")

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

    if "qwen" in m_lower or "gemma" in m_lower or "embeddinggemma" in m_lower:
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
            f"Registered adapters: Qwen/Gemma, Nomic-BERT, MiniLM/BERT."
        )
