"""
rerank.py — MLX-native Reranker Adapter for Qwen3-Reranker Models
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional
import numpy as np

from .protocol import (
    DeadlineExceededError,
    InvalidInputError,
    MLXServerError,
    ModelUnavailableError,
    RequestCancelledError,
)

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


def infer_quantization_and_dtype(
    model_name_or_path: str,
    default_quant: Optional[str] = None,
    default_dtype: Optional[str] = None,
) -> tuple[str, str]:
    """
    Infers exact quantization format and compute dtype from local config.json or model naming.
    Never silently assumes or misnames quantization.
    """
    expanded = os.path.expanduser(model_name_or_path)
    quant_str = default_quant
    dtype_str = default_dtype or "bfloat16"

    if os.path.isdir(expanded):
        cfg_file = os.path.join(expanded, "config.json")
        if os.path.isfile(cfg_file):
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                quant = cfg.get("quantization") or cfg.get("quantization_config")
                if isinstance(quant, dict):
                    bits = quant.get("bits")
                    mode = quant.get("mode")
                    if bits == 4:
                        quant_str = f"4bit-{mode}" if mode else "4bit"
                    elif bits == 8:
                        quant_str = f"8bit-{mode}" if mode else "8bit"
                    elif quant.get("quant_type"):
                        quant_str = str(quant["quant_type"])
                    elif quant.get("bits"):
                        quant_str = f"{quant['bits']}bit"
                elif isinstance(quant, str):
                    quant_str = quant

                dtype_cfg = cfg.get("torch_dtype") or cfg.get("dtype")
                if dtype_cfg:
                    dtype_str = str(dtype_cfg)
            except Exception:
                pass

    if not quant_str:
        norm_name = os.path.basename(expanded).lower()
        if "4bit" in norm_name or "q4" in norm_name:
            quant_str = "4bit"
        elif "8bit" in norm_name or "q8" in norm_name:
            quant_str = "8bit"
        elif "mxfp8" in norm_name or "fp8" in norm_name:
            quant_str = "mxfp8"
        elif "fp16" in norm_name:
            quant_str = "fp16"
        elif "bf16" in norm_name:
            quant_str = "bf16"
        else:
            quant_str = "mxfp8"

    return quant_str, dtype_str


class RerankError(MLXServerError):
    """Raised when reranker input validation or inference fails."""
    status_code = 400
    error_type = "rerank_error"


class MLXRerankAdapter:
    """
    Reranker adapter for instruction-aware Qwen3 reranking models on Apple Silicon MLX.
    Evaluates P('yes') vs P('no') on the final generated token logits.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        "Note that the answer can only be 'yes' or 'no'."
    )
    DEFAULT_INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"

    # Official Qwen3-Reranker suffix
    THINK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n\n\n\n"

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-Reranker-4B-mxfp8",
        quantization: Optional[str] = None,
        dtype_str: Optional[str] = None,
        max_length: int = 2048,
        revision: Optional[str] = None,
        lazy_load: bool = False,
        executor: Optional[Any] = None,
        model_manager: Optional[Any] = None,
    ):
        if not _MLX_AVAILABLE:
            raise ModelUnavailableError("MLX or mlx-lm is not installed.")

        self.model_name = model_name
        inferred_quant, inferred_dtype = infer_quantization_and_dtype(model_name, quantization, dtype_str)
        self.quantization = inferred_quant
        self.dtype_str = inferred_dtype
        self.max_length = max_length
        self.revision = revision
        self.executor = executor
        self.model_manager = model_manager

        self.model: Any = None
        self.tokenizer: Any = None
        self.raw_hf_tokenizer: Any = None
        self.yes_token_id: Optional[int] = None
        self.no_token_id: Optional[int] = None

        self.model_memory_mb: float = 0.0
        self.total_requests: int = 0
        self.total_pairs_scored: int = 0
        self.total_latency_ms: float = 0.0

        if not lazy_load and self.model_manager is None:
            self.load()

    def load_via_manager(self, timeout_s: float = 120.0):
        if self.model_manager is not None:
            self.model_manager.ensure_loaded("rerank", timeout_s=timeout_s)
        else:
            self.load()

    def is_loaded(self) -> bool:
        return self.model is not None and self.yes_token_id is not None

    def _get_active_memory_mb(self) -> float:
        try:
            if hasattr(mx, "get_active_memory"):
                return mx.get_active_memory() / (1024 * 1024)
            if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                return mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def load(self):
        """Loads model and dynamically resolves yes/no token IDs."""
        if self.model is not None:
            return

        t0 = time.time()
        mem_before = self._get_active_memory_mb()

        try:
            model, tokenizer_wrap = mlx_lm.load(
                self.model_name,
                revision=self.revision,
            )
        except Exception as exc:
            raise ModelUnavailableError(f"Failed to load rerank model '{self.model_name}': {exc}")

        self.model = model
        self.tokenizer = tokenizer_wrap
        self.raw_hf_tokenizer = getattr(tokenizer_wrap, "_tokenizer", tokenizer_wrap)

        self._resolve_token_ids()

        self.model_memory_mb = max(0.0, self._get_active_memory_mb() - mem_before)
        elapsed = time.time() - t0
        print(
            f"[mlx-rerank] Loaded '{self.model_name}' ✓ (yes={self.yes_token_id}, no={self.no_token_id}, "
            f"{self.model_memory_mb:.1f}MB Metal) in {elapsed:.2f}s"
        )

    def _resolve_token_ids(self):
        """Dynamically resolve yes/no token IDs in the chat suffix context."""
        if self.raw_hf_tokenizer is None:
            raise RerankError("Tokenizer not loaded")

        suffix_tokens = self.raw_hf_tokenizer.encode(self.THINK_SUFFIX, add_special_tokens=False)
        yes_context = self.raw_hf_tokenizer.encode(self.THINK_SUFFIX + "yes", add_special_tokens=False)
        no_context = self.raw_hf_tokenizer.encode(self.THINK_SUFFIX + "no", add_special_tokens=False)

        if (
            len(yes_context) == len(suffix_tokens) + 1
            and len(no_context) == len(suffix_tokens) + 1
        ):
            yes_id = yes_context[-1]
            no_id = no_context[-1]
        else:
            yes_tokens = self.raw_hf_tokenizer.encode("yes", add_special_tokens=False)
            no_tokens = self.raw_hf_tokenizer.encode("no", add_special_tokens=False)
            if not yes_tokens or not no_tokens:
                raise RerankError(f"Could not resolve yes/no token IDs for model '{self.model_name}'")
            yes_id = yes_tokens[0]
            no_id = no_tokens[0]

        if yes_id == no_id:
            raise RerankError(f"Ambiguous or identical yes/no token IDs ({yes_id}) for model '{self.model_name}'")

        self.yes_token_id = int(yes_id)
        self.no_token_id = int(no_id)

    def unload(self):
        """Unloads model weights from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self.model_memory_mb = 0.0

    def _format_pair(self, query: str, document: str) -> str:
        """Formats query-document pair with the official Qwen3-Reranker prompt."""
        if not isinstance(document, str) or not document.strip():
            raise RerankError("Document text is empty or whitespace.")
        return (
            f"<|im_start|>system\n{self.DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n<Instruct>: {self.DEFAULT_INSTRUCT}\n\n<Query>: {query}\n\n<Document>: {document}"
            f"{self.THINK_SUFFIX}"
        )

    def score_pairs_sync(
        self,
        query: str,
        documents: list[str],
        batch_size: int = 1,
        deadline: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> list[float]:
        """Synchronous score execution running on the GPU execution owner thread."""
        if not isinstance(query, str) or not query.strip():
            raise RerankError("Query must be a non-empty string.")
        if not isinstance(documents, list):
            raise RerankError("Documents must be a list of strings.")
        if len(documents) == 0:
            return []

        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise RerankError(f"Document at index {i} is not a string (type={type(doc).__name__})")
            if not doc.strip():
                raise RerankError(f"Document at index {i} is empty or whitespace")

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1 or batch_size > 32:
            raise RerankError(f"batch_size must be an integer in [1, 32], got {batch_size}")

        if deadline is not None and time.monotonic() > deadline:
            raise DeadlineExceededError("Request deadline exceeded before reranking started")
        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelledError("Request cancelled before reranking started")

        if not self.is_loaded():
            if self.model_manager is not None:
                self.model_manager.ensure_loaded("rerank", deadline=deadline, cancel_event=cancel_event)
            else:
                self.load()

        t0 = time.time()

        # Precompute query token budget once for the entire batch
        query_text = f"<Instruct>: {self.DEFAULT_INSTRUCT}\n\n<Query>: {query}\n\n<Document>: "
        query_toks = len(self.raw_hf_tokenizer.encode(query_text))
        safe_doc_budget = max(64, self.max_length - query_toks - 120)

        # Single-pass format and tokenize per document
        seqs: list[list[int]] = []
        for doc in documents:
            formatted = self._format_pair(query, doc)
            tokens = self.raw_hf_tokenizer.encode(formatted)
            if len(tokens) == 0:
                raise RerankError("Empty token sequence after formatting")
            if len(tokens) > self.max_length:
                raise RerankError(
                    f"Document token length ({len(tokens)}) exceeds max safe budget ({self.max_length}) for reranker"
                )
            seqs.append(tokens)

        pad_id = self.raw_hf_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.raw_hf_tokenizer.eos_token_id
        if pad_id is None:
            raise RerankError("Tokenizer has neither pad nor EOS token for batch padding")

        scores: list[float] = []
        for start in range(0, len(seqs), batch_size):
            if cancel_event and cancel_event.is_set():
                raise RequestCancelledError("Rerank request cancelled by client")
            if deadline is not None and time.monotonic() > deadline:
                raise DeadlineExceededError(
                    f"Rerank deadline exceeded ({len(scores)}/{len(documents)} documents scored)"
                )

            batch = seqs[start:start + batch_size]
            width = max(len(s) for s in batch)
            padded = [s + [pad_id] * (width - len(s)) for s in batch]
            lengths = [len(s) for s in batch]
            input_ids = mx.array(padded, dtype=mx.int32)

            backbone = getattr(self.model, "model", getattr(self.model, "transformer", None))
            lm_head = getattr(self.model, "lm_head", getattr(self.model, "head", None))

            # Optimization: avoid full-vocab transient allocation where feasible
            if backbone is not None and lm_head is not None:
                hidden_states = backbone(input_ids)
                rows = mx.arange(len(batch))
                cols = mx.array([ln - 1 for ln in lengths], dtype=mx.int32)
                last_hidden = hidden_states[rows, cols]

                # Check if lm_head is standard unquantized Linear with float weights
                is_unquantized_linear = (
                    hasattr(lm_head, "weight")
                    and hasattr(lm_head.weight, "dtype")
                    and mx.issubdtype(lm_head.weight.dtype, mx.floating)
                    and not (hasattr(nn, "QuantizedLinear") and isinstance(lm_head, nn.QuantizedLinear))
                )

                if is_unquantized_linear:
                    w_yes = lm_head.weight[self.yes_token_id]
                    w_no = lm_head.weight[self.no_token_id]
                    diff_w = w_yes - w_no
                    diffs = mx.matmul(last_hidden, diff_w)
                    if hasattr(lm_head, "bias") and lm_head.bias is not None:
                        diffs = diffs + (lm_head.bias[self.yes_token_id] - lm_head.bias[self.no_token_id])
                else:
                    # QuantizedLinear or callable head: evaluate last token logits only [batch, vocab] (~300KB)
                    last_logits = lm_head(last_hidden)
                    diffs = last_logits[:, self.yes_token_id] - last_logits[:, self.no_token_id]

                diffs_f32 = diffs.astype(mx.float32)
                del hidden_states, last_hidden, input_ids
            else:
                logits = self.model(input_ids)
                rows = mx.arange(len(batch))
                cols = mx.array([ln - 1 for ln in lengths], dtype=mx.int32)
                last_logits = logits[rows, cols]
                diffs = last_logits[:, self.yes_token_id] - last_logits[:, self.no_token_id]
                diffs_f32 = diffs.astype(mx.float32)
                del logits, last_logits, input_ids

            mx.eval(diffs_f32)
            diffs_np = np.array(diffs_f32, dtype=np.float64)
            scores.extend(float(1.0 / (1.0 + np.exp(-d))) for d in diffs_np)

            del diffs_f32

        latency_ms = (time.time() - t0) * 1000
        self.total_requests += 1
        self.total_pairs_scored += len(documents)
        self.total_latency_ms += latency_ms
        if self.model_manager is not None:
            self.model_manager.touch("rerank")

        return scores

    def score_pairs(
        self,
        query: str,
        documents: list[str],
        batch_size: int = 1,
        timeout_s: float | None = 100.0,
        cancel_event: Optional[threading.Event] = None,
        deadline: Optional[float] = None,
    ) -> list[float]:
        """Scores pairs via the executor queue (or sync if no executor)."""
        now = time.monotonic()
        to = timeout_s if timeout_s is not None else 300.0
        dl = deadline if deadline is not None else (now + to)
        if dl is not None and time.monotonic() > dl:
            raise DeadlineExceededError(f"Request deadline expired before reranking started (deadline={dl})")
        if cancel_event is not None and cancel_event.is_set():
            raise RequestCancelledError("Request cancelled before reranking started")

        rem = max(0.01, dl - time.monotonic())

        if self.executor:
            return self.executor.submit(
                lambda: self.score_pairs_sync(
                    query, documents, batch_size=batch_size,
                    deadline=dl,
                    cancel_event=cancel_event,
                ),
                priority=0,  # interactive priority
                timeout_s=rem,
                cancel_event=cancel_event,
                description=f"Rerank {len(documents)} docs",
            )
        return self.score_pairs_sync(query, documents, batch_size=batch_size, deadline=dl, cancel_event=cancel_event)

    def warmup(self):
        print("[mlx-rerank] Running GPU warmup...")
        self.score_pairs("warmup query", ["warmup document"], timeout_s=30.0)
        print("[mlx-rerank] Warmup complete ✓")

    def get_descriptor(self) -> dict[str, Any]:
        return {
            "version": 1,
            "backend": "mlx",
            "model": self.model_name,
            "revision": self.revision or "",
            "quantization": self.quantization,
            "dtype": self.dtype_str,
            "maxTokens": self.max_length,
            "yesTokenId": self.yes_token_id,
            "noTokenId": self.no_token_id,
        }

    def shutdown(self):
        if self.executor and hasattr(self.executor, "is_owner_thread") and self.executor.is_owner_thread():
            self.unload()
        elif self.executor and hasattr(self.executor, "is_worker_alive") and self.executor.is_worker_alive():
            try:
                self.executor.submit(self.unload, priority=0, timeout_s=10.0, description="Unload rerank")
            except Exception:
                if hasattr(self.executor, "join_worker"):
                    self.executor.join_worker(timeout=5.0)
                if not self.executor.is_worker_alive():
                    self.unload()
        else:
            self.unload()
