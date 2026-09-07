"""
rerank.py — MLX-native Reranker Adapter for Qwen3-Reranker Models
"""

import time
import numpy as np
from typing import Any, Optional

try:
    import mlx.core as mx
    import mlx_lm
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


class RerankError(ValueError):
    """Raised when reranker input validation or inference fails."""
    pass


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

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-Reranker-4B-mxfp8",
        quantization: str = "mxfp8",
        dtype_str: str = "bfloat16",
        max_length: int = 2048,
        revision: Optional[str] = None,
        lazy_load: bool = False,
    ):
        if not _MLX_AVAILABLE:
            raise RerankError("MLX or mlx-lm is not installed.")

        self.model_name = model_name
        self.quantization = quantization
        self.dtype_str = dtype_str
        self.max_length = max_length
        self.revision = revision

        self.model: Any = None
        self.tokenizer: Any = None
        self.raw_hf_tokenizer: Any = None
        self.yes_token_id: Optional[int] = None
        self.no_token_id: Optional[int] = None

        self.model_memory_mb: float = 0.0
        self.total_requests: int = 0
        self.total_pairs_scored: int = 0
        self.total_latency_ms: float = 0.0

        if not lazy_load:
            self.load()

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

        model, tokenizer_wrap = mlx_lm.load(
            self.model_name,
            revision=self.revision,
        )
        self.model = model
        self.tokenizer = tokenizer_wrap
        self.raw_hf_tokenizer = getattr(tokenizer_wrap, "_tokenizer", tokenizer_wrap)

        # Dynamically resolve yes / no token IDs from the loaded tokenizer
        yes_tokens = self.raw_hf_tokenizer.encode("yes", add_special_tokens=False)
        no_tokens = self.raw_hf_tokenizer.encode("no", add_special_tokens=False)

        if not yes_tokens or not no_tokens:
            raise RerankError(f"Could not resolve yes/no token IDs for model '{self.model_name}'")

        self.yes_token_id = int(yes_tokens[0])
        self.no_token_id = int(no_tokens[0])

        self.model_memory_mb = max(0.0, self._get_active_memory_mb() - mem_before)
        elapsed = time.time() - t0
        print(
            f"[mlx-rerank] Loaded '{self.model_name}' ✓ (yes={self.yes_token_id}, no={self.no_token_id}, "
            f"{self.model_memory_mb:.1f}MB Metal) in {elapsed:.2f}s"
        )

    # Official Qwen3-Reranker suffix: the 4B/8B rerankers descend from
    # thinking-capable bases and only emit the trained yes/no signal AFTER
    # the thinking-close marker (verified against the official model card,
    # Sep 7 2026). The 0.6B has an explicit LogitScore head and is insensitive.
    THINK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n\n\n\n"

    def _format_pair(self, query: str, document: str) -> str:
        """Formats query-document pair with the official Qwen3-Reranker prompt.

        Uses the manual official format rather than the tokenizer's chat
        template: some community conversions ship broken/lossy chat templates
        (verified: Qwen3-Reranker-0.6B-4bit's template silently drops message
        content — see docs/benchmarks/mlx-reranker-4b-mxfp8-defect.md).
        Truncates the document (never the query) to fit max_length.
        """
        query_text = f"<Instruct>: {self.DEFAULT_INSTRUCT}\n\n<Query>: {query}\n\n<Document>: "
        query_toks = len(self.raw_hf_tokenizer.encode(query_text))

        # Overhead for the chat scaffolding (system prompt, im_start/im_end tags)
        # ~ 100 tokens
        safe_doc_budget = max(64, self.max_length - query_toks - 120)

        doc_toks = self.raw_hf_tokenizer.encode(document)
        if len(doc_toks) > safe_doc_budget:
            truncated_doc = self.raw_hf_tokenizer.decode(doc_toks[:safe_doc_budget])
        else:
            truncated_doc = document

        return (
            f"<|im_start|>system\n{self.DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n<Instruct>: {self.DEFAULT_INSTRUCT}\n\n<Query>: {query}\n\n<Document>: {truncated_doc}"
            f"{self.THINK_SUFFIX}"
        )

    def score_pairs(
        self,
        query: str,
        documents: list[str],
        batch_size: int = 4,
        timeout_s: float | None = 100.0,
    ) -> list[float]:
        """
        Scores a query against a list of documents.
        Returns a list of float scores in [0.0, 1.0] matching documents order.

        Pairs run in micro-batches (one forward pass per batch, logits
        gathered at each row's true final position — numerically identical to
        single-pair scoring). `timeout_s` bounds total wall-clock time; on
        expiry raises RerankError instead of hanging the HTTP worker past the
        client's deadline. None disables the deadline (tests only).
        """
        if not isinstance(query, str) or not query.strip():
            raise RerankError("Query must be a non-empty string.")
        if not isinstance(documents, list):
            raise RerankError("Documents must be a list of strings.")
        if len(documents) == 0:
            return []

        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise RerankError(f"Document at index {i} is not a string (type={type(doc).__name__})")

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1 or batch_size > 32:
            raise RerankError(f"batch_size must be an integer in [1, 32], got {batch_size}")

        if self.model is None:
            self.load()

        t0 = time.time()

        def _check_deadline():
            if timeout_s is not None and (time.time() - t0) > timeout_s:
                raise RerankError(
                    f"Rerank deadline exceeded ({timeout_s}s for {len(documents)} documents)"
                )

        # Format + tokenize everything up front (CPU-side, cheap).
        seqs = [self.raw_hf_tokenizer.encode(self._format_pair(query, doc)) for doc in documents]
        if any(len(s) == 0 for s in seqs):
            raise RerankError("Empty token sequence after formatting (query/document too long?)")

        pad_id = self.raw_hf_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.raw_hf_tokenizer.eos_token_id
        if pad_id is None:
            raise RerankError("Tokenizer has neither pad nor EOS token for batch padding")

        scores: list[float] = []
        for start in range(0, len(seqs), batch_size):
            _check_deadline()
            batch = seqs[start:start + batch_size]
            width = max(len(s) for s in batch)
            padded = [s + [pad_id] * (width - len(s)) for s in batch]
            lengths = [len(s) for s in batch]
            input_ids = mx.array(padded, dtype=mx.int32)

            logits = self.model(input_ids)
            # Gather each row's TRUE final position (padding must not shift it).
            rows = mx.arange(len(batch))
            cols = mx.array([ln - 1 for ln in lengths], dtype=mx.int32)
            last_logits = logits[rows, cols]
            diffs = last_logits[:, self.yes_token_id] - last_logits[:, self.no_token_id]
            # Cast before NumPy handoff: quantized models emit bfloat16,
            # which has no PEP 3118 buffer format and crashes np.array().
            diffs_f32 = diffs.astype(mx.float32)
            mx.eval(diffs_f32)
            diffs_np = np.array(diffs_f32, dtype=np.float64)
            scores.extend(float(1.0 / (1.0 + np.exp(-d))) for d in diffs_np)

            del logits, last_logits, diffs, diffs_f32, input_ids

        latency_ms = (time.time() - t0) * 1000
        self.total_requests += 1
        self.total_pairs_scored += len(documents)
        self.total_latency_ms += latency_ms

        return scores

    def warmup(self):
        """Runs warmup inference passes."""
        print("[mlx-rerank] Running GPU warmup...")
        self.score_pairs("warmup query", ["warmup document"])
        print("[mlx-rerank] Warmup complete ✓")

    def get_descriptor(self) -> dict[str, Any]:
        """Returns the canonical rerank descriptor."""
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
