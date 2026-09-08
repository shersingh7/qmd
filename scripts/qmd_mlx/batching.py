"""
batching.py — Token-once batching, length bucketing, memory budgeting, and order restoration
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Callable, List, Optional
import numpy as np

from .adapters.tokenization import TokenizedBatch, TokenizerHelper
from .protocol import (
    DeadlineExceededError,
    OutOfMemoryError,
    RequestCancelledError,
)


def get_system_ram_gb() -> float:
    """Detects total physical system RAM in gigabytes."""
    try:
        if sys.platform == "darwin":
            import subprocess
            res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                bytes_mem = int(res.stdout.strip())
                return bytes_mem / (1024 ** 3)
    except Exception:
        pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024 ** 3)
    except Exception:
        return 16.0  # Safe default


def calculate_default_max_batch_tokens(
    custom_budget: int = 0,
    model_params_b: float = 0.6,
    system_ram_gb_fn: Optional[Callable[[], float]] = None,
) -> int:
    """
    Calculates safe GPU token budget based on physical system RAM and model size.
    """
    if custom_budget > 0:
        return max(512, min(custom_budget, 65536))

    ram_fn = system_ram_gb_fn or get_system_ram_gb
    ram_gb = ram_fn()
    if ram_gb <= 8.5:
        base = 4096
    elif ram_gb <= 16.5:
        base = 8192
    elif ram_gb <= 24.5:
        base = 12288
    elif ram_gb <= 36.5:
        base = 16384
    else:
        base = 32768

    try:
        scale = 0.6 / float(model_params_b or 0.6)
    except (TypeError, ValueError):
        scale = 1.0
    scale = min(1.0, max(0.1, scale))
    return max(1024, int(base * scale))


class BatchPlanner:
    """
    Plans execution batches with single-pass tokenization, length-based sorting
    to minimize padding waste, and exact order restoration.
    """

    def __init__(
        self,
        max_batch_tokens: Optional[int] = None,
        model_params_b: float = 0.6,
        system_ram_gb_fn: Optional[Callable[[], float]] = None,
    ):
        self._explicit_budget = max_batch_tokens or 0
        self._ram_fn = system_ram_gb_fn
        self.max_batch_tokens = max_batch_tokens or calculate_default_max_batch_tokens(
            model_params_b=model_params_b,
            system_ram_gb_fn=self._ram_fn,
        )

    def tune_for_model(self, model_params_b: float) -> int:
        """Re-scale the token budget once the loaded model's size is known."""
        if self._explicit_budget > 0:
            return self.max_batch_tokens
        self.max_batch_tokens = calculate_default_max_batch_tokens(
            model_params_b=model_params_b,
            system_ram_gb_fn=self._ram_fn,
        )
        return self.max_batch_tokens

    def plan_micro_batches(
        self,
        tokenized_batch: TokenizedBatch,
    ) -> tuple[list[TokenizedBatch], list[list[int]]]:
        """
        Groups tokenized sequences by length into micro-batches bounded by max_batch_tokens.
        Returns (list of TokenizedBatch sub-batches, list of original index positions).
        """
        total_items = len(tokenized_batch)
        if total_items == 0:
            return [], []

        if total_items == 1:
            return [tokenized_batch], [[0]]

        lengths = tokenized_batch.lengths
        sorted_pos = sorted(range(total_items), key=lambda i: lengths[i])
        sorted_lengths = [lengths[i] for i in sorted_pos]

        micro_batches_indices: List[List[int]] = []
        current_batch: List[int] = []
        current_max_len = 0

        for pos, token_len in zip(sorted_pos, sorted_lengths):
            cand_max_len = max(current_max_len, token_len)
            cand_tokens = (len(current_batch) + 1) * cand_max_len

            if current_batch and (cand_tokens > self.max_batch_tokens or len(current_batch) >= 64):
                micro_batches_indices.append(current_batch)
                current_batch = [pos]
                current_max_len = token_len
            else:
                current_batch.append(pos)
                current_max_len = cand_max_len

        if current_batch:
            micro_batches_indices.append(current_batch)

        sub_batches = [tokenized_batch.slice(indices) for indices in micro_batches_indices]
        return sub_batches, micro_batches_indices

    @staticmethod
    def restore_ordering(
        results: list[np.ndarray],
        micro_batches_indices: list[list[int]],
        total_items: int,
    ) -> np.ndarray:
        """Restores the original input order from micro-batch results."""
        if total_items == 0:
            return np.empty((0, 0), dtype=np.float32)
        sorted_embeddings = np.concatenate(results, axis=0) if len(results) > 1 else results[0]
        dims = sorted_embeddings.shape[1]
        reordered = np.empty((total_items, dims), dtype=sorted_embeddings.dtype)

        current_row = 0
        for batch_indices in micro_batches_indices:
            for orig_pos in batch_indices:
                reordered[orig_pos] = sorted_embeddings[current_row]
                current_row += 1

        return reordered

    def plan_and_execute_tokenized(
        self,
        tokenized_batch: TokenizedBatch,
        embed_fn: Callable[[TokenizedBatch], np.ndarray],
        deadline: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> np.ndarray:
        """
        Executes embedding directly from a pre-tokenized TokenizedBatch.
        Avoids all redundant text tokenization and checks deadline between micro-batches.
        """
        total_items = len(tokenized_batch)
        if total_items == 0:
            return np.empty((0, 0), dtype=np.float32)

        sub_batches, micro_batches_indices = self.plan_micro_batches(tokenized_batch)

        results: List[np.ndarray] = []
        for batch_idx, sub_batch in enumerate(sub_batches):
            if cancel_event and cancel_event.is_set():
                raise RequestCancelledError("Request cancelled by client")
            if deadline and time.monotonic() > deadline:
                raise DeadlineExceededError(
                    f"Deadline exceeded after {batch_idx}/{len(sub_batches)} micro-batches"
                )

            batch_result = self.execute_sub_batch_with_retry(sub_batch, embed_fn)
            results.append(batch_result)

        return self.restore_ordering(results, micro_batches_indices, total_items)

    def plan_and_execute(
        self,
        texts: list[str],
        tokenizer: Any,
        embed_fn: Callable[[list[str]], np.ndarray],
        max_length: int = 2048,
    ) -> np.ndarray:
        """
        Backwards-compatible wrapper that tokenizes texts once into TokenizedBatch.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        raw_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
        tokenized_batch = TokenizerHelper.tokenize_once(texts, raw_tokenizer, max_length=max_length)

        def _adapted_embed(sub_batch: TokenizedBatch) -> np.ndarray:
            sub_texts = [texts[i] for i in sub_batch.original_indices]
            return embed_fn(sub_texts)

        return self.plan_and_execute_tokenized(tokenized_batch, _adapted_embed)

    def execute_sub_batch_with_retry(
        self,
        batch: TokenizedBatch,
        embed_fn: Callable[[TokenizedBatch], np.ndarray],
        depth: int = 0,
    ) -> np.ndarray:
        """
        Executes a sub-batch. If a memory allocation failure occurs, bisects the batch
        up to max depth 3 (halves batch size) and retries.
        """
        try:
            return embed_fn(batch)
        except Exception as e:
            if isinstance(e, OutOfMemoryError):
                is_oom = True
            else:
                err_msg = str(e).lower()
                is_oom = (
                    "metal buffer allocation failed" in err_msg
                    or ("metal" in err_msg and "out of memory" in err_msg)
                    or ("[metal]" in err_msg and ("allocation" in err_msg or "memory" in err_msg))
                    or ("mlx" in err_msg and "out of memory" in err_msg)
                )
            if is_oom:
                # Reduce future batch token budget on OOM to prevent future failures
                self.max_batch_tokens = max(512, self.max_batch_tokens // 2)
                if len(batch) > 1 and depth < 3:
                    mid = len(batch) // 2
                    sub1 = batch.slice(list(range(0, mid)))
                    sub2 = batch.slice(list(range(mid, len(batch))))
                    res1 = self.execute_sub_batch_with_retry(sub1, embed_fn, depth=depth + 1)
                    res2 = self.execute_sub_batch_with_retry(sub2, embed_fn, depth=depth + 1)
                    return np.concatenate([res1, res2], axis=0)
                raise OutOfMemoryError(f"Metal allocation failed during embedding forward pass: {e}")
            raise

    # Backwards compatible alias
    _execute_with_retry = execute_sub_batch_with_retry
