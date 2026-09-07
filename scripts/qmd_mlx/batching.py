"""
batching.py — Token-once batching, length bucketing, memory budgeting, and order restoration
"""

import os
import sys
import numpy as np
from typing import Any, Callable, Optional


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


def calculate_default_max_batch_tokens(custom_budget: int = 0) -> int:
    """
    Calculates safe GPU token budget based on physical system RAM.
    Leaves ample headroom for macOS system services and UI.
    """
    if custom_budget > 0:
        return max(512, min(custom_budget, 65536))

    ram_gb = get_system_ram_gb()
    if ram_gb <= 8.5:
        return 4096
    elif ram_gb <= 16.5:
        return 8192
    elif ram_gb <= 24.5:
        return 12288
    elif ram_gb <= 36.5:
        return 16384
    else:
        return 32768


class BatchPlanner:
    """
    Plans execution batches with single-pass tokenization, length-based sorting
    to minimize padding waste, and exact order restoration.
    """

    def __init__(self, max_batch_tokens: Optional[int] = None):
        self.max_batch_tokens = max_batch_tokens or calculate_default_max_batch_tokens()

    def plan_and_execute(
        self,
        texts: list[str],
        tokenizer: Any,
        embed_fn: Callable[[list[str]], np.ndarray],
        max_length: int = 2048,
    ) -> np.ndarray:
        """
        Executes embedding with length-based batching and restores original order.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if len(texts) == 1:
            # Single text fast path
            return embed_fn(texts)

        # 1. Single-pass token length measurement
        # Encode or inspect text lengths
        try:
            # Use tokenizer to get approximate or exact lengths
            lengths = [len(tokenizer.encode(t)) for t in texts]
        except Exception:
            # Fallback approximate length (words * 1.3)
            lengths = [max(1, int(len(t.split()) * 1.3)) for t in texts]

        # 2. Sort indices by text token length
        sorted_indices = sorted(range(len(texts)), key=lambda i: lengths[i])
        sorted_texts = [texts[i] for i in sorted_indices]
        sorted_lengths = [min(lengths[i], max_length) for i in sorted_indices]

        # 3. Form micro-batches bounded by max_batch_tokens
        micro_batches: list[list[str]] = []
        current_batch: list[str] = []
        current_max_len = 0

        for text, token_len in zip(sorted_texts, sorted_lengths):
            cand_max_len = max(current_max_len, token_len)
            cand_tokens = (len(current_batch) + 1) * cand_max_len

            if current_batch and (cand_tokens > self.max_batch_tokens or len(current_batch) >= 64):
                micro_batches.append(current_batch)
                current_batch = [text]
                current_max_len = token_len
            else:
                current_batch.append(text)
                current_max_len = cand_max_len

        if current_batch:
            micro_batches.append(current_batch)

        # 4. Execute micro-batches with OOM retry
        results: list[np.ndarray] = []
        for batch in micro_batches:
            batch_result = self._execute_with_retry(batch, embed_fn)
            results.append(batch_result)

        # 5. Concatenate sorted results
        sorted_embeddings = np.concatenate(results, axis=0) if len(results) > 1 else results[0]

        # 6. Restore original input ordering
        dims = sorted_embeddings.shape[1]
        reordered = np.empty((len(texts), dims), dtype=sorted_embeddings.dtype)
        for sorted_pos, orig_idx in enumerate(sorted_indices):
            reordered[orig_idx] = sorted_embeddings[sorted_pos]

        return reordered

    def _execute_with_retry(
        self,
        batch: list[str],
        embed_fn: Callable[[list[str]], np.ndarray],
    ) -> np.ndarray:
        """Executes a batch; if an OOM error occurs, halves the batch and retries recursively."""
        try:
            return embed_fn(batch)
        except Exception as e:
            err_msg = str(e).lower()
            if ("out of memory" in err_msg or "metal" in err_msg or "alloc" in err_msg) and len(batch) > 1:
                mid = len(batch) // 2
                res1 = self._execute_with_retry(batch[:mid], embed_fn)
                res2 = self._execute_with_retry(batch[mid:], embed_fn)
                return np.concatenate([res1, res2], axis=0)
            raise
