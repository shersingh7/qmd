"""
tokenization.py — Single-pass tokenization, length-aware batch containers, and exact order tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np

from ..protocol import InvalidInputError


@dataclass
class TokenizedBatch:
    """
    Container for tokenized sequences. Holds raw unpadded token arrays,
    exact lengths, and original indices to enable single-pass tokenization.
    """
    token_ids: List[List[int]]
    lengths: List[int]
    original_indices: List[int]
    pad_token_id: int
    texts: Optional[List[str]] = None

    def __len__(self) -> int:
        return len(self.token_ids)

    def slice(self, indices: List[int]) -> TokenizedBatch:
        """Slices a subset of sequences maintaining proper original indices."""
        sub_ids = [self.token_ids[i] for i in indices]
        sub_lens = [self.lengths[i] for i in indices]
        sub_orig = [self.original_indices[i] for i in indices]
        sub_texts = [self.texts[i] for i in indices] if self.texts else None
        return TokenizedBatch(
            token_ids=sub_ids,
            lengths=sub_lens,
            original_indices=sub_orig,
            pad_token_id=self.pad_token_id,
            texts=sub_texts,
        )

    def pad_micro_batch(
        self,
        padding_side: str = "right",
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        Pads only this micro-batch to its own maximum length.
        Returns (padded_input_ids, attention_mask, lengths) as numpy arrays.
        """
        if not self.token_ids:
            return (
                np.empty((0, 0), dtype=np.int32),
                np.empty((0, 0), dtype=np.int32),
                [],
            )

        batch_size = len(self.token_ids)
        max_len = max(self.lengths) if self.lengths else 1
        if max_len == 0:
            max_len = 1

        padded_ids = np.full((batch_size, max_len), self.pad_token_id, dtype=np.int32)
        attn_mask = np.zeros((batch_size, max_len), dtype=np.int32)

        for i, (seq, length) in enumerate(zip(self.token_ids, self.lengths)):
            if length == 0:
                # Keep 1 pad token to avoid empty dimension errors
                continue
            if padding_side == "right":
                padded_ids[i, :length] = seq[:length]
                attn_mask[i, :length] = 1
            else:
                # Left padding
                start = max_len - length
                padded_ids[i, start:] = seq[:length]
                attn_mask[i, start:] = 1

        return padded_ids, attn_mask, list(self.lengths)


class TokenizerHelper:
    """Helper for single-pass tokenization and padding configuration."""

    @staticmethod
    def get_pad_token_id(tokenizer: Any) -> int:
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = 0
        return int(pad_id)

    @staticmethod
    def tokenize_once(
        texts: List[str],
        raw_tokenizer: Any,
        max_length: int = 2048,
    ) -> TokenizedBatch:
        """
        Tokenizes a list of texts exactly once.
        Returns a TokenizedBatch containing unpadded token lists and lengths.
        """
        pad_id = TokenizerHelper.get_pad_token_id(raw_tokenizer)

        if not texts:
            return TokenizedBatch(
                token_ids=[],
                lengths=[],
                original_indices=[],
                pad_token_id=pad_id,
                texts=[],
            )

        # Fast path: use tokenizer's batch encode if available
        token_ids: List[List[int]] = []
        lengths: List[int] = []

        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                raise InvalidInputError(f"Text at index {i} is empty or whitespace")
            # Encode without padding to get exact unpadded tokens
            try:
                ids = raw_tokenizer.encode(t, add_special_tokens=True)
            except TypeError:
                ids = raw_tokenizer.encode(t)
            if len(ids) > max_length:
                raise InvalidInputError(
                    f"Text at index {i} token length ({len(ids)}) exceeds max_length ({max_length})"
                )
            token_ids.append(ids)
            lengths.append(len(ids))

        return TokenizedBatch(
            token_ids=token_ids,
            lengths=lengths,
            original_indices=list(range(len(texts))),
            pad_token_id=pad_id,
            texts=texts,
        )
