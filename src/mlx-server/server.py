#!/usr/bin/env python3
"""
MLX Embedding Server for Qwen3-Embedding-8B-4bit-DWQ
=====================================================

Apple Silicon-native embedding server using MLX and HuggingFace Transformers.
Provides OpenAI-compatible /v1/embeddings endpoint for QMD integration.

Model: Qwen3-Embedding-8B-4bit-DWQ (mlx-community)
- 4096 dimensions, ~4GB (4-bit DWQ quantization)
- Loads from HuggingFace cache automatically

Usage:
    python3 server.py                      # Default: port 8080
    MLX_PORT=9000 python3 server.py       # Custom port

Endpoints:
    POST /v1/embeddings   OpenAI-compatible embedding endpoint
    GET  /health          Health check + model info
    POST /embed_batch     Raw batch embedding (no OpenAI wrapper)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("qmd.mlx_server")

DEFAULT_MODEL = "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

# =============================================================================
# Model resolution helpers
# =============================================================================

def _cache_dir_for_repo(repo_id: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"


def _find_cached_snapshot(repo_id: str) -> Path | None:
    repo_cache = _cache_dir_for_repo(repo_id)
    snapshots_dir = repo_cache / "snapshots"
    if not snapshots_dir.exists():
        return None

    ref_file = repo_cache / "refs" / "main"
    if ref_file.exists():
        revision = ref_file.read_text().strip()
        if revision:
            candidate = snapshots_dir / revision
            if candidate.exists():
                return candidate

    snapshots = sorted(
        (p for p in snapshots_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def _resolve_model_source(model_ref: str) -> tuple[str, bool]:
    explicit = Path(model_ref).expanduser()
    if explicit.exists():
        return str(explicit), True
    cached = _find_cached_snapshot(model_ref)
    if cached:
        return str(cached), True
    return model_ref, False


def _normalize_inputs(raw: Sequence[str]) -> list[str]:
    texts = [str(t) for t in raw]
    if not texts:
        raise ValueError("at least one input string is required")
    if any(t == "" for t in texts):
        raise ValueError("input strings must be non-empty")
    return texts


# =============================================================================
# Embedding service
# =============================================================================

@dataclass
class EmbeddingBatch:
    embeddings: list[list[float]]
    prompt_tokens: int


class EmbeddingService:
    """
    Loads and runs Qwen3-Embedding-8B-4bit-DWQ via MLX.

    Key design decisions:
    - Uses model.model (the transformer backbone) NOT model (the full LM head).
      model() returns logits (batch, seq, vocab=151665).
      model.model() returns hidden states (batch, seq, hidden=4096) — correct for embeddings.
    - Last-token pooling: takes the hidden state of the last non-padding token.
    - L2 normalization applied to all embeddings.
    - bfloat16 -> float32 conversion before numpy to avoid PEP 3118 buffer errors.
    - mlx_lm TokenizerWrapper.__call__ doesn't support padding kwargs, so we use
      the underlying transformers tokenizer via tokenizer._tokenizer.
    """

    def __init__(self, model_ref: str) -> None:
        self.model_ref = model_ref
        self.status = "loading"
        self.error: str | None = None
        self.embedding_dim: int | None = None
        self.loaded_from_cache = False
        self.resolved_model_path: str | None = None
        self.max_seq_len = 32768
        self.model: Any = None  # Full model (lm head + backbone)
        self.backbone: Any = None  # Transformer backbone only (for hidden states)
        self.tokenizer: Any = None  # TokenizerWrapper (mlx_lm)
        self._encoder_tok: Any = None  # Underlying transformers tokenizer
        self.lock = threading.Lock()
        self.started_at = time.time()

    def start_loading(self) -> None:
        t = threading.Thread(target=self._load, name="mlx-model-loader", daemon=True)
        t.start()

    def _load(self) -> None:
        self.status = "loading"
        self.error = None

        model_source, cached = _resolve_model_source(self.model_ref)
        self.loaded_from_cache = cached
        self.resolved_model_path = model_source

        LOGGER.info("Loading MLX embedding model from %s", model_source)

        try:
            from mlx_lm import load as mlx_load

            # mlx_lm.load returns (model, tokenizer, config_dict)
            self.model, self.tokenizer, config = mlx_load(model_source, return_config=True)

            # The backbone (transformer encoder) produces hidden states of shape (batch, seq, 4096).
            # We use this for embeddings instead of the full LM head.
            self.backbone = getattr(self.model, "model", None)
            if self.backbone is None:
                raise RuntimeError("MLX model does not expose .model attribute (transformer backbone)")

            # mlx_lm's TokenizerWrapper.__call__ doesn't support padding=True etc.
            # The underlying transformers tokenizer does — access via _tokenizer.
            self._encoder_tok = self.tokenizer._tokenizer
            pad_tok = self._encoder_tok.pad_token or self._encoder_tok.eos_token or "<|endoftext|>"
            if self._encoder_tok.pad_token_id is None:
                self._encoder_tok.pad_token = pad_tok
            self._encoder_tok.padding_side = "right"

            # Infer embedding dimension from config (should be 4096 for Qwen3-8B)
            hidden = config.get("hidden_size")
            if isinstance(hidden, int) and hidden > 0:
                self.embedding_dim = hidden

            self.max_seq_len = int(config.get("max_position_embeddings", 32768))

            self.status = "ok"
            LOGGER.info(
                "MLX embedding model ready (dim=%s, cache=%s)",
                self.embedding_dim or "unknown",
                "hit" if cached else "miss",
            )
        except Exception as exc:
            self.status = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Failed to load MLX embedding model")

    def assert_ready(self) -> None:
        if self.status == "error":
            raise RuntimeError(self.error or "model failed to load")
        if self.status != "ok":
            raise RuntimeError("model is still loading")

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        """
        Embed a batch of texts using last-token pooling.

        Pipeline per text:
        1. Tokenize with padding (transformers tokenizer, not TokenizerWrapper)
        2. Forward through model.model (backbone) → (batch, seq, 4096)
        3. Take last non-padding token's hidden state
        4. L2-normalize → 4096-d embedding
        """
        self.assert_ready()

        normalized = _normalize_inputs(texts)
        prompt_tokens = 0
        results: list[list[float] | None] = [None] * len(normalized)

        import mlx.core as mx

        # Group by sequence length for efficient batching
        by_len: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
        for idx, text in enumerate(normalized):
            # Encode: returns dict with input_ids (1, seq) and attention_mask (1, seq)
            enc = self._encoder_tok(
                [text],  # Must be a list, not a plain string
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_seq_len,
                padding=True,  # Pad to longest in batch (only 1 here)
                return_attention_mask=True,
                return_tensors="np",
            )
            ids = np.asarray(enc["input_ids"], dtype=np.int32)[0]  # (seq,)
            mask = np.asarray(enc["attention_mask"], dtype=np.int32)[0]  # (seq,)
            by_len.setdefault(len(ids), []).append((idx, ids, mask))
            prompt_tokens += int(mask.sum())

        with self.lock:
            for seq_len, items in sorted(by_len.items()):
                # Stack to batch: (batch, seq)
                batch_ids = np.stack([ids for _, ids, _ in items])
                batch_masks = np.stack([mask for _, _, mask in items])

                # Forward through backbone (NOT the full model)
                mx_ids = mx.array(batch_ids)  # (batch, seq)
                hidden = self.backbone(mx_ids)  # (batch, seq, 4096)

                # bfloat16 -> float32 conversion (required before np.array to avoid PEP 3118 error)
                hidden_f32 = hidden.astype(mx.float32)
                mx.eval(hidden_f32)
                hidden_np = np.array(hidden_f32)  # (batch, seq, 4096)

                # Last-token pooling: find last non-padding position per sequence
                last_indices = np.array(
                    [int(m.sum()) - 1 for m in batch_masks],
                    dtype=np.int32,
                )
                embeddings = hidden_np[np.arange(len(items)), last_indices, :]

                # L2 normalize
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.clip(norms, 1e-12, None)

                if self.embedding_dim is None:
                    self.embedding_dim = int(embeddings.shape[-1])

                for local_i, (orig_idx, _, _) in enumerate(items):
                    results[orig_idx] = embeddings[local_i].astype(np.float32).tolist()

        return EmbeddingBatch(
            embeddings=[e if e is not None else [] for e in results],
            prompt_tokens=prompt_tokens,
        )


# =============================================================================
# FastAPI app
# =============================================================================

MODEL_REF = os.environ.get("MLX_MODEL_PATH", DEFAULT_MODEL).strip() or DEFAULT_MODEL
HOST = os.environ.get("MLX_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
PORT = int(os.environ.get("MLX_PORT", str(DEFAULT_PORT)))
SERVICE = EmbeddingService(MODEL_REF)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    SERVICE.start_loading()
    yield


app = FastAPI(
    title="QMD MLX Embedding Server",
    version="1.0.0",
    lifespan=lifespan,
)


class EmbedRequest(BaseModel):
    input: str | list[str] = Field(..., description="Single text or list of texts")
    model: str | None = None


class BatchRequest(BaseModel):
    input: list[str] | None = None
    inputs: list[str] | None = None
    model: str | None = None


def _get_inputs(req: EmbedRequest | BatchRequest) -> list[str]:
    if isinstance(req, EmbedRequest):
        return _normalize_inputs([req.input] if isinstance(req.input, str) else req.input)
    if req.input is not None:
        return _normalize_inputs(req.input)
    if req.inputs is not None:
        return _normalize_inputs(req.inputs)
    raise ValueError("body must include 'input' or 'inputs'")


@app.get("/health")
async def health() -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": SERVICE.status,
        "model": SERVICE.model_ref,
        "uptime_seconds": round(time.time() - SERVICE.started_at, 2),
    }
    if SERVICE.embedding_dim is not None:
        out["embedding_dim"] = SERVICE.embedding_dim
    if SERVICE.resolved_model_path:
        out["model_path"] = SERVICE.resolved_model_path
    if SERVICE.loaded_from_cache:
        out["cached"] = True
    if SERVICE.error:
        out["error"] = SERVICE.error
    return out


@app.post("/v1/embeddings")
async def create_embeddings(req: EmbedRequest) -> dict[str, Any]:
    try:
        batch = await asyncio.to_thread(SERVICE.embed_texts, _get_inputs(req))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Embedding failed")
        raise HTTPException(500, str(exc)) from exc

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": emb}
            for i, emb in enumerate(batch.embeddings)
        ],
        "model": req.model or SERVICE.model_ref,
        "usage": {"prompt_tokens": batch.prompt_tokens, "total_tokens": batch.prompt_tokens},
    }


@app.post("/embed_batch")
async def embed_batch(req: BatchRequest) -> dict[str, Any]:
    try:
        batch = await asyncio.to_thread(SERVICE.embed_texts, _get_inputs(req))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Batch embedding failed")
        raise HTTPException(500, str(exc)) from exc

    return {
        "embeddings": batch.embeddings,
        "model": req.model or SERVICE.model_ref,
        "count": len(batch.embeddings),
        "embedding_dim": SERVICE.embedding_dim,
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
