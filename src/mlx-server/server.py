#!/usr/bin/env python3
"""
MLX Server for QMD — Embedding, Reranking & Generation
======================================================
Apple Silicon-native server using MLX and mlx_lm.
Provides OpenAI-compatible endpoints for QMD integration.

Models:
  Embedding:  Qwen3-Embedding-8B-4bit-DWQ  (4096 dims, ~4GB)
  Reranker:   Qwen3-Reranker-8B-mxfp8       (~7.8GB)
  Generation: Qwen3-8B-MLX-4bit             (~4.3GB)

Usage:
    python3 server.py                      # Default: port 8080
    MLX_PORT=9000 python3 server.py        # Custom port

Endpoints:
    POST /v1/embeddings       OpenAI-compatible embedding endpoint
    POST /v1/rerank           Rerank documents by relevance
    POST /v1/generate         Text generation (raw prompt)
    POST /v1/chat/completions Chat-style generation (OpenAI-compatible)
    GET  /health              Health check + model status
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("qmd.mlx_server")

# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_EMBED_MODEL = "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"
DEFAULT_RERANK_MODEL = "mlx-community/Qwen3-Reranker-8B-mxfp8"
DEFAULT_GENERATE_MODEL = "Qwen/Qwen3-8B-MLX-4bit"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
MAX_EMBED_INPUT_SIZE = 128          # Max texts per embedding request
MAX_CONCURRENT_EMBED_REQUESTS = 2   # Queue depth limit — prevents thread pool starvation
EMBED_REQUEST_TIMEOUT_S = 300       # 5 min timeout for a single embed request

# ─── Model resolution helpers ─────────────────────────────────────────────────

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


# ─── Embedding Service ────────────────────────────────────────────────────────

@dataclass
class EmbeddingBatch:
    embeddings: list[list[float]]
    prompt_tokens: int


class EmbeddingService:
    """
    Loads and runs Qwen3-Embedding-8B-4bit-DWQ via MLX.

    Uses model.model (transformer backbone) for hidden states,
    NOT the full LM head. Last-token pooling + L2 normalization.
    """

    def __init__(self, model_ref: str) -> None:
        self.model_ref = model_ref
        self.status = "loading"
        self.error: str | None = None
        self.embedding_dim: int | None = None
        self.loaded_from_cache = False
        self.resolved_model_path: str | None = None
        self.max_seq_len = 32768
        self.model: Any = None
        self.backbone: Any = None
        self.tokenizer: Any = None
        self._encoder_tok: Any = None
        self.lock = threading.Lock()
        self.started_at = time.time()
        # Thermal throttle detection: track per-batch latency
        self._baseline_latency_s: float | None = None
        self._thermal_pause_s = 0.5  # Pause duration when throttled

    def start_loading(self) -> None:
        t = threading.Thread(target=self._load, name="mlx-embed-loader", daemon=True)
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
            self.model, self.tokenizer, config = mlx_load(model_source, return_config=True)
            self.backbone = getattr(self.model, "model", None)
            if self.backbone is None:
                raise RuntimeError("MLX model does not expose .model attribute (transformer backbone)")
            self._encoder_tok = self.tokenizer._tokenizer
            pad_tok = self._encoder_tok.pad_token or self._encoder_tok.eos_token or ""
            if self._encoder_tok.pad_token_id is None:
                self._encoder_tok.pad_token = pad_tok
            self._encoder_tok.padding_side = "right"
            hidden = config.get("hidden_size")
            if isinstance(hidden, int) and hidden > 0:
                self.embedding_dim = hidden
            self.max_seq_len = int(config.get("max_position_embeddings", 32768))
            self.status = "ok"
            LOGGER.info("MLX embedding model ready (dim=%s, cache=%s)", self.embedding_dim or "unknown", "hit" if cached else "miss")
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
        self.assert_ready()
        normalized = _normalize_inputs(texts)
        prompt_tokens = 0
        results: list[list[float] | None] = [None] * len(normalized)
        import gc
        import mlx.core as mx

        # Tokenize all texts individually (with truncation) then collect lengths
        tokenized: list[tuple[int, np.ndarray, np.ndarray]] = []
        for idx, text in enumerate(normalized):
            enc = self._encoder_tok(
                [text],
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_seq_len,
                padding=True,
                return_attention_mask=True,
                return_tensors="np",
            )
            ids = np.asarray(enc["input_ids"], dtype=np.int32)[0]
            mask = np.asarray(enc["attention_mask"], dtype=np.int32)[0]
            prompt_tokens += int(mask.sum())
            tokenized.append((idx, ids, mask))

        # Single padded forward pass — much faster than grouping by length.
        # Pad all token IDs to the max length in this batch, right-pad with 0s.
        max_len = max(len(ids) for _, ids, _ in tokenized)
        batch_size = len(tokenized)

        batch_ids = np.zeros((batch_size, max_len), dtype=np.int32)
        batch_masks = np.zeros((batch_size, max_len), dtype=np.int32)
        last_indices = np.zeros(batch_size, dtype=np.int32)

        for i, (_, ids, mask) in enumerate(tokenized):
            seq_len = len(ids)
            batch_ids[i, :seq_len] = ids
            batch_masks[i, :seq_len] = mask
            # Last valid token index = position of last non-padding token
            last_indices[i] = int(mask.sum()) - 1

        with self.lock:
            t0 = time.monotonic()
            mx_ids = mx.array(batch_ids)
            hidden = self.backbone(mx_ids)
            hidden_f32 = hidden.astype(mx.float32)
            mx.eval(hidden_f32)
            hidden_np = np.array(hidden_f32)

            # Explicitly free MLX intermediate tensors to prevent
            # gradual memory growth over long embed runs
            del mx_ids, hidden, hidden_f32, batch_ids, batch_masks
            mx.eval()  # evaluate any remaining lazy ops
            gc.collect()

            # Extract the last valid token's hidden state for each text
            embeddings = hidden_np[np.arange(batch_size), last_indices, :]
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-12, None)
            if self.embedding_dim is None:
                self.embedding_dim = int(embeddings.shape[-1])
            for i, (orig_idx, _, _) in enumerate(tokenized):
                results[orig_idx] = embeddings[i].astype(np.float32).tolist()

            elapsed = time.monotonic() - t0

        # ── Thermal throttle detection ──────────────────────────────────
        # Measure latency per batch. If >2x the baseline (first successful
        # batch), Apple Silicon likely hit thermal limits. Pause briefly so
        # the SoC can cool down and avoid cascading slowness.
        if self._baseline_latency_s is None:
            self._baseline_latency_s = elapsed
            LOGGER.info("Embed baseline latency: %.3fs for %d texts", elapsed, batch_size)
        elif elapsed > self._baseline_latency_s * 2:
            LOGGER.warning(
                "Thermal throttle detected: %.3fs > 2x baseline %.3fs — pausing %.1fs",
                elapsed, self._baseline_latency_s, self._thermal_pause_s,
            )
            time.sleep(self._thermal_pause_s)

        return EmbeddingBatch(
            embeddings=[e if e is not None else [] for e in results],
            prompt_tokens=prompt_tokens,
        )


# ─── Reranker Service ─────────────────────────────────────────────────────────

class RerankerService:
    """
    Loads Qwen3-Reranker-8B-mxfp8 via MLX.

    Uses the model's built-in relevance scoring — feeds query+document
    pairs and extracts yes/no logit probabilities as relevance scores.
    """

    def __init__(self, model_ref: str) -> None:
        self.model_ref = model_ref
        self.status = "not_loaded"
        self.error: str | None = None
        self.loaded_from_cache = False
        self.resolved_model_path: str | None = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.lock = threading.Lock()
        self._load_lock = threading.Lock()  # Prevents concurrent start_loading() calls
        self._ready_event = threading.Event()   # Set when loading completes (ok or error)
        self.started_at = time.time()

    def start_loading(self) -> None:
        # Prevent concurrent start_loading() calls (TOCTOU race fix)
        if not self._load_lock.acquire(blocking=False):
            LOGGER.info("Model %s already loading, skipping duplicate start", self.model_ref)
            return
        try:
            t = threading.Thread(target=self._load, name="mlx-rerank-loader", daemon=True)
            t.start()
        finally:
            self._load_lock.release()

    def _load(self) -> None:
        self.status = "loading"
        self.error = None
        model_source, cached = _resolve_model_source(self.model_ref)
        self.loaded_from_cache = cached
        self.resolved_model_path = model_source
        LOGGER.info("Loading MLX reranker model from %s", model_source)

        try:
            from mlx_lm import load as mlx_load
            self.model, self.tokenizer = mlx_load(model_source)
            self.status = "ok"
            LOGGER.info("MLX reranker model ready (cache=%s)", "hit" if cached else "miss")
        except Exception as exc:
            self.status = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Failed to load MLX reranker model")
        finally:
            self._ready_event.set()

    def wait_ready(self, timeout: float = 300) -> bool:
        """Wait for model to finish loading. Returns True if ready, False on timeout."""
        return self._ready_event.wait(timeout=timeout)

    def assert_ready(self) -> None:
        if self.status == "error":
            raise RuntimeError(self.error or "reranker model failed to load")
        if self.status != "ok":
            raise RuntimeError(f"reranker model is {self.status}")

    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        """
        Score query-document pairs using the Qwen3 Reranker.

        The reranker uses a token-level scoring approach:
        Format each pair as: "<|im_start|>user\n{instruction}\n{doc}<|im_end|><|im_start|>assistant\n"
        Then extract the logit for "yes" token as relevance score.
        """
        self.assert_ready()
        import mlx.core as mx
        from mlx_lm.utils import TokenizerWrapper

        results = []

        # Get token IDs for yes/no
        yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        no_id = self.tokenizer.convert_tokens_to_ids("no")

        instruction = f"Instruct: Given a query, determine the relevance of each document to the query.\nQuery: {query}"

        with self.lock:
            for idx, doc_text in enumerate(documents):
                prompt = f"<|im_start|>user\n{instruction}\n{doc_text}<|im_end|>\n<|im_start|>assistant\n"

                tokens = self.tokenizer.encode(prompt)
                mx_tokens = mx.array([tokens])
                logits = self.model(mx_tokens)
                # Get last token logits, softmax
                last_logits = logits[0, -1, :].astype(mx.float32)
                mx.eval(last_logits)

                # Extract yes/no logit scores and compute probability
                yes_logit = float(last_logits[yes_id])
                no_logit = float(last_logits[no_id])

                # Sigmoid-like scoring: relevance = exp(yes) / (exp(yes) + exp(no))
                import math
                max_logit = max(yes_logit, no_logit)
                exp_yes = math.exp(yes_logit - max_logit)
                exp_no = math.exp(no_logit - max_logit)
                score = exp_yes / (exp_yes + exp_no) if (exp_yes + exp_no) > 0 else 0.5

                results.append({
                    "index": idx,
                    "relevance_score": score,
                    "text": doc_text[:200],
                })

        return results


# ─── Generation Service ────────────────────────────────────────────────────────

class GenerateService:
    """
    Loads Qwen3-8B-MLX-4bit for query expansion / text generation.
    """

    def __init__(self, model_ref: str) -> None:
        self.model_ref = model_ref
        self.status = "not_loaded"
        self.error: str | None = None
        self.loaded_from_cache = False
        self.resolved_model_path: str | None = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.lock = threading.Lock()
        self._load_lock = threading.Lock()  # Prevents concurrent start_loading() calls
        self._ready_event = threading.Event()   # Set when loading completes (ok or error)
        self.started_at = time.time()

    def start_loading(self) -> None:
        # Prevent concurrent start_loading() calls (TOCTOU race fix)
        if not self._load_lock.acquire(blocking=False):
            LOGGER.info("Model %s already loading, skipping duplicate start", self.model_ref)
            return
        try:
            t = threading.Thread(target=self._load, name="mlx-generate-loader", daemon=True)
            t.start()
        finally:
            self._load_lock.release()

    def _load(self) -> None:
        self.status = "loading"
        self.error = None
        model_source, cached = _resolve_model_source(self.model_ref)
        self.loaded_from_cache = cached
        self.resolved_model_path = model_source
        LOGGER.info("Loading MLX generate model from %s", model_source)

        try:
            from mlx_lm import load as mlx_load
            self.model, self.tokenizer = mlx_load(model_source)
            self.status = "ok"
            LOGGER.info("MLX generate model ready (cache=%s)", "hit" if cached else "miss")
        except Exception as exc:
            self.status = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Failed to load MLX generate model")
        finally:
            self._ready_event.set()

    def wait_ready(self, timeout: float = 300) -> bool:
        """Wait for model to finish loading. Returns True if ready, False on timeout."""
        return self._ready_event.wait(timeout=timeout)

    def assert_ready(self) -> None:
        if self.status == "error":
            raise RuntimeError(self.error or "generate model failed to load")
        if self.status != "ok":
            raise RuntimeError(f"generate model is {self.status}")

    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> dict:
        """Generate text from a raw prompt."""
        self.assert_ready()
        import mlx.core as mx

        from mlx_lm import generate as mlx_generate

        with self.lock:
            response = mlx_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )

        return {
            "text": response,
            "model": self.model_ref,
            "done": True,
        }

    def chat_completion(self, messages: list[dict], max_tokens: int = 600, temperature: float = 0.7) -> dict:
        """Generate text from chat-style messages."""
        self.assert_ready()

        # Format messages into prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)

        import mlx.core as mx
        from mlx_lm import generate as mlx_generate

        with self.lock:
            response = mlx_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )

        return {
            "choices": [{"message": {"content": response}}],
            "model": self.model_ref,
        }


# ─── FastAPI App ───────────────────────────────────────────────────────────────

EMBED_MODEL = os.environ.get("MLX_MODEL_PATH", DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL
RERANK_MODEL = os.environ.get("MLX_RERANK_MODEL", DEFAULT_RERANK_MODEL).strip() or DEFAULT_RERANK_MODEL
GENERATE_MODEL = os.environ.get("MLX_GENERATE_MODEL", DEFAULT_GENERATE_MODEL).strip() or DEFAULT_GENERATE_MODEL
HOST = os.environ.get("MLX_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
PORT = int(os.environ.get("MLX_PORT", str(DEFAULT_PORT)))

# Lazy loading flags — models load on first request to save memory
EMBED_SERVICE = EmbeddingService(EMBED_MODEL)
RERANK_SERVICE = RerankerService(RERANK_MODEL)
GENERATE_SERVICE = GenerateService(GENERATE_MODEL)

# Start embedding model immediately (needed for qmd embed)
# Reranker and generator load lazily on first request
EMBED_SERVICE.start_loading()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    yield


app = FastAPI(
    title="QMD MLX Server",
    version="2.0.0",
    lifespan=lifespan,
)


# ─── Request/Response Models ──────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    input: str | list[str] = Field(..., description="Single text or list of texts")
    model: str | None = None

    @property
    def input_count(self) -> int:
        return 1 if isinstance(self.input, str) else len(self.input)


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    model: str | None = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict] = Field(..., description="Chat messages")
    max_tokens: int = 600
    temperature: float = 0.7


# ─── Concurrency limiter ─────────────────────────────────────────────────────
_embed_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBED_REQUESTS)

# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, Any]:
    models = {
        "embed": EMBED_SERVICE.status,
        "rerank": RERANK_SERVICE.status,
        "generate": GENERATE_SERVICE.status,
    }
    out: dict[str, Any] = {
        "status": EMBED_SERVICE.status,
        "models": models,
    }
    if EMBED_SERVICE.embedding_dim is not None:
        out["embedding_dim"] = EMBED_SERVICE.embedding_dim
    if EMBED_SERVICE.resolved_model_path:
        out["model_path"] = EMBED_SERVICE.resolved_model_path
    if EMBED_SERVICE.loaded_from_cache:
        out["cached"] = True
    if EMBED_SERVICE.error:
        out["error"] = EMBED_SERVICE.error
    return out


class TokenizeRequest(BaseModel):
    text: str | list[str] = Field(..., description="Text or list of texts to tokenize")
    model: str | None = None


@app.post("/v1/tokenize")
async def tokenize_texts(req: TokenizeRequest) -> dict[str, Any]:
    """Tokenize text using the embedding model's tokenizer.

    Returns token counts for accurate chunking — replaces the
    pseudo-tokenization (text.length/4) in the TS client.
    """
    texts = [req.text] if isinstance(req.text, str) else req.text
    if not texts:
        raise HTTPException(400, "At least one text is required")

    if EMBED_SERVICE.status != "ok":
        raise HTTPException(503, f"Embedding model not ready (status={EMBED_SERVICE.status})")

    try:
        results = await asyncio.to_thread(_tokenize_sync, texts)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    return {
        "object": "list",
        "data": [{"index": i, "tokens": t} for i, t in enumerate(results)],
        "model": req.model or EMBED_SERVICE.model_ref,
    }


def _tokenize_sync(texts: list[str]) -> list[list[int]]:
    """Synchronous tokenizer call — runs in thread pool."""
    tok = EMBED_SERVICE._encoder_tok
    if tok is None:
        raise RuntimeError("Tokenizer not loaded")
    results: list[list[int]] = []
    for text in texts:
        enc = tok(
            [text],
            add_special_tokens=False,
            truncation=True,
            max_length=EMBED_SERVICE.max_seq_len,
        )
        ids = enc["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        results.append(ids[0] if isinstance(ids, list) and len(ids) > 0 else list(ids))
    return results


@app.post("/v1/embeddings")
async def create_embeddings(req: EmbedRequest) -> dict[str, Any]:
    if req.input_count > MAX_EMBED_INPUT_SIZE:
        raise HTTPException(400, f"Too many inputs: {req.input_count} (max {MAX_EMBED_INPUT_SIZE})")

    async with _embed_semaphore:
        try:
            batch = await asyncio.wait_for(
                asyncio.to_thread(EMBED_SERVICE.embed_texts, _normalize_inputs([req.input] if isinstance(req.input, str) else req.input)),
                timeout=EMBED_REQUEST_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, f"Embedding request timed out after {EMBED_REQUEST_TIMEOUT_S}s")
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
        "model": req.model or EMBED_SERVICE.model_ref,
        "usage": {"prompt_tokens": batch.prompt_tokens, "total_tokens": batch.prompt_tokens},
    }


@app.post("/v1/rerank")
async def rerank_documents(req: RerankRequest) -> dict[str, Any]:
    # Lazy-load reranker on first request — wait for load to complete
    if RERANK_SERVICE.status == "not_loaded":
        RERANK_SERVICE.start_loading()
    if RERANK_SERVICE.status == "loading":
        loaded = await asyncio.to_thread(RERANK_SERVICE.wait_ready, 300)
        if not loaded:
            raise HTTPException(504, "Reranker model loading timed out")

    try:
        results = await asyncio.to_thread(RERANK_SERVICE.rerank, req.query, req.documents)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Reranking failed")
        raise HTTPException(500, str(exc)) from exc

    return {
        "results": results,
        "model": req.model or RERANK_SERVICE.model_ref,
    }


@app.post("/v1/generate")
async def generate_text(req: GenerateRequest) -> dict[str, Any]:
    # Lazy-load generator on first request — wait for load to complete
    if GENERATE_SERVICE.status == "not_loaded":
        GENERATE_SERVICE.start_loading()
    if GENERATE_SERVICE.status == "loading":
        loaded = await asyncio.to_thread(GENERATE_SERVICE.wait_ready, 300)
        if not loaded:
            raise HTTPException(504, "Generate model loading timed out")

    try:
        result = await asyncio.to_thread(GENERATE_SERVICE.generate, req.prompt, req.max_tokens, req.temperature)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Generation failed")
        raise HTTPException(500, str(exc)) from exc

    return result


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> dict[str, Any]:
    # Lazy-load generator on first request — wait for load to complete
    if GENERATE_SERVICE.status == "not_loaded":
        GENERATE_SERVICE.start_loading()
    if GENERATE_SERVICE.status == "loading":
        loaded = await asyncio.to_thread(GENERATE_SERVICE.wait_ready, 300)
        if not loaded:
            raise HTTPException(504, "Generate model loading timed out")

    try:
        result = await asyncio.to_thread(GENERATE_SERVICE.chat_completion, req.messages, req.max_tokens, req.temperature)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Chat completion failed")
        raise HTTPException(500, str(exc)) from exc

    return result


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )