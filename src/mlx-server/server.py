#!/usr/bin/env python3
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
from transformers import AutoTokenizer

LOGGER = logging.getLogger("qmd.mlx_server")

DEFAULT_MODEL = "mlx-community/Qwen3-Embedding-8B-4bit-DWQ"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", HF_HOME / "hub"))


def cache_dir_for_repo(repo_id: str) -> Path:
    return HF_HUB_CACHE / f"models--{repo_id.replace('/', '--')}"


def find_cached_snapshot(repo_id: str) -> Path | None:
    repo_cache_dir = cache_dir_for_repo(repo_id)
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    ref_file = repo_cache_dir / "refs" / "main"
    if ref_file.exists():
        revision = ref_file.read_text().strip()
        if revision:
            candidate = snapshots_dir / revision
            if candidate.exists():
                return candidate

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return None

    return sorted(snapshots, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def resolve_model_source(model_ref: str) -> tuple[str, bool]:
    explicit_path = Path(model_ref).expanduser()
    if explicit_path.exists():
        return str(explicit_path), True

    cached_snapshot = find_cached_snapshot(model_ref)
    if cached_snapshot is not None:
        return str(cached_snapshot), True

    return model_ref, False


def normalize_inputs(raw_inputs: Sequence[str]) -> list[str]:
    inputs = [text if isinstance(text, str) else str(text) for text in raw_inputs]
    if not inputs:
        raise ValueError("at least one input string is required")
    if any(text == "" for text in inputs):
        raise ValueError("input strings must be non-empty")
    return inputs


@dataclass
class EmbeddingBatch:
    embeddings: list[list[float]]
    prompt_tokens: int


class EmbeddingService:
    def __init__(self, model_ref: str) -> None:
        self.model_ref = model_ref
        self.status = "loading"
        self.error: str | None = None
        self.embedding_dim: int | None = None
        self.loaded_from_cache = False
        self.resolved_model_path: str | None = None
        self.max_length = 40960
        self.model: Any = None
        self.encoder: Any = None
        self.tokenizer: Any = None
        self.lock = threading.Lock()
        self.started_at = time.time()

    def start_loading(self) -> None:
        thread = threading.Thread(target=self.load, name="mlx-model-loader", daemon=True)
        thread.start()

    def load(self) -> None:
        self.status = "loading"
        self.error = None

        model_source, loaded_from_cache = resolve_model_source(self.model_ref)
        self.loaded_from_cache = loaded_from_cache
        self.resolved_model_path = model_source

        LOGGER.info("Loading MLX embedding model from %s", model_source)

        try:
            from mlx_lm import load as mlx_load

            model, _, config = mlx_load(model_source, return_config=True)
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=True,
                local_files_only=loaded_from_cache,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            tokenizer.padding_side = "right"

            encoder = getattr(model, "model", None)
            if encoder is None:
                raise RuntimeError("loaded MLX model does not expose encoder hidden states")

            self.model = model
            self.encoder = encoder
            self.tokenizer = tokenizer
            self.max_length = int(
                config.get("max_position_embeddings")
                or getattr(tokenizer, "model_max_length", 40960)
            )

            hidden_size = config.get("hidden_size")
            if isinstance(hidden_size, int) and hidden_size > 0:
                self.embedding_dim = hidden_size

            self.status = "ok"
            LOGGER.info(
                "MLX embedding model ready (dim=%s, cache=%s)",
                self.embedding_dim or "unknown",
                "hit" if loaded_from_cache else "miss",
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
        if self.encoder is None or self.tokenizer is None:
            raise RuntimeError("model is not fully initialized")

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.assert_ready()

        prepared: list[tuple[np.ndarray, np.ndarray]] = []
        prompt_tokens = 0

        for text in normalize_inputs(texts):
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
            )
            input_ids = np.asarray(encoded["input_ids"], dtype=np.int32).reshape(-1)
            attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int32).reshape(-1)
            if input_ids.size == 0:
                raise ValueError("input strings must produce at least one token")
            prepared.append((input_ids, attention_mask))
            prompt_tokens += int(attention_mask.sum())

        results: list[list[float] | None] = [None] * len(prepared)

        import mlx.core as mx

        grouped: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
        for index, (input_ids, attention_mask) in enumerate(prepared):
            grouped.setdefault(int(input_ids.shape[0]), []).append((index, input_ids, attention_mask))

        with self.lock:
            for _, items in sorted(grouped.items(), key=lambda entry: entry[0]):
                batch_input_ids = np.stack([input_ids for _, input_ids, _ in items], axis=0)
                hidden_states = self.encoder(mx.array(batch_input_ids))
                mx.eval(hidden_states)
                hidden_np = np.array(hidden_states)

                last_indices = np.array(
                    [int(attention_mask.sum()) - 1 for _, _, attention_mask in items],
                    dtype=np.int32,
                )
                batch_embeddings = hidden_np[np.arange(len(items)), last_indices, :]
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / np.clip(norms, 1e-12, None)

                if self.embedding_dim is None:
                    self.embedding_dim = int(batch_embeddings.shape[-1])

                for batch_index, (original_index, _, _) in enumerate(items):
                    results[original_index] = batch_embeddings[batch_index].astype(np.float32).tolist()

        return EmbeddingBatch(
            embeddings=[embedding if embedding is not None else [] for embedding in results],
            prompt_tokens=prompt_tokens,
        )


class OpenAIEmbeddingRequest(BaseModel):
    input: str | list[str] = Field(..., description="Input text or list of input texts")
    model: str | None = Field(default=None, description="Model identifier")


class BatchEmbeddingRequest(BaseModel):
    input: list[str] | None = None
    inputs: list[str] | None = None
    model: str | None = None


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


def get_request_inputs(request: OpenAIEmbeddingRequest | BatchEmbeddingRequest) -> list[str]:
    if isinstance(request, OpenAIEmbeddingRequest):
        if isinstance(request.input, str):
            return normalize_inputs([request.input])
        return normalize_inputs(request.input)

    if request.input is not None:
        return normalize_inputs(request.input)
    if request.inputs is not None:
        return normalize_inputs(request.inputs)

    raise ValueError("request body must include 'input' or 'inputs'")


def build_openai_response(model_name: str, batch: EmbeddingBatch) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": embedding,
            }
            for index, embedding in enumerate(batch.embeddings)
        ],
        "model": model_name,
        "usage": {
            "prompt_tokens": batch.prompt_tokens,
            "total_tokens": batch.prompt_tokens,
        },
    }


async def run_embeddings(inputs: list[str], model_name: str) -> dict[str, Any]:
    try:
        batch = await asyncio.to_thread(SERVICE.embed_texts, inputs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Embedding request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return build_openai_response(model_name, batch)


@app.get("/health")
async def health() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": SERVICE.status,
        "model": SERVICE.model_ref,
        "uptime_seconds": round(time.time() - SERVICE.started_at, 2),
    }
    if SERVICE.embedding_dim is not None:
        payload["embedding_dim"] = SERVICE.embedding_dim
    if SERVICE.resolved_model_path is not None:
        payload["model_path"] = SERVICE.resolved_model_path
    if SERVICE.loaded_from_cache:
        payload["cached"] = True
    if SERVICE.error is not None:
        payload["error"] = SERVICE.error
    return payload


@app.post("/v1/embeddings")
async def create_embeddings(request: OpenAIEmbeddingRequest) -> dict[str, Any]:
    inputs = get_request_inputs(request)
    return await run_embeddings(inputs, request.model or SERVICE.model_ref)


@app.post("/embed_batch")
async def embed_batch(request: BatchEmbeddingRequest) -> dict[str, Any]:
    inputs = get_request_inputs(request)
    response = await run_embeddings(inputs, request.model or SERVICE.model_ref)
    response["embedding_dim"] = SERVICE.embedding_dim
    return response


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level=os.environ.get("LOG_LEVEL", "info"))
