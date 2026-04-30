#!/usr/bin/env python3
"""
QMD MLX Embedding Server — Apple Silicon GPU-Accelerated Embeddings via MLX

A lightweight HTTP server serving MLX-native embedding models for QMD,
giving you full Metal GPU acceleration on Apple Silicon without GGUF overhead.

Usage:
    python scripts/mlx_embed_server.py --model mlx-community/nomic-embed-text-v2-moe --port 8787
    python scripts/mlx_embed_server.py --model mlx-community/Qwen3-Embedding-8B --quantization q4_0

Environment:
    MLX_EMBED_MODEL        — model identifier (HF repo or local path)
    MLX_EMBED_PORT         — port to listen on (default 8787)
    MLX_EMBED_MAX_LENGTH   — max tokens per input (default 512)
    MLX_EMBED_QUANTIZATION — quantization format: bf16, q8_0, q4_0 (default bf16)
    MLX_MAX_BATCH_TOKENS   — max total tokens per GPU forward pass (default 8192)

Endpoints:
    POST /embed       {"texts": [...], "dims": 256, "is_query": false}
    POST /embed-bin   Same as /embed but returns raw float32 bytes (no JSON overhead)
    GET  /health      -> {"status": "ok", "model": ..., "dims": ..., "ready": true}
    GET  /ready       -> {"ready": true|false}
    GET  /memory      -> {"active_memory_mb": ..., "peak_memory_mb": ...}
    GET  /models      -> {"loaded": ..., "available": [...]}

Features:
    - L2-normalized embeddings with mean pooling
    - Matryoshka Representation Learning (MRL): request any dimension <= native
    - Adaptive batch splitting (prevents GPU OOM on large batches)
    - Binary wire format for zero-JSON-overhead bulk transfers
    - Quantization support (bf16, q8_0, q4_0)
    - GPU memory monitoring
"""

import argparse
import http.server
import json
import os
import socketserver
import struct
import time
from typing import Any, Optional

# ── Lazy-load MLX (only if available at startup) ────────────────────────────
_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import numpy as np
    from transformers import AutoTokenizer as _AutoTokenizer
    _MLX_AVAILABLE = True
except ImportError as e:
    print(f"[mlx-server] WARNING: MLX not available ({e})")
    print("[mlx-server] Install: pip install mlx mlx-lm transformers numpy safetensors")
    # Don't raise — let the server start in degraded mode and report errors per-request

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_PORT = int(os.getenv("MLX_EMBED_PORT", "8787"))
DEFAULT_MAX_LENGTH = int(os.getenv("MLX_EMBED_MAX_LENGTH", "512"))
DEFAULT_MODEL = os.getenv("MLX_EMBED_MODEL", "nomic-ai/nomic-embed-text-v2-moe")
DEFAULT_QUANTIZATION = os.getenv("MLX_EMBED_QUANTIZATION", "bf16")
MAX_BATCH_TOKENS = int(os.getenv("MLX_MAX_BATCH_TOKENS", "8192"))

# ── Model Cache ─────────────────────────────────────────────────────────────

_model_cache: dict[str, Any] = {}
_tokenizer_cache: dict[str, Any] = {}
_model_dims: dict[str, int] = {}
_model_is_asymmetric: dict[str, bool] = {}
_model_name: Optional[str] = None
_peak_memory: float = 0.0


def _get_memory_mb() -> float:
    """Get current active MLX GPU memory in megabytes."""
    try:
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            return mx.metal.get_active_memory() / (1024 * 1024)
        if hasattr(mx, "get_active_memory"):
            return mx.get_active_memory() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _update_peak():
    global _peak_memory
    current = _get_memory_mb()
    if current > _peak_memory:
        _peak_memory = current


def _load_model(model_name: str, quantization: str = DEFAULT_QUANTIZATION):
    """Load an MLX embedding model and its tokenizer. Caches globally."""
    global _model_cache, _tokenizer_cache, _model_dims, _model_is_asymmetric, _model_name

    # Cache key includes quantization so different quants don't collide
    cache_key = f"{model_name}__{quantization}"
    if cache_key in _model_cache:
        return _model_cache[cache_key], _tokenizer_cache[cache_key], _model_dims[cache_key]

    if not _MLX_AVAILABLE:
        raise RuntimeError("MLX is not installed. Run: pip install mlx mlx-lm transformers numpy")

    print(f"[mlx-server] Loading {model_name} (quant={quantization})...")
    t0 = time.time()

    # Determine if the model uses asymmetric embeddings (query vs doc prompts differ)
    is_asymmetric = any(tok in model_name.lower() for tok in ["nomic", "qwen", "embeddinggemma", "e5"])

    # ── Path A: Try mlx_lm.load for native MLX models ──
    try:
        import mlx_lm
        from mlx_lm.utils import load as load_mlx

        load_kwargs = {}
        if quantization in ("q4_0", "q4"):
            load_kwargs["quantize"] = True
            load_kwargs["quantization_group_size"] = 64
            load_kwargs["quantization_bits"] = 4
        elif quantization in ("q8_0", "q8"):
            load_kwargs["quantize"] = True
            load_kwargs["quantization_group_size"] = 64
            load_kwargs["quantization_bits"] = 8

        model, tokenizer = load_mlx(model_name, **load_kwargs)

        # Infer dimensions from model config
        config = getattr(model, "config", {})
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            # Probe: run a dummy forward pass
            dummy = mx.zeros((1, 8), dtype=mx.int32)
            out = model(dummy)
            if isinstance(out, tuple):
                out = out[0]
            if hasattr(out, "last_hidden_state"):
                hidden_size = out.last_hidden_state.shape[-1]
            else:
                hidden_size = out.shape[-1]
            del out
            del dummy

        _model_cache[cache_key] = model
        _tokenizer_cache[cache_key] = tokenizer
        _model_dims[cache_key] = int(hidden_size)
        _model_is_asymmetric[cache_key] = is_asymmetric
    except Exception as e1:
        # ── Path B: sentence-transformers fallback ──
        print(f"[mlx-server] mlx_lm load failed ({e1}), trying sentence-transformers...")
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, trust_remote_code=True)
            tokenizer = _AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            dims = model.get_sentence_embedding_dimension()

            _model_cache[cache_key] = model
            _tokenizer_cache[cache_key] = tokenizer
            _model_dims[cache_key] = dims
            _model_is_asymmetric[cache_key] = is_asymmetric
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load {model_name} with both mlx_lm and sentence-transformers.\n"
                f"  mlx_lm error: {e1}\n"
                f"  st error: {e2}\n"
                f"Ensure the model has MLX weights available on HuggingFace."
            ) from e2

    elapsed = time.time() - t0
    _model_name = model_name
    _update_peak()
    print(f"[mlx-server] Loaded {model_name} ({_model_dims[cache_key]}d, {quantization}) in {elapsed:.2f}s")
    return _model_cache[cache_key], _tokenizer, _model_dims[cache_key]


def _get_tokenizer(model_name: str, quantization: str = DEFAULT_QUANTIZATION):
    """Get tokenizer for a model, loading if needed."""
    cache_key = f"{model_name}__{quantization}"
    if cache_key in _tokenizer_cache:
        return _tokenizer_cache[cache_key]
    _, tokenizer, _ = _load_model(model_name, quantization)
    return tokenizer


# ── Embedding Logic ─────────────────────────────────────────────────────────

def _mean_pooling(hidden_states: "mx.array", attention_mask: "mx.array") -> "mx.array":
    """Mean pooling: weighted average over token embeddings by attention mask."""
    mask = mx.expand_dims(attention_mask, -1).astype(mx.float32)
    sum_emb = mx.sum(hidden_states * mask, axis=1)
    sum_mask = mx.clip(mx.sum(mask, axis=1), a_min=1e-9)
    return sum_emb / sum_mask


def _l2_normalize(vectors: "mx.array") -> "mx.array":
    """L2 normalize embeddings to unit vectors."""
    norms = mx.sqrt(mx.sum(vectors ** 2, axis=-1, keepdims=True))
    return vectors / mx.clip(norms, a_min=1e-12)


def _truncate_mrl(vectors: "mx.array", dims: int) -> "mx.array":
    """Matryoshka truncation: slice to requested dims and re-normalize."""
    if vectors.shape[-1] <= dims:
        return vectors
    truncated = vectors[..., :dims]
    return _l2_normalize(truncated)


def _forward_pass(model, input_ids: "mx.array", attention_mask: "mx.array") -> "mx.array":
    """
    Run a forward pass through the model and return the last hidden states.
    Handles multiple output formats (tuple, HF output, raw tensor).
    The output is an mx.array — NOT YET EVALUATED.
    """
    outputs = model(input_ids)

    # Handle sentence-transformers style models
    if hasattr(model, "encode"):
        raise TypeError("sentence_transformers model — use _encode_st() path instead")

    # Handle HuggingFace-style output (BaseModelOutput, CausalLMOutput, etc.)
    if hasattr(outputs, "last_hidden_state"):
        hidden_states = outputs.last_hidden_state
    elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
        hidden_states = outputs.hidden_states[-1]
    elif isinstance(outputs, tuple):
        hidden_states = outputs[0]
    else:
        hidden_states = outputs

    return hidden_states


def _embed_batch(
    model,
    tokenizer,
    texts: list[str],
    native_dims: int,
    max_length: int,
    requested_dims: Optional[int] = None,
    is_query: bool = False,
) -> "mx.array":
    """
    Embed a single (possibly pre-split) batch through MLX.
    Returns evaluated, normalized mx.array of shape (len(texts), actual_dims).
    """
    inputs = tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = mx.array(inputs["input_ids"], dtype=mx.int32)
    attention_mask = mx.array(inputs["attention_mask"], dtype=mx.int32)

    # Handle sentence-transformers models (no native MLX tensor ops)
    if hasattr(model, "encode"):
        embeddings = np.stack([model.encode(t) for t in texts])
        result = mx.array(embeddings)
    else:
        hidden_states = _forward_pass(model, input_ids, attention_mask)
        pooled = _mean_pooling(hidden_states, attention_mask)
        result = _l2_normalize(pooled)

    # CRITICAL: mx.eval() forces execution of the lazy computation graph.
    # Without this, .tolist() may return garbage (all zeros or uninitialized memory).
    mx.eval(result)
    _update_peak()

    # Matryoshka truncation: slice to requested dims
    actual_dims = requested_dims if requested_dims else native_dims
    if actual_dims < result.shape[-1]:
        result = _truncate_mrl(result, actual_dims)
        mx.eval(result)

    return result


def _estimate_tokens(texts: list[str], tokenizer) -> int:
    """Quick token count estimate for batch-split decisions."""
    total = 0
    for text in texts:
        total += len(tokenizer.encode(text))
    return total


def embed_texts(
    texts: list[str],
    model_name: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    quantization: str = DEFAULT_QUANTIZATION,
    requested_dims: Optional[int] = None,
    is_query: bool = False,
) -> list[list[float]]:
    """
    Embed a batch of texts using MLX with adaptive batch splitting.
    Returns a list of float lists.
    """
    cache_key = f"{model_name}__{quantization}"
    model, tokenizer, native_dims = _load_model(model_name, quantization)

    if not texts:
        return []

    # ── Adaptive batch splitting ──────────────────────────────────────────
    total_tokens = _estimate_tokens(texts, tokenizer)
    max_seq_len = max(len(tokenizer.encode(t)) for t in texts)

    # Split if total token budget exceeded or any single seq is too long
    if total_tokens > MAX_BATCH_TOKENS and len(texts) > 1:
        # Split into sub-batches
        all_results: list["mx.array"] = []
        sub_batch: list[str] = []
        sub_tokens = 0

        for text in texts:
            n_tok = len(tokenizer.encode(text))
            if sub_batch and (sub_tokens + n_tok > MAX_BATCH_TOKENS * 0.8 or len(sub_batch) >= 32):
                result = _embed_batch(model, tokenizer, sub_batch, native_dims, max_length, requested_dims, is_query)
                all_results.append(result)
                sub_batch = []
                sub_tokens = 0
            sub_batch.append(text)
            sub_tokens += n_tok

        if sub_batch:
            result = _embed_batch(model, tokenizer, sub_batch, native_dims, max_length, requested_dims, is_query)
            all_results.append(result)

        # Concatenate results
        if len(all_results) == 1:
            combined = all_results[0]
        else:
            combined = mx.concatenate(all_results, axis=0)
            mx.eval(combined)

        vecs = combined.tolist()
        del combined
        return vecs

    # ── Single batch (within budget) ──────────────────────────────────────
    result = _embed_batch(model, tokenizer, texts, native_dims, max_length, requested_dims, is_query)
    vecs = result.tolist()
    del result
    return vecs


# ── HTTP Handler ────────────────────────────────────────────────────────────

class MLXHandler(http.server.BaseHTTPRequestHandler):
    model: str = DEFAULT_MODEL
    max_length: int = DEFAULT_MAX_LENGTH
    quantization: str = DEFAULT_QUANTIZATION
    ready: bool = False

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, embeddings: list[list[float]], dims: int):
        """Send embeddings as raw float32 binary: count(int32) | dims(int32) | data(float32*)"""
        count = len(embeddings)
        header = struct.pack("<ii", count, dims)
        data = b"".join(struct.pack(f"<{dims}f", *vec) for vec in embeddings)
        body = header + data

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # quiet by default

    # ── GET endpoints ─────────────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/health":
            cache_key = f"{self.model}__{self.quantization}"
            self._send_json({
                "status": "ok" if self.ready else "degraded",
                "model": self.model,
                "dims": _model_dims.get(cache_key),
                "ready": self.ready,
            })
        elif self.path == "/ready":
            self._send_json({"ready": self.ready})
        elif self.path == "/memory":
            try:
                active = _get_memory_mb()
                self._send_json({
                    "active_memory_mb": round(active, 1),
                    "peak_memory_mb": round(_peak_memory, 1),
                })
            except Exception:
                self._send_json({"active_memory_mb": 0, "peak_memory_mb": 0})
        elif self.path == "/models":
            self._send_json({
                "loaded": _model_name,
                "quantization": self.quantization,
                "dims": _model_dims.get(f"{self.model}__{self.quantization}"),
            })
        else:
            self._send_json({"error": "Not found"}, 404)

    # ── POST endpoints ────────────────────────────────────────────────────

    def do_POST(self):
        if self.path not in ("/embed", "/embed-bin"):
            self._send_json({"error": "Not found"}, 404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json({"error": "Empty body"}, 400)
            return

        try:
            body = self.rfile.read(content_length)
            payload = json.loads(body)
            texts = payload.get("texts", [])
            if not texts:
                self._send_json({"error": "texts field is required and non-empty"}, 400)
                return
            if not isinstance(texts, list):
                self._send_json({"error": "texts must be a list"}, 400)
                return

            # Check for Matryoshka dimensions override
            requested_dims = payload.get("dims")  # int or None
            if requested_dims is not None:
                requested_dims = int(requested_dims)

            # Check for asymmetric query flag
            is_query = bool(payload.get("is_query", False))

            embeddings = embed_texts(
                texts,
                self.model,
                self.max_length,
                self.quantization,
                requested_dims,
                is_query,
            )

            actual_dims = len(embeddings[0]) if embeddings else (
                requested_dims if requested_dims else 0
            )

            if self.path == "/embed-bin":
                self._send_binary(embeddings, actual_dims)
            else:
                self._send_json({
                    "embeddings": embeddings,
                    "model": self.model,
                    "dims": actual_dims,
                })

        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
        except Exception as e:
            import traceback
            print(f"[mlx-server] Error during embed:", file=__import__('sys').stderr)
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QMD MLX Embedding Server")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID or local path")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max tokens per input")
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION,
                        choices=["bf16", "fp16", "q8_0", "q4_0", "q8", "q4"],
                        help="Model quantization (default: bf16)")
    parser.add_argument("--preload", action="store_true", help="Preload model at startup (recommended)")
    args = parser.parse_args()

    MLXHandler.model = args.model
    MLXHandler.max_length = args.max_length
    MLXHandler.quantization = args.quantization

    if args.preload:
        print(f"[mlx-server] Pre-loading {args.model} (--quantization {args.quantization})...")
        try:
            _, _, dims = _load_model(args.model, args.quantization)
            MLXHandler.ready = True
            print(f"[mlx-server] Model ready ✓ ({dims}d, {_get_memory_mb():.0f} MB GPU)")
        except Exception as e:
            print(f"[mlx-server] Pre-load FAILED: {e}")
            MLXHandler.ready = False
    else:
        MLXHandler.ready = True  # Lazy-load on first request

    server = ThreadedHTTPServer(("127.0.0.1", args.port), MLXHandler)
    print(f"[mlx-server] Listening → http://127.0.0.1:{args.port}")
    print(f"[mlx-server] Model: {args.model} | Quant: {args.quantization} | Max len: {args.max_length}")
    print(f"[mlx-server] Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mlx-server] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
