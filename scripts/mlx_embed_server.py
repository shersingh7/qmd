#!/usr/bin/env python3
"""
QMD MLX Embedding Server

A lightweight HTTP server that serves MLX embedding models so QMD can use
Apple Silicon GPU acceleration via native MLX (not GGUF/llama.cpp).

Usage:
    python scripts/mlx_embed_server.py --model nomic-ai/nomic-embed-text-v2-moe --port 8787
    python scripts/mlx_embed_server.py --model mlx-community/nomicai-modernbert-embed-base-bf16 --port 8787

Environment:
    MLX_EMBED_MODEL -- model identifier (HF repo or local path)
    MLX_EMBED_PORT  -- port to listen on (default 8787)
    MLX_EMBED_MAX_LENGTH -- max tokens per input (default 512)

Endpoints:
    POST /embed   {"texts": [...]}  -> {"embeddings": [...], "model": ..., "dims": ...}
    GET  /health  -> {"status": "ok", "model": ..., "dims": ...}
    GET  /ready   -> {"ready": true|false}

The server normalizes embeddings (L2 norm) and batches inputs efficiently.
"""

import argparse
import asyncio
import http.server
import json
import os
import socketserver
import threading
import time
from typing import Any, Optional

try:
    import mlx.core as mx
    from transformers import AutoTokenizer
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install mlx-embeddings transformers numpy")
    raise

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_PORT = int(os.getenv("MLX_EMBED_PORT", "8787"))
DEFAULT_MAX_LENGTH = int(os.getenv("MLX_EMBED_MAX_LENGTH", "512"))
DEFAULT_MODEL = os.getenv("MLX_EMBED_MODEL", "nomic-ai/nomic-embed-text-v2-moe")

# ── Model Cache ─────────────────────────────────────────────────────────────

_model_cache: dict[str, Any] = {}
_tokenizer_cache: dict[str, Any] = {}
_model_dims: dict[str, int] = {}
_model_name: Optional[str] = None


def _load_model(model_name: str):
    """Load an MLX embedding model and its tokenizer."""
    global _model_cache, _tokenizer_cache, _model_dims, _model_name

    if model_name in _model_cache:
        return _model_cache[model_name], _tokenizer_cache[model_name], _model_dims[model_name]

    print(f"[mlx-server] Loading model {model_name}...")
    t0 = time.time()

    try:
        # Try mlx-embeddings style loader first (handles MLX-optimized models)
        import mlx_lm
        from mlx_lm.utils import load as load_mlx

        model, tokenizer = load_mlx(model_name)
        _model_cache[model_name] = model
        _tokenizer_cache[model_name] = tokenizer

        # Infer dimensions from model config
        config = getattr(model, "config", {})
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            # Fallback: inspect first output
            dummy = mx.zeros((1, 16))
            with mx.stream:
                out = model(dummy)
            if isinstance(out, tuple):
                out = out[0]
            hidden_size = out.shape[-1]
        _model_dims[model_name] = hidden_size
    except Exception:
        # Fallback: use transformers with MLX backend
        from sentence_transformers import SentenceTransformer

        # For MLX models on HF, load with trust_remote_code if needed
        model = SentenceTransformer(model_name, trust_remote_code=True)
        # Wrap to expose MLX-compatible interface
        _model_cache[model_name] = model
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _tokenizer_cache[model_name] = tokenizer
        _model_dims[model_name] = model.get_sentence_embedding_dimension()

    elapsed = time.time() - t0
    print(f"[mlx-server] Loaded {model_name} ({_model_dims[model_name]}d) in {elapsed:.2f}s")
    _model_name = model_name
    return _model_cache[model_name], _tokenizer, _model_dims[model_name]


# ── Embedding Logic ─────────────────────────────────────────────────────────

def _mean_pooling(hidden_states, attention_mask):
    """Mean pooling over token embeddings weighted by attention mask."""
    mask = mx.expand_dims(attention_mask, axis=-1).astype(mx.float32)
    sum_emb = mx.sum(hidden_states * mask, axis=1)
    sum_mask = mx.clip(mx.sum(mask, axis=1), a_min=1e-9)
    return sum_emb / sum_mask


def _normalize(vectors):
    """L2 normalize embeddings."""
    norms = mx.sqrt(mx.sum(vectors ** 2, axis=-1, keepdims=True))
    return vectors / mx.clip(norms, a_min=1e-12)


def embed_texts(texts: list[str], model_name: str, max_length: int = DEFAULT_MAX_LENGTH) -> list[list[float]]:
    """Embed a batch of texts using MLX. Returns a list of float lists."""
    model, tokenizer, dims = _load_model(model_name)

    inputs = tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # Convert to MLX arrays
    input_ids = mx.array(inputs["input_ids"])
    attention_mask = mx.array(inputs["attention_mask"])

    # Forward pass
    with mx.stream:
        outputs = model(input_ids)
        # Handle sentence-transformers style models
        if hasattr(model, "encode"):
            # It's a SentenceTransformer — use its encode
            embeddings = np.stack([model.encode(t) for t in texts])
            return [e.tolist() for e in embeddings]

        # Standard MLX HF model output
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        pooled = _mean_pooling(hidden_states, attention_mask)
        normed = _normalize(pooled)

    # Evaluate to CPU
    result = normed.tolist()
    return result


# ── HTTP Handler ────────────────────────────────────────────────────────────

class MLXHandler(http.server.BaseHTTPRequestHandler):
    model: str = DEFAULT_MODEL
    max_length: int = DEFAULT_MAX_LENGTH
    ready: bool = False

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Override to disable default logging; uncomment if you want it
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send_json({
                "status": "ok",
                "model": self.model,
                "dims": _model_dims.get(self.model, None),
                "ready": self.ready,
            })
        elif self.path == "/ready":
            self._send_json({"ready": self.ready})
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path != "/embed":
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

            embeddings = embed_texts(texts, self.model, self.max_length)
            self._send_json({
                "embeddings": embeddings,
                "model": self.model,
                "dims": _model_dims.get(self.model, len(embeddings[0]) if embeddings else None),
            })
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
        except Exception as e:
            print(f"[mlx-server] Error during embed: {e}")
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
    parser.add_argument("--preload", action="store_true", help="Preload model at startup")
    args = parser.parse_args()

    MLXHandler.model = args.model
    MLXHandler.max_length = args.max_length

    if args.preload:
        print(f"[mlx-server] Pre-loading {args.model}...")
        try:
            _, _, dims = _load_model(args.model)
            MLXHandler.ready = True
            print(f"[mlx-server] Model ready ({dims}d)")
        except Exception as e:
            print(f"[mlx-server] Pre-load failed: {e}")
            MLXHandler.ready = False
    else:
        MLXHandler.ready = True  # Lazy-load on first request

    server = ThreadedHTTPServer(("127.0.0.1", args.port), MLXHandler)
    print(f"[mlx-server] Listening on http://127.0.0.1:{args.port}")
    print(f"[mlx-server] Model: {args.model}")
    print(f"[mlx-server] Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mlx-server] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
