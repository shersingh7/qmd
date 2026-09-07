#!/usr/bin/env python3
"""
QMD MLX Embedding Server — Metal-Native Apple Silicon GPU Acceleration

Usage:
    python scripts/mlx_embed_server.py --model mlx-community/nomic-embed-text-v1.5 --port 8787 --preload
    python scripts/mlx_embed_server.py --model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ --preload
    python scripts/mlx_embed_server.py --model sentence-transformers/all-MiniLM-L6-v2 --preload

Environment:
    MLX_EMBED_MODEL       — model identifier (HF repo or local path)
    MLX_EMBED_PORT        — port (default 8787)
    MLX_EMBED_MAX_LENGTH  — max tokens per input (default 2048)
    MLX_EMBED_QUANT       — quantization: bf16, fp16, q8_0, q4_0 (default bf16)
    MLX_EMBED_DTYPE       — compute dtype: float32, float16, bfloat16 (default float32)
    MLX_MAX_BATCH_TOKENS  — max total tokens per GPU pass (default: auto from VRAM)

Endpoints:
    GET  /health          -> Process status, model descriptor, readiness
    GET  /ready           -> 200 {"ready": true} or 503 {"ready": false}
    GET  /descriptor     -> Canonical EmbeddingDescriptor JSON
    GET  /memory          -> {active_mb, peak_mb, model_mb}
    GET  /stats           -> {total_requests, avg_ms, compiled_shapes, uptime_sec}
    POST /embed           {"texts": [...], "dims": 768, "is_query": false}
    POST /embed-bin       Binary wire: returns int32(count,dims) + float32 array
    POST /tokenize        {"texts": [...]} -> {"tokens": [...], "counts": [...]}
"""

import argparse
import os
import sys
import time

# Ensure repo root is on sys.path so scripts.qmd_mlx or qmd_mlx can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from qmd_mlx.server import start_server


def main():
    default_model = os.getenv("MLX_EMBED_MODEL", "mlx-community/nomic-embed-text-v1.5")
    default_port = int(os.getenv("MLX_EMBED_PORT", "8787"))
    default_max_length = int(os.getenv("MLX_EMBED_MAX_LENGTH", "2048"))
    default_quant = os.getenv("MLX_EMBED_QUANT", "bf16")
    default_dtype = os.getenv("MLX_EMBED_DTYPE", "float32")
    default_max_tokens = int(os.getenv("MLX_MAX_BATCH_TOKENS", "0"))
    default_rerank = os.getenv("MLX_RERANK_MODEL", "")
    default_generate = os.getenv("MLX_GENERATE_MODEL", "")

    parser = argparse.ArgumentParser(description="QMD MLX Embedding Server (Apple Silicon Metal Native)")
    parser.add_argument("--model", default=default_model, help="Hugging Face repo or local path")
    parser.add_argument("--port", type=int, default=default_port, help="Port to listen on (default: 8787)")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--max-length", type=int, default=default_max_length, help="Max tokens per sequence")
    parser.add_argument("--quantization", default=default_quant, choices=["bf16", "fp16", "q8_0", "q4_0", "q8", "q4"])
    parser.add_argument("--dtype", default=default_dtype, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--preload", action="store_true", default=True, help="Preload model at startup")
    parser.add_argument("--no-preload", action="store_false", dest="preload", help="Lazy load model on first request")
    parser.add_argument("--no-warmup", action="store_true", help="Skip Metal GPU warmup passes")
    parser.add_argument("--max-batch-tokens", type=int, default=default_max_tokens, help="Max tokens per GPU micro-batch")
    parser.add_argument("--rerank-model", default=default_rerank, help="Optional rerank model (HF repo or local path); enables /rerank")
    parser.add_argument("--generate-model", default=default_generate, help="Optional generation model (HF repo or local path); enables /generate")

    args = parser.parse_args()

    print(f"[mlx-server] Starting server on http://{args.host}:{args.port}")
    print(f"[mlx-server] Model: {args.model} | Dtype: {args.dtype} | MaxLength: {args.max_length}")
    if args.rerank_model:
        print(f"[mlx-server] Rerank: {args.rerank_model}")
    if args.generate_model:
        print(f"[mlx-server] Generate: {args.generate_model}")

    server, thread = start_server(
        model_name=args.model,
        port=args.port,
        bind_host=args.host,
        quantization=args.quantization,
        dtype_str=args.dtype,
        max_length=args.max_length,
        preload=args.preload,
        warmup=not args.no_warmup,
        max_batch_tokens=args.max_batch_tokens,
        rerank_model=args.rerank_model or None,
        generate_model=args.generate_model or None,
    )

    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[mlx-server] Shutting down...")
        server.shutdown()
        server.server_close()
        print("[mlx-server] Stopped.")


if __name__ == "__main__":
    main()
