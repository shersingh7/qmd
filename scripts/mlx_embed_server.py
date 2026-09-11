#!/usr/bin/env python3
"""
QMD MLX Embedding Server — Metal-Native Apple Silicon GPU Acceleration

Usage:
    python scripts/mlx_embed_server.py --model mlx-community/nomic-embed-text-v1.5 --port 8787 --control-port 8788 --preload
    python scripts/mlx_embed_server.py --model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ --preload
    python scripts/mlx_embed_server.py --model sentence-transformers/all-MiniLM-L6-v2 --preload

Environment:
    MLX_EMBED_MODEL       — model identifier (HF repo or local path)
    MLX_EMBED_PORT        — inference port (default 8787)
    MLX_CONTROL_PORT      — control port (default: auto port+1)
    MLX_INSTANCE_TOKEN    — unique server instance token for watchdog binding
    MLX_EMBED_MAX_LENGTH  — max tokens per input (default 2048)
    MLX_EMBED_QUANT       — quantization: bf16, fp16, q8_0, q4_0 (default bf16)
    MLX_EMBED_DTYPE       — compute dtype: float32, float16, bfloat16 (default float32)
    MLX_MAX_BATCH_TOKENS  — max total tokens per GPU pass (default: auto from VRAM)

Endpoints (Inference Port):
    GET  /health          -> Process status, model descriptor, readiness (legacy/fallback)
    GET  /ready           -> 200 {"ready": true} or 503 {"ready": false}
    GET  /descriptor     -> Canonical EmbeddingDescriptor JSON
    GET  /memory          -> {active_mb, peak_mb, model_mb}
    GET  /stats           -> {total_requests, avg_ms, compiled_shapes, uptime_sec}
    POST /embed           {"texts": [...], "dims": 768, "is_query": false}
    POST /embed-bin       Binary wire: returns int32(count,dims) + float32 array
    POST /tokenize        {"texts": [...]} -> {"tokens": [...], "counts": [...]}
    POST /rerank          {"query": "...", "documents": [...]}
    POST /generate        {"prompt": "...", "max_tokens": 128}

Endpoints (Dedicated Control Port):
    GET  /health          -> Process status, instance token, worker progress (isolated capacity)
    GET  /ready           -> 200 / 503
    GET  /descriptor     -> Canonical EmbeddingDescriptor JSON
    GET  /memory          -> {active_mb, peak_mb, model_mb}
    GET  /stats           -> {total_requests, avg_ms, compiled_shapes, uptime_sec}
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
    default_control_port = int(os.getenv("MLX_CONTROL_PORT", "0")) if os.getenv("MLX_CONTROL_PORT") else None
    default_max_length = int(os.getenv("MLX_EMBED_MAX_LENGTH", "2048"))
    default_quant = os.getenv("MLX_EMBED_QUANT", "bf16")
    default_dtype = os.getenv("MLX_EMBED_DTYPE", "float32")
    default_max_tokens = int(os.getenv("MLX_MAX_BATCH_TOKENS", "0"))
    default_rerank = os.getenv("MLX_RERANK_MODEL", "")
    default_generate = os.getenv("MLX_GENERATE_MODEL", "")

    parser = argparse.ArgumentParser(description="QMD MLX Embedding Server (Apple Silicon Metal Native)")
    parser.add_argument("--model", default=default_model, help="Hugging Face repo or local path")
    parser.add_argument("--no-embed", action="store_true", help="Do not load embedding model (single-stage mode for rerank/generate)")
    parser.add_argument("--port", type=int, default=default_port, help="Inference port to listen on (default: 8787)")
    parser.add_argument("--control-port", type=int, default=default_control_port, help="Dedicated loopback control port (default: auto port+1)")
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
    effective_embed_model = None if (args.no_embed or not args.model or args.model.lower() in ("none", "null")) else args.model

    server, thread = start_server(
        model_name=effective_embed_model,
        port=args.port,
        bind_host=args.host,
        control_port=args.control_port,
        quantization=args.quantization,
        dtype_str=args.dtype,
        max_length=args.max_length,
        preload=args.preload,
        warmup=not args.no_warmup,
        max_batch_tokens=args.max_batch_tokens,
        rerank_model=args.rerank_model or None,
        generate_model=args.generate_model or None,
    )

    print(f"[mlx-server] Inference server: http://{args.host}:{server.server_address[1]} | Control server: http://127.0.0.1:{server.control_port}")
    if effective_embed_model:
        print(f"[mlx-server] Model: {effective_embed_model} | Dtype: {args.dtype} | MaxLength: {args.max_length} | InstanceToken: {server.instance_token}")
    else:
        print(f"[mlx-server] Single-stage server (no embedding loaded) | InstanceToken: {server.instance_token}")
    if args.rerank_model:
        print(f"[mlx-server] Rerank: {args.rerank_model}")
    if args.generate_model:
        print(f"[mlx-server] Generate: {args.generate_model}")

    import signal
    import threading

    stop_event = threading.Event()

    def _sig_handler(signum, frame):
        print(f"\n[mlx-server] Received signal {signum}, initiating graceful shutdown...")
        stop_event.set()

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    try:
        while thread.is_alive() and not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print("[mlx-server] Shutting down...")
        server.stop()
        server.server_close()
        if thread.is_alive():
            thread.join(timeout=10.0)
        if server._stopped_event.is_set() and not thread.is_alive():
            print("[mlx-server] Stopped.")
        else:
            print("[mlx-server] Shutdown incomplete: server thread or worker still active.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
