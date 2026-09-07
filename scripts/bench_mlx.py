#!/usr/bin/env python3
"""
bench_mlx.py — Metal-Native Apple Silicon MLX Embedding Benchmark Harness
"""

import argparse
import json
import os
import platform
import socket
import sys
import time
import requests
import numpy as np
from typing import Optional, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from qmd_mlx.server import start_server
from qmd_mlx.protocol import decode_binary_embeddings


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_benchmark(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batches: list[int] = [1, 8, 32],
    iterations: int = 5,
    warmup_runs: int = 2,
    output_json: Optional[str] = None,
):
    port = get_free_port()
    print(f"[bench] Starting MLX server on port {port} with model {model_name}...")
    server, thread = start_server(
        model_name=model_name,
        port=port,
        bind_host="127.0.0.1",
        preload=True,
        warmup=True,
    )

    base_url = f"http://127.0.0.1:{port}"

    # Wait for ready
    for _ in range(100):
        try:
            r = requests.get(f"{base_url}/ready", timeout=1)
            if r.status_code == 200 and r.json().get("ready"):
                break
        except Exception:
            pass
        time.sleep(0.1)

    # Fetch descriptor
    desc = requests.get(f"{base_url}/descriptor").json()
    print(f"[bench] Model descriptor: {desc['model']} ({desc['outputDimensions']}d, {desc['pooling']})")

    # Sample texts of varying lengths
    sample_corpus = [
        "Apple Silicon M-series chips feature unified memory architecture for zero-copy CPU-GPU data sharing.",
        "SQLite FTS5 provides fast full-text search with BM25 ranking algorithms.",
        "Vector embeddings map semantic document concepts into continuous vector spaces.",
        "Reciprocal Rank Fusion merges rankings from BM25 and vector similarity searches.",
        "Tree-sitter AST chunking splits code files at class, function, and interface boundaries.",
        "Metal performance shaders compile compute graphs directly into optimized GPU kernels.",
        "Zero-copy binary serialization eliminates JSON serialization overhead in high-throughput pipelines.",
        "Local RAG pipelines on macOS achieve high throughput with low memory footprint.",
    ]

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "version": platform.version(),
            "python": platform.python_version(),
        },
        "descriptor": desc,
        "benchmarks": [],
    }

    try:
        for batch_size in batches:
            texts = [sample_corpus[i % len(sample_corpus)] for i in range(batch_size)]
            payload = {"texts": texts}

            # 1. Benchmark JSON endpoint
            json_latencies: list[float] = []
            for run_idx in range(warmup_runs + iterations):
                t0 = time.perf_counter()
                r = requests.post(f"{base_url}/embed", json=payload, timeout=30)
                t1 = time.perf_counter()
                if r.status_code != 200:
                    print(f"[bench] Error on batch {batch_size} JSON: {r.text}")
                    continue
                if run_idx >= warmup_runs:
                    json_latencies.append((t1 - t0) * 1000)

            # 2. Benchmark Binary endpoint
            bin_latencies: list[float] = []
            for run_idx in range(warmup_runs + iterations):
                t0 = time.perf_counter()
                r = requests.post(f"{base_url}/embed-bin", json=payload, timeout=30)
                t1 = time.perf_counter()
                if r.status_code != 200:
                    print(f"[bench] Error on batch {batch_size} Binary: {r.text}")
                    continue
                if run_idx >= warmup_runs:
                    arr, count, dims = decode_binary_embeddings(r.content)
                    bin_latencies.append((t1 - t0) * 1000)

            if json_latencies:
                json_p50 = float(np.percentile(json_latencies, 50))
                json_p95 = float(np.percentile(json_latencies, 95))
                json_tps = (batch_size / (json_p50 / 1000.0))
            else:
                json_p50, json_p95, json_tps = 0.0, 0.0, 0.0

            if bin_latencies:
                bin_p50 = float(np.percentile(bin_latencies, 50))
                bin_p95 = float(np.percentile(bin_latencies, 95))
                bin_tps = (batch_size / (bin_p50 / 1000.0))
            else:
                bin_p50, bin_p95, bin_tps = 0.0, 0.0, 0.0

            mem = requests.get(f"{base_url}/memory").json()

            bench_entry = {
                "batch_size": batch_size,
                "json": {
                    "p50_ms": round(json_p50, 2),
                    "p95_ms": round(json_p95, 2),
                    "texts_per_sec": round(json_tps, 1),
                },
                "binary": {
                    "p50_ms": round(bin_p50, 2),
                    "p95_ms": round(bin_p95, 2),
                    "texts_per_sec": round(bin_tps, 1),
                },
                "memory_mb": mem,
            }
            results["benchmarks"].append(bench_entry)

            print(
                f"[bench] Batch {batch_size:2d} | "
                f"JSON: {json_p50:6.2f}ms ({json_tps:6.1f} texts/s) | "
                f"Binary: {bin_p50:6.2f}ms ({bin_tps:6.1f} texts/s) | "
                f"Metal: {mem.get('active_mb', 0):.0f}MB"
            )

    finally:
        server.shutdown()
        server.server_close()

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[bench] Saved results to {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX embedding server")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", default="docs/benchmarks/mlx-benchmark.json")
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        batches=args.batches,
        iterations=args.iterations,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
