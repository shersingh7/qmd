#!/usr/bin/env python3
"""
qmd-mlx-rerank-smoke.py — CLI Entrypoint for MLX Reranker Single-Stage Smoke Qualification
"""

import argparse
import json
import os
import sys

# Ensure repo root is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == "scripts" else current_dir
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.qmd_mlx.rerank_smoke import run_rerank_smoke


def main():
    parser = argparse.ArgumentParser(description="MLX Reranker Single-Stage Smoke Qualification")
    parser.add_argument(
        "--model-path",
        default="~/.cache/qmd/models/qwen3-reranker-4b-mlx-4bit",
        help="Path to local quantized MLX reranker weights",
    )
    parser.add_argument("--port", type=int, default=8797, help="Inference port (default: 8797)")
    parser.add_argument("--control-port", type=int, default=8798, help="Control port (default: 8798)")
    parser.add_argument("--timeout-s", type=float, default=60.0, help="Wall-clock ceiling in seconds (default: 60)")
    parser.add_argument("--min-headroom-mb", type=float, default=6000.0, help="Required memory headroom in MB")
    parser.add_argument("--output-json", default="docs/reviews/artifacts/phase4-rerank-4b-smoke.json", help="Path to write JSON artifact")

    args = parser.parse_args()

    print(f"[rerank-smoke] Starting single-stage qualification for: {args.model_path}")
    report = run_rerank_smoke(
        model_path=args.model_path,
        port=args.port,
        control_port=args.control_port,
        timeout_s=args.timeout_s,
        min_headroom_mb=args.min_headroom_mb,
        output_json=args.output_json,
    )

    print(f"\n[rerank-smoke] Status: {report['status'].upper()}")
    if report.get("metrics"):
        print(f"[rerank-smoke] Startup: {report['metrics'].get('startup_time_s')}s | Fixture Latency p50: {report['metrics'].get('fixture_p50_ms')}ms | Batch 4-docs: {report['metrics'].get('batch_4docs_latency_ms')}ms")
    if report.get("errors"):
        for err in report["errors"]:
            print(f"[rerank-smoke] ERROR: {err}", file=sys.stderr)

    if report.get("status") == "passed":
        print(f"[rerank-smoke] Report artifact written to: {args.output_json}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
