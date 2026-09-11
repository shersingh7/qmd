#!/usr/bin/env python3
"""
qmd-mlx-generate-smoke.py — CLI Entrypoint for MLX Generation / Query Expansion Smoke Qualification
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

from scripts.qmd_mlx.generate_smoke import run_generate_smoke


def main():
    parser = argparse.ArgumentParser(description="MLX Generation Single-Stage Smoke Qualification")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-1.7B-4bit",
        help="Model identifier (local path or HF repo)",
    )
    parser.add_argument("--port", type=int, default=8797, help="Inference port (default: 8797)")
    parser.add_argument("--control-port", type=int, default=8798, help="Control port (default: 8798)")
    parser.add_argument("--timeout-s", type=float, default=60.0, help="Wall-clock ceiling in seconds (default: 60)")
    parser.add_argument("--min-headroom-mb", type=float, default=3000.0, help="Required memory headroom in MB")
    parser.add_argument("--output-json", default="docs/reviews/artifacts/phase4-generate-1.7b-smoke.json", help="Path to write JSON artifact")

    args = parser.parse_args()

    print(f"[gen-smoke] Starting single-stage qualification for: {args.model}")
    report = run_generate_smoke(
        model_name_or_path=args.model,
        port=args.port,
        control_port=args.control_port,
        timeout_s=args.timeout_s,
        min_headroom_mb=args.min_headroom_mb,
        output_json=args.output_json,
    )

    print(f"\n[gen-smoke] Status: {report['status'].upper()}")
    if report.get("metrics"):
        print(f"[gen-smoke] Startup: {report['metrics'].get('startup_time_s')}s | Generate Latency p50: {report['metrics'].get('generate_p50_ms')}ms")
    if report.get("errors"):
        for err in report["errors"]:
            print(f"[gen-smoke] ERROR: {err}", file=sys.stderr)

    if report.get("status") == "passed":
        print(f"[gen-smoke] Report artifact written to: {args.output_json}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
