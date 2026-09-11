#!/usr/bin/env python3
"""
qmd-mlx-smoke.py — Bounded Single-Model Smoke Test Runner for MLX Qualification

Guarantees & Constraints:
- 100% Offline: Enforces HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1.
- Local weights only: Model path must exist locally; downloads are strictly prohibited.
- Endpoint isolation: Uses dedicated loopback ports (default 8797/8798) with zero collision on 8787.
- Watchdog-owned child: Disposable server child managed with unique instance token and wall-clock ceiling.
- Guaranteed cleanup: Process handles are terminated and reaped on all exit paths.
- Rehearsal mode: --rehearsal runs full suite using disposable fake child without loading real weights.
- Opt-in real mode: --real-model is required for real model execution alongside memory preflight check.
- Non-destructive: Live index, GGUF daemon, and services are untouched (read-only inspection).
- Strict telemetry: Captures actual measured latencies and metrics or explicit "not_run" (never fabricated data).
- Disclaimer: Synthetic numeric checks verify runtime determinism and resource bounding, NOT semantic quality.

Usage:
    # 1. Inspect local model metadata (read-only, no weights load):
    python3 scripts/qmd-mlx-smoke.py --model-path ~/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine --inventory-only

    # 2. Run offline rehearsal with fake adapter disposable child (NO real weights):
    python3 scripts/qmd-mlx-smoke.py --rehearsal

    # 3. Dry-run preflight check (no spawn):
    python3 scripts/qmd-mlx-smoke.py --rehearsal --dry-run --json

    # 4. Future real-model qualification (explicit opt-in + refreshed preflight):
    python3 scripts/qmd-mlx-smoke.py --real-model --model-path ~/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine --port 8797 --control-port 8798 --timeout-s 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

# Ensure repository root and scripts directory are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == "scripts" else current_dir
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
scripts_dir = os.path.join(repo_root, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Enforce offline Hugging Face operation
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from scripts.qmd_mlx.smoke import (
    SmokeRunner,
    SmokeRunnerConfig,
    inspect_model_metadata,
)
from scripts.qmd_mlx.watchdog import (
    SystemMemorySampler,
    SystemMetricsError,
    is_numeric_loopback,
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded Single-Model MLX Smoke Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local directory path to existing MLX model weights and config",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Loopback IP interface (strictly numeric 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8797,
        help="Inference HTTP port (must be distinct from production 8787)",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=8798,
        help="Dedicated loopback control HTTP port",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="Hard wall-clock execution ceiling in seconds",
    )
    parser.add_argument(
        "--min-headroom-mb",
        type=float,
        default=2048.0,
        help="Minimum required available memory headroom before spawning",
    )
    parser.add_argument(
        "--rehearsal",
        action="store_true",
        help="Run offline rehearsal with disposable fake child (NO real weights loaded)",
    )
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="Explicit opt-in required to execute against real local MLX weights",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Inspect model metadata and running services read-only, then exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and system preflight without spawning server child",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Format output as structured JSON",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Write structured JSON report to specified file path",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Run comprehensive diagnostic suite (repeated singletons, duplicate batch, mixed lengths, position permutations)",
    )
    parser.add_argument(
        "--fake-fail-consistency",
        action="store_true",
        help="Simulate consistency failure in synthetic rehearsal mode for robustness regression testing",
    )

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2

    # 1. Inventory-only mode
    if args.inventory_only:
        sampler = SystemMemorySampler()
        try:
            headroom = sampler.get_memory_headroom_mb()
            installed_ram = sampler.get_installed_ram_mb()
            swap_used = sampler.get_swap_used_mb()
        except SystemMetricsError as e:
            print(f"Error sampling memory metrics: {e}", file=sys.stderr)
            return 2

        inv_data = {
            "system_telemetry": {
                "installed_ram_mb": installed_ram,
                "headroom_mb": headroom,
                "swap_used_mb": swap_used,
            },
            "model_metadata": None,
        }

        if args.model_path:
            try:
                resolved_path = os.path.expanduser(args.model_path)
                meta = inspect_model_metadata(resolved_path)
                inv_data["model_metadata"] = meta.to_dict()
            except Exception as e:
                print(f"Error inspecting model metadata: {e}", file=sys.stderr)
                return 2

        if args.json:
            print(json.dumps(inv_data, indent=2))
        else:
            print("=== System Resource Telemetry (Read-Only) ===")
            print(f"  Installed RAM:      {installed_ram:.1f} MB")
            print(f"  Available Headroom: {headroom:.1f} MB")
            print(f"  Swap Used:          {swap_used:.1f} MB")
            if inv_data["model_metadata"]:
                m = inv_data["model_metadata"]
                print("\n=== Model Metadata Inventory (Read-Only) ===")
                print(f"  Model Path:         {m['model_path']}")
                print(f"  Model Type:         {m['model_type']}")
                print(f"  Architectures:      {m['architectures']}")
                print(f"  Hidden Size:        {m['hidden_size']}")
                print(f"  Hidden Layers:      {m['num_hidden_layers']}")
                print(f"  Attention Heads:    {m['num_attention_heads']}")
                print(f"  KV Heads:           {m['num_key_value_heads']}")
                print(f"  Quantization:       {m['quantization']}")
                print(f"  Safetensors:        {m['has_safetensors']}")
                print(f"  Tokenizer:          {m['has_tokenizer']}")
                print(f"  Total Size:         {m['total_file_size_bytes'] / (1024*1024):.1f} MB")
                print(f"  Estimated Model MB: {m.get('estimated_memory_mb', 0):.1f} MB ({m.get('params_b', 0):.1f}B params)")
                print(f"  Required Headroom:  {m.get('conservative_required_headroom_mb', 2048):.1f} MB (model-aware)")
        return 0

    # 2. Rehearsal vs Real Model checks
    resolved_model_path = os.path.expanduser(args.model_path) if args.model_path else None
    if not args.rehearsal and not args.real_model and not args.dry_run:
        print(
            "Error: Must specify either --rehearsal (for zero-weight offline rehearsal) "
            "or --real-model (with --model-path for real weights).",
            file=sys.stderr,
        )
        return 2

    if args.real_model and not resolved_model_path:
        print("Error: --real-model requires --model-path <PATH> to existing local weights.", file=sys.stderr)
        return 2

    config = SmokeRunnerConfig(
        model_path=resolved_model_path,
        host=args.host,
        port=args.port,
        control_port=args.control_port,
        timeout_s=args.timeout_s,
        min_headroom_mb=args.min_headroom_mb,
        use_fake_child=args.rehearsal,
        fake_fail_consistency=args.fake_fail_consistency,
        dry_run=args.dry_run,
        diagnostic=args.diagnostic,
        real_model_opt_in=args.real_model,
        output_file=args.output_file,
    )

    runner = SmokeRunner(config)

    report = None
    exit_code = 0
    try:
        report = runner.run()
        exit_code = 0 if report.get("status") in ("passed", "dry_run_completed") else 1
    except Exception as e:
        exit_code = 1
        report = getattr(runner, "last_report", None)
        if not report:
            report = {
                "status": "failed",
                "error": str(e),
                "duration_s": 0.0,
                "fixtures": {},
                "child_cleanup": {"cleaned": True, "pid": None, "exit_code": None},
            }
        else:
            report["status"] = "failed"
            if "error" not in report or not report["error"]:
                report["error"] = str(e)

    if args.output_file:
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[qmd-mlx-smoke] JSON report written to {args.output_file}")
        except Exception as e:
            print(f"Error writing output file {args.output_file}: {e}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("\n=== MLX Single-Model Smoke Qualification Report ===")
        print(f"Status:             {report.get('status', '').upper()}")
        print(f"Mode:               {'Rehearsal (Synthetic Adapter)' if config.use_fake_child else 'Real Model'}")
        print(f"Duration:           {report.get('duration_s', 0):.2f}s")
        print(f"Endpoints:          Inference http://{config.host}:{config.port} | Control http://{config.host}:{config.control_port}")
        
        pre = report.get("preflight", {})
        print(f"Preflight Headroom: {pre.get('headroom_mb', 0):.1f} MB (min required: {config.min_headroom_mb:.1f} MB)")

        if report.get("error"):
            print(f"\n[!] Failure Cause:  {report.get('error')}")

        fixtures = report.get("fixtures", {})
        if fixtures:
            print("\nFixture Verification Results:")
            for name, fix in fixtures.items():
                st = fix.get("status", "unknown").upper()
                lat = fix.get("latency_ms")
                lat_str = f"{lat:.2f}ms" if lat is not None else "N/A"
                print(f"  [{st}] {name:<15} (latency: {lat_str})")
                if name == "consistency" and "cosine_similarity" in fix:
                    print(f"        Cosine Sim: {fix['cosine_similarity']:.6f} (tol >= {fix['cosine_tolerance']}), Max Diff: {fix['max_absolute_difference']:.6e}")
                elif name == "timing" and "p50_ms" in fix:
                    print(f"        Iterations: {fix['iterations']}, min: {fix['min_ms']}ms, p50: {fix['p50_ms']}ms, p95: {fix['p95_ms']}ms, max: {fix['max_ms']}ms")

        diagnostics = report.get("diagnostics", {})
        if diagnostics:
            print("\nDiagnostic Mode Detailed Results:")
            if "repeated_singleton" in diagnostics:
                d_rs = diagnostics["repeated_singleton"]
                print(f"  Repeated Singleton:  [{d_rs.get('status', '').upper()}] Cosine: {d_rs.get('cosine_similarity', 0):.6f}, MaxDiff: {d_rs.get('max_absolute_difference', 0):.6e}")
            if "duplicate_batch" in diagnostics:
                d_db = diagnostics["duplicate_batch"]
                print(f"  Duplicate Batch:     [{d_db.get('status', '').upper()}] Min Cosine vs Sing: {d_db.get('min_cosine_similarity', 0):.6f}, MaxDiff: {d_db.get('max_absolute_difference', 0):.6e}")
            if "mixed_lengths" in diagnostics:
                d_ml = diagnostics["mixed_lengths"]
                print(f"  Mixed Lengths:       [{d_ml.get('status', '').upper()}] Row0 Cosine vs Sing: {d_ml.get('singleton_vs_row0_cosine', 0):.6f}, MaxDiff: {d_ml.get('singleton_vs_row0_max_diff', 0):.6e}")
            if "position_permutations" in diagnostics:
                d_pp = diagnostics["position_permutations"]
                print(f"  Position Permutes:   [{d_pp.get('status', '').upper()}]")
                for p in d_pp.get("permutations", []):
                    print(f"    - Target Pos {p.get('target_position_in_batch')}: Cosine: {p.get('cosine_vs_singleton', 0):.6f}, MaxDiff: {p.get('max_diff_vs_singleton', 0):.6e}")

        cleanup = report.get("child_cleanup", {})
        if cleanup:
            print(f"\nChild Process Cleanup: Cleaned={cleanup.get('cleaned')}, PID={cleanup.get('pid')}, ExitCode={cleanup.get('exit_code')}")

        print(f"\nDisclaimer: {report.get('disclaimer')}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
