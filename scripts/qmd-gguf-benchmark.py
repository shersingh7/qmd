#!/usr/bin/env python3
"""
qmd-gguf-benchmark.py — Bounded GGUF Embedding Qualification Runner

Executes bounded pilot (<=30 requests, <=120s) and bounded soak (<=100 requests, <=180s)
benchmarks for GGUF embedding models via node-llama-cpp under active Watchdog supervision
with strict process isolation, memory safety enforcement, and structured JSON reporting.

Usage:
    # 1. Offline Rehearsal (Pilot Mode, Synthetic, <=30 requests):
    python3 scripts/qmd-gguf-benchmark.py --rehearsal --mode pilot --json

    # 2. Real-Model Bounded Pilot (Explicit opt-in, <=30 requests, <=120s):
    python3 scripts/qmd-gguf-benchmark.py --real-model \
        --model-path ~/.cache/qmd/models/hf_Qwen_Qwen3-Embedding-4B-Q4_K_M.gguf \
        --mode pilot \
        --output-file docs/reviews/artifacts/phase4-gguf-4b-pilot.json
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

from scripts.qmd_gguf.benchmark import (
    GGUFBenchmarkConfig,
    GGUFBenchmarkRunner,
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded GGUF Embedding Qualification Runner (node-llama-cpp)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local file path to existing GGUF model weights",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Loopback IP interface (strictly numeric 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8795,
        help="Inference HTTP port (must be distinct from production 8787)",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=8796,
        help="Dedicated loopback control HTTP port",
    )
    parser.add_argument(
        "--mode",
        choices=["pilot", "soak"],
        default="pilot",
        help="Benchmark mode: 'pilot' (<=30 reqs, <=120s) or 'soak' (<=100 reqs, <=180s)",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="Hard wall-clock execution ceiling (defaults to 120s for pilot, 180s for soak)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Maximum request iterations (defaults to 30 for pilot, 100 for soak)",
    )
    parser.add_argument(
        "--min-headroom-mb",
        type=float,
        default=6000.0,
        help="Minimum required available memory headroom before spawning",
    )
    parser.add_argument(
        "--contexts",
        type=int,
        default=2,
        help="Number of parallel embedding contexts to allocate in node-llama-cpp",
    )
    parser.add_argument(
        "--rehearsal",
        action="store_true",
        help="Run offline rehearsal with synthetic node server (NO real weights loaded)",
    )
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="Explicit opt-in required to execute against real local GGUF weights",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and preflight without spawning server child",
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

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2

    mode = args.mode
    default_timeout = 120.0 if mode == "pilot" else 180.0
    timeout_s = min(float(args.timeout_s or default_timeout), 180.0)

    default_requests = 30 if mode == "pilot" else 100
    max_requests = min(int(args.max_requests or default_requests), 100)

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

    config = GGUFBenchmarkConfig(
        model_path=resolved_model_path,
        host=args.host,
        port=args.port,
        control_port=args.control_port,
        mode=mode,
        timeout_s=timeout_s,
        max_requests=max_requests,
        min_headroom_mb=args.min_headroom_mb,
        use_fake_child=args.rehearsal,
        dry_run=args.dry_run,
        real_model_opt_in=args.real_model,
        output_file=args.output_file,
        contexts=args.contexts,
    )

    runner = GGUFBenchmarkRunner(config)

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
                "child_cleanup": {"cleaned": True, "pid": None, "exit_code": None},
            }
        else:
            report["status"] = "failed"
            if "error" not in report or not report["error"]:
                report["error"] = str(e)

    if args.output_file:
        try:
            out_dir = os.path.dirname(os.path.abspath(args.output_file))
            os.makedirs(out_dir, exist_ok=True)
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[qmd-gguf-benchmark] JSON report written to {args.output_file}")
        except Exception as e:
            print(f"Error writing output file {args.output_file}: {e}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("\n=== GGUF Sustained Embedding Qualification Report ===")
        print(f"Status:             {report.get('status', '').upper()}")
        print(f"Mode:               {mode.upper()} ({'Synthetic Rehearsal' if config.use_fake_child else 'Real Model'})")
        print(f"Duration:           {report.get('duration_s', 0):.2f}s (timeout: {config.timeout_s}s)")
        print(f"Endpoints:          Inference http://{config.host}:{config.port} | Control http://{config.host}:{config.control_port}")

        pre = report.get("preflight", {})
        print(f"Preflight Headroom: {pre.get('headroom_mb', 0):.1f} MB (min required: {config.min_headroom_mb:.1f} MB)")

        cold = report.get("cold_startup", {})
        if cold:
            print(f"Cold Startup Time:  {cold.get('cold_startup_elapsed_s', 0):.3f}s")

        warm = report.get("warmup", {})
        if warm:
            print(f"GPU Warmup:         {warm.get('warmup_elapsed_ms', 0):.2f}ms (excluded from metrics)")

        summary = report.get("metrics_summary", {})
        if summary:
            print(f"\nWorkload Metrics Summary:")
            print(f"  Total Measured Requests:     {summary.get('total_measured_requests', 0)} / {config.max_requests}")
            print(f"  Successful Embed Requests:   {summary.get('successful_embedding_requests', 0)}")
            print(f"  Expected Rejections (400):   {summary.get('rejected_boundary_requests', 0)}")
            print(f"  Successful Embedded Tokens:  {summary.get('total_successful_embedded_tokens', 0)}")
            print(f"  True Token Throughput Rate:  {summary.get('overall_effective_tokens_per_sec', 0):.1f} tokens/s")
            print(f"  Partial Run Flag:            {summary.get('partial_run', False)}")

            wall = summary.get("cumulative_wall_accounting", {})
            if wall:
                print(f"  Cumulative Wall Accounting:  Total: {wall.get('total_benchmark_elapsed_s')}s | Successful Wire: {wall.get('sum_successful_request_wire_s')}s | Rejection Wire: {wall.get('sum_rejection_request_wire_s')}s")

            ov = summary.get("overall_successful_latency_ms", {})
            if ov:
                print(f"  Successful Latency:          min: {ov.get('min')}ms, p50: {ov.get('p50')}ms, p95: {ov.get('p95')}ms, max: {ov.get('max')}ms, mean: {ov.get('mean')}ms (descriptive: {ov.get('descriptive_only')})")

            rej_lat = summary.get("expected_rejections_latency_ms", {})
            if rej_lat and rej_lat.get("count", 0) > 0:
                print(f"  Expected Rejection Latency:  p50: {rej_lat.get('p50')}ms, mean: {rej_lat.get('mean')}ms (N={rej_lat.get('count')})")

            by_strat = summary.get("by_stratum_latency_ms", {})
            if by_strat:
                print("\n  Latency by Stratum:")
                for strat, st_data in by_strat.items():
                    print(f"    - {strat:<16} (N={st_data.get('count')}): p50={st_data.get('p50')}ms, p95={st_data.get('p95')}ms, mean={st_data.get('mean')}ms")

            by_batch = summary.get("by_batch_size_latency_ms", {})
            if by_batch:
                print("\n  Latency by Batch Size:")
                for bsz, b_data in by_batch.items():
                    print(f"    - {bsz:<16} (N={b_data.get('count')}): p50={b_data.get('p50')}ms, p95={b_data.get('p95')}ms, mean={b_data.get('mean')}ms")

            tail = summary.get("concurrent_tail_latency_ms", {})
            if tail:
                solo = tail.get("solo_baseline_interactive", {})
                conc = tail.get("concurrent_interactive_under_load", {})
                bulk = tail.get("concurrent_bulk_batches", {})
                qw = tail.get("estimated_queue_wait_ms", {})
                print(f"\n  Concurrent Load Tail Latency:")
                print(f"    - Solo Baseline Interactive:  p50={solo.get('p50')}ms, p95={solo.get('p95')}ms (N={solo.get('count')})")
                print(f"    - Concurrent Under Load:      p50={conc.get('p50')}ms, p95={conc.get('p95')}ms (N={conc.get('count')})")
                print(f"    - Concurrent Bulk Batch:      p50={bulk.get('p50')}ms, p95={bulk.get('p95')}ms (N={bulk.get('count')})")
                print(f"    - Estimated Queue Wait:       p50={qw.get('p50')}ms, max={qw.get('max')}ms | Mean Slowdown: {tail.get('mean_slowdown_factor')}x")

        mem = report.get("metal_memory_telemetry", {})
        if mem:
            print(f"\nUnified Memory Telemetry:")
            print(f"  Peak Active:                 {mem.get('peak_active_mb', 0):.1f} MB")
            print(f"  Active Delta:                {mem.get('delta_active_mb', 0):.1f} MB")
            stab = mem.get("provisional_stability_assessment", {})
            if stab:
                print(f"  Stability Assessment:        {stab.get('verdict')} (Swap Growth: {stab.get('swap_growth_mb')} MB, RSS Growth: {stab.get('rss_growth_mb')} MB)")

        cleanup = report.get("child_cleanup", {})
        if cleanup:
            print(f"\nChild Process Cleanup: Cleaned={cleanup.get('cleaned')}, PID={cleanup.get('pid')}, ExitCode={cleanup.get('exit_code')}")

        qual = report.get("qualification_evaluation", {})
        if qual:
            print("\n=== Qualification Evaluation Gates (Explicit Original Criteria) ===")
            print(f"  Overall Gate Status:         {qual.get('overall_status', 'UNKNOWN').upper()}")
            gates = qual.get("gates", {})
            for gate_name, gate_info in gates.items():
                g_pass = "PASS" if gate_info.get("passed") else "FAIL"
                print(f"  - [{g_pass}] {gate_name}")
                for k, v in gate_info.items():
                    if k != "passed":
                        print(f"      {k}: {v}")
            if qual.get("failure_reasons"):
                print(f"\n  Gate Failure Reasons:")
                for r in qual.get("failure_reasons"):
                    print(f"    - {r}")

        prim = report.get("primary_failure")
        if prim:
            print(f"\n[!] PRIMARY FAILURE ROOT CAUSE:")
            print(f"    Category: {prim.get('category')}")
            print(f"    Reason:   {prim.get('reason')}")
            if prim.get("breach_type"):
                print(f"    Breach Type: {prim.get('breach_type')}")

        if report.get("error") and not prim:
            print(f"\n[!] Failure Cause:  {report.get('error')}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
