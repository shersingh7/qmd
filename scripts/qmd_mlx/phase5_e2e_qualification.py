#!/usr/bin/env python3
"""
phase5_e2e_qualification.py — Supervised End-to-End Public Retrieval & Recovery Qualification

Orchestrates sequential, isolated end-to-end qualification composed with SmokeStageSupervisor:
1. Production Indexing & Durable Recovery (cancellation, resumption, deduplication, fingerprint safety, SQLite online backup/restore).
2. Supervised MLX-only stage invocation with TS runner (GGUF stage explicitly BLOCKED pending supervised integration).
3. Integrated MCP Streamable HTTP transport retrieval probe.
4. Active MLXWatchdog supervision with continuous memory, swap, and headroom tracking.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir)) if os.path.basename(current_dir) == "qmd_mlx" else current_dir
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.qmd_mlx.supervisor import (
    SmokeHttpClient,
    SmokeStageSupervisor,
    StageSupervisorConfig,
    SubordinateProcessResult,
    run_subordinate_process,
)
from scripts.qmd_mlx.watchdog import SystemMemorySampler, SystemMetricsError


@dataclasses.dataclass
class Phase5SupervisorConfig:
    mlx_model: str = os.path.expanduser("~/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine")
    gguf_model: str = os.path.expanduser("~/.cache/qmd/models/hf_Qwen_Qwen3-Embedding-0.6B-Q8_0.gguf")
    host: str = "127.0.0.1"
    mlx_port: int = 8797
    mlx_control_port: int = 8798
    timeout_s: float = 120.0
    min_headroom_mb: float = 6000.0
    output_json: str = "docs/reviews/artifacts/phase5-isolated-e2e.json"
    dry_run: bool = False
    use_fake_child: bool = False
    skip_mlx: bool = False
    skip_gguf: bool = True


class Phase5Supervisor:
    """
    Supervises Phase 5 End-to-End qualification by composing the existing
    SmokeStageSupervisor with active MLXWatchdog monitoring, launch token authorization,
    bounded execution deadlines, and guaranteed fail-closed teardown.
    """

    def __init__(
        self,
        config: Phase5SupervisorConfig,
        sampler: Optional[SystemMemorySampler] = None,
    ):
        self.config = config
        self.sampler = sampler or SystemMemorySampler()
        self.last_report: dict[str, Any] = {}

    def _check_preflight_headroom(self) -> dict[str, Any]:
        """Validates configuration sanity, system telemetry, and headroom >= 6000MB, failing closed on any error."""
        # Validate numeric configuration parameters
        for f_name, val, min_v in [
            ("timeout_s", self.config.timeout_s, 0.001),
            ("min_headroom_mb", self.config.min_headroom_mb, 0.0),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"Config field '{f_name}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Config field '{f_name}' must not be NaN or Inf, got {val}")
            if val < min_v:
                raise ValueError(f"Config field '{f_name}' must be >= {min_v}, got {val}")

        for p_name, p_val in [("mlx_port", self.config.mlx_port), ("mlx_control_port", self.config.mlx_control_port)]:
            if isinstance(p_val, bool) or not isinstance(p_val, int):
                raise ValueError(f"Config field '{p_name}' must be an integer, got {type(p_val).__name__}")
            if not (1 <= p_val <= 65535):
                raise ValueError(f"Invalid {p_name}: {p_val} (must be between 1 and 65535)")

        if self.config.mlx_port == self.config.mlx_control_port:
            raise ValueError(f"Inference port ({self.config.mlx_port}) and control port ({self.config.mlx_control_port}) must be distinct")

        try:
            installed_ram = self.sampler.get_installed_ram_mb()
            headroom = self.sampler.get_memory_headroom_mb()
            swap_used = self.sampler.get_swap_used_mb()
            free_pct = self.sampler.get_memory_free_pct()
        except SystemMetricsError as e:
            raise SystemMetricsError(f"Preflight memory sampling failed: {e}")

        print(
            f"[Phase 5 Supervisor] Preflight: Installed RAM={installed_ram:.1f}MB, "
            f"Headroom={headroom:.1f}MB, Swap={swap_used:.1f}MB, Free={free_pct:.1f}%"
        )

        if headroom < self.config.min_headroom_mb:
            raise RuntimeError(
                f"Insufficient memory headroom for isolated qualification: "
                f"{headroom:.1f} MB available, required >= {self.config.min_headroom_mb:.1f} MB."
            )

        return {
            "installed_ram_mb": installed_ram,
            "headroom_mb": round(headroom, 1),
            "swap_used_mb": round(swap_used, 1),
            "memory_free_pct": round(free_pct, 1),
        }

    def _save_report(self, report: dict[str, Any]):
        out_file = os.path.abspath(self.config.output_json)
        try:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            report.setdefault("errors", []).append(f"Failed to write report to {out_file}: {e}")

    def run(self) -> dict[str, Any]:
        overall_t0 = time.time()
        start_mono = time.monotonic()
        launch_token = f"phase5-token-{uuid.uuid4().hex[:16]}"
        errors: list[str] = []

        # 1. Preflight Telemetry Check (min 6000MB, persisting failure report fail-closed)
        try:
            preflight_info = self._check_preflight_headroom()
        except Exception as preflight_err:
            failed_report = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "duration_s": round(time.monotonic() - start_mono, 3),
                "status": "failed",
                "errors": [f"Preflight configuration/telemetry failed: {preflight_err}"],
                "checks": {},
            }
            self._save_report(failed_report)
            self.last_report = failed_report
            return failed_report

        # 2. Explicit Model Path Check (no remote fallback or autodownload)
        expanded_mlx_path = os.path.expanduser(self.config.mlx_model)
        if not self.config.dry_run and not self.config.use_fake_child and not self.config.skip_mlx:
            if not os.path.exists(expanded_mlx_path):
                blocked_report = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "duration_s": round(time.monotonic() - start_mono, 3),
                    "status": "failed",
                    "preflight": preflight_info,
                    "errors": [
                        f"Explicit MLX model path missing: {expanded_mlx_path}. Network autodownloads are strictly prohibited."
                    ],
                    "checks": {
                        "model_weights_available": False,
                    },
                    "unfulfilled_release_gates": [
                        "MLX model weights not present in local cache directory; qualification blocked offline."
                    ],
                }
                self._save_report(blocked_report)
                self.last_report = blocked_report
                return blocked_report

        server_script = os.path.join(repo_root, "scripts", "mlx_embed_server.py")
        server_cmd = [
            sys.executable,
            server_script,
            "--model",
            expanded_mlx_path,
            "--port",
            str(self.config.mlx_port),
            "--control-port",
            str(self.config.mlx_control_port),
            "--host",
            self.config.host,
            "--preload",
        ]

        supervisor_config = StageSupervisorConfig(
            cmd=server_cmd,
            host=self.config.host,
            port=self.config.mlx_port,
            control_port=self.config.mlx_control_port,
            timeout_s=self.config.timeout_s,
            min_headroom_mb=self.config.min_headroom_mb,
            stage_name="phase5_qualification",
            model_identifier=self.config.mlx_model,
            instance_token=launch_token,
            output_file=self.config.output_json,
            dry_run=self.config.dry_run,
        )

        supervisor = SmokeStageSupervisor(config=supervisor_config, sampler=self.sampler)

        if self.config.dry_run:
            with supervisor.managed_stage() as ctx:
                ctx.record_fixture("dry_run", {"status": "skipped", "reason": "dry-run requested"})
                report = ctx.report
                report["status"] = "dry_run_completed"
                self.last_report = report
                return report

        report: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "in_progress",
            "preflight": preflight_info,
            "checks": {},
            "errors": [],
            "retrieval_comparison": {
                "gguf_06b": {
                    "status": "blocked",
                    "reason": "GGUF stage explicitly BLOCKED pending supervised integration instead of unmanaged subprocess.run",
                },
                "mlx_4b": {
                    "status": "not_run" if self.config.skip_mlx else "in_progress",
                },
            },
            "unfulfilled_release_gates": [
                "MLX 4B concurrent interactive latency under bulk load (642.06ms) violates <=200.0ms target; hardware/driver bounds remain unproven.",
                "1000-batch sustained load memory test unfulfilled for live promotion.",
                "Full-corpus evaluation (MS MARCO/BEIR) pending; small public fixture results labeled as Preliminary Release Smoke.",
                "GGUF stage explicitly BLOCKED pending supervised integration.",
                "Fresh-process crash recovery across OS boundaries marked pending; only in-process checkpoint recovery verified.",
            ],
        }

        ctx = None
        try:
            with supervisor.managed_stage() as ctx:
                # 3. Check supervisor health before invoking TS child
                ctx.check_breach("server_ready_preflight")

                # Verify descriptor
                desc_status, desc = ctx.client.get_json(
                    f"{ctx.base_url}/descriptor",
                    deadline=ctx.deadline,
                    timeout_s=2.0,
                )
                if desc_status != 200 or not isinstance(desc, dict):
                    raise RuntimeError(f"Descriptor check on {ctx.ctrl_url}/descriptor failed: HTTP {desc_status} {desc}")

                # 4. Execute TS runner child under strict bounded deadline and launch token approval
                ts_runner_script = os.path.join(repo_root, "scripts", "phase5_e2e_runner.ts")
                tmp_ts_report = f"/tmp/qmd-phase5-ts-{uuid.uuid4().hex[:8]}.json"

                bun_path = shutil.which("bun") or "bun"
                ts_cmd = [
                    bun_path,
                    ts_runner_script,
                    "--launch-token",
                    launch_token,
                    "--skip-gguf",  # GGUF stage explicitly BLOCKED pending supervised integration
                    "--mlx-port",
                    str(self.config.mlx_port),
                    "--output-json",
                    tmp_ts_report,
                ]

                ts_env = os.environ.copy()
                ts_env["QMD_PHASE5_LAUNCH_TOKEN"] = launch_token
                ts_env["QMD_EMBED_BACKEND"] = "mlx"
                ts_env["QMD_MLX_EMBED_URL"] = ctx.base_url
                ts_env["QMD_MLX_FALLBACK"] = "0"
                ts_env["HF_HUB_OFFLINE"] = "1"
                ts_env["TRANSFORMERS_OFFLINE"] = "1"
                ts_env["HF_DATASETS_OFFLINE"] = "1"

                ts_res = run_subordinate_process(
                    cmd=ts_cmd,
                    deadline=ctx.deadline,
                    check_breach=ctx.check_breach,
                    cwd=repo_root,
                    env=ts_env,
                    capture_log_prefix="phase5_ts_",
                )

                ctx.check_breach("ts_runner_completed")

                if ts_res.timed_out:
                    err_msg = f"TypeScript Phase 5 runner timed out after {ts_res.duration_s}s (PID {ts_res.pid})"
                    errors.append(err_msg)
                    ctx.add_error(err_msg)
                elif ts_res.breached:
                    err_msg = f"Supervisor watchdog breach during TypeScript runner execution: {ts_res.error}"
                    errors.append(err_msg)
                    ctx.add_error(err_msg)
                elif ts_res.exit_code != 0:
                    err_msg = f"TypeScript Phase 5 runner failed with exit code {ts_res.exit_code}:\n{ts_res.captured_logs}"
                    errors.append(err_msg)
                    ctx.add_error(err_msg)

                ts_report_data: dict[str, Any] = {}
                if os.path.exists(tmp_ts_report):
                    try:
                        with open(tmp_ts_report, "r", encoding="utf-8") as f:
                            ts_report_data = json.load(f)
                    finally:
                        try:
                            os.unlink(tmp_ts_report)
                        except Exception:
                            pass

                # Consolidate TS checks and recovery results
                ts_checks = ts_report_data.get("checks", {})
                report["checks"] = {
                    "shadow_target_safety": ts_checks.get("shadowTargetSafety", False),
                    "durable_indexing_cancellation": ts_checks.get("durableIndexingCancellation", False),
                    "durable_indexing_resumption": ts_checks.get("durableIndexingResumption", False),
                    "vector_deduplication_reconciled": ts_checks.get("vectorDeduplicationReconciled", False),
                    "fingerprint_mismatch_refused": ts_checks.get("fingerprintMismatchRefused", False),
                    "sqlite_online_backup_restored": ts_checks.get("sqliteOnlineBackupRestored", False),
                    "restored_retrieval_identical": ts_checks.get("restoredRetrievalIdentical", False),
                    "mcp_transport_probe_passed": ts_checks.get("mcpTransportProbePassed", False),
                    "mlx4b_retrieval_complete": ts_checks.get("mlx4bRetrievalComplete", False),
                    "gguf06b_stage_blocked": True,
                }
                report["corpus"] = ts_report_data.get("corpus", {})
                report["recovery"] = ts_report_data.get("recovery", {})
                report["retrieval_comparison"]["mlx_4b"] = ts_report_data.get("retrieval", {}).get("mlx4b", {
                    "status": "completed" if ts_checks.get("mlx4bRetrievalComplete") else "failed",
                })
                report["mcp_probe"] = ts_report_data.get("mcpProbe", {})

                # Read final telemetry metrics
                st_mem, r_mem = ctx.client.get_json(f"{ctx.ctrl_url}/memory", deadline=ctx.deadline, timeout_s=2.0)
                if st_mem == 200 and isinstance(r_mem, dict):
                    report["memory_metrics"] = r_mem

                st_stats, r_stats = ctx.client.get_json(f"{ctx.ctrl_url}/stats", deadline=ctx.deadline, timeout_s=2.0)
                if st_stats == 200 and isinstance(r_stats, dict):
                    report["stats_metrics"] = r_stats

                all_passed = (
                    all(report["checks"].values())
                    and ts_res.exit_code == 0
                    and not ts_res.timed_out
                    and not ts_res.breached
                    and len(errors) == 0
                )
                report["status"] = "passed" if all_passed else "failed"

        except Exception as e:
            report["status"] = "failed"
            errors.append(str(e))
            report["errors"] = errors
        finally:
            report["duration_s"] = round(time.time() - overall_t0, 3)
            report["errors"] = errors
            if ctx is not None:
                report["supervision"] = ctx.report
                if ctx.report.get("errors"):
                    report["status"] = "failed"
            self._save_report(report)
            self.last_report = report

        return report


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Isolated End-to-End Qualification Supervisor")
    parser.add_argument("--output-json", default="docs/reviews/artifacts/phase5-isolated-e2e.json")
    parser.add_argument("--mlx-port", type=int, default=8797)
    parser.add_argument("--mlx-control-port", type=int, default=8798)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--min-headroom-mb", type=float, default=6000.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = Phase5SupervisorConfig(
        mlx_port=args.mlx_port,
        mlx_control_port=args.mlx_control_port,
        timeout_s=args.timeout_s,
        min_headroom_mb=args.min_headroom_mb,
        output_json=args.output_json,
        dry_run=args.dry_run,
    )

    supervisor = Phase5Supervisor(config)
    report = supervisor.run()
    sys.exit(0 if report.get("status") == "passed" else 1)


if __name__ == "__main__":
    main()
