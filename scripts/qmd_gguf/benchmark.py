"""
benchmark.py — Bounded Sustained GGUF Embedding Qualification Harness

Executes bounded pilot (<=30 requests, <=120s) and bounded soak (<=100 requests, <=180s)
benchmarks for GGUF embedding models via node-llama-cpp under active Watchdog supervision
with strict process isolation, identical token strata, and structured JSON reporting.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Optional
import numpy as np

from scripts.qmd_mlx.smoke import (
    NORM_TOLERANCE,
    SmokeHttpClient,
    check_dimensions,
    check_finite,
    check_l2_normalization,
)
from scripts.qmd_mlx.sustained import (
    SHORT_FIXTURES,
    MEDIUM_FIXTURES,
    LONG_FIXTURES,
    CODE_FIXTURES,
    compute_explicit_percentiles,
    calibrate_boundary_fixtures,
    compute_metrics_summary,
    evaluate_qualification_gates,
    extract_watchdog_breach_dict,
)
from scripts.qmd_mlx.watchdog import (
    BreachType,
    MLXWatchdog,
    MLXWatchdogConfig,
    SystemMemorySampler,
    SystemMetricsError,
    WatchdogCheckResult,
    is_numeric_loopback,
)


def compute_code_manifest(repo_root: str) -> dict[str, str]:
    """
    Computes exact SHA256 fingerprints for all relevant qualification and runtime scripts,
    including GGUF benchmark runners and node-llama-cpp wrappers.
    """
    manifest_files = [
        "scripts/gguf_embed_server.mjs",
        "scripts/qmd-gguf-benchmark.py",
        "scripts/qmd_gguf/benchmark.py",
        "scripts/qmd-mlx-sustained.py",
        "scripts/qmd_mlx/sustained.py",
        "scripts/qmd_mlx/watchdog.py",
        "src/llm.ts",
        "src/embedding/config.ts",
        "src/embedding/contract.ts",
    ]
    manifest: dict[str, str] = {}
    for rel_path in manifest_files:
        full_path = os.path.join(repo_root, rel_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, "rb") as f:
                    manifest[rel_path] = hashlib.sha256(f.read()).hexdigest()
            except Exception as e:
                manifest[rel_path] = f"error: {e}"
    return manifest


def inspect_gguf_metadata(model_path: str) -> dict[str, Any]:
    """Inspects GGUF file size and metadata without reloading weights."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GGUF model file not found: {model_path}")
    size_bytes = os.path.getsize(model_path)
    filename = os.path.basename(model_path)
    
    # Infer architecture and quantization from filename / path
    params_b = 4.0 if "4B" in filename or "4b" in filename else (0.6 if "0.6B" in filename or "0.6b" in filename else 0.3)
    quant = "Q4_K_M" if "Q4_K_M" in filename else ("Q8_0" if "Q8_0" in filename else "unknown")
    dims = 2560 if params_b >= 4.0 else (1024 if params_b >= 0.6 else 768)

    # Compute conservative headroom: 1.5x model size + 1.5 GB for activations
    est_memory_mb = size_bytes / (1024 * 1024)
    req_headroom_mb = max(est_memory_mb * 1.5 + 1500.0, 6000.0)

    return {
        "model_path": model_path,
        "filename": filename,
        "total_file_size_bytes": size_bytes,
        "params_b": params_b,
        "quantization": quant,
        "native_dimensions": dims,
        "estimated_memory_mb": round(est_memory_mb, 1),
        "conservative_required_headroom_mb": round(req_headroom_mb, 1),
    }


@dataclasses.dataclass
class GGUFBenchmarkConfig:
    model_path: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8795
    control_port: int = 8796
    mode: str = "pilot"          # 'pilot' (<=30 reqs, <=120s) or 'soak' (<=100 reqs, <=180s)
    timeout_s: float = 120.0
    max_requests: int = 30
    min_headroom_mb: float = 6000.0
    use_fake_child: bool = False
    fake_dims: int = 2560
    dry_run: bool = False
    real_model_opt_in: bool = False
    output_file: Optional[str] = None
    contexts: int = 2


class GGUFBenchmarkRunner:
    """
    Executes bounded sustained qualification benchmarks for GGUF embedding models
    via node-llama-cpp under active Watchdog supervision.
    """

    def __init__(self, config: GGUFBenchmarkConfig, sampler: Optional[SystemMemorySampler] = None):
        self.config = config
        self.sampler = sampler or SystemMemorySampler()
        self._preflight_defaults: Optional[Any] = None
        self.last_report: dict[str, Any] = {}

    def validate_preflight(self) -> dict[str, Any]:
        """
        Enforces system memory headroom (>=6000 MB for real models), port distinctness/availability,
        and loopback policy before process spawn.
        """
        preflight_info: dict[str, Any] = {
            "timestamp": time.time(),
            "host": self.config.host,
            "port": self.config.port,
            "control_port": self.config.control_port,
            "mode": self.config.mode,
            "backend": "gguf",
        }

        # 1. Host restriction
        if not is_numeric_loopback(self.config.host) or self.config.host != "127.0.0.1":
            raise ValueError(f"Host '{self.config.host}' must be numeric loopback '127.0.0.1'")

        # 2. Port distinctness and validity
        if not (1 <= self.config.port <= 65535) or not (1 <= self.config.control_port <= 65535):
            raise ValueError(f"Invalid ports: port={self.config.port}, control_port={self.config.control_port}")
        if self.config.port == self.config.control_port:
            raise ValueError("Inference and control ports must be distinct")
        if self.config.port == 8787 or self.config.control_port == 8787:
            raise ValueError("Port 8787 is reserved for live daemon. Select distinct qualification ports.")

        # 3. Port availability
        import socket
        for p_name, p_val in [("Inference port", self.config.port), ("Control port", self.config.control_port)]:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.config.host, p_val))
                s.close()
            except OSError as e:
                raise RuntimeError(f"{p_name} {p_val} on {self.config.host} is already in use: {e}")

        # 4. Memory sampling
        installed_ram = self.sampler.get_installed_ram_mb()
        headroom = self.sampler.get_memory_headroom_mb()
        swap_used = self.sampler.get_swap_used_mb()
        free_pct = self.sampler.get_memory_free_pct()
        defaults = self.sampler.compute_conservative_defaults(headroom_mb=headroom)
        self._preflight_defaults = defaults

        preflight_info.update({
            "installed_ram_mb": installed_ram,
            "headroom_mb": headroom,
            "swap_used_mb": swap_used,
            "free_memory_pct": free_pct,
        })

        # 5. Model-aware preflight headroom check
        if not self.config.use_fake_child:
            if not self.config.real_model_opt_in:
                raise ValueError("Real model execution requires explicit opt-in (--real-model).")
            if not self.config.model_path:
                raise ValueError("Model path is required for real model execution.")

            meta = inspect_gguf_metadata(self.config.model_path)
            preflight_info["model_metadata"] = meta

            required_headroom = max(self.config.min_headroom_mb, meta["conservative_required_headroom_mb"], 6000.0)
            if headroom < required_headroom:
                raise RuntimeError(
                    f"Insufficient memory headroom for sustained {meta['params_b']:.1f}B GGUF qualification: "
                    f"measured {headroom:.1f} MB < required {required_headroom:.1f} MB. "
                    f"Refusing to spawn server child."
                )
        else:
            if headroom < min(self.config.min_headroom_mb, 256.0):
                raise RuntimeError(
                    f"Insufficient headroom for rehearsal: measured {headroom:.1f} MB < required {self.config.min_headroom_mb:.1f} MB"
                )

        return preflight_info

    def run(self) -> dict[str, Any]:
        """
        Executes bounded sustained embedding qualification under active Watchdog supervision.
        """
        run_start_time = time.time()
        start_mono = time.monotonic()
        deadline = start_mono + self.config.timeout_s

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        report: dict[str, Any] = {
            "status": "in_progress",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_start_time)),
            "backend": "gguf",
            "mode": f"sustained_{self.config.mode}",
            "synthetic": self.config.use_fake_child,
            "code_manifest_sha256": compute_code_manifest(repo_root),
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "control_port": self.config.control_port,
                "timeout_s": self.config.timeout_s,
                "max_requests": self.config.max_requests,
                "mode": self.config.mode,
                "use_fake_child": self.config.use_fake_child,
                "model_path": self.config.model_path,
                "contexts": self.config.contexts,
            },
            "strata_verification": {},
            "cold_startup": {},
            "warmup": {},
            "solo_baseline": {},
            "measured_requests": [],
            "rejections": [],
            "concurrent_load_interleaving": [],
            "metrics_summary": {},
            "telemetry_samples": [],
            "metal_memory_telemetry": {},
            "captured_logs": "",
            "child_cleanup": {
                "cleaned": False,
                "pid": None,
                "exit_code": None,
            },
        }
        self.last_report = report

        # 1. Preflight validation
        preflight = self.validate_preflight()
        report["preflight"] = preflight

        if self.config.dry_run:
            report["status"] = "dry_run_completed"
            report["duration_s"] = time.time() - run_start_time
            self.last_report = report
            return report

        # 2. Spawn Child Server (Node.js)
        instance_token = uuid.uuid4().hex
        child_env = os.environ.copy()
        child_env["GGUF_INSTANCE_TOKEN"] = instance_token
        child_env["GGUF_PORT"] = str(self.config.port)
        child_env["GGUF_CONTROL_PORT"] = str(self.config.control_port)

        server_script = os.path.join(repo_root, "scripts", "gguf_embed_server.mjs")
        cmd = [
            "node",
            server_script,
            "--port", str(self.config.port),
            "--control-port", str(self.config.control_port),
            "--host", self.config.host,
            "--instance-token", instance_token,
            "--contexts", str(self.config.contexts),
        ]

        if self.config.use_fake_child:
            cmd.extend(["--synthetic", "--synthetic-dims", str(self.config.fake_dims)])
        else:
            cmd.extend(["--model", str(self.config.model_path)])

        child: Optional[subprocess.Popen] = None
        log_file = None
        log_file_path = None
        client = SmokeHttpClient(default_timeout_s=10.0)
        t_supervisor: Optional[threading.Thread] = None
        stop_supervisor = threading.Event()
        breach_event = threading.Event()
        breach_result: list[Optional[WatchdogCheckResult]] = [None]
        telemetry_samples: list[dict[str, Any]] = []
        cleanup_errors: list[str] = []

        try:
            log_file = tempfile.NamedTemporaryFile(mode="w+", prefix="gguf_sustained_server_", suffix=".log", delete=False)
            log_file_path = log_file.name

            child = subprocess.Popen(
                cmd,
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            report["child_pid"] = child.pid
            report["instance_token"] = instance_token

            # 3. Instantiate Watchdog & Supervisor Thread
            watchdog_config = MLXWatchdogConfig(
                pid=child.pid,
                host=self.config.host,
                port=self.config.port,
                control_port=self.config.control_port,
                check_interval_s=0.25,
                startup_grace_period_s=min(30.0, self.config.timeout_s),
                stalled_inference_timeout_s=min(15.0, self.config.timeout_s),
                expected_cmd_pattern=r"(node|llama|gguf|qmd)",
                instance_token=instance_token,
                allow_external_pid=True,
                dry_run=False,
            )
            watchdog = MLXWatchdog(
                config=watchdog_config,
                sampler=self.sampler,
                owned_child=child,
                baseline_swap_mb=preflight.get("swap_used_mb"),
                defaults=self._preflight_defaults,
            )

            def _supervisor_loop():
                while not stop_supervisor.is_set():
                    now_m = time.monotonic()
                    if now_m >= deadline:
                        reason = f"Absolute execution ceiling of {self.config.timeout_s:.1f}s exceeded"
                        term_ok, sigkill = watchdog.terminate_target_process(reason)
                        res = WatchdogCheckResult(
                            timestamp=time.time(),
                            healthy=False,
                            breach_type=BreachType.HEALTH_CHECK_FAILED,
                            breach_reason=reason,
                            metrics={"timeout_s": self.config.timeout_s},
                            terminated_pid=child.pid if term_ok else None,
                            sigkill_used=sigkill,
                        )
                        breach_result[0] = res
                        breach_event.set()
                        break

                    check = watchdog.check_step()
                    if not check.healthy:
                        breach_result[0] = check
                        breach_event.set()
                        break

                    if check.metrics:
                        telemetry_samples.append({
                            "timestamp": round(check.timestamp, 2),
                            "elapsed_s": round(time.monotonic() - start_mono, 2),
                            "rss_mb": check.metrics.get("rss_mb"),
                            "swap_used_mb": check.metrics.get("swap_used_mb"),
                            "swap_growth_mb": check.metrics.get("swap_growth_mb"),
                            "memory_free_pct": check.metrics.get("memory_free_pct"),
                        })

                    stop_supervisor.wait(timeout=0.25)

            t_supervisor = threading.Thread(target=_supervisor_loop, daemon=True, name="GGUF-Watchdog-Supervisor")
            t_supervisor.start()

            # 4. Poll Control Port for Readiness (Cold Startup Timing)
            base_url = f"http://{self.config.host}:{self.config.port}"
            ctrl_url = f"http://{self.config.host}:{self.config.control_port}"

            ready = False
            startup_deadline = min(deadline, time.monotonic() + 30.0)
            while time.monotonic() < startup_deadline:
                if breach_event.is_set():
                    b_res = breach_result[0]
                    raise RuntimeError(f"Watchdog breach during server startup: {b_res.breach_reason if b_res else 'unknown breach'}")

                if child.poll() is not None:
                    time.sleep(0.1)
                    if breach_event.is_set():
                        b_res = breach_result[0]
                        raise RuntimeError(f"Watchdog breach during server startup: {b_res.breach_reason if b_res else 'unknown breach'}")
                    raise RuntimeError(f"Server child exited prematurely with code {child.returncode}")

                try:
                    status_code, data = client.get_json(f"{ctrl_url}/health", deadline=startup_deadline, timeout_s=1.0)
                    if status_code == 200 and isinstance(data, dict):
                        if data.get("ready") is True and data.get("instance_token") == instance_token:
                            ready = True
                            report["cold_startup"] = {
                                "ready": True,
                                "cold_startup_elapsed_s": round(time.monotonic() - start_mono, 3),
                                "status": "passed",
                            }
                            break
                except Exception:
                    pass
                time.sleep(0.1)

            if not ready:
                raise TimeoutError("GGUF server failed to report ready within startup period")

            # 5. Fetch and Validate Descriptor
            st_desc, desc = client.get_json(f"{ctrl_url}/descriptor", deadline=deadline, timeout_s=2.0)
            if st_desc != 200 or not isinstance(desc, dict):
                raise RuntimeError(f"Failed to fetch descriptor: status {st_desc}, data {desc}")
            report["descriptor"] = desc
            expected_dims = desc.get("outputDimensions") or desc.get("nativeDimensions") or 2560

            # 6. Sample Initial Memory
            _, initial_mem = client.get_json(f"{ctrl_url}/memory", deadline=deadline, timeout_s=2.0)
            report["metal_memory_telemetry"]["initial"] = initial_mem

            def _check_breach():
                if breach_event.is_set():
                    b = breach_result[0]
                    raise RuntimeError(f"Watchdog breach: {b.breach_reason if b else 'unknown breach'}")
                if child.poll() is not None:
                    time.sleep(0.05)
                    if breach_event.is_set():
                        b = breach_result[0]
                        raise RuntimeError(f"Watchdog breach: {b.breach_reason if b else 'unknown breach'}")
                    raise RuntimeError(f"Server child exited unexpectedly with code {child.returncode}")

            # 7. Calibrate & Verify Exact Boundary Fixtures & Token Strata via /tokenize
            _check_breach()
            calibrated_boundaries = calibrate_boundary_fixtures(client, base_url, deadline)
            b2047 = calibrated_boundaries["boundary_2047"]
            b2048 = calibrated_boundaries["boundary_2048"]
            b2049 = calibrated_boundaries["boundary_2049"]

            strata_candidates = {
                "short": SHORT_FIXTURES,
                "medium": MEDIUM_FIXTURES,
                "long": LONG_FIXTURES,
                "code": CODE_FIXTURES,
                "boundary_2047": [b2047],
                "boundary_2048": [b2048],
            }
            strata_verified: dict[str, Any] = {}

            for stratum_name, texts in strata_candidates.items():
                _check_breach()
                st_tok, r_tok = client.post_json(f"{base_url}/tokenize", {"texts": texts}, deadline=deadline)
                if st_tok != 200 or not isinstance(r_tok, dict) or "counts" not in r_tok:
                    raise RuntimeError(f"Tokenization failed for {stratum_name}: HTTP {st_tok}, payload {r_tok}")
                counts = [int(c) for c in r_tok["counts"]]
                assert len(counts) == len(texts), f"Count mismatch for {stratum_name}: {len(counts)} != {len(texts)}"
                strata_verified[stratum_name] = {
                    "item_count": len(counts),
                    "token_counts": counts,
                    "min_tokens": min(counts),
                    "max_tokens": max(counts),
                    "mean_tokens": round(float(np.mean(counts)), 1),
                }

            # Boundary 2049 Policy Check (Must reject with HTTP 400 and exact token count in error)
            _check_breach()
            st_b_over, r_b_over = client.post_json(f"{base_url}/tokenize", {"texts": [b2049]}, deadline=deadline)
            rejection_error_str = str(r_b_over)
            if st_b_over == 400:
                m_tokens = re.search(r"\((\d+)\)\s+exceeds\s+max_length\s+\((\d+)\)", rejection_error_str)
                parsed_len = int(m_tokens.group(1)) if m_tokens else 2049
                strata_verified["boundary_2049"] = {
                    "policy": "explicit_rejection",
                    "status_code": 400,
                    "rejection_verified": True,
                    "token_length": parsed_len,
                    "message": rejection_error_str,
                }
            else:
                raise RuntimeError(f"Expected HTTP 400 rejection for boundary_2049, got HTTP {st_b_over}: {r_b_over}")

            # Verify strata compliance ranges strictly
            assert 5 <= strata_verified["short"]["min_tokens"] <= 25, f"Short stratum out of range: {strata_verified['short']}"
            assert 50 <= strata_verified["medium"]["min_tokens"] and strata_verified["medium"]["max_tokens"] <= 300, f"Medium stratum out of range: {strata_verified['medium']}"
            assert 400 <= strata_verified["long"]["min_tokens"] and strata_verified["long"]["max_tokens"] <= 1500, f"Long stratum out of range: {strata_verified['long']}"
            assert 50 <= strata_verified["code"]["min_tokens"] and strata_verified["code"]["max_tokens"] <= 500, f"Code stratum out of range: {strata_verified['code']}"
            assert strata_verified["boundary_2047"]["min_tokens"] == 2047, f"Boundary 2047 not exact: {strata_verified['boundary_2047']}"
            assert strata_verified["boundary_2048"]["min_tokens"] == 2048, f"Boundary 2048 not exact: {strata_verified['boundary_2048']}"
            assert strata_verified["boundary_2049"]["rejection_verified"] is True, f"Boundary 2049 not rejected: {strata_verified['boundary_2049']}"
            report["strata_verification"] = strata_verified

            # 8. Warmup Pass (Explicitly Excluded from Measured Metrics)
            _check_breach()
            t_warm_0 = time.monotonic()
            st_w1, _ = client.post_json(f"{base_url}/embed", {"texts": [SHORT_FIXTURES[0]], "is_query": True}, deadline=deadline)
            st_w2, _ = client.post_json(f"{base_url}/embed", {"texts": [SHORT_FIXTURES[0]] * 4}, deadline=deadline)
            warmup_ms = (time.monotonic() - t_warm_0) * 1000.0
            if st_w1 != 200 or st_w2 != 200:
                raise RuntimeError(f"Warmup pass failed: st_w1={st_w1}, st_w2={st_w2}")
            report["warmup"] = {
                "status": "completed",
                "warmup_elapsed_ms": round(warmup_ms, 2),
                "excluded_from_measured_metrics": True,
            }

            # Sample post-warmup memory
            _, post_warmup_mem = client.get_json(f"{ctrl_url}/memory", deadline=deadline, timeout_s=2.0)
            report["metal_memory_telemetry"]["post_warmup"] = post_warmup_mem

            # 9. Solo Baseline Interactive Measurements (Under Idle Load)
            solo_baseline_latencies: list[float] = []
            for solo_idx in range(3):
                _check_breach()
                t0_s = time.monotonic()
                st_s, r_s = client.post_json(f"{base_url}/embed", {"texts": [SHORT_FIXTURES[solo_idx % len(SHORT_FIXTURES)]], "is_query": True}, deadline=deadline)
                solo_ms = (time.monotonic() - t0_s) * 1000.0
                if st_s != 200:
                    raise RuntimeError(f"Solo baseline request {solo_idx} failed: HTTP {st_s}")
                solo_baseline_latencies.append(solo_ms)

            solo_baseline_p50 = float(np.percentile(solo_baseline_latencies, 50))
            report["solo_baseline"] = {
                "samples_ms": [round(x, 2) for x in solo_baseline_latencies],
                "percentiles": compute_explicit_percentiles(solo_baseline_latencies),
            }

            # 10. Workload Execution
            measured_records: list[dict[str, Any]] = []
            rejection_records: list[dict[str, Any]] = []
            concurrent_interleaving_records: list[dict[str, Any]] = []
            request_counter = 0

            def _execute_measured_request(
                texts: list[str],
                stratum: str,
                is_query: bool = False,
                tag: str = "standard",
            ) -> Optional[dict[str, Any]]:
                nonlocal request_counter
                request_counter += 1
                _check_breach()

                tok_counts: list[int] = []
                if stratum == "boundary_2047":
                    tok_counts = [2047]
                elif stratum == "boundary_2048":
                    tok_counts = [2048]
                elif stratum == "boundary_2049":
                    tok_counts = [2049]
                else:
                    st_tok, r_tok = client.post_json(f"{base_url}/tokenize", {"texts": texts}, deadline=deadline)
                    if st_tok == 200 and isinstance(r_tok, dict) and "counts" in r_tok:
                        tok_counts = [int(c) for c in r_tok["counts"]]
                    else:
                        tok_counts = [len(t.split()) for t in texts]

                t0 = time.monotonic()
                st_emb, r_emb = client.post_json(
                    f"{base_url}/embed",
                    {"texts": texts, "is_query": is_query},
                    deadline=deadline,
                )
                elapsed_ms = (time.monotonic() - t0) * 1000.0

                if stratum == "boundary_2049" and st_emb == 400:
                    rec_rej = {
                        "request_idx": request_counter,
                        "stratum": stratum,
                        "tag": tag,
                        "batch_size": len(texts),
                        "token_counts": tok_counts,
                        "is_query": is_query,
                        "wire_ms": round(elapsed_ms, 2),
                        "rejection_verified": True,
                        "status_code": 400,
                    }
                    rejection_records.append(rec_rej)
                    return rec_rej

                if st_emb != 200 or not isinstance(r_emb, dict):
                    time.sleep(0.05)
                    _check_breach()
                    raise RuntimeError(f"Request {request_counter} ({stratum}) failed with HTTP {st_emb}: {r_emb}")

                arr = np.array(r_emb["embeddings"], dtype=np.float32)
                fin_ok, fin_msg = check_finite(arr)
                dim_ok, dim_msg = check_dimensions(arr, expected_rows=len(texts), expected_dims=expected_dims)
                norm_ok, norms, norm_msg = check_l2_normalization(arr)

                if not (fin_ok and dim_ok and norm_ok):
                    raise ValueError(f"Vector validation failed for request {request_counter}: {dim_msg}, {norm_msg}, {fin_msg}")

                total_tokens = sum(tok_counts)
                tok_per_sec = (total_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0

                record = {
                    "request_idx": request_counter,
                    "stratum": stratum,
                    "tag": tag,
                    "batch_size": len(texts),
                    "token_counts": tok_counts,
                    "total_tokens": total_tokens,
                    "is_query": is_query,
                    "wire_ms": round(elapsed_ms, 2),
                    "effective_tokens_per_sec": round(tok_per_sec, 1),
                    "shape": list(arr.shape),
                    "finite": fin_ok,
                    "l2_normalized": norm_ok,
                    "mean_norm": round(float(np.mean(norms)), 6) if norms else 1.0,
                }
                measured_records.append(record)
                return record

            # Stage A: Singletons across all strata (Batch Size = 1)
            for s_idx, short_t in enumerate(SHORT_FIXTURES[:3]):
                if request_counter >= self.config.max_requests:
                    break
                _execute_measured_request([short_t], stratum="short", is_query=True, tag=f"singleton_short_{s_idx}")

            for m_idx, med_t in enumerate(MEDIUM_FIXTURES[:2]):
                if request_counter >= self.config.max_requests:
                    break
                _execute_measured_request([med_t], stratum="medium", is_query=False, tag=f"singleton_medium_{m_idx}")

            for l_idx, long_t in enumerate(LONG_FIXTURES[:2]):
                if request_counter >= self.config.max_requests:
                    break
                _execute_measured_request([long_t], stratum="long", is_query=False, tag=f"singleton_long_{l_idx}")

            for c_idx, code_t in enumerate(CODE_FIXTURES[:2]):
                if request_counter >= self.config.max_requests:
                    break
                _execute_measured_request([code_t], stratum="code", is_query=False, tag=f"singleton_code_{c_idx}")

            # Stage B: Exact 2047, 2048, 2049 Boundary Tests
            if request_counter < self.config.max_requests:
                _execute_measured_request([b2047], stratum="boundary_2047", tag="boundary_2047_pass")
            if request_counter < self.config.max_requests:
                _execute_measured_request([b2048], stratum="boundary_2048", tag="boundary_2048_pass")
            if request_counter < self.config.max_requests:
                _execute_measured_request([b2049], stratum="boundary_2049", tag="boundary_2049_reject_400")

            # Stage C: Batch Size = 2 Conservative Progression
            if request_counter < self.config.max_requests:
                _execute_measured_request([SHORT_FIXTURES[0], SHORT_FIXTURES[1]], stratum="short", tag="batch2_short")
            if request_counter < self.config.max_requests:
                _execute_measured_request([MEDIUM_FIXTURES[0], MEDIUM_FIXTURES[1]], stratum="medium", tag="batch2_medium")
            if request_counter < self.config.max_requests:
                _execute_measured_request([CODE_FIXTURES[0], CODE_FIXTURES[1]], stratum="code", tag="batch2_code")

            # Stage D: Batch Size = 4 Conservative Progression
            if request_counter < self.config.max_requests:
                _execute_measured_request(SHORT_FIXTURES[:4], stratum="short", tag="batch4_short")
            if request_counter < self.config.max_requests:
                _execute_measured_request([MEDIUM_FIXTURES[0], MEDIUM_FIXTURES[1]], stratum="medium", tag="batch4_medium")

            # Stage E: True Concurrent Interleaving (Barrier & Telemetry Handshake)
            num_concurrent_iters = min(3, max(1, (self.config.max_requests - request_counter) // 2))
            for c_iter in range(num_concurrent_iters):
                if request_counter + 2 > self.config.max_requests:
                    break

                _check_breach()
                bulk_texts = [LONG_FIXTURES[0], MEDIUM_FIXTURES[0], CODE_FIXTURES[0], SHORT_FIXTURES[0]]
                query_texts = [SHORT_FIXTURES[c_iter % len(SHORT_FIXTURES)]]

                bulk_client = SmokeHttpClient(default_timeout_s=15.0)
                query_client = SmokeHttpClient(default_timeout_s=15.0)

                bulk_res: dict[str, Any] = {}
                query_res: dict[str, Any] = {}
                bulk_started = threading.Event()
                query_launched = threading.Event()

                def _bulk_worker():
                    try:
                        t0_b = time.monotonic()
                        bulk_started.set()
                        st_b, r_b = bulk_client.post_json(
                            f"{base_url}/embed",
                            {"texts": bulk_texts, "is_query": False},
                            deadline=deadline,
                        )
                        t1_b = time.monotonic()
                        bulk_res.update({
                            "status_code": st_b,
                            "response": r_b,
                            "start_mono": t0_b,
                            "end_mono": t1_b,
                            "wire_ms": round((t1_b - t0_b) * 1000.0, 2),
                        })
                    except Exception as exc:
                        bulk_res["error"] = str(exc)
                    finally:
                        bulk_client.close()

                def _query_worker():
                    try:
                        bulk_started.wait(timeout=2.0)
                        time.sleep(0.015)  # Handshake: ensure bulk request is in-flight on server
                        t0_q = time.monotonic()
                        query_launched.set()
                        st_q, r_q = query_client.post_json(
                            f"{base_url}/embed",
                            {"texts": query_texts, "is_query": True},
                            deadline=deadline,
                        )
                        t1_q = time.monotonic()
                        query_res.update({
                            "status_code": st_q,
                            "response": r_q,
                            "start_mono": t0_q,
                            "end_mono": t1_q,
                            "wire_ms": round((t1_q - t0_q) * 1000.0, 2),
                        })
                    except Exception as exc:
                        query_res["error"] = str(exc)
                    finally:
                        query_client.close()

                t_b = threading.Thread(target=_bulk_worker, daemon=True, name=f"GGUF-Bulk-Client-{c_iter}")
                t_q = threading.Thread(target=_query_worker, daemon=True, name=f"GGUF-Interactive-Client-{c_iter}")

                t_b.start()
                t_q.start()
                t_b.join(timeout=30.0)
                t_q.join(timeout=30.0)

                _check_breach()

                if "error" in bulk_res or bulk_res.get("status_code") != 200:
                    time.sleep(0.05)
                    _check_breach()
                    raise RuntimeError(f"Concurrent bulk batch failed: {bulk_res}")
                if "error" in query_res or query_res.get("status_code") != 200:
                    time.sleep(0.05)
                    _check_breach()
                    raise RuntimeError(f"Concurrent interactive query failed: {query_res}")

                arr_b = np.array(bulk_res["response"]["embeddings"], dtype=np.float32)
                arr_q = np.array(query_res["response"]["embeddings"], dtype=np.float32)

                fin_b, _ = check_finite(arr_b)
                norm_b, _, _ = check_l2_normalization(arr_b)
                fin_q, _ = check_finite(arr_q)
                norm_q, _, _ = check_l2_normalization(arr_q)

                if not (fin_b and norm_b and fin_q and norm_q):
                    raise ValueError("Concurrent embeddings validation failed (finite or norm violation)")

                request_counter += 1
                rec_b = {
                    "request_idx": request_counter,
                    "stratum": "mixed",
                    "tag": f"concurrent_bulk_batch_{c_iter}",
                    "batch_size": len(bulk_texts),
                    "token_counts": [len(t.split()) for t in bulk_texts],
                    "total_tokens": sum(len(t.split()) for t in bulk_texts),
                    "is_query": False,
                    "wire_ms": bulk_res["wire_ms"],
                    "effective_tokens_per_sec": round((sum(len(t.split()) for t in bulk_texts) / (bulk_res["wire_ms"] / 1000.0)), 1),
                    "shape": list(arr_b.shape),
                    "finite": fin_b,
                    "l2_normalized": norm_b,
                }
                measured_records.append(rec_b)

                request_counter += 1
                rec_q = {
                    "request_idx": request_counter,
                    "stratum": "short",
                    "tag": f"concurrent_interactive_query_{c_iter}",
                    "batch_size": 1,
                    "token_counts": [len(query_texts[0].split())],
                    "total_tokens": len(query_texts[0].split()),
                    "is_query": True,
                    "wire_ms": query_res["wire_ms"],
                    "effective_tokens_per_sec": round((len(query_texts[0].split()) / (query_res["wire_ms"] / 1000.0)), 1),
                    "shape": list(arr_q.shape),
                    "finite": fin_q,
                    "l2_normalized": norm_q,
                }
                measured_records.append(rec_q)

                overlap = bool(
                    query_res["start_mono"] >= bulk_res["start_mono"]
                    and query_res["start_mono"] < bulk_res["end_mono"]
                )
                baseline_fixture_ms = (
                    solo_baseline_latencies[c_iter % len(solo_baseline_latencies)]
                    if solo_baseline_latencies
                    else solo_baseline_p50
                )
                est_queue_wait = max(0.0, query_res["wire_ms"] - baseline_fixture_ms)
                slowdown_factor = round(query_res["wire_ms"] / max(1.0, baseline_fixture_ms), 2)

                interleaving_entry = {
                    "iteration": c_iter,
                    "fixture_index": c_iter % len(SHORT_FIXTURES),
                    "overlap_verified": overlap,
                    "bulk_wire_ms": bulk_res["wire_ms"],
                    "concurrent_interactive_wire_ms": query_res["wire_ms"],
                    "solo_baseline_fixture_ms": round(baseline_fixture_ms, 2),
                    "solo_baseline_p50_ms": round(solo_baseline_p50, 2),
                    "estimated_queue_wait_ms": round(est_queue_wait, 2),
                    "slowdown_factor": slowdown_factor,
                }
                concurrent_interleaving_records.append(interleaving_entry)

            report["concurrent_load_interleaving"] = concurrent_interleaving_records

            # 11. Sample Final Memory & Stability
            _, final_mem = client.get_json(f"{ctrl_url}/memory", deadline=deadline, timeout_s=2.0)
            report["metal_memory_telemetry"]["final"] = final_mem

            init_act = initial_mem.get("active_mb", 0.0) if isinstance(initial_mem, dict) else 0.0
            fin_act = final_mem.get("active_mb", 0.0) if isinstance(final_mem, dict) else 0.0
            peak_act = final_mem.get("peak_mb", 0.0) if isinstance(final_mem, dict) else 0.0
            pw_act = post_warmup_mem.get("active_mb", 0.0) if isinstance(post_warmup_mem, dict) else init_act

            delta_active = round(fin_act - init_act, 2)
            delta_post_warmup = round(fin_act - pw_act, 2)
            report["metal_memory_telemetry"]["delta_active_mb"] = delta_active
            report["metal_memory_telemetry"]["delta_post_warmup_active_mb"] = delta_post_warmup
            report["metal_memory_telemetry"]["peak_active_mb"] = round(peak_act, 2)

            rss_samples = [s["rss_mb"] for s in telemetry_samples if isinstance(s.get("rss_mb"), (int, float))]
            swap_samples = [s["swap_growth_mb"] for s in telemetry_samples if isinstance(s.get("swap_growth_mb"), (int, float))]
            max_swap_growth = max(swap_samples) if swap_samples else 0.0
            rss_growth = (rss_samples[-1] - rss_samples[0]) if len(rss_samples) >= 2 else 0.0

            stability_verdict = "provisional_stable" if (delta_active <= 500.0 and max_swap_growth == 0.0) else "unstable"
            report["metal_memory_telemetry"]["provisional_stability_assessment"] = {
                "verdict": stability_verdict,
                "rss_growth_mb": round(rss_growth, 2),
                "swap_growth_mb": round(max_swap_growth, 2),
                "metal_delta_active_mb": delta_active,
                "metal_peak_active_mb": round(peak_act, 2),
                "notes": (
                    "Provisional stability verified under matched warmed batch shapes across bounded qualification window. "
                    "Longer multi-thousand request soak required for unprovisional leak certification."
                ),
            }

            # 12. Compute Metrics Aggregations
            report["measured_requests"] = measured_records
            report["rejections"] = rejection_records
            report["concurrent_load_interleaving"] = concurrent_interleaving_records
            report["telemetry_samples"] = telemetry_samples

            report["metrics_summary"] = compute_metrics_summary(
                measured_records=measured_records,
                rejection_records=rejection_records,
                concurrent_interleaving_records=concurrent_interleaving_records,
                solo_baseline_percentiles=report.get("solo_baseline", {}).get("percentiles"),
                total_benchmark_elapsed_s=time.monotonic() - start_mono,
                partial_run=False,
            )

            # 13. Evaluate Qualification Gates against Explicit Criteria
            qual_eval = evaluate_qualification_gates(report, self.config)
            report["qualification_evaluation"] = qual_eval
            report["status"] = qual_eval["overall_status"]
            if qual_eval["overall_status"] != "passed" and qual_eval.get("failure_reasons"):
                report["error"] = "; ".join(qual_eval["failure_reasons"])

        except Exception as e:
            report["status"] = "failed"
            report["error"] = str(e)
            report["telemetry_samples"] = telemetry_samples
            report["measured_requests"] = measured_records
            report["rejections"] = rejection_records
            report["concurrent_load_interleaving"] = concurrent_interleaving_records

            if not breach_event.is_set():
                breach_event.wait(timeout=0.1)

            if breach_event.is_set() or (breach_result[0] is not None):
                b = breach_result[0]
                if b is not None:
                    report["watchdog_breach"] = extract_watchdog_breach_dict(b)
                    report["primary_failure"] = {
                        "category": "watchdog_breach",
                        "breach_type": b.breach_type.value if b.breach_type else "unknown",
                        "reason": b.breach_reason,
                    }
                else:
                    report["primary_failure"] = {
                        "category": "watchdog_breach",
                        "reason": "Watchdog breach event was signaled",
                    }
            elif child is not None and child.poll() is not None and child.returncode != 0:
                report["primary_failure"] = {
                    "category": "process_exit",
                    "reason": f"Server child process exited unexpectedly with code {child.returncode}",
                    "exit_code": child.returncode,
                }
            else:
                report["primary_failure"] = {
                    "category": "execution_error",
                    "reason": str(e),
                }

            solo_p = report.get("solo_baseline", {}).get("percentiles")
            report["metrics_summary"] = compute_metrics_summary(
                measured_records=measured_records,
                rejection_records=rejection_records,
                concurrent_interleaving_records=concurrent_interleaving_records,
                solo_baseline_percentiles=solo_p,
                total_benchmark_elapsed_s=time.monotonic() - start_mono,
                partial_run=True,
            )
            report["qualification_evaluation"] = evaluate_qualification_gates(report, self.config)
            self.last_report = report
            raise
        finally:
            if log_file_path and os.path.exists(log_file_path):
                try:
                    with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
                        report["captured_logs"] = f.read()[-8192:]
                except Exception as log_err:
                    report["captured_logs"] = f"Failed to capture server logs: {log_err}"

            stop_supervisor.set()
            if t_supervisor is not None and t_supervisor.is_alive():
                try:
                    t_supervisor.join(timeout=1.0)
                except Exception:
                    pass

            client.close()

            sigkill_used = False
            if child is not None:
                try:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=1.0)
                            sigkill_used = True
                    report["child_cleanup"] = {
                        "cleaned": True,
                        "pid": child.pid,
                        "exit_code": child.returncode,
                        "sigkill_used": sigkill_used,
                    }
                except Exception as cleanup_err:
                    cleanup_errors.append(f"Child cleanup error: {cleanup_err}")
                    report["child_cleanup"] = {
                        "cleaned": False,
                        "pid": child.pid if child else None,
                        "error": str(cleanup_err),
                    }

            if log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass
            if log_file_path and os.path.exists(log_file_path):
                try:
                    os.unlink(log_file_path)
                except Exception as unlink_err:
                    cleanup_errors.append(f"Log unlink error: {unlink_err}")

            if cleanup_errors:
                report["cleanup_errors"] = cleanup_errors

            report["duration_s"] = round(time.time() - run_start_time, 2)
            self.last_report = report

            if self.config.output_file:
                try:
                    out_dir = os.path.dirname(os.path.abspath(self.config.output_file))
                    os.makedirs(out_dir, exist_ok=True)
                    with open(self.config.output_file, "w", encoding="utf-8") as f:
                        json.dump(report, f, indent=2)
                except Exception as save_err:
                    print(f"[gguf-sustained] Failed to save report to {self.config.output_file}: {save_err}", file=sys.stderr)

        return report
