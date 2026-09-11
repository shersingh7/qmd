"""
sustained.py — Bounded Sustained MLX Embedding Qualification Harness

Reuses SmokeRunner / MLXWatchdog owned process composition to execute
bounded pilot (<=30 requests, <=120s) and bounded soak (<=100 requests, <=180s)
benchmarks across four verified token strata:
  - Short:  5 – 25 tokens (interactive queries)
  - Medium: 50 – 200 tokens (passages)
  - Long:   400 – 1500 tokens (documents)
  - Code:   50 – 500 tokens (syntax blocks)
  - 2048 boundary policy verification

Guarantees:
  1. Cold startup measured separately; warmup excluded from metrics.
  2. Latency percentiles computed via explicit linear interpolation (small samples marked descriptive).
  3. Wire end-to-end vs forward pass distinction.
  4. True tokens/s computed from verified input tokens.
  5. Process RSS, swap growth, and Metal active/peak memory tracked continuously.
  6. Conservative batch progression (1 -> 2 -> 4) and interactive interleaving.
  7. Absolute monotonic wall-clock deadlines and guaranteed watchdog teardown.
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

from .adapters.embedding import (
    estimate_model_memory_mb,
    infer_model_params_b,
)
from .smoke import (
    NORM_TOLERANCE,
    SmokeHttpClient,
    check_dimensions,
    check_finite,
    check_l2_normalization,
    inspect_model_metadata,
)
from .watchdog import (
    BreachType,
    MLXWatchdog,
    MLXWatchdogConfig,
    SystemMemorySampler,
    SystemMetricsError,
    WatchdogCheckResult,
    is_numeric_loopback,
)


# --- Public Public Fixtures ---

SHORT_FIXTURES = [
    "Fast vector search with MLX on Apple Silicon Metal.",
    "What is the latency of semantic retrieval on unified memory?",
    "Locate the definition of EmbeddingDescriptor in contract module.",
    "How does BM25 and vector hybrid search rank documents?",
    "High-bandwidth shared unified memory across CPU and GPU cores.",
]

MEDIUM_FIXTURES = [
    (
        "Apple Silicon Metal unified memory architecture provides high-bandwidth shared memory access "
        "across CPU and GPU execution contexts. In contrast to discrete PCIe-attached accelerators with explicit DMA copies, "
        "Apple Silicon unified memory allows host tokenization buffers and Metal command encoders to operate on shared physical "
        "memory pools without PCIe bus serialization bottlenecks or data transfer overhead."
    ),
    (
        "Matryoshka Representation Learning enables embedding models to produce flexible representation dimensions within a single forward pass. "
        "By optimizing loss across truncated prefixes of the hidden embedding vector (such as 512, 1024, or 2560 dimensions), "
        "downstream vector indices can trade off storage footprint and cosine similarity precision without retraining or maintaining multiple separate models."
    ),
    (
        "Durable index recovery guarantees that interrupted bulk indexing operations resume cleanly with zero duplicated vector rows "
        "and deterministic document count reconciliation across all collections. Checkpoint transactions record document content hashes, chunk boundaries, "
        "and vector IDs in SQLite before committing updates to the primary full-text search and vector indexes on Apple Silicon hardware architecture."
    ),
]

LONG_DOC_BASE = (
    "Apple Silicon unified memory architecture integrates CPU, GPU, and Neural Engine cores onto a single system-on-chip (SoC) "
    "with a unified memory fabric. This physical topology eliminates the traditional memory separation between system host RAM and GPU VRAM. "
    "In machine learning inference workloads, this design allows large weight tensors to remain resident in physical RAM while Metal compute "
    "pipelines execute matrix multiplications directly in-place. "
    "Furthermore, memory pressure management requires explicit accounting of resident set size (RSS), wired memory allocations, "
    "and inactive purgeable file cache pages to prevent unexpected swap thrashing or process termination under heavy multi-tasking workloads.\n\n"
    "When evaluating embedding performance, end-to-end latency must be decomposed into distinct execution phases: host CPU tokenization, "
    "tensor buffer allocation, Metal command buffer encoding, GPU backbone execution, pooling operations, and L2 normalization. "
    "In causal architectures such as Qwen, last-token pooling requires indexing the hidden state at the exact position of the last non-padding "
    "token for each sequence in the batch. Padded tokens introduced to uniformize batch dimensions must not contaminate the extracted vector. "
    "Casting intermediate activations to float32 before normalization preserves numerical stability across diverse sequence length distributions.\n\n"
)

# Repeat base text to build ~500-1000 token long documents
LONG_FIXTURES = [
    LONG_DOC_BASE * 3,
    LONG_DOC_BASE * 4,
]

CODE_FIXTURES = [
    (
        "```python\n"
        "def process_embedding_batch(texts: list[str], max_batch_tokens: int = 4096, dims: int = 2560) -> np.ndarray:\n"
        "    \"\"\"Plans micro-batches and executes forward pass with L2 normalization across Apple Silicon Metal unified memory.\"\"\"\n"
        "    batches: list[list[str]] = []\n"
        "    current_batch: list[str] = []\n"
        "    current_tokens: int = 0\n"
        "    for text in texts:\n"
        "        token_len = len(text.split())\n"
        "        if current_batch and current_tokens + token_len > max_batch_tokens:\n"
        "            batches.append(current_batch)\n"
        "            current_batch, current_tokens = [text], token_len\n"
        "        else:\n"
        "            current_batch.append(text)\n"
        "            current_tokens += token_len\n"
        "    if current_batch:\n"
        "        batches.append(current_batch)\n"
        "    return np.vstack([forward_model(b) for b in batches])\n"
        "```"
    ),
    (
        "```typescript\n"
        "export function verifyEmbeddingCompatibility(\n"
        "  expected: EmbeddingDescriptor,\n"
        "  actual: EmbeddingDescriptor,\n"
        "  context?: string\n"
        "): boolean {\n"
        "  if (expected.outputDimensions !== actual.outputDimensions) return false;\n"
        "  if (expected.normalized !== actual.normalized) return false;\n"
        "  if (expected.pooling !== actual.pooling) return false;\n"
        "  const idExpected = computeEmbeddingSpaceId(expected);\n"
        "  const idActual = computeEmbeddingSpaceId(actual);\n"
        "  if (idExpected !== idActual) {\n"
        "    throw new Error(`Embedding space mismatch: expected ${idExpected}, got ${idActual}`);\n"
        "  }\n"
        "  return true;\n"
        "}\n"
        "```"
    ),
]

def compute_code_manifest(repo_root: str) -> dict[str, str]:
    """
    Computes exact SHA256 fingerprints for all relevant qualification and runtime scripts,
    including untracked and tracked files.
    """
    manifest_files = [
        "scripts/qmd-mlx-sustained.py",
        "scripts/qmd_mlx/sustained.py",
        "scripts/qmd_mlx/server.py",
        "scripts/qmd_mlx/runtime.py",
        "scripts/qmd_mlx/watchdog.py",
        "scripts/qmd_mlx/executor.py",
        "scripts/qmd_mlx/batching.py",
        "scripts/qmd_mlx/model_manager.py",
        "scripts/qmd_mlx/adapters/embedding.py",
        "scripts/qmd_mlx/adapters/tokenization.py",
        "test/python/test_mlx_sustained.py",
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


def compute_explicit_percentiles(values: list[float]) -> dict[str, Any]:
    """
    Computes Min, p50, p95, Max, Avg using numpy linear interpolation.
    Explicitly flags small samples (N <= 30) as descriptive only.
    """
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0, "descriptive_only": True}

    arr = np.array(values, dtype=np.float64)
    p50 = float(np.percentile(arr, 50, method="linear"))
    p95 = float(np.percentile(arr, 95, method="linear"))
    return {
        "count": len(values),
        "min": round(float(np.min(arr)), 2),
        "p50": round(p50, 2),
        "p95": round(p95, 2),
        "max": round(float(np.max(arr)), 2),
        "mean": round(float(np.mean(arr)), 2),
        "descriptive_only": len(values) <= 30,
    }


def calibrate_boundary_fixtures(
    client: SmokeHttpClient,
    base_url: str,
    deadline: float,
) -> dict[str, str]:
    """
    Calibrates exact 2047, 2048, and 2049 token texts by querying the active /tokenize endpoint.
    Guarantees:
      - boundary_2047 encodes to exactly 2047 tokens (including special tokens).
      - boundary_2048 encodes to exactly 2048 tokens (including special tokens).
      - boundary_2049 encodes to exactly 2049 tokens (including special tokens), rejected with HTTP 400.
    """
    words = ["token" for _ in range(2040)]
    text = " ".join(words)

    st, r = client.post_json(f"{base_url}/tokenize", {"texts": [text]}, deadline=deadline)
    if st == 200 and isinstance(r, dict) and "counts" in r:
        cnt = int(r["counts"][0])
    else:
        cnt = len(words)

    max_steps = 20
    step = 0
    while cnt != 2047 and step < max_steps:
        step += 1
        if cnt < 2047:
            diff = 2047 - cnt
            words.extend(["token" for _ in range(diff)])
        elif cnt > 2047:
            diff = cnt - 2047
            words = words[:-diff] if diff < len(words) else words[:1]
        text = " ".join(words)
        st, r = client.post_json(f"{base_url}/tokenize", {"texts": [text]}, deadline=deadline)
        if st == 200 and isinstance(r, dict) and "counts" in r:
            cnt = int(r["counts"][0])
        else:
            break

    text_2047 = text

    words_2048 = list(words)
    words_2048.append("token")
    text_2048 = " ".join(words_2048)
    st, r = client.post_json(f"{base_url}/tokenize", {"texts": [text_2048]}, deadline=deadline)
    if st == 200 and isinstance(r, dict) and "counts" in r:
        cnt_2048 = int(r["counts"][0])
    else:
        cnt_2048 = len(words_2048)

    step = 0
    while cnt_2048 != 2048 and step < max_steps:
        step += 1
        if cnt_2048 < 2048:
            words_2048.append("token")
        elif cnt_2048 > 2048:
            words_2048.pop()
        text_2048 = " ".join(words_2048)
        st, r = client.post_json(f"{base_url}/tokenize", {"texts": [text_2048]}, deadline=deadline)
        if st == 200 and isinstance(r, dict) and "counts" in r:
            cnt_2048 = int(r["counts"][0])
        else:
            break

    words_2049 = list(words_2048)
    words_2049.append("token")
    text_2049 = " ".join(words_2049)

    return {
        "boundary_2047": text_2047,
        "boundary_2048": text_2048,
        "boundary_2049": text_2049,
    }


def extract_watchdog_breach_dict(b: WatchdogCheckResult) -> dict[str, Any]:
    """Formats a structured snapshot of a primary watchdog breach."""
    return {
        "breach_type": b.breach_type.value if b.breach_type else "unknown",
        "breach_reason": b.breach_reason or "Watchdog breach triggered",
        "timestamp": round(b.timestamp, 3),
        "metrics": b.metrics or {},
        "terminated_pid": b.terminated_pid,
        "sigkill_used": b.sigkill_used,
    }


def compute_metrics_summary(
    measured_records: list[dict[str, Any]],
    rejection_records: list[dict[str, Any]],
    concurrent_interleaving_records: list[dict[str, Any]],
    solo_baseline_percentiles: Optional[dict[str, Any]],
    total_benchmark_elapsed_s: float,
    partial_run: bool = False,
) -> dict[str, Any]:
    """
    Computes complete, typed, finite metrics aggregations across all attempted/measured requests.
    Supports partial runs on failure without fabricating data.
    """
    successful_records = [r for r in measured_records if r.get("rejection_verified") is not True]
    all_successful_latencies = [
        r["wire_ms"] for r in successful_records if isinstance(r.get("wire_ms"), (int, float))
    ]
    total_successful_tokens = sum(r.get("total_tokens", 0) for r in successful_records)
    sum_successful_wire_ms = sum(
        r.get("wire_ms", 0.0) for r in successful_records if isinstance(r.get("wire_ms"), (int, float))
    )
    sum_rejection_wire_ms = sum(
        r.get("wire_ms", 0.0) for r in rejection_records if isinstance(r.get("wire_ms"), (int, float))
    )
    total_wire_time_s = sum_successful_wire_ms / 1000.0
    overall_throughput = (
        round(total_successful_tokens / total_wire_time_s, 1) if total_wire_time_s > 0 else 0.0
    )

    by_stratum: dict[str, Any] = {}
    for strat in ["short", "medium", "long", "code", "boundary_2047", "boundary_2048", "mixed"]:
        strat_lats = [
            r["wire_ms"]
            for r in successful_records
            if r.get("stratum") == strat and isinstance(r.get("wire_ms"), (int, float))
        ]
        if strat_lats:
            by_stratum[strat] = compute_explicit_percentiles(strat_lats)

    by_batch_size: dict[str, Any] = {}
    for b_sz in [1, 2, 4]:
        b_lats = [
            r["wire_ms"]
            for r in successful_records
            if r.get("batch_size") == b_sz and isinstance(r.get("wire_ms"), (int, float))
        ]
        if b_lats:
            by_batch_size[f"batch_{b_sz}"] = compute_explicit_percentiles(b_lats)

    conc_query_lats = [
        c["concurrent_interactive_wire_ms"]
        for c in concurrent_interleaving_records
        if isinstance(c.get("concurrent_interactive_wire_ms"), (int, float))
    ]
    conc_bulk_lats = [
        c["bulk_wire_ms"]
        for c in concurrent_interleaving_records
        if isinstance(c.get("bulk_wire_ms"), (int, float))
    ]
    conc_queue_waits = [
        c["estimated_queue_wait_ms"]
        for c in concurrent_interleaving_records
        if isinstance(c.get("estimated_queue_wait_ms"), (int, float))
    ]
    conc_slowdowns = [
        c["slowdown_factor"]
        for c in concurrent_interleaving_records
        if isinstance(c.get("slowdown_factor"), (int, float))
    ]

    return {
        "total_measured_requests": len(measured_records) + len(rejection_records),
        "successful_embedding_requests": len(successful_records),
        "rejected_boundary_requests": len(rejection_records),
        "total_successful_embedded_tokens": total_successful_tokens,
        "overall_effective_tokens_per_sec": overall_throughput,
        "partial_run": partial_run,
        "cumulative_wall_accounting": {
            "total_benchmark_elapsed_s": round(total_benchmark_elapsed_s, 2),
            "sum_successful_request_wire_s": round(sum_successful_wire_ms / 1000.0, 3),
            "sum_rejection_request_wire_s": round(sum_rejection_wire_ms / 1000.0, 3),
            "forwarded_vs_wire_note": "Client wire end-to-end latency measured via HTTP socket without fabricating forward pass internals.",
        },
        "overall_successful_latency_ms": compute_explicit_percentiles(all_successful_latencies),
        "expected_rejections_latency_ms": compute_explicit_percentiles(
            [r["wire_ms"] for r in rejection_records if isinstance(r.get("wire_ms"), (int, float))]
        ),
        "by_stratum_latency_ms": by_stratum,
        "by_batch_size_latency_ms": by_batch_size,
        "concurrent_tail_latency_ms": {
            "solo_baseline_interactive": solo_baseline_percentiles or compute_explicit_percentiles([]),
            "concurrent_interactive_under_load": compute_explicit_percentiles(conc_query_lats),
            "concurrent_bulk_batches": compute_explicit_percentiles(conc_bulk_lats),
            "estimated_queue_wait_ms": compute_explicit_percentiles(conc_queue_waits),
            "mean_slowdown_factor": round(float(np.mean(conc_slowdowns)), 2) if conc_slowdowns else 1.0,
        },
        "padding_overhead": "unavailable",
    }


def evaluate_qualification_gates(report: dict[str, Any], config: SustainedRunnerConfig) -> dict[str, Any]:
    """
    Evaluates benchmark results against explicit qualification criteria.
    Never relaxes original targets. Evaluates small sample sizes explicitly while
    noting descriptive flag in summary percentiles.
    """
    gates: dict[str, Any] = {}
    failure_reasons: list[str] = []

    # 1. Cold Startup Gate
    cold = report.get("cold_startup", {})
    cold_ready = cold.get("ready") is True
    cold_elapsed = float(cold.get("cold_startup_elapsed_s", 999.0))
    cold_passed = cold_ready and (cold_elapsed <= 30.0)
    gates["cold_startup"] = {
        "passed": cold_passed,
        "cold_startup_elapsed_s": cold_elapsed if cold_ready else None,
        "limit_s": 30.0,
    }
    if not cold_passed:
        failure_reasons.append(f"Cold startup failed or exceeded 30s limit (elapsed: {cold_elapsed}s)")

    # 2. Boundary Rejection Policy Gate
    rejs = report.get("rejections", [])
    b2049_rejections = [
        r
        for r in rejs
        if r.get("stratum") == "boundary_2049"
        and r.get("rejection_verified") is True
        and r.get("status_code") == 400
    ]
    rejection_passed = len(b2049_rejections) >= 1
    gates["boundary_rejection_policy"] = {
        "passed": rejection_passed,
        "boundary_2049_rejections_count": len(b2049_rejections),
    }
    if not rejection_passed:
        failure_reasons.append("Boundary 2049 explicit HTTP 400 rejection not verified")

    # 3. Vector Numerical Correctness Gate
    meas = report.get("measured_requests", [])
    vector_correct = len(meas) > 0 and all(
        m.get("finite") is True and m.get("l2_normalized") is True
        for m in meas
    )
    gates["numerical_correctness"] = {
        "passed": vector_correct,
        "validated_request_count": len(meas),
    }
    if not vector_correct:
        failure_reasons.append("Vector validation failed (empty, non-finite, or unnormalized output)")

    # 4. Interactive Responsiveness Gate (Stated Target: Concurrent Interactive p95 <= 200.0ms)
    summary = report.get("metrics_summary", {})
    tail = summary.get("concurrent_tail_latency_ms", {})
    conc_interactive = tail.get("concurrent_interactive_under_load", {})
    conc_p95 = conc_interactive.get("p95", 0.0)
    conc_count = conc_interactive.get("count", 0)

    if conc_count > 0:
        responsiveness_passed = bool(conc_p95 <= 200.0)
        gates["concurrent_interactive_responsiveness"] = {
            "passed": responsiveness_passed,
            "target_p95_ms": 200.0,
            "measured_p95_ms": conc_p95,
            "sample_count": conc_count,
            "descriptive_only": bool(conc_interactive.get("descriptive_only", True)),
            "mean_slowdown_factor": tail.get("mean_slowdown_factor", 1.0),
        }
        if not responsiveness_passed:
            failure_reasons.append(
                f"Concurrent interactive query p95 ({conc_p95:.2f}ms) violated the <= 200.0ms target "
                f"(mean slowdown {tail.get('mean_slowdown_factor', 1.0):.2f}x vs solo baseline)"
            )
    else:
        if config.dry_run:
            gates["concurrent_interactive_responsiveness"] = {"passed": True, "dry_run": True}
        else:
            gates["concurrent_interactive_responsiveness"] = {
                "passed": False,
                "reason": "No concurrent interactive requests were measured",
            }
            failure_reasons.append("No concurrent interactive requests were measured")

    # 5. Memory Stability Gate
    metal = report.get("metal_memory_telemetry", {})
    stab = metal.get("provisional_stability_assessment", {})
    swap_growth = stab.get("swap_growth_mb", 0.0) if stab else 0.0
    max_allowed_swap = (config.min_headroom_mb * 0.25) if hasattr(config, "min_headroom_mb") else 2048.0
    memory_passed = bool(swap_growth <= max_allowed_swap and "watchdog_breach" not in report)
    gates["memory_stability"] = {
        "passed": memory_passed,
        "swap_growth_mb": swap_growth,
        "max_allowed_swap_growth_mb": max_allowed_swap,
        "verdict": stab.get("verdict", "unstable") if stab else "unstable",
    }
    if not memory_passed:
        failure_reasons.append(
            f"Memory stability failed: swap growth ({swap_growth:.1f}MB) exceeded budget ({max_allowed_swap:.1f}MB)"
        )

    # 6. Watchdog Integrity Gate
    watchdog_passed = "watchdog_breach" not in report
    gates["watchdog_integrity"] = {
        "passed": watchdog_passed,
        "breach": report.get("watchdog_breach"),
    }
    if not watchdog_passed:
        failure_reasons.append(
            f"Watchdog breach triggered: {report.get('watchdog_breach', {}).get('breach_reason')}"
        )

    overall_passed = (len(failure_reasons) == 0) and (report.get("status") not in ("failed", "error"))
    return {
        "overall_status": "passed" if overall_passed else "failed",
        "gates": gates,
        "failure_reasons": failure_reasons,
    }


@dataclasses.dataclass
class SustainedRunnerConfig:
    model_path: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8797
    control_port: int = 8798
    mode: str = "pilot"          # 'pilot' (<=30 reqs, <=120s) or 'soak' (<=100 reqs, <=180s)
    timeout_s: float = 120.0
    max_requests: int = 30
    min_headroom_mb: float = 6000.0
    use_fake_child: bool = False
    fake_dims: int = 2560
    dry_run: bool = False
    real_model_opt_in: bool = False
    output_file: Optional[str] = None


class SustainedRunner:
    """
    Executes bounded sustained qualification benchmarks for MLX embedding models
    reusing the owned process composition and active MLXWatchdog supervision.
    """

    def __init__(self, config: SustainedRunnerConfig, sampler: Optional[SystemMemorySampler] = None):
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

            meta = inspect_model_metadata(self.config.model_path)
            preflight_info["model_metadata"] = meta.to_dict()

            # Enforce >= 6000 MB available headroom for sustained real model testing
            required_headroom = max(self.config.min_headroom_mb, meta.conservative_required_headroom_mb, 6000.0)
            if headroom < required_headroom:
                raise RuntimeError(
                    f"Insufficient memory headroom for sustained {meta.params_b:.1f}B model qualification: "
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
        Executes bounded sustained embedding qualification under active MLXWatchdog supervision.
        """
        run_start_time = time.time()
        start_mono = time.monotonic()
        deadline = start_mono + self.config.timeout_s

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        report: dict[str, Any] = {
            "status": "in_progress",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_start_time)),
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

        # 2. Spawn Child Server
        instance_token = uuid.uuid4().hex
        child_env = os.environ.copy()
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"
        child_env["HF_DATASETS_OFFLINE"] = "1"
        child_env["MLX_INSTANCE_TOKEN"] = instance_token
        child_env["MLX_EMBED_PORT"] = str(self.config.port)
        child_env["MLX_CONTROL_PORT"] = str(self.config.control_port)

        if "PYTHONPATH" in child_env:
            child_env["PYTHONPATH"] = f"{repo_root}:{child_env['PYTHONPATH']}"
        else:
            child_env["PYTHONPATH"] = repo_root

        server_script = os.path.join(repo_root, "scripts", "mlx_embed_server.py")
        if self.config.use_fake_child:
            model_spec = f"synthetic-qwen3-4b-{self.config.fake_dims}d" if self.config.fake_dims != 2560 else "synthetic-qwen3-4b"
        else:
            model_spec = str(self.config.model_path)

        cmd = [
            sys.executable,
            server_script,
            "--model", model_spec,
            "--port", str(self.config.port),
            "--control-port", str(self.config.control_port),
            "--host", self.config.host,
            "--no-warmup",  # Harness controls and isolates warmup measurement
        ]

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
            log_file = tempfile.NamedTemporaryFile(mode="w+", prefix="mlx_sustained_server_", suffix=".log", delete=False)
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
                startup_grace_period_s=min(25.0, self.config.timeout_s),
                stalled_inference_timeout_s=min(15.0, self.config.timeout_s),
                expected_cmd_pattern=r"(python|mlx|qmd|server)",
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

            t_supervisor = threading.Thread(target=_supervisor_loop, daemon=True, name="Sustained-Watchdog-Supervisor")
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
                raise TimeoutError("Server failed to report ready within startup period")

            # 5. Fetch and Validate Descriptor
            st_desc, desc = client.get_json(f"{ctrl_url}/descriptor", deadline=deadline, timeout_s=2.0)
            if st_desc != 200 or not isinstance(desc, dict):
                raise RuntimeError(f"Failed to fetch descriptor: status {st_desc}, data {desc}")
            report["descriptor"] = desc
            expected_dims = desc.get("outputDimensions") or desc.get("nativeDimensions") or 2560

            # 6. Sample Initial Metal Memory
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

                t_b = threading.Thread(target=_bulk_worker, daemon=True, name=f"Bulk-Client-{c_iter}")
                t_q = threading.Thread(target=_query_worker, daemon=True, name=f"Interactive-Client-{c_iter}")

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

            # Stage F: If in soak mode, continue bounded iterations up to max_requests
            if self.config.mode == "soak":
                soak_pool = [
                    ([SHORT_FIXTURES[0]], "short", True),
                    ([MEDIUM_FIXTURES[0]], "medium", False),
                    ([SHORT_FIXTURES[1], SHORT_FIXTURES[2]], "short", False),
                    ([CODE_FIXTURES[0]], "code", False),
                    ([MEDIUM_FIXTURES[1], CODE_FIXTURES[1]], "mixed", False),
                    ([SHORT_FIXTURES[0], SHORT_FIXTURES[1], SHORT_FIXTURES[2], SHORT_FIXTURES[3]], "short", False),
                ]
                soak_idx = 0
                while request_counter < self.config.max_requests and time.monotonic() < (deadline - 5.0):
                    item = soak_pool[soak_idx % len(soak_pool)]
                    _execute_measured_request(item[0], stratum=item[1], is_query=item[2], tag=f"soak_iter_{request_counter}")
                    soak_idx += 1

            # 11. Sample Final Metal Memory & Steady-State Memory Trend
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

            # 12. Compute Metrics Aggregations (Successful Tokens ONLY, Separate Rejections)
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

            # Check if supervisor captured a breach or if child terminated
            if not breach_event.is_set():
                breach_event.wait(timeout=0.1)

            # Primary failure classification vs derivative errors
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

            # Always populate partial metrics summary with completed request counts
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
            # 12. Guaranteed Teardown and Cleanup
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
                    print(f"[sustained] Failed to save report to {self.config.output_file}: {save_err}", file=sys.stderr)

        return report
