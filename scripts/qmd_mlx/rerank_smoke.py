"""
rerank_smoke.py — Bounded Single-Stage MLX Reranker Smoke Test Runner & Fixture Suite

Safety & Isolation Guarantees:
1. 100% Offline: Enforces HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1.
2. Single-Stage Isolation: Runs reranker ONLY (no embedding model loaded into Metal).
3. Dedicated Loopback Ports: Uses distinct loopback ports (zero collision with live service 8787).
4. Memory-Aware Preflight: Verifies conservative RAM headroom (>= 6000MB for 4B reranker).
5. Watchdog Supervision: Disposable child monitored continuously by MLXWatchdog with strict wall-clock ceiling.
6. Guaranteed Cleanup: Child processes reaped on all exit paths; sentinel/parent untouched.
7. Public Judged Fixtures: Numerical, rank ordering, and consistency checks on public technical pairs.
8. Measured Descriptor Verification: Reads cached config.json and asserts exact measured quantization.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .rerank import infer_quantization_and_dtype
from .supervisor import SmokeHttpClient, SmokeStageSupervisor, StageSupervisorConfig
from .watchdog import SystemMemorySampler


# Public Judged Fixture Pairs for Reranker Qualification
RERANK_FIXTURES = [
    {
        "query": "REST API versioning best practices",
        "rel_doc": (
            "When designing web APIs, versioning is critical for backward compatibility. "
            "Common strategies include URI path versioning (/v1/users), custom request headers (X-API-Version: 2), "
            "and Accept header content negotiation. URI path versioning is the most transparent for caching proxies."
        ),
        "irrel_doc": (
            "To bake chocolate chip cookies, combine 2 cups of all-purpose flour, 1 tsp baking soda, and 1/2 tsp salt. "
            "Cream 1 cup unsalted butter with 3/4 cup brown sugar and bake at 375F for 10 minutes."
        ),
    },
    {
        "query": "Distributed consensus leader election",
        "rel_doc": (
            "Raft is a distributed consensus algorithm designed to be equivalent to Paxos in fault-tolerance. "
            "It decomposes consensus into leader election, log replication, and safety. Nodes transition between Leader, Follower, and Candidate."
        ),
        "irrel_doc": (
            "Employees accrue 15 days of paid time off per calendar year. Vacation requests must be submitted through HR portal two weeks in advance."
        ),
    },
]

MULTI_DOC_BATCH = [
    "Chocolate chip cookie recipe with butter and sugar.",
    "Raft and Paxos are distributed consensus algorithms for replicated state machines.",
    "Remote work vacation request and paid time off policy.",
    "REST API endpoint design and URI path versioning guidelines.",
]


@dataclasses.dataclass
class RerankSmokeConfig:
    model_path: str = "~/.cache/qmd/models/qwen3-reranker-4b-mlx-4bit"
    host: str = "127.0.0.1"
    port: int = 8797
    control_port: int = 8798
    timeout_s: float = 60.0
    min_headroom_mb: float = 6000.0
    output_json: Optional[str] = None
    dry_run: bool = False
    use_fake_child: bool = False


class RerankSmokeRunner:
    """
    Orchestrates bounded, isolated single-stage MLX reranker smoke qualification
    under active MLXWatchdog supervision with exact quantization inspection.
    """

    def __init__(self, config: RerankSmokeConfig, sampler: Optional[SystemMemorySampler] = None):
        self.config = config
        self.sampler = sampler or SystemMemorySampler()
        self.last_report: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        expanded_path = os.path.expanduser(self.config.model_path)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        server_script = os.path.join(repo_root, "scripts", "mlx_embed_server.py")

        cmd = [
            sys.executable,
            server_script,
            "--no-embed",
            "--rerank-model",
            expanded_path,
            "--port",
            str(self.config.port),
            "--control-port",
            str(self.config.control_port),
            "--host",
            self.config.host,
            "--preload",
        ]

        expected_quant, expected_dtype = infer_quantization_and_dtype(expanded_path)

        supervisor_config = StageSupervisorConfig(
            cmd=cmd,
            host=self.config.host,
            port=self.config.port,
            control_port=self.config.control_port,
            timeout_s=self.config.timeout_s,
            min_headroom_mb=self.config.min_headroom_mb,
            stage_name="rerank",
            model_identifier=self.config.model_path,
            output_file=self.config.output_json,
            dry_run=self.config.dry_run,
        )

        supervisor = SmokeStageSupervisor(config=supervisor_config, sampler=self.sampler)

        # Local model existence check before spawning
        if not self.config.dry_run and not self.config.use_fake_child:
            if not os.path.exists(expanded_path):
                blocked_report = {
                    "stage": "rerank",
                    "model": self.config.model_path,
                    "port": self.config.port,
                    "control_port": self.config.control_port,
                    "status": "blocked",
                    "errors": [f"Model path does not exist locally: {expanded_path}"],
                }
                if self.config.output_json:
                    supervisor._save_report(blocked_report)
                self.last_report = blocked_report
                return blocked_report

        with supervisor.managed_stage() as ctx:
            if self.config.dry_run:
                ctx.record_fixture("dry_run", {"status": "skipped", "reason": "dry-run mode requested"})
                return ctx.report

            client = ctx.client
            base_url = ctx.base_url
            ctrl_url = ctx.ctrl_url
            deadline = ctx.deadline

            # 1. Descriptor Schema & Quantization Check
            ctx.check_breach("descriptor_fetch")
            desc_status, desc = client.get_json(f"{ctrl_url}/descriptor", deadline=deadline, timeout_s=2.0)
            if desc_status != 200 or not isinstance(desc, dict):
                ctx.add_error(f"Descriptor check failed: HTTP {desc_status} {desc}")
                ctx.report["status"] = "failed"
                return ctx.report

            ctx.report["descriptor"] = desc
            yes_id = desc.get("yesTokenId")
            no_id = desc.get("noTokenId")
            if yes_id is None or no_id is None or yes_id == no_id:
                ctx.add_error(f"Invalid dynamic token IDs: yes={yes_id}, no={no_id}")
                ctx.report["status"] = "failed"
                return ctx.report
            ctx.record_check("dynamic_token_resolution", True)

            # Assert descriptor quantization reflects actual configuration
            desc_quant = desc.get("quantization")
            if desc_quant != expected_quant:
                ctx.add_error(
                    f"Descriptor quantization mismatch: server reported '{desc_quant}', "
                    f"actual model config specifies '{expected_quant}'"
                )
                ctx.report["status"] = "failed"
                return ctx.report
            ctx.record_check("quantization_descriptor_match", True)

            # 2. Public Judged Fixture Ranking Discrimination Check
            ctx.check_breach("fixture_ranking")
            latencies = []
            fixture_results = []
            for i, fix in enumerate(RERANK_FIXTURES):
                ctx.check_breach(f"fixture_{i}")
                t0 = time.monotonic()
                payload = {
                    "query": fix["query"],
                    "documents": [fix["rel_doc"], fix["irrel_doc"]],
                }
                st_code, r_data = client.post_json(f"{base_url}/rerank", payload, deadline=deadline, timeout_s=10.0)
                lat = (time.monotonic() - t0) * 1000.0
                latencies.append(lat)

                if st_code != 200 or not isinstance(r_data, dict):
                    ctx.add_error(f"Fixture {i} failed with HTTP {st_code}: {r_data}")
                    ctx.report["status"] = "failed"
                    return ctx.report

                scores = r_data.get("scores", [])
                if len(scores) != 2:
                    ctx.add_error(f"Fixture {i} returned {len(scores)} scores, expected 2")
                    ctx.report["status"] = "failed"
                    return ctx.report

                for s in scores:
                    if not (isinstance(s, (int, float)) and math.isfinite(s) and 0.0 <= s <= 1.0):
                        ctx.add_error(f"Fixture {i} score {s} out of finite [0, 1] range")
                        ctx.report["status"] = "failed"
                        return ctx.report

                if scores[0] <= scores[1]:
                    ctx.add_error(
                        f"Fixture {i} discrimination failure: rel ({scores[0]:.4f}) <= irrel ({scores[1]:.4f})"
                    )
                    ctx.report["status"] = "failed"
                    return ctx.report

                fixture_results.append({
                    "query": fix["query"],
                    "scores": scores,
                    "discriminated": bool(scores[0] > scores[1]),
                    "latency_ms": round(lat, 2),
                })

            ctx.record_check("ranking_discrimination", True)
            ctx.record_metric("fixture_p50_ms", round(sorted(latencies)[len(latencies) // 2], 2))
            ctx.record_fixture("judged_pairs", fixture_results)

            # 3. Multi-Document Batch & Singleton Consistency Check
            ctx.check_breach("multi_doc_batch")
            t0 = time.monotonic()
            batch_query = "Distributed consensus algorithms"
            batch_payload = {"query": batch_query, "documents": MULTI_DOC_BATCH}
            st_b, r_batch = client.post_json(f"{base_url}/rerank", batch_payload, deadline=deadline, timeout_s=10.0)
            batch_lat = (time.monotonic() - t0) * 1000.0
            ctx.record_metric("batch_4docs_latency_ms", round(batch_lat, 2))

            if st_b != 200 or not isinstance(r_batch, dict):
                ctx.add_error(f"Batch request failed with HTTP {st_b}: {r_batch}")
                ctx.report["status"] = "failed"
                return ctx.report

            batch_scores = r_batch.get("scores", [])
            if len(batch_scores) != 4:
                ctx.add_error(f"Batch returned {len(batch_scores)} scores, expected 4")
                ctx.report["status"] = "failed"
                return ctx.report

            best_idx = int(max(range(len(batch_scores)), key=lambda idx: batch_scores[idx]))
            if best_idx != 1:
                ctx.add_error(f"Batch top ranked doc was index {best_idx}, expected index 1")
                ctx.report["status"] = "failed"
                return ctx.report

            # Singleton consistency check: score doc 1 individually
            ctx.check_breach("singleton_consistency")
            st_s, r_single = client.post_json(
                f"{base_url}/rerank",
                {"query": batch_query, "documents": [MULTI_DOC_BATCH[1]]},
                deadline=deadline,
                timeout_s=10.0,
            )
            if st_s != 200 or not isinstance(r_single, dict):
                ctx.add_error(f"Singleton request failed with HTTP {st_s}: {r_single}")
                ctx.report["status"] = "failed"
                return ctx.report

            single_score = r_single.get("scores", [])[0]
            batch_score_1 = batch_scores[1]
            score_diff = abs(single_score - batch_score_1)
            if score_diff > 1e-3:
                ctx.add_error(
                    f"Singleton vs batch score inconsistency: single={single_score:.6f} vs batch={batch_score_1:.6f} (diff={score_diff:.6e})"
                )
                ctx.report["status"] = "failed"
                return ctx.report

            ctx.record_check("singleton_batch_consistency", True)
            ctx.record_fixture("consistency", {
                "singleton_score": single_score,
                "batch_score": batch_score_1,
                "abs_diff": score_diff,
                "passed": bool(score_diff <= 1e-3),
            })

            # 4. Input Boundary & Rejection Checks (must return HTTP 400)
            ctx.check_breach("invalid_input_probes")
            bad_cases = [
                {"query": "", "documents": ["Valid doc"]},
                {"query": "   ", "documents": ["Valid doc"]},
                {"query": "Valid query", "documents": []},
                {"query": "Valid query", "documents": [""]},
                {"query": "Valid query", "documents": ["   "]},
                {"query": "Valid query", "documents": "not a list"},
            ]
            for idx, bc in enumerate(bad_cases):
                st_bad, _ = client.post_json(f"{base_url}/rerank", bc, deadline=deadline, timeout_s=3.0)
                if st_bad != 400:
                    ctx.add_error(f"Bad input case {idx} returned HTTP {st_bad}, expected 400")
                    ctx.report["status"] = "failed"
                    return ctx.report
            ctx.record_check("invalid_input_rejection", True)

            # 5. Over-Budget Document Length Rejection (> 2048 tokens rejected with HTTP 400)
            ctx.check_breach("long_doc_rejection")
            long_doc = "Apple Silicon unified memory architecture Metal GPU compute. " * 350
            st_long, _ = client.post_json(
                f"{base_url}/rerank",
                {"query": "Apple Silicon", "documents": [long_doc]},
                deadline=deadline,
                timeout_s=5.0,
            )
            if st_long != 400:
                ctx.add_error(f"Over-budget document returned HTTP {st_long}, expected 400")
                ctx.report["status"] = "failed"
                return ctx.report
            ctx.record_check("max_length_rejection", True)

            # 6. Memory & Stats Telemetry
            ctx.check_breach("telemetry_capture")
            st_mem, r_mem = client.get_json(f"{ctrl_url}/memory", deadline=deadline, timeout_s=2.0)
            if st_mem == 200 and isinstance(r_mem, dict):
                ctx.record_metric("memory", r_mem)

            st_stats, r_stats = client.get_json(f"{ctrl_url}/stats", deadline=deadline, timeout_s=2.0)
            if st_stats == 200 and isinstance(r_stats, dict):
                ctx.record_metric("stats", r_stats)

        self.last_report = supervisor.last_report
        return supervisor.last_report


def run_rerank_smoke(
    model_path: str = "~/.cache/qmd/models/qwen3-reranker-4b-mlx-4bit",
    port: int = 8797,
    control_port: int = 8798,
    timeout_s: float = 60.0,
    min_headroom_mb: float = 6000.0,
    output_json: Optional[str] = None,
    dry_run: bool = False,
    sampler: Optional[SystemMemorySampler] = None,
) -> dict[str, Any]:
    cfg = RerankSmokeConfig(
        model_path=model_path,
        port=port,
        control_port=control_port,
        timeout_s=timeout_s,
        min_headroom_mb=min_headroom_mb,
        output_json=output_json,
        dry_run=dry_run,
    )
    runner = RerankSmokeRunner(cfg, sampler=sampler)
    return runner.run()
