"""
generate_smoke.py — Bounded Single-Stage MLX Generation / Query Expansion Smoke Runner

Safety & Isolation Guarantees:
1. 100% Offline: Enforces HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1.
2. Single-Stage Isolation: Runs query expansion generator ONLY (no embedding model loaded).
3. Dedicated Loopback Ports: Uses distinct loopback ports (zero collision with live service 8787).
4. Memory-Aware Preflight: Verifies conservative RAM headroom (>= 3000MB for 1.7B model).
5. Watchdog Supervision: Disposable child process actively supervised by MLXWatchdog.
6. Guaranteed Cleanup: Child process reaped on all exit paths; parent/sentinel untouched.
7. Shared-Contract TS Protocol:
   - Uses exact production prompt builder and parser via TypeScript bridge (scripts/expansion_bridge.ts).
   - Strictly enforces typed lines (lex:, vec:, hyde:) without loose keyword/prose acceptance.
   - Rejects empty, echo-only, thinking-tag, or unrelated completions.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .supervisor import SmokeHttpClient, SmokeStageSupervisor, StageSupervisorConfig
from .watchdog import SystemMemorySampler


# Public Production Held-Out Prompts for Query Expansion Qualification
GENERATE_TEST_PROMPTS = [
    {
        "query": "distributed database indexing",
        "intent": None,
        "max_tokens": 150,
        "expected_keywords": ["index", "database", "distributed", "b-tree", "partition", "sharding"],
    },
    {
        "query": "web application REST API authentication",
        "intent": None,
        "max_tokens": 150,
        "expected_keywords": ["auth", "api", "token", "rest", "security", "oauth", "jwt"],
    },
    {
        "query": "vector similarity search ANN algorithms",
        "intent": None,
        "max_tokens": 150,
        "expected_keywords": ["vector", "similarity", "search", "ann", "algorithm"],
    },
    {
        "query": "cache invalidation strategies distributed systems",
        "intent": None,
        "max_tokens": 150,
        "expected_keywords": ["cache", "invalidation", "distributed", "systems", "strategy"],
    },
]


class TSExpansionBridge:
    """
    Lossless bridge to the production pure TypeScript expansion protocol module
    (src/expansion/protocol.ts) via Bun/Node CLI IPC.
    """

    def __init__(self, repo_root: Optional[str] = None):
        if repo_root is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.repo_root = repo_root
        self.bridge_script = os.path.join(repo_root, "scripts", "expansion_bridge.ts")
        self.runner = self._resolve_runner()

    def _resolve_runner(self) -> list[str]:
        bun_path = shutil.which("bun") or os.path.expanduser("~/.bun/bin/bun")
        if os.path.exists(bun_path) and os.access(bun_path, os.X_OK):
            return [bun_path, "run", self.bridge_script]
        node_path = shutil.which("node") or os.path.expanduser("~/.local/bin/node")
        if os.path.exists(node_path):
            npx_path = shutil.which("npx")
            if npx_path:
                return [npx_path, "tsx", self.bridge_script]
            return [node_path, self.bridge_script]
        return ["bun", "run", self.bridge_script]

    def build_prompt(self, query: str, intent: Optional[str] = None) -> str:
        payload = json.dumps({"action": "prompt", "query": query, "intent": intent})
        proc = subprocess.run(
            self.runner + ["--json"],
            input=payload,
            text=True,
            capture_output=True,
            cwd=self.repo_root,
            check=True,
        )
        res = json.loads(proc.stdout)
        return res["prompt"]

    def parse(self, text: str, query: str = "", include_lexical: bool = True) -> dict[str, Any]:
        payload = json.dumps({
            "action": "parse",
            "text": text,
            "query": query,
            "includeLexical": include_lexical,
        })
        proc = subprocess.run(
            self.runner + ["--json"],
            input=payload,
            text=True,
            capture_output=True,
            cwd=self.repo_root,
            check=True,
        )
        return json.loads(proc.stdout)


_DEFAULT_BRIDGE: Optional[TSExpansionBridge] = None


def get_default_bridge() -> TSExpansionBridge:
    global _DEFAULT_BRIDGE
    if _DEFAULT_BRIDGE is None:
        _DEFAULT_BRIDGE = TSExpansionBridge()
    return _DEFAULT_BRIDGE


def parse_expansion(text: str, query: str = "", bridge: Optional[TSExpansionBridge] = None) -> dict[str, Any]:
    """
    Parses generation output according to the production QMD query expansion format.
    Extracts 'lex:', 'vec:', 'hyde:' lines, and captures invalid/unstructured lines.
    Uses the pure TypeScript protocol bridge when available.
    """
    try:
        b = bridge or get_default_bridge()
        res = b.parse(text=text, query=query, include_lexical=True)
        return {
            "lex": res.get("lex", []),
            "vec": res.get("vec", []),
            "hyde": res.get("hyde", []),
            "invalid": res.get("invalidLines", []),
            "keywords": [],  # Prose keywords deprecated; typed lines only
            "queryables": res.get("queryables", []),
            "usable": res.get("usable", False),
            "quality_message": res.get("qualityMessage", ""),
        }
    except Exception:
        # Fallback pure-python parser mirroring TS protocol
        result: dict[str, Any] = {
            "lex": [],
            "vec": [],
            "hyde": [],
            "invalid": [],
            "keywords": [],
            "queryables": [],
            "usable": False,
            "quality_message": "",
        }
        raw_text = text or ""
        if not raw_text.strip():
            result["quality_message"] = "Completion text is empty"
            return result

        if "<think>" in raw_text or "</think>" in raw_text:
            result["quality_message"] = "Completion contains unstripped thinking tags"
            return result

        query_lower = query.lower().strip()
        query_terms = [t for t in re.sub(r"[^a-z0-9\s]", " ", query_lower).split() if len(t) >= 2]

        def has_relevance(c: str) -> bool:
            if not query_terms:
                return True
            c_low = c.lower()
            return any(t in c_low for t in query_terms) or query_lower in c_low

        seen = set()
        for raw_line in raw_text.split("\n"):
            line = raw_line.strip()
            if not line or line.startswith("```"):
                if line.startswith("```"):
                    result["invalid"].append(line)
                continue

            match = re.match(r"^[-*•\d.\s]*(lex|vec|hyde)\s*:\s*(.+)$", line, re.IGNORECASE)
            if not match:
                result["invalid"].append(line)
                continue

            type_str = match.group(1).lower()
            content = match.group(2).strip().strip("\"'`*")
            if not content:
                result["invalid"].append(line)
                continue

            c_low = content.lower()
            if c_low in ("<query>", "</query>", "<intent>", "</intent>") or c_low.startswith("/no_think"):
                result["invalid"].append(line)
                continue
            if c_low in (
                "(keyword or synonym phrase)",
                "(conceptual or semantic reformulation)",
                "(hypothetical sentence answering the query)",
            ):
                result["invalid"].append(line)
                continue

            if query and not has_relevance(content):
                result["invalid"].append(line)
                continue

            key = f"{type_str}:{content}"
            if key in seen:
                continue
            seen.add(key)

            result[type_str].append(content)
            result["queryables"].append({"type": type_str, "text": content})

        if not result["queryables"]:
            result["usable"] = False
            result["quality_message"] = f"No valid typed lines parsed; {len(result['invalid'])} invalid lines rejected"
        else:
            result["usable"] = True
            result["quality_message"] = "Valid usable query expansion"
        return result


def validate_expansion_quality(
    query: str,
    text: str,
    parsed: dict[str, Any],
    expected_keywords: Optional[list[str]] = None,
) -> tuple[bool, str]:
    """
    Validates whether the completion produces a usable query expansion rather than
    preamble fluff, echo prompt, or empty output.
    """
    if not text or not text.strip():
        return False, "Completion is empty"

    if "<think>" in text or "</think>" in text:
        return False, "Completion contains unstripped thinking tags"

    # Must contain structured typed lines
    has_typed_items = bool(parsed.get("lex") or parsed.get("vec") or parsed.get("hyde") or parsed.get("queryables"))
    if not has_typed_items:
        msg = parsed.get("quality_message") or "No valid typed expansion lines (lex/vec/hyde) found"
        return False, msg

    cleaned = text.strip()

    # Detect preamble-only without usable expansions
    preamble_pattern = re.compile(
        r"^(sure|certainly|here are|to expand|i can help|below are|expanded query)",
        re.IGNORECASE,
    )
    if preamble_pattern.match(cleaned) and not has_typed_items:
        return False, f"Completion contains only introductory preamble without actual expansions: '{cleaned}'"

    # Detect verbatim prompt echo
    query_lower = query.lower()
    if cleaned.lower() == query_lower or cleaned.lower() == f"/no_think expand this search query: {query_lower}":
        return False, "Completion is a verbatim echo of the query/prompt"

    # Check query term relevance across typed items
    all_content = " ".join(parsed.get("lex", []) + parsed.get("vec", []) + parsed.get("hyde", [])).lower()
    query_terms = [t for t in re.sub(r"[^a-z0-9\s]", " ", query_lower).split() if len(t) >= 2]
    term_matches = [t for t in query_terms if t in all_content]
    if query_terms and not term_matches:
        return False, f"Parsed expansions have zero overlap with query terms ({query_terms})"

    # Check expected keyword matches if specified
    if expected_keywords:
        kw_matches = [kw for kw in expected_keywords if kw.lower() in all_content]
        if not kw_matches:
            return False, f"Completion contains none of the expected technical keywords ({expected_keywords[:4]}...)"

    return True, "Valid usable query expansion"


@dataclasses.dataclass
class GenerateSmokeConfig:
    model_name_or_path: str = "mlx-community/Qwen3-1.7B-4bit"
    host: str = "127.0.0.1"
    port: int = 8797
    control_port: int = 8798
    timeout_s: float = 60.0
    min_headroom_mb: float = 3000.0
    output_json: Optional[str] = None
    dry_run: bool = False
    use_fake_child: bool = False


class GenerateSmokeRunner:
    """
    Orchestrates bounded, isolated single-stage MLX generation / query expansion smoke qualification
    under active MLXWatchdog supervision with production prompt and parser verification.
    """

    def __init__(self, config: GenerateSmokeConfig, sampler: Optional[SystemMemorySampler] = None):
        self.config = config
        self.sampler = sampler or SystemMemorySampler()
        self.last_report: dict[str, Any] = {}
        self.bridge = TSExpansionBridge()

    def run(self) -> dict[str, Any]:
        expanded_path = os.path.expanduser(self.config.model_name_or_path)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        server_script = os.path.join(repo_root, "scripts", "mlx_embed_server.py")

        cmd = [
            sys.executable,
            server_script,
            "--no-embed",
            "--generate-model",
            expanded_path,
            "--port",
            str(self.config.port),
            "--control-port",
            str(self.config.control_port),
            "--host",
            self.config.host,
            "--preload",
        ]

        supervisor_config = StageSupervisorConfig(
            cmd=cmd,
            host=self.config.host,
            port=self.config.port,
            control_port=self.config.control_port,
            timeout_s=self.config.timeout_s,
            min_headroom_mb=self.config.min_headroom_mb,
            stage_name="generate",
            model_identifier=self.config.model_name_or_path,
            output_file=self.config.output_json,
            dry_run=self.config.dry_run,
        )

        supervisor = SmokeStageSupervisor(config=supervisor_config, sampler=self.sampler)

        # Local weights existence check before spawning
        if not self.config.dry_run and not self.config.use_fake_child:
            hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            is_local_dir = os.path.exists(expanded_path) and os.path.isdir(expanded_path)
            repo_clean = self.config.model_name_or_path.replace("/", "--")
            hf_repo_dir = os.path.join(hf_cache_dir, f"models--{repo_clean}")
            has_hf_cache = os.path.exists(hf_repo_dir)

            if not (is_local_dir or has_hf_cache):
                blocked_report = {
                    "stage": "generate",
                    "model": self.config.model_name_or_path,
                    "port": self.config.port,
                    "control_port": self.config.control_port,
                    "status": "blocked",
                    "errors": [
                        f"Generation model weights missing locally: neither {expanded_path} nor {hf_repo_dir} exists"
                    ],
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

            # 1. Descriptor Schema Check
            ctx.check_breach("descriptor_fetch")
            desc_status, desc = client.get_json(f"{ctrl_url}/descriptor", deadline=deadline, timeout_s=2.0)
            if desc_status != 200 or not isinstance(desc, dict):
                ctx.add_error(f"Descriptor check failed: HTTP {desc_status} {desc}")
                ctx.report["status"] = "failed"
                return ctx.report

            ctx.report["descriptor"] = desc
            if desc.get("kind") != "generate":
                ctx.add_error(f"Descriptor kind is '{desc.get('kind')}', expected 'generate'")
                ctx.report["status"] = "failed"
                return ctx.report
            ctx.record_check("descriptor_valid", True)

            # 2. Public Query Expansion Evaluation with Production TS Protocol
            ctx.check_breach("query_expansion_probes")
            latencies = []
            completions_report = []
            functional_all_ok = True
            usable_all_ok = True

            for i, item in enumerate(GENERATE_TEST_PROMPTS):
                ctx.check_breach(f"generate_prompt_{i}")
                
                # Build exact production prompt via TS bridge
                prompt = self.bridge.build_prompt(item["query"], intent=item.get("intent"))
                t0 = time.monotonic()
                payload = {
                    "prompt": prompt,
                    "max_tokens": item["max_tokens"],
                    "temperature": 0.0,
                }
                
                # Run 1
                st_code, r1 = client.post_json(f"{base_url}/generate", payload, deadline=deadline, timeout_s=15.0)
                lat = (time.monotonic() - t0) * 1000.0
                latencies.append(lat)

                if st_code != 200 or not isinstance(r1, dict):
                    ctx.add_error(f"Prompt {i} failed: HTTP {st_code} {r1}")
                    functional_all_ok = False
                    usable_all_ok = False
                    ctx.report["status"] = "failed"
                    return ctx.report

                text1 = r1.get("text", "")
                if not text1 or not text1.strip():
                    ctx.add_error(f"Prompt {i} returned empty completion text")
                    functional_all_ok = False
                    usable_all_ok = False
                    ctx.report["status"] = "failed"
                    return ctx.report

                # Thinking tag check
                if "<think>" in text1 or "</think>" in text1:
                    ctx.add_error(f"Prompt {i} contains unstripped thinking tags")
                    functional_all_ok = False
                    usable_all_ok = False
                    ctx.report["status"] = "failed"
                    return ctx.report

                # Run 2 (determinism check)
                ctx.check_breach(f"generate_determinism_{i}")
                _, r2 = client.post_json(f"{base_url}/generate", payload, deadline=deadline, timeout_s=15.0)
                text2 = r2.get("text", "") if isinstance(r2, dict) else ""
                deterministic = (text1 == text2)
                if not deterministic:
                    ctx.add_error(f"Prompt {i} failed deterministic decoding (run1 != run2)")
                    functional_all_ok = False

                # Production TS format & quality parse
                parsed = parse_expansion(text1, query=item["query"], bridge=self.bridge)
                usable, quality_msg = validate_expansion_quality(
                    query=item["query"],
                    text=text1,
                    parsed=parsed,
                    expected_keywords=item.get("expected_keywords"),
                )
                if not usable:
                    ctx.add_error(f"Prompt {i} expansion quality failure: {quality_msg}")
                    usable_all_ok = False

                completions_report.append({
                    "query": item["query"],
                    "prompt": prompt,
                    "full_completion": text1,
                    "character_count": len(text1),
                    "parsed_structure": {
                        "lex": parsed["lex"],
                        "vec": parsed["vec"],
                        "hyde": parsed["hyde"],
                        "invalid": parsed["invalid"],
                        "queryables": parsed.get("queryables", []),
                    },
                    "usable_expansion": usable,
                    "quality_message": quality_msg,
                    "deterministic": deterministic,
                    "latency_ms": round(lat, 2),
                })

            ctx.record_check("functional_generation", functional_all_ok)
            ctx.record_check("usable_expansion", usable_all_ok)
            ctx.record_metric("completions", completions_report)
            if latencies:
                ctx.record_metric("generate_p50_ms", round(sorted(latencies)[len(latencies) // 2], 2))

            if not functional_all_ok or not usable_all_ok:
                ctx.report["status"] = "failed"
                return ctx.report

            # 3. Invalid Input Rejection Checks (must return HTTP 400)
            ctx.check_breach("invalid_input_probes")
            bad_inputs = [
                {"prompt": ""},
                {"prompt": "   "},
                {"prompt": "valid prompt", "max_tokens": 0},
                {"prompt": "valid prompt", "max_tokens": 999999},
            ]
            for idx, bi in enumerate(bad_inputs):
                st_bad, _ = client.post_json(f"{base_url}/generate", bi, deadline=deadline, timeout_s=3.0)
                if st_bad != 400:
                    ctx.add_error(f"Bad input {idx} returned HTTP {st_bad}, expected 400")
                    ctx.report["status"] = "failed"
                    return ctx.report
            ctx.record_check("invalid_input_rejection", True)

            # 4. Memory & Stats Telemetry
            ctx.check_breach("telemetry_capture")
            st_mem, r_mem = client.get_json(f"{ctrl_url}/memory", deadline=deadline, timeout_s=2.0)
            if st_mem == 200 and isinstance(r_mem, dict):
                ctx.record_metric("memory", r_mem)

            st_stats, r_stats = client.get_json(f"{ctrl_url}/stats", deadline=deadline, timeout_s=2.0)
            if st_stats == 200 and isinstance(r_stats, dict):
                ctx.record_metric("stats", r_stats)

        self.last_report = supervisor.last_report
        return supervisor.last_report


def run_generate_smoke(
    model_name_or_path: str = "mlx-community/Qwen3-1.7B-4bit",
    port: int = 8797,
    control_port: int = 8798,
    timeout_s: float = 60.0,
    min_headroom_mb: float = 3000.0,
    output_json: Optional[str] = None,
    dry_run: bool = False,
    sampler: Optional[SystemMemorySampler] = None,
) -> dict[str, Any]:
    cfg = GenerateSmokeConfig(
        model_name_or_path=model_name_or_path,
        port=port,
        control_port=control_port,
        timeout_s=timeout_s,
        min_headroom_mb=min_headroom_mb,
        output_json=output_json,
        dry_run=dry_run,
    )
    runner = GenerateSmokeRunner(cfg, sampler=sampler)
    return runner.run()
