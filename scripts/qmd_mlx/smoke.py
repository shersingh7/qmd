"""
smoke.py — Bounded Single-Model Embedding Smoke Runner & Fixture Suite

Safety & Isolation Guarantees:
1. Purely offline: Requires HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1.
2. Local path enforcement: Requires existing local weights directory; rejects remote repo IDs and downloads.
3. Endpoint isolation: Requires distinct loopback ports (127.0.0.1) that do not conflict with live services (8787).
4. Watchdog-owned lifecycle: Spawns server child with unique instance token, actively supervised by MLXWatchdog.
5. Model-aware headroom: Dynamically calculates required RAM from model architecture/file sizes and activation margins.
6. Guaranteed cleanup: Reaps supervisor, watcher, and server child processes on every exit path; sentinel untouched.
7. Isolated probe transport: HTTP probes use trust_env=False, allow_redirects=False, and bounded response buffers.
8. Measured telemetry: Captures ongoing measured memory snapshots throughout run (never fabricated data).
9. Synthetic fixture disclaimer: Numerical checks verify runtime determinism, numerical stability, and bounding;
   they do NOT evaluate retrieval semantic quality or compare MLX vs GGUF performance.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Optional
import numpy as np
import requests

from .adapters.embedding import (
    estimate_model_memory_mb,
    infer_model_params_b,
)
from .supervisor import SmokeHttpClient
from .watchdog import (
    BreachType,
    MLXWatchdog,
    MLXWatchdogConfig,
    SystemMemorySampler,
    SystemMetricsError,
    WatchdogCheckResult,
    is_numeric_loopback,
)


# --- Fixture Texts ---

SINGLETON_FIXTURE = "The quick brown fox jumps over the lazy dog."
SINGLETON_TEXT = SINGLETON_FIXTURE

BATCH_FIXTURES = [
    "Apple Silicon Metal unified memory acceleration.",
    "Vector search enables semantic retrieval across local markdown documentation and codebases.",
    "```python\ndef embed_batch(texts: list[str]) -> np.ndarray:\n    return mlx_model(texts)\n```",
    "自然语言处理 and multilingual text representations on macOS.",
    "Special symbols & punctuation: !@#$%^&*()_+-=[]{}|;':\",./<>?",
]

# Long text exceeding typical chunk lengths (~1000 tokens / 4000 characters)
LONG_INPUT_FIXTURE = (
    "Apple Silicon unified memory architecture provides high-bandwidth shared access between CPU and GPU cores. "
    * 35
)

# Tolerances
NORM_TOLERANCE = 1e-4
COSINE_SIMILARITY_TOLERANCE_FP32 = 0.9999
COSINE_SIMILARITY_TOLERANCE_BF16 = 0.999
MAX_ABS_DIFF_TOLERANCE_FP32 = 1e-3
MAX_ABS_DIFF_TOLERANCE_BF16 = 5e-3


@dataclasses.dataclass
class ModelMetadata:
    model_path: str
    model_type: Optional[str] = None
    architectures: list[str] = dataclasses.field(default_factory=list)
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    quantization: Optional[dict[str, Any]] = None
    vocab_size: Optional[int] = None
    has_safetensors: bool = False
    has_tokenizer: bool = False
    total_file_size_bytes: int = 0
    params_b: float = 0.6
    estimated_memory_mb: float = 0.0
    conservative_required_headroom_mb: float = 2048.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def inspect_model_metadata(model_path: str) -> ModelMetadata:
    """
    Inspects model directory configuration and weights metadata strictly read-only.
    Derives model parameter count and conservative memory headroom requirement.
    Never loads weights into MLX or RAM.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not os.path.isdir(model_path):
        raise ValueError(f"Model path must be a directory: {model_path}")

    config_path = os.path.join(model_path, "config.json")
    config: dict[str, Any] = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse config.json at {config_path}: {e}")

    # Inspect files
    has_safetensors = False
    has_tokenizer = False
    total_size = 0

    for root, _, files in os.walk(model_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                total_size += os.path.getsize(fpath)
            except OSError:
                pass
            if fname.endswith(".safetensors") or fname == "model.safetensors.index.json" or fname.endswith(".bin"):
                has_safetensors = True
            if fname in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
                has_tokenizer = True

    quant = config.get("quantization") or config.get("quantization_config")
    quant_str = quant.get("quant_type") if isinstance(quant, dict) else (quant if isinstance(quant, str) else None)

    # Calculate conservative memory requirement
    params_b = infer_model_params_b(os.path.basename(model_path), config)
    estimated_mb = estimate_model_memory_mb(model_path, quant_str, params_b)
    weights_size_mb = total_size / (1024 * 1024)
    base_mb = max(weights_size_mb, estimated_mb)
    activation_margin_mb = max(1024.0, base_mb * 0.35 + 512.0)
    conservative_required_headroom_mb = round(base_mb + activation_margin_mb, 1)

    # 4B and larger models require conservative minimum headroom ceiling
    if params_b >= 3.0:
        conservative_required_headroom_mb = max(conservative_required_headroom_mb, 3500.0)

    return ModelMetadata(
        model_path=os.path.abspath(model_path),
        model_type=config.get("model_type"),
        architectures=config.get("architectures", []),
        hidden_size=config.get("hidden_size"),
        num_hidden_layers=config.get("num_hidden_layers"),
        num_attention_heads=config.get("num_attention_heads"),
        num_key_value_heads=config.get("num_key_value_heads"),
        max_position_embeddings=config.get("max_position_embeddings"),
        quantization=quant,
        vocab_size=config.get("vocab_size"),
        has_safetensors=has_safetensors,
        has_tokenizer=has_tokenizer,
        total_file_size_bytes=total_size,
        params_b=params_b,
        estimated_memory_mb=round(estimated_mb, 1),
        conservative_required_headroom_mb=conservative_required_headroom_mb,
    )


# --- Numerical Validators ---

def check_finite(arr: np.ndarray) -> tuple[bool, str]:
    """Checks that all elements in the array are finite (no NaN, no Inf)."""
    if not isinstance(arr, np.ndarray):
        return False, f"Expected numpy ndarray, got {type(arr)}"
    if arr.size == 0:
        return False, "Array is empty (size 0)"
    if not np.all(np.isfinite(arr)):
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        return False, f"Array contains non-finite values (NaN: {nan_count}, Inf: {inf_count})"
    return True, "All values are finite"


def check_dimensions(arr: np.ndarray, expected_rows: int, expected_dims: Optional[int] = None) -> tuple[bool, str]:
    """Checks that array matches expected shape [expected_rows, expected_dims]."""
    if not isinstance(arr, np.ndarray):
        return False, f"Expected numpy ndarray, got {type(arr)}"
    if arr.ndim != 2:
        return False, f"Expected 2D array [rows, dims], got ndim={arr.ndim} shape={arr.shape}"
    if arr.shape[0] != expected_rows:
        return False, f"Expected {expected_rows} rows, got {arr.shape[0]}"
    if expected_dims is not None and arr.shape[1] != expected_dims:
        return False, f"Expected {expected_dims} dims, got {arr.shape[1]}"
    return True, f"Shape {arr.shape} matches expected [{expected_rows}, {expected_dims or '*'}]"


def check_l2_normalization(arr: np.ndarray, tol: float = NORM_TOLERANCE) -> tuple[bool, list[float], str]:
    """Checks that each row vector has unit L2 norm within specified tolerance."""
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        return False, [], "Invalid array for norm check"
    norms = np.linalg.norm(arr, axis=1).tolist()
    for i, n in enumerate(norms):
        if not math.isfinite(n) or abs(n - 1.0) > tol:
            return False, norms, f"Row {i} L2 norm {n:.7f} outside tolerance [1.0 - {tol}, 1.0 + {tol}]"
    return True, norms, f"All {len(norms)} vectors normalized (mean norm: {np.mean(norms):.6f})"


def check_batch_singleton_consistency(
    singleton_vec: np.ndarray,
    batch_vec: np.ndarray,
    tol_cos: float = COSINE_SIMILARITY_TOLERANCE_FP32,
    tol_diff: float = MAX_ABS_DIFF_TOLERANCE_FP32,
) -> tuple[bool, float, float, str]:
    """
    Computes cosine similarity and max absolute difference between a singleton embedding
    and the same text embedded inside a batch.
    """
    s = singleton_vec.flatten()
    b = batch_vec.flatten()
    if s.shape != b.shape:
        return False, 0.0, 0.0, f"Shape mismatch: singleton {s.shape} vs batch {b.shape}"

    norm_s = np.linalg.norm(s)
    norm_b = np.linalg.norm(b)
    if norm_s == 0 or norm_b == 0:
        return False, 0.0, 0.0, "Zero norm encountered during consistency check"

    cos_sim = float(np.dot(s, b) / (norm_s * norm_b))
    max_diff = float(np.max(np.abs(s - b)))

    if cos_sim < tol_cos:
        return (
            False,
            cos_sim,
            max_diff,
            f"Cosine similarity {cos_sim:.6f} below tolerance {tol_cos:.6f}",
        )
    if max_diff > tol_diff:
        return (
            False,
            cos_sim,
            max_diff,
            f"Max absolute difference {max_diff:.6e} exceeds tolerance {tol_diff:.6e}",
        )
    return True, cos_sim, max_diff, f"Consistency verified (cos_sim={cos_sim:.6f}, max_diff={max_diff:.6e})"


# --- Bounded HTTP Client ---

class SmokeHttpClient:
    """
    Isolated, wall-clock deadline-bounded HTTP client for smoke probes.
    Enforces:
    - trust_env=False (no system HTTP_PROXY / HTTPS_PROXY interference).
    - allow_redirects=False (no redirection on loopback endpoints).
    - Monotonic wall-clock deadline bounding across connect, send, and receive.
    - Maximum response body bytes cap.
    """

    def __init__(self, default_timeout_s: float = 10.0, max_response_bytes: int = 10 * 1024 * 1024):
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.max_redirects = 0
        self.default_timeout_s = default_timeout_s
        self.max_response_bytes = max_response_bytes

    def request(
        self,
        method: str,
        url: str,
        json_data: Optional[dict[str, Any]] = None,
        deadline: Optional[float] = None,
        per_request_timeout_s: Optional[float] = None,
    ) -> requests.Response:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise TimeoutError("Absolute execution deadline exceeded before initiating HTTP request")

        rem = (deadline - now) if deadline is not None else (per_request_timeout_s or self.default_timeout_s)
        sock_timeout = max(0.001, min(per_request_timeout_s or self.default_timeout_s, rem))

        try:
            resp = self.session.request(
                method=method,
                url=url,
                json=json_data,
                timeout=sock_timeout,
                allow_redirects=False,
                stream=True,
            )

            chunks = []
            bytes_received = 0
            for chunk in resp.iter_content(chunk_size=8192):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Absolute execution deadline exceeded while reading HTTP response")
                if chunk:
                    bytes_received += len(chunk)
                    if bytes_received > self.max_response_bytes:
                        raise ValueError(f"HTTP response exceeded byte limit of {self.max_response_bytes} bytes")
                    chunks.append(chunk)

            resp._content = b"".join(chunks)
            return resp
        except (requests.exceptions.Timeout, socket.timeout):
            raise TimeoutError("HTTP request socket timed out or deadline exceeded")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HTTP probe request failed: {e}")

    def get_json(
        self,
        url: str,
        deadline: Optional[float] = None,
        timeout_s: Optional[float] = None,
    ) -> tuple[int, Any]:
        resp = self.request("GET", url, deadline=deadline, per_request_timeout_s=timeout_s)
        try:
            return resp.status_code, resp.json()
        except Exception as e:
            return resp.status_code, {"raw": resp.text, "error": str(e)}

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        deadline: Optional[float] = None,
        timeout_s: Optional[float] = None,
    ) -> tuple[int, Any]:
        resp = self.request("POST", url, json_data=payload, deadline=deadline, per_request_timeout_s=timeout_s)
        try:
            return resp.status_code, resp.json()
        except Exception as e:
            return resp.status_code, {"raw": resp.text, "error": str(e)}

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass


# --- Configuration & Runner ---

@dataclasses.dataclass
class SmokeRunnerConfig:
    model_path: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8797
    control_port: int = 8798
    timeout_s: float = 60.0
    min_headroom_mb: float = 2048.0
    use_fake_child: bool = False
    fake_dims: int = 2560
    fake_hang_on_embed: bool = False
    fake_fail_consistency: bool = False
    dry_run: bool = False
    diagnostic: bool = False
    real_model_opt_in: bool = False
    max_rss_mb: Optional[float] = None
    max_swap_growth_mb: Optional[float] = None
    min_free_memory_pct: float = 12.0
    output_file: Optional[str] = None


class SmokeRunner:
    """
    Orchestrates bounded, isolated single-model embedding smoke qualification.
    Child server process is owned and actively monitored in real time by MLXWatchdog.
    """

    def __init__(self, config: SmokeRunnerConfig, sampler: Optional[SystemMemorySampler] = None):
        self.config = config
        self.sampler = sampler or SystemMemorySampler()
        self._preflight_defaults: Optional[Any] = None
        self.last_report: dict[str, Any] = {}

    def validate_preflight(self) -> dict[str, Any]:
        """
        Validates system telemetry, headroom, loopback policy, port availability, and model path.
        Fails closed on any error before process spawning.
        """
        preflight_info: dict[str, Any] = {
            "timestamp": time.time(),
            "host": self.config.host,
            "port": self.config.port,
            "control_port": self.config.control_port,
        }

        # 1. Host restriction (strict numeric IPv4 loopback)
        if not is_numeric_loopback(self.config.host) or self.config.host != "127.0.0.1":
            raise ValueError(f"Host '{self.config.host}' must be numeric loopback '127.0.0.1'")

        # 2. Port distinctness and validity
        if not (1 <= self.config.port <= 65535):
            raise ValueError(f"Invalid port: {self.config.port}")
        if not (1 <= self.config.control_port <= 65535):
            raise ValueError(f"Invalid control port: {self.config.control_port}")
        if self.config.port == self.config.control_port:
            raise ValueError(f"Inference port ({self.config.port}) and control port ({self.config.control_port}) must be distinct")

        # 3. Prevent collision with live production daemon (8787)
        if self.config.port == 8787 or self.config.control_port == 8787:
            raise ValueError("Refusing to use port 8787; port 8787 is reserved for live daemon. Choose distinct qualification ports.")

        # 4. Port availability check
        for p_name, p_val in [("Inference port", self.config.port), ("Control port", self.config.control_port)]:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.config.host, p_val))
                s.close()
            except OSError as e:
                raise RuntimeError(f"{p_name} {p_val} on {self.config.host} is already in use: {e}")

        # 5. Memory telemetry & headroom check
        try:
            installed_ram = self.sampler.get_installed_ram_mb()
            headroom = self.sampler.get_memory_headroom_mb()
            swap_used = self.sampler.get_swap_used_mb()
            free_pct = self.sampler.get_memory_free_pct()
            defaults = self.sampler.compute_conservative_defaults(headroom_mb=headroom)
            self._preflight_defaults = defaults
            defaults_dict = dataclasses.asdict(defaults) if dataclasses.is_dataclass(defaults) else (defaults.__dict__ if hasattr(defaults, "__dict__") else {})
            preflight_info.update({
                "installed_ram_mb": installed_ram,
                "headroom_mb": headroom,
                "swap_used_mb": swap_used,
                "free_memory_pct": free_pct,
                "watchdog_defaults": defaults_dict,
            })
        except SystemMetricsError as e:
            raise SystemMetricsError(f"Preflight memory sampling failed: {e}")

        # 6. Model-aware preflight headroom check
        if not self.config.use_fake_child:
            if not self.config.real_model_opt_in:
                raise ValueError(
                    "Real model execution requires explicit opt-in (--real-model). "
                    "Use --rehearsal for bounded offline testing with synthetic model adapter."
                )
            if not self.config.model_path:
                raise ValueError("Model path is required for real model execution")
            meta = inspect_model_metadata(self.config.model_path)
            preflight_info["model_metadata"] = meta.to_dict()

            required_headroom = max(self.config.min_headroom_mb, meta.conservative_required_headroom_mb)
            if headroom < required_headroom:
                raise RuntimeError(
                    f"Insufficient memory headroom for {meta.params_b:.1f}B model: "
                    f"measured {headroom:.1f} MB < required {required_headroom:.1f} MB "
                    f"(base weights {meta.estimated_memory_mb:.1f} MB + activation margin "
                    f"{meta.conservative_required_headroom_mb - meta.estimated_memory_mb:.1f} MB). "
                    f"Refusing to load real model."
                )
        else:
            if headroom < min(self.config.min_headroom_mb, 256.0):
                raise RuntimeError(
                    f"Insufficient memory headroom for rehearsal: measured {headroom:.1f} MB < required {self.config.min_headroom_mb:.1f} MB"
                )

        return preflight_info

    def run(self) -> dict[str, Any]:
        """
        Executes bounded single-model smoke qualification workflow under active watchdog supervision.
        """
        run_start_time = time.time()
        start_mono = time.monotonic()
        deadline = start_mono + self.config.timeout_s

        report: dict[str, Any] = {
            "status": "in_progress",
            "start_time": run_start_time,
            "mode": "rehearsal_synthetic" if self.config.use_fake_child else "real_model",
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "control_port": self.config.control_port,
                "timeout_s": self.config.timeout_s,
                "use_fake_child": self.config.use_fake_child,
                "fake_fail_consistency": self.config.fake_fail_consistency,
                "dry_run": self.config.dry_run,
                "diagnostic": self.config.diagnostic,
                "model_path": self.config.model_path,
            },
            "fixtures": {},
            "telemetry_samples": [],
            "captured_logs": "",
            "child_cleanup": {
                "cleaned": False,
                "pid": None,
                "exit_code": None,
            },
            "disclaimer": (
                "Synthetic numerical checks verify runtime determinism, numerical stability, "
                "and bounded resource behavior; they do NOT evaluate semantic retrieval quality."
            ),
        }
        self.last_report = report

        # Step 1: Preflight
        preflight = self.validate_preflight()
        report["preflight"] = preflight

        if self.config.dry_run:
            report["status"] = "dry_run_completed"
            report["fixtures"] = {
                "singleton": {"status": "not_run", "reason": "dry_run requested"},
                "batch": {"status": "not_run", "reason": "dry_run requested"},
                "consistency": {"status": "not_run", "reason": "dry_run requested"},
                "long_input": {"status": "not_run", "reason": "dry_run requested"},
                "timing": {"status": "not_run", "reason": "dry_run requested"},
            }
            report["duration_s"] = time.time() - run_start_time
            self.last_report = report
            return report

        # Step 2: Spawn Child Server
        instance_token = uuid.uuid4().hex
        child_env = os.environ.copy()
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"
        child_env["HF_DATASETS_OFFLINE"] = "1"
        child_env["MLX_INSTANCE_TOKEN"] = instance_token
        child_env["MLX_EMBED_PORT"] = str(self.config.port)
        child_env["MLX_CONTROL_PORT"] = str(self.config.control_port)
        if self.config.fake_hang_on_embed:
            child_env["MLX_HANG_ON_EMBED"] = "1"
        if self.config.fake_fail_consistency:
            child_env["MLX_FAKE_FAIL_CONSISTENCY"] = "1"

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if "PYTHONPATH" in child_env:
            child_env["PYTHONPATH"] = f"{repo_root}:{child_env['PYTHONPATH']}"
        else:
            child_env["PYTHONPATH"] = repo_root

        server_script = os.path.join(repo_root, "scripts", "mlx_embed_server.py")
        if self.config.use_fake_child:
            model_spec = f"synthetic-qwen3-4b-{self.config.fake_dims}d" if self.config.fake_dims != 2560 else "synthetic-qwen3-4b"
            cmd = [
                sys.executable,
                server_script,
                "--model",
                model_spec,
                "--port",
                str(self.config.port),
                "--control-port",
                str(self.config.control_port),
                "--host",
                self.config.host,
                "--no-warmup",
            ]
        else:
            cmd = [
                sys.executable,
                server_script,
                "--model",
                str(self.config.model_path),
                "--port",
                str(self.config.port),
                "--control-port",
                str(self.config.control_port),
                "--host",
                self.config.host,
                "--no-warmup",
            ]

        child: Optional[subprocess.Popen] = None
        log_file = None
        log_file_path = None
        client = SmokeHttpClient(default_timeout_s=5.0)
        t_supervisor: Optional[threading.Thread] = None
        stop_supervisor = threading.Event()
        breach_event = threading.Event()
        breach_result: list[Optional[WatchdogCheckResult]] = [None]
        telemetry_samples: list[dict[str, Any]] = []
        cleanup_errors: list[str] = []

        try:
            # Bounded log file prevents pipe buffer deadlocks
            log_file = tempfile.NamedTemporaryFile(mode="w+", prefix="mlx_smoke_server_", suffix=".log", delete=False)
            log_file_path = log_file.name

            child = subprocess.Popen(
                cmd,
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            report["child_pid"] = child.pid
            report["instance_token"] = instance_token

            # Step 3: Instantiate MLXWatchdog & Background Supervisor Thread
            watchdog_config = MLXWatchdogConfig(
                pid=child.pid,
                host=self.config.host,
                port=self.config.port,
                control_port=self.config.control_port,
                max_rss_mb=self.config.max_rss_mb,
                max_swap_growth_mb=self.config.max_swap_growth_mb,
                min_free_memory_pct=self.config.min_free_memory_pct,
                check_interval_s=0.25,
                startup_grace_period_s=min(15.0, self.config.timeout_s),
                stalled_inference_timeout_s=min(10.0, self.config.timeout_s),
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

            t_supervisor = threading.Thread(target=_supervisor_loop, daemon=True, name="Smoke-Watchdog-Supervisor")
            t_supervisor.start()

            # Step 4: Poll Control Port for Readiness
            base_url = f"http://{self.config.host}:{self.config.port}"
            ctrl_url = f"http://{self.config.host}:{self.config.control_port}"

            ready = False
            startup_deadline = min(deadline, time.monotonic() + 30.0)
            while time.monotonic() < startup_deadline:
                if breach_event.is_set():
                    b_res = breach_result[0]
                    raise RuntimeError(f"Watchdog breach during server startup: {b_res.breach_reason if b_res else 'unknown breach'}")

                if child.poll() is not None:
                    # Allow supervisor a short window to register breach if watchdog delivered a kill signal
                    for _ in range(5):
                        if breach_event.is_set() or (watchdog.last_check_result and not watchdog.last_check_result.healthy):
                            break
                        time.sleep(0.05)
                    if breach_event.is_set():
                        b_res = breach_result[0]
                        raise RuntimeError(f"Watchdog breach during server startup: {b_res.breach_reason if b_res else 'unknown breach'}")
                    if watchdog.last_check_result and not watchdog.last_check_result.healthy:
                        raise RuntimeError(f"Watchdog breach during server startup: {watchdog.last_check_result.breach_reason}")

                    log_tail = ""
                    try:
                        with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
                            log_tail = f.read()[-2000:]
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Server child exited prematurely with code {child.returncode}. Server logs:\n{log_tail}"
                    )

                try:
                    status_code, data = client.get_json(f"{ctrl_url}/health", deadline=startup_deadline, timeout_s=1.0)
                    if status_code == 200 and isinstance(data, dict):
                        if (
                            data.get("ready") is True
                            and data.get("instance_token") == instance_token
                            and data.get("pid") == child.pid
                        ):
                            ready = True
                            report["server_ready_elapsed_s"] = time.monotonic() - start_mono
                            break
                except Exception:
                    pass
                time.sleep(0.1)

            if not ready:
                if breach_event.is_set():
                    b_res = breach_result[0]
                    raise RuntimeError(f"Watchdog breach during server startup: {b_res.breach_reason if b_res else 'unknown breach'}")
                if watchdog.last_check_result and not watchdog.last_check_result.healthy:
                    raise RuntimeError(f"Watchdog breach during server startup: {watchdog.last_check_result.breach_reason}")
                raise TimeoutError(f"Server child failed to report ready on {ctrl_url}/health within startup period")

            # Step 5: Fetch & Strictly Validate Descriptor Schema
            status_desc, descriptor = client.get_json(f"{ctrl_url}/descriptor", deadline=deadline, timeout_s=2.0)
            if status_desc != 200 or not isinstance(descriptor, dict):
                raise RuntimeError(f"Failed to fetch descriptor: status {status_desc}, payload: {descriptor}")

            expected_dims = descriptor.get("outputDimensions") or descriptor.get("nativeDimensions")
            if not isinstance(expected_dims, int) or isinstance(expected_dims, bool) or expected_dims <= 0:
                raise ValueError(
                    f"Invalid or missing dimensions in descriptor schema (got {expected_dims}): {descriptor}"
                )

            report["descriptor"] = descriptor

            fixtures_report: dict[str, Any] = {}

            def _check_breach_before_request(stage_name: str):
                if breach_event.is_set():
                    b_res = breach_result[0]
                    raise RuntimeError(f"Watchdog breach before {stage_name}: {b_res.breach_reason if b_res else 'unknown breach'}")
                if watchdog.last_check_result and not watchdog.last_check_result.healthy:
                    raise RuntimeError(f"Watchdog breach before {stage_name}: {watchdog.last_check_result.breach_reason}")
                if child.poll() is not None:
                    for _ in range(5):
                        if breach_event.is_set() or (watchdog.last_check_result and not watchdog.last_check_result.healthy):
                            break
                        time.sleep(0.05)
                    if breach_event.is_set():
                        b_res = breach_result[0]
                        raise RuntimeError(f"Watchdog breach before {stage_name}: {b_res.breach_reason if b_res else 'unknown breach'}")
                    if watchdog.last_check_result and not watchdog.last_check_result.healthy:
                        raise RuntimeError(f"Watchdog breach before {stage_name}: {watchdog.last_check_result.breach_reason}")
                    raise RuntimeError(f"Server child exited unexpectedly with code {child.returncode} before {stage_name}")

            # Measure token lengths via /tokenize if available
            single_tok_len = None
            try:
                _, tok_res = client.post_json(f"{base_url}/tokenize", {"texts": [SINGLETON_TEXT]}, deadline=deadline)
                if isinstance(tok_res, dict) and "counts" in tok_res and tok_res["counts"]:
                    single_tok_len = int(tok_res["counts"][0])
            except Exception:
                pass

            # Step 6: Fixture 1 — Singleton Embedding
            _check_breach_before_request("singleton")
            t0 = time.monotonic()
            st_single, r_single = client.post_json(
                f"{base_url}/embed",
                {"texts": [SINGLETON_TEXT]},
                deadline=deadline,
            )
            single_elapsed_ms = (time.monotonic() - t0) * 1000.0

            if st_single != 200 or not isinstance(r_single, dict):
                raise RuntimeError(f"Singleton embed failed with HTTP {st_single}: {r_single}")

            single_arr = np.array(r_single["embeddings"], dtype=np.float32)

            fin_ok, fin_msg = check_finite(single_arr)
            dim_ok, dim_msg = check_dimensions(single_arr, expected_rows=1, expected_dims=expected_dims)
            norm_ok, norms, norm_msg = check_l2_normalization(single_arr)

            fixtures_report["singleton"] = {
                "status": "passed" if (fin_ok and dim_ok and norm_ok) else "failed",
                "latency_ms": single_elapsed_ms,
                "shape": list(single_arr.shape),
                "finite": fin_ok,
                "finite_msg": fin_msg,
                "dimension_match": dim_ok,
                "dimension_msg": dim_msg,
                "l2_norm": norms[0] if norms else None,
                "norm_msg": norm_msg,
                "token_length": single_tok_len,
            }
            report["fixtures"] = fixtures_report
            if not (fin_ok and dim_ok and norm_ok):
                raise ValueError(f"Singleton fixture validation failed: {dim_msg}, {norm_msg}, {fin_msg}")

            # Step 7: Fixture 2 — Mixed-Length Batch Embedding
            _check_breach_before_request("batch")
            batch_input = [SINGLETON_TEXT] + BATCH_FIXTURES

            batch_tok_lens = None
            try:
                _, tok_b_res = client.post_json(f"{base_url}/tokenize", {"texts": batch_input}, deadline=deadline)
                if isinstance(tok_b_res, dict) and "counts" in tok_b_res:
                    batch_tok_lens = [int(x) for x in tok_b_res["counts"]]
            except Exception:
                pass

            t0 = time.monotonic()
            st_batch, r_batch = client.post_json(
                f"{base_url}/embed",
                {"texts": batch_input},
                deadline=deadline,
            )
            batch_elapsed_ms = (time.monotonic() - t0) * 1000.0

            if st_batch != 200 or not isinstance(r_batch, dict):
                raise RuntimeError(f"Batch embed failed with HTTP {st_batch}: {r_batch}")

            batch_arr = np.array(r_batch["embeddings"], dtype=np.float32)

            b_fin_ok, b_fin_msg = check_finite(batch_arr)
            b_dim_ok, b_dim_msg = check_dimensions(batch_arr, expected_rows=len(batch_input), expected_dims=expected_dims)
            b_norm_ok, b_norms, b_norm_msg = check_l2_normalization(batch_arr)

            fixtures_report["batch"] = {
                "status": "passed" if (b_fin_ok and b_dim_ok and b_norm_ok) else "failed",
                "latency_ms": batch_elapsed_ms,
                "count": len(batch_input),
                "shape": list(batch_arr.shape),
                "finite": b_fin_ok,
                "dimension_match": b_dim_ok,
                "mean_l2_norm": float(np.mean(b_norms)) if b_norms else None,
                "norm_msg": b_norm_msg,
                "token_lengths": batch_tok_lens,
            }
            report["fixtures"] = fixtures_report
            if not (b_fin_ok and b_dim_ok and b_norm_ok):
                raise ValueError(f"Batch fixture validation failed: {b_dim_msg}, {b_norm_msg}, {b_fin_msg}")

            # Step 8: Fixture 3 — Singleton vs Batch Consistency
            _check_breach_before_request("consistency")
            cons_ok, cos_sim, max_diff, cons_msg = check_batch_singleton_consistency(
                single_arr[0],
                batch_arr[0],
                tol_cos=COSINE_SIMILARITY_TOLERANCE_FP32,
                tol_diff=MAX_ABS_DIFF_TOLERANCE_FP32,
            )
            fixtures_report["consistency"] = {
                "status": "passed" if cons_ok else "failed",
                "cosine_similarity": cos_sim,
                "max_absolute_difference": max_diff,
                "cosine_tolerance": COSINE_SIMILARITY_TOLERANCE_FP32,
                "max_abs_diff_tolerance": MAX_ABS_DIFF_TOLERANCE_FP32,
                "reference_shapes": {
                    "singleton": list(single_arr.shape),
                    "batch": list(batch_arr.shape),
                },
                "token_lengths": {
                    "singleton": single_tok_len,
                    "batch_target": batch_tok_lens[0] if batch_tok_lens else None,
                    "batch_all": batch_tok_lens,
                },
                "message": cons_msg,
            }
            report["fixtures"] = fixtures_report
            if not cons_ok:
                fixtures_report["long_input"] = {"status": "not_run", "reason": "aborted due to prior failure"}
                fixtures_report["timing"] = {"status": "not_run", "reason": "aborted due to prior failure"}
                report["status"] = "failed"
                report["error"] = f"Singleton vs Batch consistency failed: {cons_msg}"
                if not self.config.diagnostic:
                    raise ValueError(f"Singleton vs Batch consistency failed: {cons_msg}")

            if cons_ok:
                # Step 9: Fixture 4 — Long-Input Handling Policy
                _check_breach_before_request("long_input")
                t0 = time.monotonic()
                st_long, r_long = client.post_json(
                    f"{base_url}/embed",
                    {"texts": [LONG_INPUT_FIXTURE]},
                    deadline=deadline,
                )
                long_elapsed_ms = (time.monotonic() - t0) * 1000.0

                if st_long != 200 or not isinstance(r_long, dict):
                    raise RuntimeError(f"Long-input embed failed with HTTP {st_long}: {r_long}")

                long_arr = np.array(r_long["embeddings"], dtype=np.float32)

                l_fin_ok, _ = check_finite(long_arr)
                l_dim_ok, _ = check_dimensions(long_arr, expected_rows=1, expected_dims=expected_dims)
                l_norm_ok, l_norms, _ = check_l2_normalization(long_arr)

                fixtures_report["long_input"] = {
                    "status": "passed" if (l_fin_ok and l_dim_ok and l_norm_ok) else "failed",
                    "latency_ms": long_elapsed_ms,
                    "input_character_count": len(LONG_INPUT_FIXTURE),
                    "shape": list(long_arr.shape),
                    "finite": l_fin_ok,
                    "l2_norm": l_norms[0] if l_norms else None,
                }
                report["fixtures"] = fixtures_report
                if not (l_fin_ok and l_dim_ok and l_norm_ok):
                    raise ValueError("Long input fixture validation failed")

                # Step 10: Fixture 5 — Repeated Requests Timing
                _check_breach_before_request("timing")
                timing_latencies: list[float] = []
                for _ in range(5):
                    _check_breach_before_request("timing_iteration")
                    t_iter = time.monotonic()
                    st_rep, r_rep = client.post_json(
                        f"{base_url}/embed",
                        {"texts": ["Timing repeatability probe."]},
                        deadline=deadline,
                    )
                    if st_rep != 200:
                        raise RuntimeError(f"Repeated request failed with HTTP {st_rep}: {r_rep}")
                    timing_latencies.append((time.monotonic() - t_iter) * 1000.0)

                timing_latencies_sorted = sorted(timing_latencies)
                fixtures_report["timing"] = {
                    "status": "passed",
                    "iterations": len(timing_latencies),
                    "latencies_ms": [round(x, 2) for x in timing_latencies],
                    "min_ms": round(min(timing_latencies), 2),
                    "p50_ms": round(timing_latencies_sorted[len(timing_latencies_sorted) // 2], 2),
                    "p95_ms": round(timing_latencies_sorted[int(len(timing_latencies_sorted) * 0.95)], 2),
                    "max_ms": round(max(timing_latencies), 2),
                    "avg_ms": round(float(np.mean(timing_latencies)), 2),
                }

            # Step 11: Diagnostic Mode Suite (if requested)
            if self.config.diagnostic:
                diagnostics_report: dict[str, Any] = {}

                # 1. Repeated Singletons Probe
                _check_breach_before_request("diagnostic_repeated_singleton")
                t0 = time.monotonic()
                st_s1, r_s1 = client.post_json(f"{base_url}/embed", {"texts": [SINGLETON_TEXT]}, deadline=deadline)
                lat_s1 = (time.monotonic() - t0) * 1000.0

                t0 = time.monotonic()
                st_s2, r_s2 = client.post_json(f"{base_url}/embed", {"texts": [SINGLETON_TEXT]}, deadline=deadline)
                lat_s2 = (time.monotonic() - t0) * 1000.0

                if st_s1 == 200 and st_s2 == 200 and isinstance(r_s1, dict) and isinstance(r_s2, dict):
                    v1 = np.array(r_s1["embeddings"][0], dtype=np.float32)
                    v2 = np.array(r_s2["embeddings"][0], dtype=np.float32)
                    rep_ok, rep_cos, rep_diff, rep_msg = check_batch_singleton_consistency(
                        v1, v2, tol_cos=COSINE_SIMILARITY_TOLERANCE_FP32, tol_diff=MAX_ABS_DIFF_TOLERANCE_FP32
                    )
                    diagnostics_report["repeated_singleton"] = {
                        "status": "passed" if rep_ok else "failed",
                        "latencies_ms": [round(lat_s1, 2), round(lat_s2, 2)],
                        "cosine_similarity": rep_cos,
                        "max_absolute_difference": rep_diff,
                        "token_length": single_tok_len,
                        "shape": list(v1.shape),
                        "message": rep_msg,
                    }

                # 2. Same-Length Duplicate Batch Probe
                _check_breach_before_request("diagnostic_duplicate_batch")
                dup_texts = [SINGLETON_TEXT] * 4
                t0 = time.monotonic()
                st_dup, r_dup = client.post_json(f"{base_url}/embed", {"texts": dup_texts}, deadline=deadline)
                lat_dup = (time.monotonic() - t0) * 1000.0

                if st_dup == 200 and isinstance(r_dup, dict):
                    dup_arr = np.array(r_dup["embeddings"], dtype=np.float32)
                    dup_cosines = []
                    dup_diffs = []
                    for row_idx in range(len(dup_texts)):
                        _, c, d, _ = check_batch_singleton_consistency(
                            single_arr[0], dup_arr[row_idx], tol_cos=0.0, tol_diff=1.0
                        )
                        dup_cosines.append(c)
                        dup_diffs.append(d)

                    diagnostics_report["duplicate_batch"] = {
                        "status": "passed" if min(dup_cosines) >= COSINE_SIMILARITY_TOLERANCE_FP32 else "failed",
                        "latency_ms": round(lat_dup, 2),
                        "count": len(dup_texts),
                        "token_lengths": [single_tok_len] * len(dup_texts) if single_tok_len else None,
                        "shape": list(dup_arr.shape),
                        "singleton_vs_row_cosines": dup_cosines,
                        "singleton_vs_row_max_diffs": dup_diffs,
                        "min_cosine_similarity": min(dup_cosines),
                        "max_absolute_difference": max(dup_diffs),
                    }

                # 3. Mixed Lengths Batch Probe
                _check_breach_before_request("diagnostic_mixed_lengths")
                mixed_texts = [
                    SINGLETON_TEXT,
                    "Fast vector search with MLX.",
                    "Apple Silicon unified memory architecture provides high-bandwidth shared memory access across CPU and GPU cores.",
                    LONG_INPUT_FIXTURE[:300],
                ]
                mixed_lens = None
                try:
                    _, m_tok = client.post_json(f"{base_url}/tokenize", {"texts": mixed_texts}, deadline=deadline)
                    if isinstance(m_tok, dict) and "counts" in m_tok:
                        mixed_lens = [int(x) for x in m_tok["counts"]]
                except Exception:
                    pass

                t0 = time.monotonic()
                st_mix, r_mix = client.post_json(f"{base_url}/embed", {"texts": mixed_texts}, deadline=deadline)
                lat_mix = (time.monotonic() - t0) * 1000.0

                if st_mix == 200 and isinstance(r_mix, dict):
                    mix_arr = np.array(r_mix["embeddings"], dtype=np.float32)
                    _, mix_cos, mix_diff, mix_msg = check_batch_singleton_consistency(
                        single_arr[0], mix_arr[0], tol_cos=COSINE_SIMILARITY_TOLERANCE_FP32, tol_diff=MAX_ABS_DIFF_TOLERANCE_FP32
                    )
                    diagnostics_report["mixed_lengths"] = {
                        "status": "passed" if mix_cos >= COSINE_SIMILARITY_TOLERANCE_FP32 else "failed",
                        "latency_ms": round(lat_mix, 2),
                        "count": len(mixed_texts),
                        "token_lengths": mixed_lens,
                        "shape": list(mix_arr.shape),
                        "singleton_vs_row0_cosine": mix_cos,
                        "singleton_vs_row0_max_diff": mix_diff,
                        "message": mix_msg,
                    }

                # 4. Position Permutations Probe
                _check_breach_before_request("diagnostic_position_permutations")
                perm_results = []
                for target_idx in range(len(mixed_texts)):
                    perm = list(mixed_texts)
                    perm[0], perm[target_idx] = perm[target_idx], perm[0]
                    p_lens = list(mixed_lens) if mixed_lens else None
                    if p_lens:
                        p_lens[0], p_lens[target_idx] = p_lens[target_idx], p_lens[0]

                    t0 = time.monotonic()
                    st_p, r_p = client.post_json(f"{base_url}/embed", {"texts": perm}, deadline=deadline)
                    lat_p = (time.monotonic() - t0) * 1000.0

                    if st_p == 200 and isinstance(r_p, dict):
                        p_arr = np.array(r_p["embeddings"], dtype=np.float32)
                        target_pos_in_perm = target_idx
                        _, p_cos, p_diff, _ = check_batch_singleton_consistency(
                            single_arr[0], p_arr[target_pos_in_perm], tol_cos=0.0, tol_diff=1.0
                        )
                        perm_results.append({
                            "target_position_in_batch": target_pos_in_perm,
                            "latency_ms": round(lat_p, 2),
                            "token_lengths": p_lens,
                            "cosine_vs_singleton": p_cos,
                            "max_diff_vs_singleton": p_diff,
                        })

                diagnostics_report["position_permutations"] = {
                    "status": "passed" if all(p["cosine_vs_singleton"] >= COSINE_SIMILARITY_TOLERANCE_FP32 for p in perm_results) else "failed",
                    "permutations": perm_results,
                }

                report["diagnostics"] = diagnostics_report

            report["fixtures"] = fixtures_report
            report["telemetry_samples"] = telemetry_samples
            report["status"] = "failed" if report.get("error") else "passed"

        except Exception as e:
            report["status"] = "failed"
            if "error" not in report or not report["error"]:
                report["error"] = str(e)
            report["telemetry_samples"] = telemetry_samples
            self.last_report = report
            raise
        finally:
            # Step 12: Guaranteed Cleanup and Log Capture
            # 1. Capture bounded log tail before closing or unlinking
            if log_file_path and os.path.exists(log_file_path):
                try:
                    with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
                        report["captured_logs"] = f.read()[-8192:]
                except Exception as log_err:
                    report["captured_logs"] = f"Failed to capture server logs: {log_err}"

            # 2. Stop supervisor
            stop_supervisor.set()
            if t_supervisor is not None and t_supervisor.is_alive():
                try:
                    t_supervisor.join(timeout=1.0)
                except Exception:
                    pass

            client.close()

            # 3. Clean up child process
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

            # 4. Clean up log file
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

            report["duration_s"] = time.time() - run_start_time
            self.last_report = report

        return report
