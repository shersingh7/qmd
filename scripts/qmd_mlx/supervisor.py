"""
supervisor.py — Reusable Owned-Process Supervisor for Bounded MLX Stage Qualification

Guarantees & Invariants:
1. Active MLXWatchdog Lifecycle: Server child process is continuously supervised by
   an instantiated MLXWatchdog in a dedicated background thread.
2. TokenPID & Endpoint Binding: Enforces unique MLX_INSTANCE_TOKEN, strict numeric loopback (127.0.0.1),
   and matches PID and instance token on control /health probes.
3. Continuous Measured Telemetry: Continuously samples RSS, swap usage, swap growth, and memory
   pressure throughout execution (never fabricated data).
4. Wall-Clock Deadline Bounding: Enforces monotonic execution deadlines across startup, probes, and teardown.
5. Structured Breach Handling & Process Isolation: Watchdog breaches or deadline overruns terminate ONLY
   the owned child process. Sentinel / parent processes survive intact.
6. Guaranteed Teardown & Saved JSON: In finally blocks, captures bounded logs, stops supervisor thread,
   reaps child, unlinks log file, and returns a complete, structured JSON report even on exceptions or breaches.
7. Isolated HTTP Transport: Probes use SmokeHttpClient (trust_env=False, allow_redirects=False,
   monotonic deadline enforcement, bounded streaming response bodies).
"""

from __future__ import annotations

import contextlib
import dataclasses
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
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
import requests

from .watchdog import (
    BreachType,
    MLXWatchdog,
    MLXWatchdogConfig,
    SystemMemorySampler,
    SystemMetricsError,
    WatchdogCheckResult,
    is_numeric_loopback,
)


@dataclasses.dataclass
class SubordinateProcessResult:
    exit_code: Optional[int]
    captured_logs: str
    duration_s: float
    timed_out: bool = False
    breached: bool = False
    pid: Optional[int] = None
    error: Optional[str] = None
    sigkill_used: bool = False
    barrier_reached: bool = False


def run_subordinate_process(
    cmd: list[str],
    deadline: float,
    check_breach: Optional[Callable[[str], None]] = None,
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    poll_interval_s: float = 0.05,
    capture_log_prefix: str = "subordinate_",
    max_log_bytes: int = 65536,
    barrier_sentinel: Optional[str] = None,
    kill_on_barrier: bool = False,
) -> SubordinateProcessResult:
    """
    Executes a bounded subordinate child process under strict monotonic deadline monitoring,
    periodic breach checks against the parent supervisor/watchdog, temporary file-backed
    log capture, and guaranteed fail-closed termination of ONLY the owned child process.
    """
    t0 = time.monotonic()
    if t0 >= deadline:
        raise TimeoutError("Absolute execution deadline exceeded before initiating subordinate process")

    log_file = None
    log_file_path = None
    proc: Optional[subprocess.Popen] = None
    timed_out = False
    breached = False
    barrier_reached = False
    sigkill_used = False
    err_msg: Optional[str] = None

    try:
        log_file = tempfile.NamedTemporaryFile(
            mode="w+",
            prefix=capture_log_prefix,
            suffix=".log",
            delete=False,
        )
        log_file_path = log_file.name

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        last_read_pos = 0
        while True:
            now = time.monotonic()
            if now >= deadline:
                timed_out = True
                err_msg = f"Subordinate process (PID {proc.pid}) exceeded monotonic deadline"
                break

            if check_breach is not None:
                try:
                    check_breach("subordinate_process_wait")
                except Exception as b_err:
                    breached = True
                    err_msg = str(b_err)
                    break

            if barrier_sentinel is not None:
                try:
                    with open(log_file_path, "r", encoding="utf-8", errors="replace") as lf:
                        lf.seek(last_read_pos)
                        new_content = lf.read()
                        last_read_pos = lf.tell()
                        if barrier_sentinel in new_content:
                            barrier_reached = True
                            if kill_on_barrier:
                                break
                except Exception:
                    pass

            if proc.poll() is not None:
                break

            time.sleep(poll_interval_s)

    except Exception as e:
        err_msg = str(e)
    finally:
        # Guaranteed cleanup of ONLY the owned child process (identity guard)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
                    sigkill_used = True
            except Exception as clean_err:
                if not err_msg:
                    err_msg = f"Failed to terminate subordinate child PID {proc.pid}: {clean_err}"

        # Capture bounded logs
        captured_logs = ""
        if log_file_path and os.path.exists(log_file_path):
            try:
                with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
                    captured_logs = f.read()[-max_log_bytes:]
            except Exception as log_e:
                captured_logs = f"Failed to read subordinate logs: {log_e}"

        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass
        if log_file_path and os.path.exists(log_file_path):
            try:
                os.unlink(log_file_path)
            except Exception:
                pass

    duration = time.monotonic() - t0
    exit_code = proc.returncode if proc is not None else None

    return SubordinateProcessResult(
        exit_code=exit_code,
        captured_logs=captured_logs,
        duration_s=round(duration, 3),
        timed_out=timed_out,
        breached=breached,
        pid=proc.pid if proc else None,
        error=err_msg,
        sigkill_used=sigkill_used,
        barrier_reached=barrier_reached,
    )



class SmokeHttpClient:
    """
    Isolated, wall-clock deadline-bounded HTTP client for smoke probes.
    Enforces:
    - trust_env=False (no system HTTP_PROXY / HTTPS_PROXY interference).
    - allow_redirects=False (no redirection on loopback endpoints).
    - Monotonic wall-clock deadline bounding across connect, send, and receive.
    - Maximum response body bytes cap to prevent memory exhaustion.
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


@dataclasses.dataclass
class StageSupervisorConfig:
    cmd: list[str]
    host: str = "127.0.0.1"
    port: int = 8797
    control_port: int = 8798
    timeout_s: float = 60.0
    min_headroom_mb: float = 2048.0
    max_rss_mb: Optional[float] = None
    max_swap_growth_mb: Optional[float] = None
    min_free_memory_pct: float = 12.0
    check_interval_s: float = 0.25
    startup_grace_period_s: float = 30.0
    stalled_inference_timeout_s: float = 10.0
    stage_name: str = "stage"
    model_identifier: Optional[str] = None
    instance_token: Optional[str] = None
    extra_env: Optional[dict[str, str]] = None
    output_file: Optional[str] = None
    dry_run: bool = False
    expected_cmd_pattern: str = r"(python|mlx|qmd|server)"


@dataclasses.dataclass
class SmokeStageContext:
    supervisor: SmokeStageSupervisor
    client: SmokeHttpClient
    child: subprocess.Popen
    watchdog: MLXWatchdog
    instance_token: str
    base_url: str
    ctrl_url: str
    start_mono: float
    deadline: float
    report: dict[str, Any]

    def check_breach(self, operation_label: str = "operation"):
        """Verifies that no watchdog breach or unexpected child exit occurred before proceeding."""
        self.supervisor.check_breach(operation_label)

    def record_check(self, key: str, value: bool):
        self.report.setdefault("checks", {})[key] = value

    def record_metric(self, key: str, value: Any):
        self.report.setdefault("metrics", {})[key] = value

    def record_fixture(self, key: str, value: Any):
        self.report.setdefault("fixtures", {})[key] = value

    def add_error(self, err: str):
        self.report.setdefault("errors", []).append(err)


class SmokeStageSupervisor:
    """
    Supervises a spawned MLX server child process under active MLXWatchdog monitoring.
    Enforces strict lifecycle guarantees, non-blocking telemetry, bounded execution,
    isolated child termination on breaches, and complete saved JSON output.
    """

    def __init__(
        self,
        config: StageSupervisorConfig,
        sampler: Optional[SystemMemorySampler] = None,
    ):
        self.config = config
        self.sampler = sampler or SystemMemorySampler()
        self._preflight_defaults: Optional[Any] = None
        self._preflight_info: dict[str, Any] = {}
        self.breach_event = threading.Event()
        self.breach_result: list[Optional[WatchdogCheckResult]] = [None]
        self.telemetry_samples: list[dict[str, Any]] = []
        self.watchdog: Optional[MLXWatchdog] = None
        self.child: Optional[subprocess.Popen] = None
        self.last_report: dict[str, Any] = {}

    def validate_preflight(self) -> dict[str, Any]:
        """
        Validates system telemetry, headroom, loopback policy, and port availability.
        Fails closed on any error before process spawning.
        """
        # 0. Numeric sanity checks (fail closed on NaN, inf, negative or wrong types)
        for field_name, val, min_val in [
            ("timeout_s", self.config.timeout_s, 0.001),
            ("min_headroom_mb", self.config.min_headroom_mb, 0.0),
            ("check_interval_s", self.config.check_interval_s, 0.001),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"Config field '{field_name}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Config field '{field_name}' must not be NaN or Inf, got {val}")
            if val < min_val:
                raise ValueError(f"Config field '{field_name}' must be >= {min_val}, got {val}")

        for port_name, port_val in [("port", self.config.port), ("control_port", self.config.control_port)]:
            if isinstance(port_val, bool) or not isinstance(port_val, int):
                raise ValueError(f"Config field '{port_name}' must be an integer, got {type(port_val).__name__}")
            if not (1 <= port_val <= 65535):
                raise ValueError(f"Invalid {port_name}: {port_val} (must be between 1 and 65535)")

        if self.config.max_rss_mb is not None:
            v = self.config.max_rss_mb
            if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) or v <= 0:
                raise ValueError(f"Config field 'max_rss_mb' must be positive number, got {v}")

        if self.config.max_swap_growth_mb is not None:
            v = self.config.max_swap_growth_mb
            if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) or v < 0:
                raise ValueError(f"Config field 'max_swap_growth_mb' must be non-negative number, got {v}")

        preflight_info: dict[str, Any] = {
            "timestamp": time.time(),
            "host": self.config.host,
            "port": self.config.port,
            "control_port": self.config.control_port,
            "stage": self.config.stage_name,
            "model": self.config.model_identifier,
        }

        # 1. Host restriction (strict numeric IPv4 loopback)
        if not is_numeric_loopback(self.config.host) or self.config.host != "127.0.0.1":
            raise ValueError(f"Host '{self.config.host}' must be numeric loopback '127.0.0.1'")

        # 2. Port distinctness and validity
        if self.config.port == self.config.control_port:
            raise ValueError(
                f"Inference port ({self.config.port}) and control port ({self.config.control_port}) must be distinct"
            )

        # 3. Prevent collision with live production daemon (8787)
        if self.config.port == 8787 or self.config.control_port == 8787:
            raise ValueError(
                "Refusing to use port 8787; port 8787 is reserved for live daemon. Choose distinct qualification ports."
            )

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
            defaults_dict = (
                dataclasses.asdict(defaults)
                if dataclasses.is_dataclass(defaults)
                else (defaults.__dict__ if hasattr(defaults, "__dict__") else {})
            )
            preflight_info.update({
                "installed_ram_mb": installed_ram,
                "headroom_mb": round(headroom, 1),
                "swap_used_mb": round(swap_used, 1),
                "free_memory_pct": round(free_pct, 1),
                "watchdog_defaults": defaults_dict,
            })
        except SystemMetricsError as e:
            raise SystemMetricsError(f"Preflight memory sampling failed: {e}")

        # 6. Conservative headroom enforcement
        if headroom < self.config.min_headroom_mb:
            raise RuntimeError(
                f"Insufficient memory headroom for {self.config.stage_name} stage: "
                f"measured {headroom:.1f} MB < required {self.config.min_headroom_mb:.1f} MB. "
                f"Refusing to execute."
            )

        self._preflight_info = preflight_info
        return preflight_info

    def check_breach(self, operation_label: str = "operation"):
        """Checks if a watchdog breach has fired or if child process died."""
        if self.breach_event.is_set():
            b_res = self.breach_result[0]
            reason = b_res.breach_reason if b_res else "unknown breach"
            raise RuntimeError(f"Watchdog breach during {operation_label}: {reason}")

        if self.watchdog and self.watchdog.last_check_result and not self.watchdog.last_check_result.healthy:
            raise RuntimeError(f"Watchdog breach during {operation_label}: {self.watchdog.last_check_result.breach_reason}")

        if self.child is not None and self.child.poll() is not None:
            for _ in range(5):
                if self.breach_event.is_set() or (self.watchdog and self.watchdog.last_check_result and not self.watchdog.last_check_result.healthy):
                    break
                time.sleep(0.05)
            if self.breach_event.is_set():
                b_res = self.breach_result[0]
                reason = b_res.breach_reason if b_res else "unknown breach"
                raise RuntimeError(f"Watchdog breach during {operation_label}: {reason}")
            if self.watchdog and self.watchdog.last_check_result and not self.watchdog.last_check_result.healthy:
                raise RuntimeError(f"Watchdog breach during {operation_label}: {self.watchdog.last_check_result.breach_reason}")
            raise RuntimeError(f"Server child exited unexpectedly with code {self.child.returncode} during {operation_label}")

    @contextlib.contextmanager
    def managed_stage(self) -> Generator[SmokeStageContext, None, None]:
        """
        Context manager that handles child spawning, watchdog supervision, readiness polling,
        and guaranteed cleanup on all exit paths.
        """
        run_start_time = time.time()
        start_mono = time.monotonic()
        deadline = start_mono + self.config.timeout_s

        instance_token = self.config.instance_token or f"stage-{uuid.uuid4().hex[:12]}"

        report: dict[str, Any] = {
            "instance_token": instance_token,
            "stage": self.config.stage_name,
            "model": self.config.model_identifier,
            "port": self.config.port,
            "control_port": self.config.control_port,
            "timeout_s": self.config.timeout_s,
            "status": "in_progress",
            "start_time": run_start_time,
            "checks": {},
            "metrics": {},
            "fixtures": {},
            "errors": [],
            "telemetry_samples": [],
            "captured_logs": "",
            "child_cleanup": {
                "cleaned": False,
                "pid": None,
                "exit_code": None,
            },
        }
        self.last_report = report

        # Step 1: Preflight (persisting failure report fail-closed if preflight fails)
        try:
            preflight = self.validate_preflight()
            report["preflight"] = preflight
            report["metrics"]["preflight_headroom_mb"] = preflight.get("headroom_mb")
        except Exception as preflight_err:
            report["status"] = "failed"
            report["duration_s"] = round(time.time() - run_start_time, 3)
            err_msg = f"Preflight validation failed: {preflight_err}"
            report.setdefault("errors", []).append(err_msg)
            self.last_report = report
            self._save_report(report)
            raise

        if self.config.dry_run:
            ctx = SmokeStageContext(
                supervisor=self,
                client=SmokeHttpClient(),
                child=None,  # type: ignore
                watchdog=None,  # type: ignore
                instance_token=instance_token,
                base_url=f"http://{self.config.host}:{self.config.port}",
                ctrl_url=f"http://{self.config.host}:{self.config.control_port}",
                start_mono=start_mono,
                deadline=deadline,
                report=report,
            )
            try:
                report["status"] = "dry_run_completed"
                yield ctx
            except Exception as e:
                report["status"] = "failed"
                err_str = str(e)
                if err_str not in report.get("errors", []):
                    report.setdefault("errors", []).append(err_str)
                raise
            finally:
                report["duration_s"] = round(time.time() - run_start_time, 3)
                self.last_report = report
                self._save_report(report)
            return

        # Step 2: Prepare Child Environment
        child_env = os.environ.copy()
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"
        child_env["HF_DATASETS_OFFLINE"] = "1"
        child_env["MLX_INSTANCE_TOKEN"] = instance_token
        child_env["MLX_EMBED_PORT"] = str(self.config.port)
        child_env["MLX_CONTROL_PORT"] = str(self.config.control_port)

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if "PYTHONPATH" in child_env:
            child_env["PYTHONPATH"] = f"{repo_root}:{child_env['PYTHONPATH']}"
        else:
            child_env["PYTHONPATH"] = repo_root

        if self.config.extra_env:
            child_env.update(self.config.extra_env)

        child: Optional[subprocess.Popen] = None
        log_file = None
        log_file_path = None
        client = SmokeHttpClient(default_timeout_s=5.0)
        t_supervisor: Optional[threading.Thread] = None
        stop_supervisor = threading.Event()
        self.breach_event.clear()
        self.breach_result = [None]
        self.telemetry_samples = []
        cleanup_errors: list[str] = []

        try:
            log_file = tempfile.NamedTemporaryFile(
                mode="w+",
                prefix=f"mlx_smoke_{self.config.stage_name}_",
                suffix=".log",
                delete=False,
            )
            log_file_path = log_file.name

            child = subprocess.Popen(
                self.config.cmd,
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            self.child = child
            report["child_pid"] = child.pid

            # Step 3: Instantiate MLXWatchdog & Background Supervisor Thread
            watchdog_config = MLXWatchdogConfig(
                pid=child.pid,
                host=self.config.host,
                port=self.config.port,
                control_port=self.config.control_port,
                max_rss_mb=self.config.max_rss_mb,
                max_swap_growth_mb=self.config.max_swap_growth_mb,
                min_free_memory_pct=self.config.min_free_memory_pct,
                check_interval_s=self.config.check_interval_s,
                startup_grace_period_s=min(self.config.startup_grace_period_s, self.config.timeout_s),
                stalled_inference_timeout_s=min(self.config.stalled_inference_timeout_s, self.config.timeout_s),
                expected_cmd_pattern=self.config.expected_cmd_pattern,
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
            self.watchdog = watchdog

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
                        self.breach_result[0] = res
                        self.breach_event.set()
                        break

                    check = watchdog.check_step()
                    if not check.healthy:
                        self.breach_result[0] = check
                        self.breach_event.set()
                        break

                    if check.metrics:
                        self.telemetry_samples.append({
                            "timestamp": round(check.timestamp, 2),
                            "elapsed_s": round(time.monotonic() - start_mono, 2),
                            "rss_mb": check.metrics.get("rss_mb"),
                            "swap_used_mb": check.metrics.get("swap_used_mb"),
                            "swap_growth_mb": check.metrics.get("swap_growth_mb"),
                            "memory_free_pct": check.metrics.get("memory_free_pct"),
                        })

                    stop_supervisor.wait(timeout=self.config.check_interval_s)

            t_supervisor = threading.Thread(
                target=_supervisor_loop,
                daemon=True,
                name=f"Smoke-Watchdog-{self.config.stage_name}",
            )
            t_supervisor.start()

            # Step 4: Poll Control Port for Readiness
            base_url = f"http://{self.config.host}:{self.config.port}"
            ctrl_url = f"http://{self.config.host}:{self.config.control_port}"

            ready = False
            startup_deadline = min(deadline, time.monotonic() + self.config.startup_grace_period_s)
            while time.monotonic() < startup_deadline:
                if self.breach_event.is_set():
                    b_res = self.breach_result[0]
                    raise RuntimeError(f"Watchdog breach during server startup: {b_res.breach_reason if b_res else 'unknown breach'}")

                if child.poll() is not None:
                    for _ in range(5):
                        if self.breach_event.is_set() or (watchdog.last_check_result and not watchdog.last_check_result.healthy):
                            break
                        time.sleep(0.05)
                    if self.breach_event.is_set():
                        b_res = self.breach_result[0]
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
                            report["metrics"]["startup_time_s"] = round(time.monotonic() - start_mono, 3)
                            report["checks"]["server_ready"] = True
                            break
                    elif status_code == 503:
                        pass
                except Exception:
                    pass
                time.sleep(0.1)

            if not ready:
                if self.breach_event.is_set():
                    b_res = self.breach_result[0]
                    raise RuntimeError(f"Watchdog breach during server startup: {b_res.breach_reason if b_res else 'unknown breach'}")
                if watchdog.last_check_result and not watchdog.last_check_result.healthy:
                    raise RuntimeError(f"Watchdog breach during server startup: {watchdog.last_check_result.breach_reason}")
                raise TimeoutError(f"Server child failed to report ready on {ctrl_url}/health within startup grace period")

            ctx = SmokeStageContext(
                supervisor=self,
                client=client,
                child=child,
                watchdog=watchdog,
                instance_token=instance_token,
                base_url=base_url,
                ctrl_url=ctrl_url,
                start_mono=start_mono,
                deadline=deadline,
                report=report,
            )
            yield ctx

            if report.get("errors"):
                report["status"] = "failed"
            elif report.get("status") == "in_progress":
                report["status"] = "passed"

        except Exception as e:
            report["status"] = "failed"
            err_str = str(e)
            if err_str not in report.get("errors", []):
                report.setdefault("errors", []).append(err_str)
            raise
        finally:
            # Step 5: Guaranteed Cleanup and Complete Report Generation
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

            report["telemetry_samples"] = self.telemetry_samples
            report["duration_s"] = round(time.time() - run_start_time, 3)
            self.last_report = report
            self._save_report(report)

    def _save_report(self, report: dict[str, Any]):
        if self.config.output_file:
            try:
                out_dir = os.path.dirname(os.path.abspath(self.config.output_file))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(self.config.output_file, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
            except Exception as e:
                report.setdefault("errors", []).append(f"Failed to write output JSON to {self.config.output_file}: {e}")
