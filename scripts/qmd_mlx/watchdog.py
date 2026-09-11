"""
watchdog.py — External Resource Watchdog for MLX Production Qualification

Monitors:
1. Target PID RSS (in MB) vs headroom-derived limits.
2. System memory pressure (via memory_pressure -Q / vm_stat).
3. Swap growth (via sysctl vm.swapusage) relative to baseline.
4. HTTP /health endpoint responsiveness (on dedicated control or inference port) and worker progress telemetry.
5. Stuck GPU inference detection (monotonic job age vs completed sequence).

Safety Invariants:
- Conservative defaults derived strictly from measured memory headroom (NO double-counting purgeable pages).
- Fail closed: Missing or unparseable telemetry triggers fail-safe breach rather than optimistic assumptions.
  No fabricated startup defaults in __init__ or telemetry methods.
- Strict process identity & lifecycle isolation:
  * Owned child processes by default. External attach requires explicit opt-in, non-empty start identity, and instance token.
  * Revalidates process identity and immutable start time before EVERY signal (SIGTERM and SIGKILL).
  * Never sends signals to recycled PIDs, mismatched handles, or unrelated services.
  * Never attempts to reap (os.waitpid) non-child processes.
  * Cleanly reaps owned child process handles on every exit path.
- Non-blocking execution: Probing uses strict wall-clock deadlines, numeric loopback (127.0.0.1),
  no system proxy interference, and bounded response body sizes.
- Startup grace policy: Legitimate cold model loading before bind does not trigger false watchdog kills.
- Explicit dry-run semantics: breaches are detected and reported without delivering kill signals.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from enum import Enum
from typing import Any, Callable, Optional


class BreachType(str, Enum):
    RSS_EXCEEDED = "rss_exceeded"
    SWAP_GROWTH_EXCEEDED = "swap_growth_exceeded"
    MEMORY_PRESSURE_CRITICAL = "memory_pressure_critical"
    HEALTH_CHECK_FAILED = "health_check_failed"
    STALLED_INFERENCE = "stalled_inference"
    IDENTITY_MISMATCH = "identity_mismatch"
    TELEMETRY_UNAVAILABLE = "telemetry_unavailable"
    TARGET_EXITED = "target_exited"


class SystemMetricsError(RuntimeError):
    """Raised when system telemetry or memory stats cannot be sampled."""
    pass


class TargetValidationError(ValueError):
    """Raised when target PID fails safety or identity checks."""
    pass


def is_numeric_loopback(host: str) -> bool:
    """Verifies that host is strictly a numeric IPv4 loopback IP address ('127.0.0.1' or within 127.0.0.0/8)."""
    if not isinstance(host, str) or not host.strip():
        return False
    h = host.strip()
    if h == "::1":
        return False
    parts = h.split(".")
    if len(parts) == 4 and parts[0] == "127":
        try:
            return all(0 <= int(p) <= 255 for p in parts)
        except ValueError:
            return False
    return False


def validate_config_parameters(
    host: str = "127.0.0.1",
    port: int = 8787,
    control_port: Optional[int] = None,
    max_rss_mb: Optional[float] = None,
    max_swap_growth_mb: Optional[float] = None,
    min_free_memory_pct: float = 12.0,
    health_timeout_s: float = 3.0,
    consecutive_health_failures: int = 3,
    check_interval_s: float = 1.0,
    grace_period_s: float = 3.0,
    startup_grace_period_s: float = 30.0,
    stalled_inference_timeout_s: float = 60.0,
    expected_cmd_pattern: Optional[str] = r"(python|mlx|qmd|server)",
) -> None:
    """
    Pure validation function for all PID-independent watchdog configuration parameters.
    Validates host policy, port ranges and distinctness, threshold positivity/finiteness,
    and regex compilation without any side effects or requiring a process PID.
    """
    if not is_numeric_loopback(host) or host == "::1":
        raise ValueError(
            f"Invalid host '{host}': host must be a numeric IPv4 loopback IP ('127.0.0.1'). "
            f"Hostnames ('localhost', etc.), IPv6 ('::1'), and non-loopback IPs are strictly rejected."
        )
    if not isinstance(port, int) or isinstance(port, bool) or not (1 <= port <= 65535):
        raise ValueError(f"Invalid port {port}: must be in range 1-65535")
    if control_port is not None:
        if not isinstance(control_port, int) or isinstance(control_port, bool) or not (1 <= control_port <= 65535):
            raise ValueError(f"Invalid control_port {control_port}: must be in range 1-65535")
        if port == control_port:
            raise ValueError(f"Invalid ports: port ({port}) and control_port ({control_port}) must be distinct")
    if isinstance(health_timeout_s, bool) or not (isinstance(health_timeout_s, (int, float)) and math.isfinite(health_timeout_s) and health_timeout_s > 0):
        raise ValueError(f"Invalid health_timeout_s {health_timeout_s}: must be finite and > 0")
    if isinstance(consecutive_health_failures, bool) or not (isinstance(consecutive_health_failures, int) and consecutive_health_failures >= 1):
        raise ValueError(f"Invalid consecutive_health_failures {consecutive_health_failures}: must be integer >= 1")
    if isinstance(check_interval_s, bool) or not (isinstance(check_interval_s, (int, float)) and math.isfinite(check_interval_s) and check_interval_s > 0):
        raise ValueError(f"Invalid check_interval_s {check_interval_s}: must be finite and > 0")
    if isinstance(grace_period_s, bool) or not (isinstance(grace_period_s, (int, float)) and math.isfinite(grace_period_s) and grace_period_s >= 0):
        raise ValueError(f"Invalid grace_period_s {grace_period_s}: must be finite and >= 0")
    if isinstance(startup_grace_period_s, bool) or not (isinstance(startup_grace_period_s, (int, float)) and math.isfinite(startup_grace_period_s) and startup_grace_period_s >= 0):
        raise ValueError(f"Invalid startup_grace_period_s {startup_grace_period_s}: must be finite and >= 0")
    if isinstance(stalled_inference_timeout_s, bool) or not (isinstance(stalled_inference_timeout_s, (int, float)) and math.isfinite(stalled_inference_timeout_s) and stalled_inference_timeout_s > 0):
        raise ValueError(f"Invalid stalled_inference_timeout_s {stalled_inference_timeout_s}: must be finite and > 0")
    if isinstance(min_free_memory_pct, bool) or not (isinstance(min_free_memory_pct, (int, float)) and math.isfinite(min_free_memory_pct) and 0.0 < min_free_memory_pct < 100.0):
        raise ValueError(f"Invalid min_free_memory_pct {min_free_memory_pct}: must be in (0, 100)")
    if max_rss_mb is not None:
        if isinstance(max_rss_mb, bool) or not (isinstance(max_rss_mb, (int, float)) and math.isfinite(max_rss_mb) and max_rss_mb > 0):
            raise ValueError(f"Invalid max_rss_mb {max_rss_mb}: must be finite and > 0")
    if max_swap_growth_mb is not None:
        if isinstance(max_swap_growth_mb, bool) or not (isinstance(max_swap_growth_mb, (int, float)) and math.isfinite(max_swap_growth_mb) and max_swap_growth_mb >= 0):
            raise ValueError(f"Invalid max_swap_growth_mb {max_swap_growth_mb}: must be finite and >= 0")
    if expected_cmd_pattern is not None:
        try:
            re.compile(expected_cmd_pattern)
        except re.error as e:
            raise ValueError(f"Invalid expected_cmd_pattern '{expected_cmd_pattern}': {e}")


@dataclasses.dataclass
class WatchdogDefaults:
    installed_ram_mb: float
    headroom_mb: float
    max_rss_mb: float
    max_swap_growth_mb: float
    min_free_memory_pct: float = 12.0
    health_timeout_s: float = 3.0
    consecutive_health_failures: int = 3
    check_interval_s: float = 1.0
    grace_period_s: float = 3.0
    startup_grace_period_s: float = 30.0
    stalled_inference_timeout_s: float = 60.0


@dataclasses.dataclass
class MLXWatchdogConfig:
    pid: int
    host: str = "127.0.0.1"
    port: int = 8787
    control_port: Optional[int] = None
    max_rss_mb: Optional[float] = None
    max_swap_growth_mb: Optional[float] = None
    min_free_memory_pct: float = 12.0
    health_timeout_s: float = 3.0
    consecutive_health_failures: int = 3
    check_interval_s: float = 1.0
    grace_period_s: float = 3.0
    startup_grace_period_s: float = 30.0
    stalled_inference_timeout_s: float = 60.0
    expected_cmd_pattern: Optional[str] = r"(python|mlx|qmd|server)"
    instance_token: Optional[str] = None
    allow_external_pid: bool = False
    dry_run: bool = False

    def __post_init__(self):
        # Validate target process PID
        if not isinstance(self.pid, int) or isinstance(self.pid, bool) or self.pid <= 1:
            raise ValueError(f"Invalid pid {self.pid}: must be an integer > 1")
        validate_config_parameters(
            host=self.host,
            port=self.port,
            control_port=self.control_port,
            max_rss_mb=self.max_rss_mb,
            max_swap_growth_mb=self.max_swap_growth_mb,
            min_free_memory_pct=self.min_free_memory_pct,
            health_timeout_s=self.health_timeout_s,
            consecutive_health_failures=self.consecutive_health_failures,
            check_interval_s=self.check_interval_s,
            grace_period_s=self.grace_period_s,
            startup_grace_period_s=self.startup_grace_period_s,
            stalled_inference_timeout_s=self.stalled_inference_timeout_s,
            expected_cmd_pattern=self.expected_cmd_pattern,
        )


@dataclasses.dataclass
class SystemSnapshot:
    installed_ram_mb: float
    headroom_mb: float
    memory_free_pct: float
    swap_used_mb: float
    target_rss_mb: Optional[float] = None
    target_cmdline: Optional[str] = None
    target_start_time: Optional[str] = None


@dataclasses.dataclass
class WatchdogCheckResult:
    timestamp: float
    healthy: bool
    breach_type: Optional[BreachType] = None
    breach_reason: Optional[str] = None
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    terminated_pid: Optional[int] = None
    sigkill_used: bool = False


class SystemMemorySampler:
    """
    Samples macOS system memory statistics, pressure levels, swap usage, and target PID RSS.
    Fails closed when telemetry commands fail or return unparseable outputs.
    """

    def __init__(
        self,
        cmd_runner: Optional[Callable[[list[str], float], tuple[int, str, str]]] = None,
    ):
        self._cmd_runner = cmd_runner or self._default_cmd_runner

    @staticmethod
    def _default_cmd_runner(cmd: list[str], timeout_s: float = 2.0) -> tuple[int, str, str]:
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def get_installed_ram_mb(self) -> float:
        """Returns total installed physical RAM in MB via sysctl hw.memsize. Fails closed on error."""
        rc, out, err = self._cmd_runner(["sysctl", "-n", "hw.memsize"], 2.0)
        if rc == 0 and out.strip().isdigit():
            val = float(int(out.strip()) / (1024 * 1024))
            if val > 0:
                return val
        raise SystemMetricsError(f"Failed to query physical RAM via sysctl hw.memsize: rc={rc}, err={err}")

    def get_memory_headroom_mb(self) -> float:
        """
        Calculates currently available memory headroom in MB on macOS from vm_stat.
        Headroom = (Pages free + Pages inactive + Pages speculative) * page_size.
        DOES NOT double-count Pages purgeable (purgeable pages are already counted within inactive/file cache).
        Fails closed on command failure.
        """
        rc, out, err = self._cmd_runner(["vm_stat"], 2.0)
        if rc != 0 or not out:
            raise SystemMetricsError(f"Failed to sample vm_stat: rc={rc}, err={err}")

        page_size = 16384  # Default Apple Silicon page size (16KB)
        m_page = re.search(r"page size of (\d+) bytes", out, re.IGNORECASE)
        if m_page:
            page_size = int(m_page.group(1))

        def _extract_pages(key: str) -> int:
            m = re.search(rf"{key}:\s+(\d+)\.", out)
            return int(m.group(1)) if m else 0

        pages_free = _extract_pages("Pages free")
        pages_inactive = _extract_pages("Pages inactive")
        pages_speculative = _extract_pages("Pages speculative")

        total_available_pages = pages_free + pages_inactive + pages_speculative
        available_bytes = total_available_pages * page_size
        return float(available_bytes / (1024 * 1024))

    def get_memory_free_pct(self) -> float:
        """
        Queries system memory free percentage from `memory_pressure -Q` or falls back to headroom ratio.
        Returns a float percentage in [0.0, 100.0]. Fails closed on error.
        """
        rc, out, _ = self._cmd_runner(["memory_pressure", "-Q"], 2.0)
        if rc == 0 and out:
            m = re.search(r"System-wide memory free percentage:\s*(\d+)%", out)
            if m:
                return float(m.group(1))

        # Fallback: calculate directly from measured headroom vs physical RAM
        headroom_mb = self.get_memory_headroom_mb()
        installed_mb = self.get_installed_ram_mb()
        if installed_mb > 0:
            return round(min(100.0, max(0.0, (headroom_mb / installed_mb) * 100.0)), 1)
        raise SystemMetricsError("Unable to determine system memory free percentage")

    def get_swap_used_mb(self) -> float:
        """
        Queries current system swap usage in MB via sysctl vm.swapusage.
        Fails closed on error.
        """
        rc, out, err = self._cmd_runner(["sysctl", "vm.swapusage"], 2.0)
        if rc == 0 and out:
            m = re.search(r"used\s*=\s*([0-9.]+)([KMG]?)", out)
            if m:
                val = float(m.group(1))
                unit = m.group(2).upper()
                if unit == "G":
                    return val * 1024.0
                elif unit == "K":
                    return val / 1024.0
                elif unit == "B" or unit == "":
                    return val / (1024.0 * 1024.0) if val > 1024 else val
                return val  # 'M' is default

        raise SystemMetricsError(f"Failed to query swap usage via sysctl vm.swapusage: rc={rc}, err={err}")

    def get_process_rss_mb(self, pid: int) -> Optional[float]:
        """
        Queries RSS of target PID in MB via `ps -o rss= -p <pid>`.
        Returns None ONLY if the process has genuinely exited (verified by liveness check).
        Raises SystemMetricsError on command timeout, OS errors, or unparseable output.
        """
        if pid <= 1:
            return None
        rc, out, err = self._cmd_runner(["ps", "-o", "rss=", "-p", str(pid)], 2.0)
        if rc == 0 and out.strip().isdigit():
            rss_kb = int(out.strip())
            return float(rss_kb / 1024.0)

        # If ps command returned non-zero, check whether process is genuinely dead
        if not TargetProcessValidator.is_pid_alive(pid):
            return None

        raise SystemMetricsError(f"Failed to query RSS for active PID {pid}: rc={rc}, err={err}, out='{out.strip()}'")

    def get_process_cmdline(self, pid: int) -> Optional[str]:
        """Queries full command-line of target PID via `ps -o command= -p <pid>`."""
        if pid <= 1:
            return None
        rc, out, _ = self._cmd_runner(["ps", "-o", "command=", "-p", str(pid)], 2.0)
        if rc == 0 and out.strip():
            return out.strip()
        return None

    def get_process_start_time(self, pid: int) -> Optional[str]:
        """Queries start time string of target PID via `ps -o lstart= -p <pid>` for immutable identity verification."""
        if pid <= 1:
            return None
        rc, out, _ = self._cmd_runner(["ps", "-o", "lstart=", "-p", str(pid)], 2.0)
        if rc == 0 and out.strip():
            return out.strip()
        return None

    def compute_conservative_defaults(self, headroom_mb: Optional[float] = None) -> WatchdogDefaults:
        """
        Calculates conservative limits derived strictly from measured memory headroom (NOT installed RAM).
        Does not clamp up to an arbitrary 512MB floor if actual headroom is smaller.
        Fails closed if headroom or physical RAM cannot be measured.
        """
        avail_headroom_mb = headroom_mb if headroom_mb is not None else self.get_memory_headroom_mb()
        installed_mb = self.get_installed_ram_mb()

        # Conservative RSS ceiling: max 65% of measured available headroom, strictly bounded below headroom * 0.70
        max_rss_mb = min(8192.0, max(32.0, avail_headroom_mb * 0.65))
        max_rss_mb = min(max_rss_mb, avail_headroom_mb * 0.70)

        # Conservative Swap growth budget: max 25% of headroom or 2048MB
        max_swap_growth_mb = min(2048.0, max(32.0, avail_headroom_mb * 0.25))

        return WatchdogDefaults(
            installed_ram_mb=installed_mb,
            headroom_mb=avail_headroom_mb,
            max_rss_mb=round(max_rss_mb, 1),
            max_swap_growth_mb=round(max_swap_growth_mb, 1),
            min_free_memory_pct=12.0,
            health_timeout_s=3.0,
            consecutive_health_failures=3,
            check_interval_s=1.0,
            grace_period_s=3.0,
            startup_grace_period_s=30.0,
            stalled_inference_timeout_s=60.0,
        )


class TargetProcessValidator:
    """
    Enforces strict process identity invariants to guarantee signals are delivered ONLY
    to the exact verified test MLX process and NEVER to recycled PIDs or unrelated services.
    """

    @staticmethod
    def is_pid_alive(pid: int, owned_child: Optional[subprocess.Popen] = None) -> bool:
        """
        Determines whether PID is actively executing.
        Never calls os.waitpid on non-child processes.
        """
        if pid <= 1:
            return False

        if owned_child is not None:
            if owned_child.pid == pid:
                return owned_child.poll() is None
            return False

        try:
            os.kill(pid, 0)
            res = subprocess.run(
                ["ps", "-p", str(pid), "-o", "state="],
                capture_output=True,
                text=True,
                check=False,
                timeout=1.0,
            )
            if res.returncode != 0 or not res.stdout.strip() or "Z" in res.stdout.strip():
                return False
            return True
        except OSError:
            return False

    @staticmethod
    def validate_target(
        pid: int,
        sampler: SystemMemorySampler,
        expected_start_time: Optional[str] = None,
        expected_cmd_pattern: Optional[str] = None,
        owned_child: Optional[subprocess.Popen] = None,
        allow_external_pid: bool = False,
        instance_token: Optional[str] = None,
    ) -> str:
        """
        Validates target PID safety, liveness, start time identity, and command pattern.
        Enforces owned_child targeting by default; requires explicit opt-in and instance token
        for external PIDs.
        Returns verified command-line string or raises TargetValidationError.
        """
        if pid <= 1:
            raise TargetValidationError(f"Invalid target PID {pid}: cannot target root or init processes")

        current_pid = os.getpid()
        if pid == current_pid:
            raise TargetValidationError(f"Invalid target PID {pid}: cannot target the watchdog's own process")

        # Enforce owned child policy
        if owned_child is None:
            if not allow_external_pid:
                raise TargetValidationError(
                    f"Attaching to external PID {pid} without owned_child is disabled by default. "
                    "Pass allow_external_pid=True (or --allow-external-pid) with required instance_token and start identity to opt in."
                )
            if not instance_token:
                raise TargetValidationError(
                    f"Attaching to external PID {pid} requires a non-empty instance_token for verification."
                )
        else:
            if owned_child.pid != pid:
                raise TargetValidationError(
                    f"owned_child PID {owned_child.pid} does not match target PID {pid}"
                )

        # Check process liveness without reaping non-children
        if not TargetProcessValidator.is_pid_alive(pid, owned_child=owned_child):
            raise TargetValidationError(f"Target PID {pid} does not exist or is not active")

        # Check start time identity to prevent PID recycling attacks
        current_start_time = sampler.get_process_start_time(pid)
        if not current_start_time:
            raise TargetValidationError(f"Could not determine start time for target PID {pid}")

        if expected_start_time is not None:
            if current_start_time != expected_start_time:
                raise TargetValidationError(
                    f"Target PID {pid} start time mismatch: expected '{expected_start_time}', "
                    f"got '{current_start_time}'. Process PID may have been recycled! Refusing signals."
                )

        # Verify command line signature
        cmdline = sampler.get_process_cmdline(pid)
        if not cmdline:
            raise TargetValidationError(f"Could not inspect command line for target PID {pid}")

        if expected_cmd_pattern:
            if not re.search(expected_cmd_pattern, cmdline, re.IGNORECASE):
                raise TargetValidationError(
                    f"Target PID {pid} command '{cmdline}' does not match expected pattern '{expected_cmd_pattern}'. "
                    f"Refusing to signal unrelated process."
                )

        return cmdline


class MLXWatchdog:
    """
    External operational safeguard watchdog for MLX inference server qualification.
    """

    def __init__(
        self,
        config: MLXWatchdogConfig,
        sampler: Optional[SystemMemorySampler] = None,
        health_probe_fn: Optional[Callable[[str, int, float], tuple[bool, Optional[dict | str], Optional[int]]]] = None,
        owned_child: Optional[subprocess.Popen] = None,
        baseline_swap_mb: Optional[float] = None,
        defaults: Optional[WatchdogDefaults] = None,
    ):
        self.config = config
        self.sampler = sampler or SystemMemorySampler()
        self._health_probe_fn = health_probe_fn or self._default_health_probe
        self.owned_child = owned_child

        # Validate target PID and capture immutable start time identity
        self.expected_start_time = self.sampler.get_process_start_time(config.pid)
        if not self.expected_start_time:
            raise TargetValidationError(f"Could not determine start time for target PID {config.pid}")

        self.cmdline = TargetProcessValidator.validate_target(
            pid=config.pid,
            sampler=self.sampler,
            expected_start_time=self.expected_start_time,
            expected_cmd_pattern=config.expected_cmd_pattern,
            owned_child=self.owned_child,
            allow_external_pid=config.allow_external_pid,
            instance_token=config.instance_token,
        )

        # Baseline metrics — reuse prelaunch snapshot if provided, else sample (fail closed on error)
        self.baseline_swap_mb = (
            baseline_swap_mb if baseline_swap_mb is not None else self.sampler.get_swap_used_mb()
        )
        self.defaults = defaults if defaults is not None else self.sampler.compute_conservative_defaults()

        self.max_rss_mb = config.max_rss_mb if config.max_rss_mb is not None else self.defaults.max_rss_mb
        self.max_swap_growth_mb = (
            config.max_swap_growth_mb
            if config.max_swap_growth_mb is not None
            else self.defaults.max_swap_growth_mb
        )
        self.min_free_memory_pct = config.min_free_memory_pct
        self.health_timeout_s = config.health_timeout_s
        self.consecutive_health_failures = config.consecutive_health_failures
        self.grace_period_s = config.grace_period_s
        self.startup_grace_period_s = config.startup_grace_period_s
        self.stalled_inference_timeout_s = config.stalled_inference_timeout_s

        self.start_time_mono: float = time.monotonic()
        self.consecutive_failures_count: int = 0
        self.last_completed_sequence: int = 0
        self.last_check_result: Optional[WatchdogCheckResult] = None

    @staticmethod
    def _default_health_probe(
        host: str, port: int, timeout_s: float, max_response_bytes: int = 65536
    ) -> tuple[bool, Optional[dict | str], Optional[int]]:
        """
        Performs a strictly wall-clock deadline-bounded HTTP GET /health probe over numeric loopback.
        - Enforces hard wall-clock deadline across connect, send, and receive stages.
        - Enforces maximum response body size (max_response_bytes) to prevent unbounded memory consumption.
        - Validates HTTP response code and parseable JSON payload.
        - Returns (is_healthy, payload_or_err_msg, http_status_code).
        """
        target_host = host
        t_deadline = time.monotonic() + timeout_s

        s = None
        try:
            rem = t_deadline - time.monotonic()
            if rem <= 0:
                return (False, "Health probe deadline expired before connect", None)

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(max(0.001, rem))
            s.connect((target_host, port))

            rem = t_deadline - time.monotonic()
            if rem <= 0:
                return (False, "Health probe deadline expired after connect", None)

            req = (
                f"GET /health HTTP/1.1\r\n"
                f"Host: {target_host}:{port}\r\n"
                f"User-Agent: QMD-MLX-Watchdog/2.0\r\n"
                f"Accept: application/json\r\n"
                f"Connection: close\r\n\r\n"
            ).encode("utf-8")

            s.settimeout(max(0.001, rem))
            s.sendall(req)

            resp_bytes = bytearray()
            while True:
                rem = t_deadline - time.monotonic()
                if rem <= 0:
                    return (False, "Health probe wall-clock deadline exceeded during response read", None)

                s.settimeout(max(0.001, rem))
                chunk = s.recv(4096)
                if not chunk:
                    break
                resp_bytes.extend(chunk)
                if len(resp_bytes) > max_response_bytes:
                    return (False, f"Health probe response exceeded {max_response_bytes} bytes", None)

            raw_data = bytes(resp_bytes)
            header_sep = raw_data.find(b"\r\n\r\n")
            if header_sep == -1:
                return (False, "Malformed HTTP response (no header terminator)", None)

            header_part = raw_data[:header_sep].decode("utf-8", errors="replace")
            body_part = raw_data[header_sep + 4:]

            status_line = header_part.split("\r\n")[0]
            parts = status_line.split()
            if len(parts) < 2 or not parts[1].isdigit():
                return (False, f"Malformed HTTP status line: '{status_line}'", None)

            status_code = int(parts[1])

            try:
                payload = json.loads(body_part.decode("utf-8"))
            except Exception as e:
                payload = {"raw": body_part.decode("utf-8", errors="replace"), "error": str(e)}

            return (status_code == 200, payload, status_code)

        except (socket.timeout, TimeoutError):
            return (False, "Health probe wall-clock deadline exceeded", None)
        except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError) as e:
            return (False, f"Connection error: {e}", None)
        except OSError as e:
            return (False, f"Socket error: {e}", None)
        except Exception as e:
            return (False, f"Unexpected probe error: {e}", None)
        finally:
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass

    def take_snapshot(self) -> SystemSnapshot:
        """Captures a point-in-time snapshot of system metrics and target process RSS."""
        return SystemSnapshot(
            installed_ram_mb=self.sampler.get_installed_ram_mb(),
            headroom_mb=self.sampler.get_memory_headroom_mb(),
            memory_free_pct=self.sampler.get_memory_free_pct(),
            swap_used_mb=self.sampler.get_swap_used_mb(),
            target_rss_mb=self.sampler.get_process_rss_mb(self.config.pid),
            target_cmdline=self.cmdline,
            target_start_time=self.expected_start_time,
        )

    def terminate_target_process(self, reason: str) -> tuple[bool, bool]:
        """
        Safely and exclusively terminates the target test process.
        Revalidates process identity and immutable start time before EVERY signal (SIGTERM and SIGKILL).
        Returns (terminated_successfully, sigkill_used).
        """
        pid = self.config.pid

        # Re-validate target PID before sending SIGTERM
        try:
            TargetProcessValidator.validate_target(
                pid=pid,
                sampler=self.sampler,
                expected_start_time=self.expected_start_time,
                expected_cmd_pattern=self.config.expected_cmd_pattern,
                owned_child=self.owned_child,
                allow_external_pid=self.config.allow_external_pid,
                instance_token=self.config.instance_token,
            )
        except TargetValidationError as e:
            print(f"[watchdog] Aborting termination before SIGTERM: validation failed: {e}", file=sys.stderr)
            return False, False

        print(f"[watchdog] BREACH TRIGGERED: {reason}. Terminating target PID {pid}...", file=sys.stderr)

        sigkill_used = False
        if self.owned_child is not None and self.owned_child.pid == pid:
            try:
                self.owned_child.terminate()
            except OSError:
                return True, False
        else:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                return True, False

        # Wait during grace period for graceful process exit
        t_deadline = time.monotonic() + self.grace_period_s
        while time.monotonic() < t_deadline:
            if not TargetProcessValidator.is_pid_alive(pid, owned_child=self.owned_child):
                print(f"[watchdog] Target PID {pid} exited gracefully under SIGTERM.", file=sys.stderr)
                if self.owned_child is not None and self.owned_child.pid == pid:
                    try:
                        self.owned_child.wait(timeout=1.0)
                    except Exception:
                        pass
                return True, False
            time.sleep(0.05)

        # RE-VALIDATE TARGET PID AND IMMUTABLE START TIME BEFORE SIGKILL!
        # Prevents race conditions where target process exited and another process acquired the same PID.
        try:
            TargetProcessValidator.validate_target(
                pid=pid,
                sampler=self.sampler,
                expected_start_time=self.expected_start_time,
                expected_cmd_pattern=self.config.expected_cmd_pattern,
                owned_child=self.owned_child,
                allow_external_pid=self.config.allow_external_pid,
                instance_token=self.config.instance_token,
            )
        except TargetValidationError as e:
            print(f"[watchdog] Target process exited or PID was recycled before SIGKILL. Refusing SIGKILL: {e}", file=sys.stderr)
            return True, False

        print(f"[watchdog] Target PID {pid} did not exit within {self.grace_period_s}s grace period; sending SIGKILL.", file=sys.stderr)

        if self.owned_child is not None and self.owned_child.pid == pid:
            try:
                self.owned_child.kill()
                sigkill_used = True
            except OSError:
                pass
        else:
            try:
                os.kill(pid, signal.SIGKILL)
                sigkill_used = True
            except OSError:
                pass

        t_kill_deadline = time.monotonic() + 2.0
        while time.monotonic() < t_kill_deadline:
            if not TargetProcessValidator.is_pid_alive(pid, owned_child=self.owned_child):
                print(f"[watchdog] Target PID {pid} terminated via SIGKILL.", file=sys.stderr)
                if self.owned_child is not None and self.owned_child.pid == pid:
                    try:
                        self.owned_child.wait(timeout=1.0)
                    except Exception:
                        pass
                return True, sigkill_used
            time.sleep(0.05)

        return False, sigkill_used

    def check_step(self) -> WatchdogCheckResult:
        """
        Executes a single non-blocking check cycle.
        Returns WatchdogCheckResult.
        """
        now = time.time()
        elapsed_since_start = time.monotonic() - self.start_time_mono

        # 1. Target process liveness and start time identity check
        if not TargetProcessValidator.is_pid_alive(self.config.pid, owned_child=self.owned_child):
            res = WatchdogCheckResult(
                timestamp=now,
                healthy=True,
                breach_type=BreachType.TARGET_EXITED,
                breach_reason=f"Target PID {self.config.pid} has already exited.",
                metrics={"target_pid": self.config.pid, "target_alive": False},
            )
            self.last_check_result = res
            return res

        curr_start_time = self.sampler.get_process_start_time(self.config.pid)
        if self.expected_start_time is not None and curr_start_time != self.expected_start_time:
            reason = f"PID {self.config.pid} identity changed: start time was '{self.expected_start_time}', now '{curr_start_time}'"
            res = WatchdogCheckResult(
                timestamp=now,
                healthy=False,
                breach_type=BreachType.IDENTITY_MISMATCH,
                breach_reason=reason,
                metrics={"target_pid": self.config.pid},
            )
            self.last_check_result = res
            return res

        # 2. Sample system metrics (fail-closed if telemetry is broken)
        try:
            target_rss = self.sampler.get_process_rss_mb(self.config.pid)
            if target_rss is None:
                res = WatchdogCheckResult(
                    timestamp=now,
                    healthy=True,
                    breach_type=BreachType.TARGET_EXITED,
                    breach_reason=f"Target PID {self.config.pid} has exited.",
                    metrics={"target_pid": self.config.pid, "target_alive": False},
                )
                self.last_check_result = res
                return res

            current_swap = self.sampler.get_swap_used_mb()
            free_pct = self.sampler.get_memory_free_pct()
        except SystemMetricsError as e:
            reason = f"Telemetry unavailable / failed closed: {e}"
            res = WatchdogCheckResult(
                timestamp=now,
                healthy=False,
                breach_type=BreachType.TELEMETRY_UNAVAILABLE,
                breach_reason=reason,
                metrics={"target_pid": self.config.pid, "telemetry_error": str(e)},
            )
            self.last_check_result = res
            return res

        # 3. Check RSS limit
        if target_rss > self.max_rss_mb:
            reason = (
                f"Target PID {self.config.pid} RSS ({target_rss:.1f} MB) exceeded "
                f"conservative limit ({self.max_rss_mb:.1f} MB)"
            )
            term_success, sigkill_used = (False, False)
            if not self.config.dry_run:
                term_success, sigkill_used = self.terminate_target_process(reason)

            res = WatchdogCheckResult(
                timestamp=now,
                healthy=False,
                breach_type=BreachType.RSS_EXCEEDED,
                breach_reason=reason,
                metrics={"target_pid": self.config.pid, "rss_mb": target_rss, "max_rss_mb": self.max_rss_mb},
                terminated_pid=self.config.pid if term_success else None,
                sigkill_used=sigkill_used,
            )
            self.last_check_result = res
            return res

        # 4. Check Swap Growth limit
        swap_growth = max(0.0, current_swap - self.baseline_swap_mb)
        if swap_growth > self.max_swap_growth_mb:
            reason = (
                f"System swap growth ({swap_growth:.1f} MB from baseline {self.baseline_swap_mb:.1f} MB) "
                f"exceeded maximum allowed growth ({self.max_swap_growth_mb:.1f} MB)"
            )
            term_success, sigkill_used = (False, False)
            if not self.config.dry_run:
                term_success, sigkill_used = self.terminate_target_process(reason)

            res = WatchdogCheckResult(
                timestamp=now,
                healthy=False,
                breach_type=BreachType.SWAP_GROWTH_EXCEEDED,
                breach_reason=reason,
                metrics={
                    "target_pid": self.config.pid,
                    "swap_used_mb": current_swap,
                    "baseline_swap_mb": self.baseline_swap_mb,
                    "swap_growth_mb": swap_growth,
                    "max_swap_growth_mb": self.max_swap_growth_mb,
                },
                terminated_pid=self.config.pid if term_success else None,
                sigkill_used=sigkill_used,
            )
            self.last_check_result = res
            return res

        # 5. Check System Memory Pressure
        if free_pct < self.min_free_memory_pct:
            reason = (
                f"System memory free percentage ({free_pct:.1f}%) dropped below critical threshold "
                f"({self.min_free_memory_pct:.1f}%)"
            )
            term_success, sigkill_used = (False, False)
            if not self.config.dry_run:
                term_success, sigkill_used = self.terminate_target_process(reason)

            res = WatchdogCheckResult(
                timestamp=now,
                healthy=False,
                breach_type=BreachType.MEMORY_PRESSURE_CRITICAL,
                breach_reason=reason,
                metrics={
                    "target_pid": self.config.pid,
                    "memory_free_pct": free_pct,
                    "min_free_memory_pct": self.min_free_memory_pct,
                },
                terminated_pid=self.config.pid if term_success else None,
                sigkill_used=sigkill_used,
            )
            self.last_check_result = res
            return res

        # 6. Check HTTP /health and worker stuck inference
        probe_port = self.config.control_port if self.config.control_port is not None else self.config.port
        is_healthy, probe_payload, status_code = self._health_probe_fn(
            self.config.host, probe_port, self.health_timeout_s
        )

        # Handle startup grace period
        in_startup_grace = elapsed_since_start < self.startup_grace_period_s

        if not is_healthy or not isinstance(probe_payload, dict):
            # If server is still starting and within startup grace period, do not trigger breach
            if in_startup_grace and (
                status_code is None
                or (isinstance(probe_payload, dict) and probe_payload.get("state") in ("starting", "loading"))
            ):
                res = WatchdogCheckResult(
                    timestamp=now,
                    healthy=True,
                    metrics={
                        "target_pid": self.config.pid,
                        "startup_grace": True,
                        "elapsed_s": round(elapsed_since_start, 1),
                    },
                )
                self.last_check_result = res
                return res

            self.consecutive_failures_count += 1
            if self.consecutive_failures_count >= self.consecutive_health_failures:
                reason = (
                    f"HTTP /health endpoint failed {self.consecutive_failures_count} consecutive times "
                    f"(last error: {probe_payload})"
                )
                term_success, sigkill_used = (False, False)
                if not self.config.dry_run:
                    term_success, sigkill_used = self.terminate_target_process(reason)

                res = WatchdogCheckResult(
                    timestamp=now,
                    healthy=False,
                    breach_type=BreachType.HEALTH_CHECK_FAILED,
                    breach_reason=reason,
                    metrics={
                        "target_pid": self.config.pid,
                        "consecutive_failures": self.consecutive_failures_count,
                        "probe_error": str(probe_payload),
                    },
                    terminated_pid=self.config.pid if term_success else None,
                    sigkill_used=sigkill_used,
                )
                self.last_check_result = res
                return res
        else:
            # Successful probe resets consecutive failure counter
            self.consecutive_failures_count = 0

            # 1. Verify PID binding
            resp_pid = probe_payload.get("pid")
            if resp_pid is None or resp_pid != self.config.pid:
                reason = f"HTTP server PID mismatch: endpoint reported PID {resp_pid}, expected {self.config.pid}"
                term_success, sigkill_used = (False, False)
                if not self.config.dry_run:
                    term_success, sigkill_used = self.terminate_target_process(reason)
                res = WatchdogCheckResult(
                    timestamp=now,
                    healthy=False,
                    breach_type=BreachType.IDENTITY_MISMATCH,
                    breach_reason=reason,
                    metrics={"expected_pid": self.config.pid, "reported_pid": resp_pid},
                    terminated_pid=self.config.pid if term_success else None,
                    sigkill_used=sigkill_used,
                )
                self.last_check_result = res
                return res

            # 2. Verify instance token binding
            resp_token = probe_payload.get("instance_token")
            if self.config.instance_token is not None:
                if not resp_token or resp_token != self.config.instance_token:
                    reason = f"HTTP server instance token mismatch: reported '{resp_token}', expected '{self.config.instance_token}'"
                    term_success, sigkill_used = (False, False)
                    if not self.config.dry_run:
                        term_success, sigkill_used = self.terminate_target_process(reason)
                    res = WatchdogCheckResult(
                        timestamp=now,
                        healthy=False,
                        breach_type=BreachType.IDENTITY_MISMATCH,
                        breach_reason=reason,
                        metrics={"expected_token": self.config.instance_token, "reported_token": resp_token},
                        terminated_pid=self.config.pid if term_success else None,
                        sigkill_used=sigkill_used,
                    )
                    self.last_check_result = res
                    return res
            else:
                if not resp_token or not isinstance(resp_token, str):
                    reason = f"HTTP server missing required instance token in /health payload: {resp_token}"
                    term_success, sigkill_used = (False, False)
                    if not self.config.dry_run:
                        term_success, sigkill_used = self.terminate_target_process(reason)
                    res = WatchdogCheckResult(
                        timestamp=now,
                        healthy=False,
                        breach_type=BreachType.IDENTITY_MISMATCH,
                        breach_reason=reason,
                        metrics={"reported_token": resp_token},
                        terminated_pid=self.config.pid if term_success else None,
                        sigkill_used=sigkill_used,
                    )
                    self.last_check_result = res
                    return res

            # 3. Validate typed finite progress fields
            is_idle = probe_payload.get("worker_idle")
            worker_alive = probe_payload.get("worker_alive")
            completed_sequence = probe_payload.get("completed_sequence")
            active_job_age_s = probe_payload.get("active_job_age_s")

            if (
                not isinstance(is_idle, bool)
                or not isinstance(worker_alive, bool)
                or not isinstance(completed_sequence, int)
                or isinstance(completed_sequence, bool)
                or completed_sequence < 0
            ):
                reason = f"Malformed typed progress fields in /health response: is_idle={is_idle}, worker_alive={worker_alive}, completed_sequence={completed_sequence}"
                term_success, sigkill_used = (False, False)
                if not self.config.dry_run:
                    term_success, sigkill_used = self.terminate_target_process(reason)
                res = WatchdogCheckResult(
                    timestamp=now,
                    healthy=False,
                    breach_type=BreachType.HEALTH_CHECK_FAILED,
                    breach_reason=reason,
                    metrics={"target_pid": self.config.pid, "probe_payload": probe_payload},
                    terminated_pid=self.config.pid if term_success else None,
                    sigkill_used=sigkill_used,
                )
                self.last_check_result = res
                return res

            if active_job_age_s is not None:
                if not (
                    isinstance(active_job_age_s, (int, float))
                    and not isinstance(active_job_age_s, bool)
                    and math.isfinite(active_job_age_s)
                    and active_job_age_s >= 0
                ):
                    reason = f"Invalid active_job_age_s in worker progress: {active_job_age_s}"
                    term_success, sigkill_used = (False, False)
                    if not self.config.dry_run:
                        term_success, sigkill_used = self.terminate_target_process(reason)
                    res = WatchdogCheckResult(
                        timestamp=now,
                        healthy=False,
                        breach_type=BreachType.HEALTH_CHECK_FAILED,
                        breach_reason=reason,
                        metrics={"target_pid": self.config.pid, "probe_payload": probe_payload},
                        terminated_pid=self.config.pid if term_success else None,
                        sigkill_used=sigkill_used,
                    )
                    self.last_check_result = res
                    return res

            # 4. Stuck GPU inference detection
            if (not is_idle) and (active_job_age_s is not None):
                if active_job_age_s > self.stalled_inference_timeout_s:
                    reason = (
                        f"Stalled inference detected: GPU worker active job has run for {active_job_age_s:.1f}s "
                        f"(exceeded timeout limit of {self.stalled_inference_timeout_s:.1f}s) while HTTP /health was responsive"
                    )
                    term_success, sigkill_used = (False, False)
                    if not self.config.dry_run:
                        term_success, sigkill_used = self.terminate_target_process(reason)

                    res = WatchdogCheckResult(
                        timestamp=now,
                        healthy=False,
                        breach_type=BreachType.STALLED_INFERENCE,
                        breach_reason=reason,
                        metrics={
                            "target_pid": self.config.pid,
                            "active_job_age_s": active_job_age_s,
                            "stalled_timeout_s": self.stalled_inference_timeout_s,
                            "completed_sequence": completed_sequence,
                        },
                        terminated_pid=self.config.pid if term_success else None,
                        sigkill_used=sigkill_used,
                    )
                    self.last_check_result = res
                    return res

            self.last_completed_sequence = completed_sequence

        # All checks passed
        res = WatchdogCheckResult(
            timestamp=now,
            healthy=True,
            metrics={
                "target_pid": self.config.pid,
                "rss_mb": target_rss,
                "swap_used_mb": current_swap,
                "swap_growth_mb": swap_growth,
                "memory_free_pct": free_pct,
                "health_ok": is_healthy,
                "consecutive_failures": self.consecutive_failures_count,
            },
        )
        self.last_check_result = res
        return res

    def run_loop(self, poll_interval_s: Optional[float] = None) -> int:
        """
        Runs the monitoring loop until target process exits, a breach occurs, or interrupted.
        Returns 0 if target exited cleanly, 1 if breach terminated process, 2 on error.
        """
        interval = poll_interval_s or self.config.check_interval_s
        print(
            f"[watchdog] Monitoring target PID {self.config.pid} ({self.cmdline[:40]}...)\n"
            f"[watchdog] Max RSS: {self.max_rss_mb:.1f} MB | Max Swap Growth: {self.max_swap_growth_mb:.1f} MB | "
            f"Min Free RAM: {self.min_free_memory_pct:.1f}% | Health Timeout: {self.health_timeout_s:.1f}s | "
            f"Stalled Timeout: {self.stalled_inference_timeout_s:.1f}s"
        )

        try:
            while True:
                result = self.check_step()
                if not result.healthy:
                    print(f"[watchdog] BREACH: {result.breach_reason}", file=sys.stderr)
                    return 1

                if result.breach_type == BreachType.TARGET_EXITED:
                    print(f"[watchdog] Target process {self.config.pid} has exited.")
                    return 0

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[watchdog] Interrupted by user; stopping watchdog.")
            return 0
        except Exception as e:
            print(f"[watchdog] Unexpected watchdog error: {e}", file=sys.stderr)
            return 2
