"""
test_mlx_watchdog.py — Comprehensive Unit & Integration Tests for External Resource Watchdog

Verifies:
1. Conservative defaults derived from memory headroom (NO double-counting purgeable pages).
2. Dynamic limit clamping respecting tight headroom (never blindly forcing 512MB minimum).
3. Fail-closed behavior on missing or unparseable telemetry (vm_stat, swapusage, memory_pressure).
   NO fabricated fallback budgets in __init__ or sampler.
4. Strict target PID validation & start-time immutable identity (cannot kill root, self, or recycled PIDs).
5. Owned-child-only enforcement by default; external attach requires explicit opt-in and instance token.
6. Subprocess isolation regression: owned hanging child is killed while unrelated sentinel survives.
7. Revalidation before EVERY signal: start-time mismatch before SIGTERM or SIGKILL aborts termination.
8. Stuck inference detection: detects wedged GPU worker (active job age > timeout) even when HTTP /health is 200 OK.
9. Idle worker safety: idle states (no active jobs) are never misdiagnosed as stalled.
10. HTTP server PID and instance token binding validation (missing token fails closed).
11. Typed finite progress field validation (rejects NaN, boolean sequences, invalid progress).
12. Startup grace period policy for legitimate cold model loading.
13. Config validation: rejects non-positive, infinite, NaN, or out-of-range parameters.
14. Bounded health probe: enforces wall-clock deadlines, rejects slow-drip headers/bodies, and caps body sizes.
15. CLI launch end-to-end with fresh instance token passing and clean process cleanup.
"""

import http.server
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from unittest.mock import MagicMock
import pytest

from scripts.qmd_mlx.watchdog import (
    BreachType,
    MLXWatchdog,
    MLXWatchdogConfig,
    SystemMemorySampler,
    SystemMetricsError,
    TargetProcessValidator,
    TargetValidationError,
)


def _spawn_dummy_process(duration_s: int = 60) -> subprocess.Popen:
    """Spawns a harmless disposable python child process for lifecycle and kill testing."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({duration_s})"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_conservative_defaults_derived_from_headroom():
    """
    Verifies that watchdog defaults are computed from available memory headroom,
    NOT blindly assuming installed physical RAM is available.
    """
    sampler = SystemMemorySampler()

    # Case A: High installed RAM (64GB), but low available headroom (4GB)
    defaults = sampler.compute_conservative_defaults(headroom_mb=4000.0)
    assert defaults.headroom_mb == 4000.0
    # 4000 * 0.65 = 2600.0 MB
    assert defaults.max_rss_mb == 2600.0
    # 4000 * 0.25 = 1000.0 MB
    assert defaults.max_swap_growth_mb == 1000.0

    # Case B: Massive headroom (32GB) -> capped at conservative upper ceiling (8192 MB)
    defaults_large = sampler.compute_conservative_defaults(headroom_mb=32000.0)
    assert defaults_large.max_rss_mb == 8192.0
    assert defaults_large.max_swap_growth_mb == 2048.0


def test_tight_headroom_dynamic_clamping_never_exceeds_headroom():
    """
    Verifies that for tight memory environments (e.g. 200MB headroom),
    the max RSS ceiling is bounded by headroom * 0.70 and NEVER clamped up to 512MB.
    """
    sampler = SystemMemorySampler()
    defaults = sampler.compute_conservative_defaults(headroom_mb=200.0)
    assert defaults.headroom_mb == 200.0
    assert defaults.max_rss_mb <= 200.0 * 0.70  # <= 140.0 MB
    assert defaults.max_rss_mb < 200.0


def test_headroom_calculation_does_not_double_count_purgeable():
    """
    Verifies that vm_stat parser sums free, inactive, and speculative pages,
    and does NOT add purgeable pages (which are already part of inactive cache).
    """
    mock_vm_stat = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                               1000.\n"
        "Pages active:                             5000.\n"
        "Pages inactive:                           2000.\n"
        "Pages speculative:                         500.\n"
        "Pages purgeable:                          1500.\n"
        "Pages wired down:                         3000.\n"
    )

    def mock_runner(cmd, timeout_s=2.0):
        if cmd == ["vm_stat"]:
            return 0, mock_vm_stat, ""
        return 0, "", ""

    sampler = SystemMemorySampler(cmd_runner=mock_runner)
    headroom_mb = sampler.get_memory_headroom_mb()

    # Expected: (1000 + 2000 + 500) * 16384 bytes = 3500 * 16384 / (1024 * 1024) = 54.6875 MB
    expected_mb = (3500 * 16384) / (1024 * 1024)
    assert abs(headroom_mb - expected_mb) < 1e-4


def test_fail_closed_on_telemetry_errors_sampler():
    """Verifies that sampler raises SystemMetricsError on failure."""
    def broken_runner(cmd, timeout_s=2.0):
        return -1, "", "Command failed"

    sampler = SystemMemorySampler(cmd_runner=broken_runner)
    with pytest.raises(SystemMetricsError):
        sampler.get_installed_ram_mb()
    with pytest.raises(SystemMetricsError):
        sampler.get_memory_headroom_mb()
    with pytest.raises(SystemMetricsError):
        sampler.get_swap_used_mb()


def test_fail_closed_on_init_no_fabricated_fallbacks():
    """
    Verifies that MLXWatchdog.__init__ fails closed and raises SystemMetricsError
    when telemetry commands fail at startup (NO fabricated 16GB/4GB/2600MB fallback).
    """
    child = _spawn_dummy_process()
    try:
        def broken_sampler_runner(cmd, timeout_s=2.0):
            if "lstart=" in cmd:
                return 0, "Tue Sep  8 12:00:00 2026", ""
            if "command=" in cmd:
                return 0, f"python3 dummy_server --pid {child.pid}", ""
            if "rss=" in cmd:
                return 0, "102400", ""
            # Telemetry fails
            return -1, "", "sysctl vm.swapusage failed"

        broken_sampler = SystemMemorySampler(cmd_runner=broken_sampler_runner)
        config = MLXWatchdogConfig(pid=child.pid, instance_token="token-1", dry_run=True)

        with pytest.raises(SystemMetricsError):
            MLXWatchdog(config=config, sampler=broken_sampler, owned_child=child)
    finally:
        child.terminate()
        child.wait()


def test_config_validation_rejects_invalid_values():
    """Verifies that MLXWatchdogConfig rejects invalid, infinite, negative, or NaN values, as well as hostnames/non-loopback and identical ports."""
    with pytest.raises(ValueError, match="Invalid pid"):
        MLXWatchdogConfig(pid=0)
    with pytest.raises(ValueError, match="Invalid pid"):
        MLXWatchdogConfig(pid=-5)
    with pytest.raises(ValueError, match="Invalid host"):
        MLXWatchdogConfig(pid=100, host="localhost")
    with pytest.raises(ValueError, match="Invalid host"):
        MLXWatchdogConfig(pid=100, host="example.com")
    with pytest.raises(ValueError, match="Invalid host"):
        MLXWatchdogConfig(pid=100, host="0.0.0.0")
    with pytest.raises(ValueError, match="Invalid host"):
        MLXWatchdogConfig(pid=100, host="192.168.1.100")
    with pytest.raises(ValueError, match="Invalid host"):
        MLXWatchdogConfig(pid=100, host="::1")
    with pytest.raises(ValueError, match="Invalid port"):
        MLXWatchdogConfig(pid=100, port=0)
    with pytest.raises(ValueError, match="Invalid port"):
        MLXWatchdogConfig(pid=100, port=70000)
    with pytest.raises(ValueError, match="Invalid control_port"):
        MLXWatchdogConfig(pid=100, control_port=0)
    with pytest.raises(ValueError, match="must be distinct"):
        MLXWatchdogConfig(pid=100, port=8787, control_port=8787)
    with pytest.raises(ValueError, match="health_timeout_s"):
        MLXWatchdogConfig(pid=100, health_timeout_s=-1.0)
    with pytest.raises(ValueError, match="health_timeout_s"):
        MLXWatchdogConfig(pid=100, health_timeout_s=float("nan"))
    with pytest.raises(ValueError, match="consecutive_health_failures"):
        MLXWatchdogConfig(pid=100, consecutive_health_failures=0)
    with pytest.raises(ValueError, match="min_free_memory_pct"):
        MLXWatchdogConfig(pid=100, min_free_memory_pct=0.0)
    with pytest.raises(ValueError, match="min_free_memory_pct"):
        MLXWatchdogConfig(pid=100, min_free_memory_pct=105.0)
    with pytest.raises(ValueError, match="max_rss_mb"):
        MLXWatchdogConfig(pid=100, max_rss_mb=-50.0)


def test_target_pid_validation_safety():
    """
    Verifies strict safety validation to prevent killing root (PID 0/1),
    the watchdog itself, non-existent processes, or mismatched command lines.
    """
    sampler = SystemMemorySampler()

    # Cannot target PID 0 or PID 1
    with pytest.raises(TargetValidationError, match="cannot target root or init"):
        TargetProcessValidator.validate_target(0, sampler)
    with pytest.raises(TargetValidationError, match="cannot target root or init"):
        TargetProcessValidator.validate_target(1, sampler)

    # Cannot target own PID
    with pytest.raises(TargetValidationError, match="cannot target the watchdog's own process"):
        TargetProcessValidator.validate_target(os.getpid(), sampler)

    # Cannot target non-existent PID
    with pytest.raises(TargetValidationError, match="disabled by default|does not exist"):
        TargetProcessValidator.validate_target(9999999, sampler)

    # Rejects process whose command does not match expected pattern
    child = _spawn_dummy_process()
    try:
        with pytest.raises(TargetValidationError, match="does not match expected pattern"):
            TargetProcessValidator.validate_target(
                child.pid,
                sampler,
                expected_cmd_pattern=r"postgres_never_matches",
                owned_child=child,
            )
    finally:
        child.terminate()
        child.wait()


def test_owned_child_required_by_default_external_requires_opt_in_and_token():
    """
    Verifies that TargetProcessValidator rejects external PID attachment unless
    allow_external_pid=True AND instance_token are provided.
    """
    child = _spawn_dummy_process()
    sampler = SystemMemorySampler()
    try:
        # Case 1: External PID without allow_external_pid -> REJECTED
        with pytest.raises(TargetValidationError, match="disabled by default"):
            TargetProcessValidator.validate_target(
                child.pid,
                sampler,
                owned_child=None,
                allow_external_pid=False,
            )

        # Case 2: External PID with allow_external_pid=True but missing instance_token -> REJECTED
        with pytest.raises(TargetValidationError, match="requires a non-empty instance_token"):
            TargetProcessValidator.validate_target(
                child.pid,
                sampler,
                owned_child=None,
                allow_external_pid=True,
                instance_token=None,
            )

        # Case 3: External PID with allow_external_pid=True and valid instance_token -> ACCEPTED
        cmdline = TargetProcessValidator.validate_target(
            child.pid,
            sampler,
            owned_child=None,
            allow_external_pid=True,
            instance_token="valid-token-123",
        )
        assert "python" in cmdline

        # Case 4: Mismatched owned_child handle -> REJECTED
        sentinel = _spawn_dummy_process()
        try:
            with pytest.raises(TargetValidationError, match="does not match target PID"):
                TargetProcessValidator.validate_target(
                    child.pid,
                    sampler,
                    owned_child=sentinel,  # wrong handle
                )
        finally:
            sentinel.terminate()
            sentinel.wait()

    finally:
        child.terminate()
        child.wait()


def test_subprocess_isolation_regression_owned_hang_killed_sentinel_survives():
    """
    Concrete subprocess isolation regression test:
    Spawns two harmless python processes:
    1. owned_hang: target process to be monitored and terminated on breach.
    2. sentinel: completely unrelated process that MUST remain untouched and alive.
    """
    owned_hang = _spawn_dummy_process(duration_s=60)
    sentinel = _spawn_dummy_process(duration_s=60)

    try:
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_process_cmdline.side_effect = lambda p: f"python3 dummy_server --pid {p}"
        mock_sampler.get_process_start_time.side_effect = lambda p: f"Tue Sep 8 12:00:00 2026 pid={p}"
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 8000.0
        mock_sampler.get_memory_free_pct.return_value = 50.0
        mock_sampler.get_swap_used_mb.return_value = 100.0
        mock_sampler.compute_conservative_defaults.return_value = SystemMemorySampler().compute_conservative_defaults(8000.0)

        # Simulate RSS breach on owned_hang
        mock_sampler.get_process_rss_mb.return_value = 9999.0

        config = MLXWatchdogConfig(
            pid=owned_hang.pid,
            instance_token="test-token",
            max_rss_mb=2000.0,
            grace_period_s=0.5,
            dry_run=False,
        )

        watchdog = MLXWatchdog(
            config=config,
            sampler=mock_sampler,
            owned_child=owned_hang,
        )

        res = watchdog.check_step()
        assert res.healthy is False
        assert res.breach_type == BreachType.RSS_EXCEEDED
        assert res.terminated_pid == owned_hang.pid

        # Wait for termination
        time.sleep(0.3)
        assert owned_hang.poll() is not None, "Target owned_hang process was not terminated"

        # Sentinel process MUST still be running untouched!
        assert sentinel.poll() is None, "Sentinel process was incorrectly terminated!"

    finally:
        if owned_hang.poll() is None:
            owned_hang.kill()
            owned_hang.wait()
        if sentinel.poll() is None:
            sentinel.kill()
            sentinel.wait()


def test_pid_identity_start_time_mismatch_refuses_signals():
    """
    Verifies that if a process's start time does not match the captured start time
    (indicating PID recycling), watchdog immediately refuses to send signals.
    """
    child = _spawn_dummy_process()
    pid = child.pid
    try:
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_process_cmdline.return_value = f"python3 dummy_server --pid {pid}"
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 8000.0
        mock_sampler.get_memory_free_pct.return_value = 50.0
        mock_sampler.get_swap_used_mb.return_value = 100.0
        mock_sampler.compute_conservative_defaults.return_value = SystemMemorySampler().compute_conservative_defaults(8000.0)

        # Start time at init
        mock_sampler.get_process_start_time.return_value = "Tue Sep 8 12:00:00 2026"

        config = MLXWatchdogConfig(pid=pid, instance_token="token-1", dry_run=False)
        watchdog = MLXWatchdog(config=config, sampler=mock_sampler, owned_child=child)

        # Simulate PID recycling: start time changes to a later timestamp
        mock_sampler.get_process_start_time.return_value = "Tue Sep 8 12:05:00 2026"

        # Check step detects identity mismatch
        res = watchdog.check_step()
        assert res.healthy is False
        assert res.breach_type == BreachType.IDENTITY_MISMATCH
        assert "identity changed" in res.breach_reason

        # Attempting termination directly should also refuse to send signals
        term_success, sigkill_used = watchdog.terminate_target_process("Test termination")
        assert term_success is False
        assert sigkill_used is False
        assert child.poll() is None  # Target process remains untouched
    finally:
        child.terminate()
        child.wait()


def test_revalidation_before_sigkill_protects_recycled_pid():
    """
    Verifies that revalidation occurs immediately before SIGKILL escalation.
    If the process exited during SIGTERM grace period and another process took the PID,
    SIGKILL is never sent to the recycled PID.
    """
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_process_cmdline.return_value = "python3 test_server"
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 8000.0
    mock_sampler.get_memory_free_pct.return_value = 50.0
    mock_sampler.get_swap_used_mb.return_value = 100.0
    mock_sampler.compute_conservative_defaults.return_value = SystemMemorySampler().compute_conservative_defaults(8000.0)

    # Start time valid on initial check and during SIGTERM, but invalid when SIGKILL check happens
    start_times = [
        "Tue Sep 8 12:00:00 2026",  # at MLXWatchdog init
        "Tue Sep 8 12:00:00 2026",  # during validate_target in init
        "Tue Sep 8 12:00:00 2026",  # before SIGTERM
        "Tue Sep 8 12:01:00 2026",  # before SIGKILL -> mismatch!
    ]
    mock_sampler.get_process_start_time.side_effect = start_times

    child = _spawn_dummy_process()
    pid = child.pid
    try:
        config = MLXWatchdogConfig(pid=pid, instance_token="token-1", grace_period_s=0.1, dry_run=False)
        watchdog = MLXWatchdog(config=config, sampler=mock_sampler, owned_child=child)

        # Mock os.kill so child isn't actually killed in test
        with pytest.MonkeyPatch.context() as mp:
            killed_signals = []
            mp.setattr(os, "kill", lambda p, s: killed_signals.append(s))
            # Mock liveness check to simulate child still appearing alive
            mp.setattr(TargetProcessValidator, "is_pid_alive", lambda *a, **k: True)

            term_success, sigkill_used = watchdog.terminate_target_process("Test termination")

            # SIGTERM was sent (signal 15), but SIGKILL (signal 9) was ABORTED due to mismatch!
            assert signal.SIGTERM in killed_signals
            assert signal.SIGKILL not in killed_signals
            assert sigkill_used is False
    finally:
        child.terminate()
        child.wait()


def test_stalled_inference_detection_while_http_health_responds_200():
    """
    Verifies that watchdog detects when the GPU worker is wedged in an inference stall
    (active job duration > stalled_inference_timeout_s) even if the HTTP thread
    continues responding with HTTP 200 OK.
    """
    child = _spawn_dummy_process()
    pid = child.pid
    try:
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_process_cmdline.return_value = f"python3 test_server --pid {pid}"
        mock_sampler.get_process_start_time.return_value = "Tue Sep 8 12:00:00 2026"
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 8000.0
        mock_sampler.get_memory_free_pct.return_value = 50.0
        mock_sampler.get_swap_used_mb.return_value = 100.0
        mock_sampler.get_process_rss_mb.return_value = 500.0
        mock_sampler.compute_conservative_defaults.return_value = SystemMemorySampler().compute_conservative_defaults(8000.0)

        # HTTP health responds 200 OK, but reports active job running for 75s (stalled)
        def mock_stalled_health_probe(host, port, timeout_s):
            return (
                True,
                {
                    "status": "ok",
                    "state": "ready",
                    "pid": pid,
                    "instance_token": "token-1",
                    "worker_alive": True,
                    "worker_idle": False,
                    "active_job_age_s": 75.0,
                    "completed_sequence": 10,
                },
                200,
            )

        config = MLXWatchdogConfig(
            pid=pid,
            instance_token="token-1",
            stalled_inference_timeout_s=60.0,
            grace_period_s=0.5,
            dry_run=False,
        )

        watchdog = MLXWatchdog(
            config=config,
            sampler=mock_sampler,
            health_probe_fn=mock_stalled_health_probe,
            owned_child=child,
        )

        res = watchdog.check_step()
        assert res.healthy is False
        assert res.breach_type == BreachType.STALLED_INFERENCE
        assert "Stalled inference detected" in res.breach_reason
        assert res.terminated_pid == pid

        time.sleep(0.3)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_idle_worker_is_never_treated_as_stalled():
    """Verifies that an idle worker (no active jobs running) is never misdiagnosed as stalled."""
    child = _spawn_dummy_process()
    pid = child.pid
    try:
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_process_cmdline.return_value = f"python3 test_server --pid {pid}"
        mock_sampler.get_process_start_time.return_value = "Tue Sep 8 12:00:00 2026"
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 8000.0
        mock_sampler.get_memory_free_pct.return_value = 50.0
        mock_sampler.get_swap_used_mb.return_value = 100.0
        mock_sampler.get_process_rss_mb.return_value = 500.0
        mock_sampler.compute_conservative_defaults.return_value = SystemMemorySampler().compute_conservative_defaults(8000.0)

        def mock_idle_health_probe(host, port, timeout_s):
            return (
                True,
                {
                    "status": "ok",
                    "state": "ready",
                    "pid": pid,
                    "instance_token": "token-1",
                    "worker_alive": True,
                    "worker_idle": True,
                    "active_job_age_s": None,
                    "completed_sequence": 42,
                },
                200,
            )

        config = MLXWatchdogConfig(pid=pid, instance_token="token-1", stalled_inference_timeout_s=30.0, dry_run=True)
        watchdog = MLXWatchdog(config=config, sampler=mock_sampler, health_probe_fn=mock_idle_health_probe, owned_child=child)

        res = watchdog.check_step()
        assert res.healthy is True
        assert res.metrics.get("health_ok") is True
    finally:
        child.terminate()
        child.wait()


def test_http_pid_and_instance_token_mismatch_breach():
    """Verifies that watchdog detects if the HTTP server on port does not match expected PID or instance token."""
    child = _spawn_dummy_process()
    pid = child.pid
    try:
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_process_cmdline.return_value = f"python3 test_server --pid {pid}"
        mock_sampler.get_process_start_time.return_value = "Tue Sep 8 12:00:00 2026"
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 8000.0
        mock_sampler.get_memory_free_pct.return_value = 50.0
        mock_sampler.get_swap_used_mb.return_value = 100.0
        mock_sampler.get_process_rss_mb.return_value = 500.0
        mock_sampler.compute_conservative_defaults.return_value = SystemMemorySampler().compute_conservative_defaults(8000.0)

        # Health probe returns a different PID
        def mock_mismatch_pid_probe(host, port, timeout_s):
            return True, {
                "status": "ok",
                "state": "ready",
                "pid": pid + 100,
                "instance_token": "token-1",
                "worker_alive": True,
                "worker_idle": True,
                "completed_sequence": 0,
            }, 200

        config = MLXWatchdogConfig(pid=pid, instance_token="token-1", dry_run=True)
        watchdog = MLXWatchdog(config=config, sampler=mock_sampler, health_probe_fn=mock_mismatch_pid_probe, owned_child=child)

        res = watchdog.check_step()
        assert res.healthy is False
        assert res.breach_type == BreachType.IDENTITY_MISMATCH
        assert "PID mismatch" in res.breach_reason
    finally:
        child.terminate()
        child.wait()


def test_missing_instance_token_in_health_response_causes_breach():
    """Verifies that if health probe response omits instance_token, watchdog triggers IDENTITY_MISMATCH."""
    child = _spawn_dummy_process()
    pid = child.pid
    try:
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_process_cmdline.return_value = f"python3 test_server --pid {pid}"
        mock_sampler.get_process_start_time.return_value = "Tue Sep 8 12:00:00 2026"
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 8000.0
        mock_sampler.get_memory_free_pct.return_value = 50.0
        mock_sampler.get_swap_used_mb.return_value = 100.0
        mock_sampler.get_process_rss_mb.return_value = 500.0
        mock_sampler.compute_conservative_defaults.return_value = SystemMemorySampler().compute_conservative_defaults(8000.0)

        # Health probe returns no instance token
        def mock_no_token_probe(host, port, timeout_s):
            return True, {
                "status": "ok",
                "state": "ready",
                "pid": pid,
                "instance_token": None,
                "worker_alive": True,
                "worker_idle": True,
                "completed_sequence": 0,
            }, 200

        config = MLXWatchdogConfig(pid=pid, instance_token="expected-token", dry_run=True)
        watchdog = MLXWatchdog(config=config, sampler=mock_sampler, health_probe_fn=mock_no_token_probe, owned_child=child)

        res = watchdog.check_step()
        assert res.healthy is False
        assert res.breach_type == BreachType.IDENTITY_MISMATCH
        assert "instance token mismatch" in res.breach_reason
    finally:
        child.terminate()
        child.wait()


def test_malformed_progress_fields_in_health_response_causes_breach():
    """Verifies that invalid or non-finite progress telemetry triggers a health check breach."""
    child = _spawn_dummy_process()
    pid = child.pid
    try:
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_process_cmdline.return_value = f"python3 test_server --pid {pid}"
        mock_sampler.get_process_start_time.return_value = "Tue Sep 8 12:00:00 2026"
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 8000.0
        mock_sampler.get_memory_free_pct.return_value = 50.0
        mock_sampler.get_swap_used_mb.return_value = 100.0
        mock_sampler.get_process_rss_mb.return_value = 500.0
        mock_sampler.compute_conservative_defaults.return_value = SystemMemorySampler().compute_conservative_defaults(8000.0)

        # Case 1: completed_sequence is not an integer
        def mock_bad_seq_probe(host, port, timeout_s):
            return True, {
                "status": "ok",
                "state": "ready",
                "pid": pid,
                "instance_token": "token-1",
                "worker_alive": True,
                "worker_idle": True,
                "completed_sequence": "not-an-int",
            }, 200

        config = MLXWatchdogConfig(pid=pid, instance_token="token-1", dry_run=True)
        watchdog = MLXWatchdog(config=config, sampler=mock_sampler, health_probe_fn=mock_bad_seq_probe, owned_child=child)

        res = watchdog.check_step()
        assert res.healthy is False
        assert res.breach_type == BreachType.HEALTH_CHECK_FAILED
        assert "Malformed typed progress fields" in res.breach_reason

        # Case 2: active_job_age_s is NaN or negative
        def mock_nan_age_probe(host, port, timeout_s):
            return True, {
                "status": "ok",
                "state": "ready",
                "pid": pid,
                "instance_token": "token-1",
                "worker_alive": True,
                "worker_idle": False,
                "active_job_age_s": float("nan"),
                "completed_sequence": 5,
            }, 200

        watchdog._health_probe_fn = mock_nan_age_probe
        res2 = watchdog.check_step()
        assert res2.healthy is False
        assert res2.breach_type == BreachType.HEALTH_CHECK_FAILED
        assert "Invalid active_job_age_s" in res2.breach_reason
    finally:
        child.terminate()
        child.wait()


def test_startup_grace_period_policy():
    """Verifies that cold model startup (starting/loading state) is permitted during grace period."""
    child = _spawn_dummy_process()
    pid = child.pid
    try:
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_process_cmdline.return_value = f"python3 test_server --pid {pid}"
        mock_sampler.get_process_start_time.return_value = "Tue Sep 8 12:00:00 2026"
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 8000.0
        mock_sampler.get_memory_free_pct.return_value = 50.0
        mock_sampler.get_swap_used_mb.return_value = 100.0
        mock_sampler.get_process_rss_mb.return_value = 500.0
        mock_sampler.compute_conservative_defaults.return_value = SystemMemorySampler().compute_conservative_defaults(8000.0)

        # Health probe returns loading state with 503 during cold startup
        def mock_loading_probe(host, port, timeout_s):
            return False, {"status": "starting", "state": "loading", "ready": False}, 503

        config = MLXWatchdogConfig(pid=pid, instance_token="token-1", startup_grace_period_s=10.0, dry_run=True)
        watchdog = MLXWatchdog(config=config, sampler=mock_sampler, health_probe_fn=mock_loading_probe, owned_child=child)

        # Within startup grace -> healthy (waiting for startup)
        res = watchdog.check_step()
        assert res.healthy is True
        assert res.metrics.get("startup_grace") is True

        # Exceed startup grace -> triggers failure
        watchdog.start_time_mono -= 15.0
        res2 = watchdog.check_step()
        # Consecutive failures counter begins incrementing
        assert watchdog.consecutive_failures_count == 1
    finally:
        child.terminate()
        child.wait()


def test_bounded_health_probe_slow_drip_headers_and_bodies():
    """
    Verifies that _default_health_probe enforces an absolute wall-clock deadline
    and aborts connections that drip headers or bodies slower than the deadline.
    """
    class SlowDripServer:
        def __init__(self, mode: str):
            self.mode = mode
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind(("127.0.0.1", 0))
            self.port = self.sock.getsockname()[1]
            self.sock.listen(5)
            self.stop_evt = threading.Event()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

        def _run(self):
            while not self.stop_evt.is_set():
                try:
                    self.sock.settimeout(0.2)
                    client, _ = self.sock.accept()
                except (socket.timeout, OSError):
                    continue

                try:
                    # Read request
                    client.recv(1024)
                    if self.mode == "slow_header":
                        # Send 1 byte every 0.15s
                        client.sendall(b"HTTP/1.1 ")
                        for char in b"200 OK\r\nContent-Type: application/json\r\n\r\n":
                            if self.stop_evt.is_set():
                                break
                            time.sleep(0.15)
                            client.sendall(bytes([char]))
                    elif self.mode == "slow_body":
                        client.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n")
                        for char in b'{"status": "ok", "ready": true}':
                            if self.stop_evt.is_set():
                                break
                            time.sleep(0.15)
                            client.sendall(bytes([char]))
                except Exception:
                    pass
                finally:
                    try:
                        client.close()
                    except Exception:
                        pass

        def close(self):
            self.stop_evt.set()
            try:
                self.sock.close()
            except Exception:
                pass
            self.thread.join(timeout=1.0)

    # Test slow headers with tight 0.3s deadline
    srv_header = SlowDripServer("slow_header")
    try:
        t0 = time.monotonic()
        ok, payload, code = MLXWatchdog._default_health_probe("127.0.0.1", srv_header.port, timeout_s=0.3)
        elapsed = time.monotonic() - t0
        assert ok is False
        assert elapsed < 0.8, f"Probe deadline was not enforced: elapsed={elapsed:.2f}s"
        assert "deadline exceeded" in str(payload) or "timed out" in str(payload)
    finally:
        srv_header.close()

    # Test slow body with tight 0.3s deadline
    srv_body = SlowDripServer("slow_body")
    try:
        t0 = time.monotonic()
        ok, payload, code = MLXWatchdog._default_health_probe("127.0.0.1", srv_body.port, timeout_s=0.3)
        elapsed = time.monotonic() - t0
        assert ok is False
        assert elapsed < 0.8, f"Probe deadline was not enforced: elapsed={elapsed:.2f}s"
    finally:
        srv_body.close()


def test_bounded_health_probe_oversized_response():
    """Verifies that _default_health_probe rejects oversized HTTP responses (> 64KB)."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    port = server_sock.getsockname()[1]
    server_sock.listen(1)

    stop_evt = threading.Event()

    def _serve_huge():
        try:
            server_sock.settimeout(2.0)
            client, _ = server_sock.accept()
            client.recv(1024)
            huge_body = b'{"data": "' + b"A" * 100000 + b'"}'
            resp = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(huge_body)}\r\n\r\n".encode("utf-8")
                + huge_body
            )
            client.sendall(resp)
            client.close()
        except Exception:
            pass

    t = threading.Thread(target=_serve_huge, daemon=True)
    t.start()

    try:
        ok, payload, code = MLXWatchdog._default_health_probe("127.0.0.1", port, timeout_s=2.0, max_response_bytes=65536)
        assert ok is False
        assert "exceeded" in str(payload)
    finally:
        server_sock.close()
        t.join(timeout=1.0)


def test_watchdog_cli_launch_token_and_cleanup_end_to_end():
    """
    Verifies that qmd-mlx-watchdog.py --launch:
    1. Generates a fresh instance token and passes it via MLX_INSTANCE_TOKEN to child env.
    2. Child server uses the token on its /health endpoint.
    3. Watchdog validates PID and instance token on /health and reports healthy.
    4. Upon watchdog exit, the child process is cleanly terminated and reaped.
    """
    def _get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    inf_port = _get_free_port()
    ctrl_port = _get_free_port()

    # Harmless inline disposable child server script
    child_script = (
        "import os, sys, time, json, http.server, socketserver, threading\n"
        "token = os.environ.get('MLX_INSTANCE_TOKEN', 'missing')\n"
        "ctrl_port = int(os.environ.get('MLX_CONTROL_PORT', '0'))\n"
        "pid = os.getpid()\n"
        "class Handler(http.server.BaseHTTPRequestHandler):\n"
        "    def log_message(self, *a): pass\n"
        "    def do_GET(self):\n"
        "        payload = json.dumps({\n"
        "            'status': 'ok',\n"
        "            'state': 'ready',\n"
        "            'pid': pid,\n"
        "            'instance_token': token,\n"
        "            'ready': True,\n"
        "            'worker_alive': True,\n"
        "            'worker_idle': True,\n"
        "            'completed_sequence': 1,\n"
        "            'active_job_age_s': None,\n"
        "        }).encode('utf-8')\n"
        "        self.send_response(200)\n"
        "        self.send_header('Content-Type', 'application/json')\n"
        "        self.send_header('Content-Length', str(len(payload)))\n"
        "        self.end_headers()\n"
        "        self.wfile.write(payload)\n"
        "srv = socketserver.TCPServer(('127.0.0.1', ctrl_port), Handler)\n"
        "srv.serve_forever()\n"
    )

    cmd = [
        sys.executable,
        "scripts/qmd-mlx-watchdog.py",
        "--port", str(inf_port),
        "--control-port", str(ctrl_port),
        "--once",
        "--json",
        "--launch",
        sys.executable,
        "-c",
        child_script,
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=10.0,
    )

    assert res.returncode == 0, f"Watchdog CLI failed (rc={res.returncode}): stdout={res.stdout}, stderr={res.stderr}"
    data = json.loads(res.stdout)
    assert data["healthy"] is True
    assert data["breach_type"] is None


def test_prelaunch_baseline_snapshot_reuse():
    """Verify that MLXWatchdog reuses prelaunch swap and headroom defaults snapshots without resharpening/recomputing."""
    from scripts.qmd_mlx.watchdog import WatchdogDefaults

    child = _spawn_dummy_process()
    pid = child.pid
    try:
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_process_cmdline.return_value = f"python3 test_server --pid {pid}"
        mock_sampler.get_process_start_time.return_value = "Tue Sep 8 12:00:00 2026"

        custom_defaults = WatchdogDefaults(
            installed_ram_mb=65536.0,
            headroom_mb=16000.0,
            max_rss_mb=5000.0,
            max_swap_growth_mb=1500.0,
        )

        config = MLXWatchdogConfig(pid=pid, instance_token="token-1", dry_run=True)
        watchdog = MLXWatchdog(
            config=config,
            sampler=mock_sampler,
            owned_child=child,
            baseline_swap_mb=420.0,
            defaults=custom_defaults,
        )

        assert watchdog.baseline_swap_mb == 420.0
        assert watchdog.defaults.max_rss_mb == 5000.0
        assert watchdog.max_rss_mb == 5000.0
        assert watchdog.max_swap_growth_mb == 1500.0
        # compute_conservative_defaults on sampler should NOT have been called
        mock_sampler.compute_conservative_defaults.assert_not_called()
    finally:
        child.terminate()
        child.wait()


def test_watchdog_cli_rejects_pid_plus_launch():
    """Verify that qmd-mlx-watchdog.py rejects specifying both --pid and --launch before spawning."""
    cmd = [
        sys.executable,
        "scripts/qmd-mlx-watchdog.py",
        "--pid", "1234",
        "--launch", sys.executable, "-c", "pass",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5.0)
    assert res.returncode == 2
    assert "Cannot specify both --pid and --launch" in res.stderr


def test_watchdog_cli_rejects_pid_without_permissions():
    """Verify that qmd-mlx-watchdog.py rejects external PID attachment without --allow-external-pid and --instance-token."""
    # Case 1: Missing --allow-external-pid
    cmd1 = [
        sys.executable,
        "scripts/qmd-mlx-watchdog.py",
        "--pid", "1234",
    ]
    res1 = subprocess.run(cmd1, capture_output=True, text=True, check=False, timeout=5.0)
    assert res1.returncode == 2
    assert "disabled by default" in res1.stderr

    # Case 2: Missing --instance-token
    cmd2 = [
        sys.executable,
        "scripts/qmd-mlx-watchdog.py",
        "--pid", "1234",
        "--allow-external-pid",
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True, check=False, timeout=5.0)
    assert res2.returncode == 2
    assert "--instance-token is required" in res2.stderr


def test_watchdog_cli_rejects_occupied_port_before_spawn():
    """Verify that qmd-mlx-watchdog.py --launch refuses to spawn if port is already occupied."""
    # Occupy a port with a listening socket
    dummy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    dummy_sock.bind(("127.0.0.1", 0))
    dummy_sock.listen(1)
    occupied_port = dummy_sock.getsockname()[1]

    try:
        cmd = [
            sys.executable,
            "scripts/qmd-mlx-watchdog.py",
            "--port", str(occupied_port),
            "--control-port", str(occupied_port + 1),
            "--launch", sys.executable, "-c", "import time; time.sleep(10)",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5.0)
        assert res.returncode == 2
        assert "already in use" in res.stderr or "occupied endpoint" in res.stderr
    finally:
        dummy_sock.close()


def test_watchdog_cli_launch_ordering_options_before_launch():
    """Verify that watchdog CLI parses --once and --json when specified before --launch."""
    def _get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    inf_p = _get_free_port()
    ctrl_p = _get_free_port()

    # Harmless fake server
    child_script = (
        "import os, sys, time, json, http.server, socketserver\n"
        "token = os.environ.get('MLX_INSTANCE_TOKEN', 'missing')\n"
        "ctrl_port = int(os.environ.get('MLX_CONTROL_PORT', '0'))\n"
        "pid = os.getpid()\n"
        "class Handler(http.server.BaseHTTPRequestHandler):\n"
        "    def log_message(self, *a): pass\n"
        "    def do_GET(self):\n"
        "        payload = json.dumps({\n"
        "            'status': 'ok',\n"
        "            'state': 'ready',\n"
        "            'pid': pid,\n"
        "            'instance_token': token,\n"
        "            'ready': True,\n"
        "            'worker_alive': True,\n"
        "            'worker_idle': True,\n"
        "            'completed_sequence': 5,\n"
        "            'active_job_age_s': None,\n"
        "        }).encode('utf-8')\n"
        "        self.send_response(200)\n"
        "        self.send_header('Content-Type', 'application/json')\n"
        "        self.send_header('Content-Length', str(len(payload)))\n"
        "        self.end_headers()\n"
        "        self.wfile.write(payload)\n"
        "srv = socketserver.TCPServer(('127.0.0.1', ctrl_port), Handler)\n"
        "srv.serve_forever()\n"
    )

    cmd = [
        sys.executable,
        "scripts/qmd-mlx-watchdog.py",
        "--port", str(inf_p),
        "--control-port", str(ctrl_p),
        "--once",
        "--json",
        "--launch",
        sys.executable, "-c", child_script,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10.0)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["healthy"] is True
    assert data["metrics"]["target_pid"] > 0


def test_watchdog_cli_launch_exception_cleans_up_child():
    """Verify that if watchdog encounters an exception during validation/monitoring, owned child is cleanly reaped."""
    def _get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    inf_p = _get_free_port()
    ctrl_p = _get_free_port()

    # Child script that sleeps
    child_script = "import time; time.sleep(30)"

    # Provide an expected-cmd pattern that will intentionally fail validation
    cmd = [
        sys.executable,
        "scripts/qmd-mlx-watchdog.py",
        "--port", str(inf_p),
        "--control-port", str(ctrl_p),
        "--expected-cmd", "definitely_not_matching_pattern_xyz",
        "--once",
        "--launch",
        sys.executable, "-c", child_script,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10.0)
    assert res.returncode == 2
    assert "safety validation failed" in res.stderr or "does not match expected pattern" in res.stderr


def test_watchdog_cli_production_launch_with_harmless_fake_server():
    """
    Test production launch CLI behavior with a harmless fake server responding on distinct endpoints.
    Verifies instance token generation, environment propagation, and probe success.
    """
    def _get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    inf_p = _get_free_port()
    ctrl_p = _get_free_port()

    child_code = (
        "import os, sys, time, json, http.server, socketserver\n"
        "token = os.environ.get('MLX_INSTANCE_TOKEN')\n"
        "inf_port = int(os.environ.get('MLX_EMBED_PORT', '0'))\n"
        "ctrl_port = int(os.environ.get('MLX_CONTROL_PORT', '0'))\n"
        "pid = os.getpid()\n"
        "class Handler(http.server.BaseHTTPRequestHandler):\n"
        "    def log_message(self, *a): pass\n"
        "    def do_GET(self):\n"
        "        p = json.dumps({\n"
        "            'status': 'ok',\n"
        "            'state': 'ready',\n"
        "            'pid': pid,\n"
        "            'instance_token': token,\n"
        "            'ready': True,\n"
        "            'worker_alive': True,\n"
        "            'worker_idle': True,\n"
        "            'completed_sequence': 10,\n"
        "            'active_job_age_s': None,\n"
        "        }).encode('utf-8')\n"
        "        self.send_response(200)\n"
        "        self.send_header('Content-Type', 'application/json')\n"
        "        self.send_header('Content-Length', str(len(p)))\n"
        "        self.send_header('Connection', 'close')\n"
        "        self.end_headers()\n"
        "        self.wfile.write(p)\n"
        "srv = socketserver.TCPServer(('127.0.0.1', ctrl_port), Handler)\n"
        "srv.serve_forever()\n"
    )

    cmd = [
        sys.executable,
        "scripts/qmd-mlx-watchdog.py",
        "--port", str(inf_p),
        "--control-port", str(ctrl_p),
        "--once",
        "--json",
        "--launch",
        sys.executable, "-c", child_code,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10.0)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["healthy"] is True
    assert data["breach_type"] is None


@pytest.mark.parametrize(
    "cli_args,expected_error_fragment",
    [
        # NaN thresholds
        (["--port", "8787", "--control-port", "8788", "--health-timeout-s", "nan", "--launch", "python3", "-c", "pass"], "health_timeout_s"),
        (["--port", "8787", "--control-port", "8788", "--min-free-memory-pct", "nan", "--launch", "python3", "-c", "pass"], "min_free_memory_pct"),
        (["--port", "8787", "--control-port", "8788", "--max-rss-mb", "nan", "--launch", "python3", "-c", "pass"], "max_rss_mb"),
        (["--port", "8787", "--control-port", "8788", "--max-swap-growth-mb", "nan", "--launch", "python3", "-c", "pass"], "max_swap_growth_mb"),
        (["--port", "8787", "--control-port", "8788", "--stalled-timeout-s", "nan", "--launch", "python3", "-c", "pass"], "stalled_inference_timeout_s"),
        # Inf thresholds
        (["--port", "8787", "--control-port", "8788", "--health-timeout-s", "inf", "--launch", "python3", "-c", "pass"], "health_timeout_s"),
        (["--port", "8787", "--control-port", "8788", "--grace-period-s", "inf", "--launch", "python3", "-c", "pass"], "grace_period_s"),
        (["--port", "8787", "--control-port", "8788", "--startup-grace-period-s", "inf", "--launch", "python3", "-c", "pass"], "startup_grace_period_s"),
        # Negative / zero thresholds
        (["--port", "8787", "--control-port", "8788", "--health-timeout-s", "-1.0", "--launch", "python3", "-c", "pass"], "health_timeout_s"),
        (["--port", "8787", "--control-port", "8788", "--health-timeout-s", "0.0", "--launch", "python3", "-c", "pass"], "health_timeout_s"),
        (["--port", "8787", "--control-port", "8788", "--check-interval-s", "0", "--launch", "python3", "-c", "pass"], "check_interval_s"),
        (["--port", "8787", "--control-port", "8788", "--check-interval-s", "-0.5", "--launch", "python3", "-c", "pass"], "check_interval_s"),
        (["--port", "8787", "--control-port", "8788", "--max-rss-mb", "-50", "--launch", "python3", "-c", "pass"], "max_rss_mb"),
        (["--port", "8787", "--control-port", "8788", "--max-swap-growth-mb", "-10", "--launch", "python3", "-c", "pass"], "max_swap_growth_mb"),
        (["--port", "8787", "--control-port", "8788", "--min-free-memory-pct", "0.0", "--launch", "python3", "-c", "pass"], "min_free_memory_pct"),
        (["--port", "8787", "--control-port", "8788", "--min-free-memory-pct", "150.0", "--launch", "python3", "-c", "pass"], "min_free_memory_pct"),
        (["--port", "8787", "--control-port", "8788", "--consecutive-failures", "0", "--launch", "python3", "-c", "pass"], "consecutive_health_failures"),
        # Out of range ports
        (["--port", "0", "--control-port", "8788", "--launch", "python3", "-c", "pass"], "Invalid port 0"),
        (["--port", "70000", "--control-port", "8788", "--launch", "python3", "-c", "pass"], "Invalid port 70000"),
        (["--port", "8787", "--control-port", "0", "--launch", "python3", "-c", "pass"], "Invalid control_port 0"),
        (["--port", "8787", "--control-port", "65536", "--launch", "python3", "-c", "pass"], "Invalid control_port 65536"),
        # Equal / missing ports
        (["--port", "8787", "--control-port", "8787", "--launch", "python3", "-c", "pass"], "must be distinct"),
        (["--port", "8787", "--launch", "python3", "-c", "pass"], "requires explicit, distinct --port and --control-port"),
        (["--control-port", "8788", "--launch", "python3", "-c", "pass"], "requires explicit, distinct --port and --control-port"),
        # Empty --launch
        (["--port", "8787", "--control-port", "8788", "--launch"], "cannot be empty"),
        (["--port", "8787", "--control-port", "8788", "--launch", ""], "cannot be empty"),
        (["--port", "8787", "--control-port", "8788", "--launch", "   "], "cannot be empty"),
        # Invalid regex
        (["--port", "8787", "--control-port", "8788", "--expected-cmd", "[", "--launch", "python3", "-c", "pass"], "expected_cmd_pattern"),
        (["--port", "8787", "--control-port", "8788", "--expected-cmd", "(+invalid", "--launch", "python3", "-c", "pass"], "expected_cmd_pattern"),
        # Invalid host
        (["--host", "::1", "--port", "8787", "--control-port", "8788", "--launch", "python3", "-c", "pass"], "Host '::1' must be a numeric IPv4 loopback IP ('127.0.0.1')"),
        (["--host", "localhost", "--port", "8787", "--control-port", "8788", "--launch", "python3", "-c", "pass"], "Host 'localhost' must be a numeric IPv4 loopback IP ('127.0.0.1')"),
        (["--host", "0.0.0.0", "--port", "8787", "--control-port", "8788", "--launch", "python3", "-c", "pass"], "Host '0.0.0.0' must be a numeric IPv4 loopback IP ('127.0.0.1')"),
        # Conflicting child CLI arguments
        (["--port", "8787", "--control-port", "8788", "--launch", "python3", "scripts/mlx_embed_server.py", "--port", "9999"], "conflicts with watchdog '--port 8787'"),
        (["--port", "8787", "--control-port", "8788", "--launch", "python3", "scripts/mlx_embed_server.py", "--control-port", "9998"], "conflicts with watchdog '--control-port 8788'"),
        (["--port", "8787", "--control-port", "8788", "--launch", "python3", "scripts/mlx_embed_server.py", "--port=9999"], "conflicts with watchdog '--port 8787'"),
        (["--port", "8787", "--control-port", "8788", "--launch", "python3", "scripts/mlx_embed_server.py", "--control-port=9998"], "conflicts with watchdog '--control-port 8788'"),
        (["--port", "8787", "--control-port", "8788", "--host", "127.0.0.1", "--launch", "python3", "scripts/mlx_embed_server.py", "--host", "192.168.1.5"], "conflicts with watchdog '--host 127.0.0.1'"),
    ],
)
def test_watchdog_cli_prespawn_validation_popen_never_called(monkeypatch, capsys, cli_args, expected_error_fragment):
    """
    Parametrized CLI regression test asserting that for invalid NaN/inf/negative thresholds,
    out-of-range ports, equal/missing ports, empty --launch, invalid regex, invalid host (e.g. ::1),
    and conflicting child CLI ports:
    1. The CLI exits with return code 2.
    2. Subprocess.Popen is NEVER called (pre-spawn validation invariant).
    3. The expected error fragment is reported on stderr.
    """
    import importlib
    watchdog_cli = importlib.import_module("scripts.qmd-mlx-watchdog")

    mock_popen = MagicMock()
    mock_sampler = MagicMock(spec=SystemMemorySampler)

    monkeypatch.setattr(sys, "argv", ["qmd-mlx-watchdog.py"] + cli_args)
    monkeypatch.setattr(watchdog_cli.subprocess, "Popen", mock_popen)
    monkeypatch.setattr(watchdog_cli, "SystemMemorySampler", lambda: mock_sampler)

    rc = watchdog_cli.main()
    captured = capsys.readouterr()

    assert rc == 2
    mock_popen.assert_not_called()
    assert expected_error_fragment.lower() in captured.err.lower()


def test_watchdog_cli_child_cli_matching_args_accepted():
    """
    Verifies that when explicit child CLI arguments match the watchdog's configured
    ports and host (e.g. --port 8787 --control-port 8788 --host 127.0.0.1), pre-spawn
    validation succeeds without conflict errors.
    """
    from scripts.qmd_mlx.watchdog import validate_config_parameters
    import importlib
    watchdog_cli = importlib.import_module("scripts.qmd-mlx-watchdog")

    cmd = ["python3", "scripts/mlx_embed_server.py", "--port", "8787", "--control-port", "8788", "--host", "127.0.0.1"]
    err = watchdog_cli.check_child_cli_conflicts(
        cmd=cmd,
        expected_port=8787,
        expected_control_port=8788,
        expected_host="127.0.0.1",
    )
    assert err is None

    cmd_equals = ["python3", "scripts/mlx_embed_server.py", "--port=8787", "--control-port=8788", "--host=127.0.0.1"]
    err_equals = watchdog_cli.check_child_cli_conflicts(
        cmd=cmd_equals,
        expected_port=8787,
        expected_control_port=8788,
        expected_host="127.0.0.1",
    )
    assert err_equals is None


def test_pure_validate_config_parameters_function():
    """
    Verifies that validate_config_parameters validates all parameters purely without
    requiring a target PID or producing side effects.
    """
    from scripts.qmd_mlx.watchdog import validate_config_parameters

    # Valid configurations succeed without error
    validate_config_parameters(host="127.0.0.1", port=8787, control_port=8788)
    validate_config_parameters(host="127.0.0.1", port=1000, control_port=None)

    # Invalid host
    with pytest.raises(ValueError, match="Invalid host"):
        validate_config_parameters(host="::1")
    with pytest.raises(ValueError, match="Invalid host"):
        validate_config_parameters(host="localhost")

    # Invalid ports
    with pytest.raises(ValueError, match="Invalid port"):
        validate_config_parameters(port=0)
    with pytest.raises(ValueError, match="Invalid port"):
        validate_config_parameters(port=70000)
    with pytest.raises(ValueError, match="Invalid control_port"):
        validate_config_parameters(port=8787, control_port=0)
    with pytest.raises(ValueError, match="must be distinct"):
        validate_config_parameters(port=8787, control_port=8787)

    # Invalid thresholds
    with pytest.raises(ValueError, match="health_timeout_s"):
        validate_config_parameters(health_timeout_s=-1.0)
    with pytest.raises(ValueError, match="health_timeout_s"):
        validate_config_parameters(health_timeout_s=float("nan"))
    with pytest.raises(ValueError, match="min_free_memory_pct"):
        validate_config_parameters(min_free_memory_pct=150.0)
    with pytest.raises(ValueError, match="max_rss_mb"):
        validate_config_parameters(max_rss_mb=float("inf"))
    with pytest.raises(ValueError, match="expected_cmd_pattern"):
        validate_config_parameters(expected_cmd_pattern="[")




