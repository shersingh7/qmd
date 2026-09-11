"""
test_phase5_qualification.py — Unit & Offline Integration Tests for Phase 5 Supervisor
"""

import json
import os
import subprocess
import sys
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from scripts.qmd_mlx.phase5_e2e_qualification import Phase5Supervisor, Phase5SupervisorConfig
from scripts.qmd_mlx.watchdog import SystemMemorySampler, SystemMetricsError


def test_phase5_supervisor_config_defaults():
    config = Phase5SupervisorConfig()
    assert config.mlx_port == 8797
    assert config.mlx_control_port == 8798
    assert config.timeout_s == 120.0
    assert config.min_headroom_mb == 6000.0
    assert config.host == "127.0.0.1"
    assert config.skip_gguf is True
    assert "phase5-isolated-e2e.json" in config.output_json


def test_phase5_supervisor_preflight_sufficient_headroom():
    config = Phase5SupervisorConfig(min_headroom_mb=6000.0)
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 16000.0
    mock_sampler.get_swap_used_mb.return_value = 0.0
    mock_sampler.get_memory_free_pct.return_value = 65.0

    supervisor = Phase5Supervisor(config, sampler=mock_sampler)
    info = supervisor._check_preflight_headroom()
    assert info["installed_ram_mb"] == 32768.0
    assert info["headroom_mb"] == 16000.0
    assert info["swap_used_mb"] == 0.0
    assert info["memory_free_pct"] == 65.0


def test_phase5_supervisor_preflight_insufficient_headroom_fails_closed():
    config = Phase5SupervisorConfig(min_headroom_mb=6000.0)
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 4500.0
    mock_sampler.get_swap_used_mb.return_value = 120.0
    mock_sampler.get_memory_free_pct.return_value = 15.0

    supervisor = Phase5Supervisor(config, sampler=mock_sampler)
    with pytest.raises(RuntimeError, match="Insufficient memory headroom"):
        supervisor._check_preflight_headroom()


def test_phase5_supervisor_preflight_telemetry_error_fails_closed():
    config = Phase5SupervisorConfig(min_headroom_mb=6000.0)
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.side_effect = SystemMetricsError("vm_stat failed")

    supervisor = Phase5Supervisor(config, sampler=mock_sampler)
    with pytest.raises(SystemMetricsError, match="Preflight memory sampling failed"):
        supervisor._check_preflight_headroom()


def test_phase5_supervisor_rejection_of_live_daemon_port_8787():
    # Config pointing to production port 8787 must fail closed in supervisor preflight
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 16000.0
    mock_sampler.get_swap_used_mb.return_value = 0.0
    mock_sampler.get_memory_free_pct.return_value = 65.0

    config_bad_port = Phase5SupervisorConfig(mlx_port=8787, mlx_control_port=8798)
    supervisor = Phase5Supervisor(config_bad_port, sampler=mock_sampler)

    # SmokeStageSupervisor validate_preflight refuses port 8787
    with pytest.raises(ValueError, match="Refusing to use port 8787"):
        from scripts.qmd_mlx.supervisor import StageSupervisorConfig, SmokeStageSupervisor
        s_cfg = StageSupervisorConfig(
            cmd=["dummy"],
            port=config_bad_port.mlx_port,
            control_port=config_bad_port.mlx_control_port,
        )
        s = SmokeStageSupervisor(s_cfg, sampler=mock_sampler)
        s.validate_preflight()


def test_phase5_supervisor_missing_model_fails_closed_without_autodownload():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = os.path.join(tmpdir, "report.json")
        config = Phase5SupervisorConfig(
            mlx_model="/nonexistent/model/path/qwen3-4b",
            output_json=out_json,
            min_headroom_mb=4000.0,
        )
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 16000.0
        mock_sampler.get_swap_used_mb.return_value = 0.0
        mock_sampler.get_memory_free_pct.return_value = 65.0

        supervisor = Phase5Supervisor(config, sampler=mock_sampler)
        report = supervisor.run()

        assert report["status"] == "failed"
        assert any("Explicit MLX model path missing" in err for err in report["errors"])
        assert os.path.exists(out_json)


def test_phase5_supervisor_dry_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = os.path.join(tmpdir, "dry_run_report.json")
        config = Phase5SupervisorConfig(
            dry_run=True,
            output_json=out_json,
            min_headroom_mb=4000.0,
        )
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 16000.0
        mock_sampler.get_swap_used_mb.return_value = 0.0
        mock_sampler.get_memory_free_pct.return_value = 65.0
        mock_sampler.compute_conservative_defaults.return_value = MagicMock()

        supervisor = Phase5Supervisor(config, sampler=mock_sampler)
        report = supervisor.run()

        assert report["status"] == "dry_run_completed"
        assert os.path.exists(out_json)


def test_phase5_ts_runner_launch_token_assertion():
    """Verifies that phase5_e2e_runner.ts rejects direct unmanaged CLI execution without launch token."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ts_script = os.path.join(repo_root, "scripts", "phase5_e2e_runner.ts")

    # Run without --launch-token and without QMD_PHASE5_LAUNCH_TOKEN
    env = {k: v for k, v in os.environ.items() if k != "QMD_PHASE5_LAUNCH_TOKEN"}
    proc = subprocess.run(
        ["bun", ts_script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Direct TypeScript execution rejected" in proc.stderr


def test_phase5_recovery_worker_launch_token_assertion():
    """Verifies that phase5_recovery_worker.ts rejects direct unmanaged CLI execution without launch token."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    worker_script = os.path.join(repo_root, "scripts", "phase5_recovery_worker.ts")

    env = {k: v for k, v in os.environ.items() if k != "QMD_PHASE5_LAUNCH_TOKEN"}
    proc = subprocess.run(
        ["bun", worker_script, "--db", "/tmp/dummy.sqlite"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Direct TypeScript recovery worker execution rejected" in proc.stderr


def test_subordinate_process_runner_success():
    """Verifies that run_subordinate_process completes successful commands and captures bounded logs."""
    from scripts.qmd_mlx.supervisor import run_subordinate_process
    import time

    deadline = time.monotonic() + 5.0
    res = run_subordinate_process(
        cmd=[sys.executable, "-c", "print('hello subordinate')"],
        deadline=deadline,
    )
    assert res.exit_code == 0
    assert "hello subordinate" in res.captured_logs
    assert not res.timed_out
    assert not res.breached
    assert res.error is None


def test_subordinate_process_runner_timeout():
    """Verifies that run_subordinate_process terminates and reaps stalled child upon deadline breach."""
    from scripts.qmd_mlx.supervisor import run_subordinate_process
    import time

    deadline = time.monotonic() + 0.2
    t0 = time.monotonic()
    res = run_subordinate_process(
        cmd=[sys.executable, "-c", "import time; time.sleep(10)"],
        deadline=deadline,
        poll_interval_s=0.02,
    )
    assert res.timed_out is True
    assert res.duration_s < 2.0
    assert "exceeded monotonic deadline" in (res.error or "")


def test_subordinate_process_runner_breach_detection():
    """Verifies that run_subordinate_process immediately kills child when breach callback fires."""
    from scripts.qmd_mlx.supervisor import run_subordinate_process
    import time

    call_count = [0]
    def mock_check_breach(label: str):
        call_count[0] += 1
        if call_count[0] >= 2:
            raise RuntimeError("Synthetic watchdog breach")

    deadline = time.monotonic() + 5.0
    res = run_subordinate_process(
        cmd=[sys.executable, "-c", "import time; time.sleep(10)"],
        deadline=deadline,
        check_breach=mock_check_breach,
        poll_interval_s=0.02,
    )
    assert res.breached is True
    assert "Synthetic watchdog breach" in (res.error or "")


def test_subordinate_stall_fake_nonmodel_sentinel_survives():
    """
    Proves that when a subordinate child process stalls and is reaped on deadline timeout,
    an external/sentinel non-model background process survives intact without corruption or termination.
    """
    from scripts.qmd_mlx.supervisor import run_subordinate_process
    import time

    # Spawn an independent background sentinel process
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        sentinel_pid = sentinel.pid
        assert sentinel.poll() is None, "Sentinel should be running"

        # Run subordinate runner on a stalling child with short deadline
        deadline = time.monotonic() + 0.2
        res = run_subordinate_process(
            cmd=[sys.executable, "-c", "import time; time.sleep(30)"],
            deadline=deadline,
            poll_interval_s=0.02,
        )

        assert res.timed_out is True
        assert res.pid is not None
        assert res.pid != sentinel_pid

        # Assert sentinel survived intact
        assert sentinel.poll() is None, "Sentinel process must survive subordinate process timeout and reap"
    finally:
        if sentinel.poll() is None:
            sentinel.terminate()
            try:
                sentinel.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                sentinel.kill()
                sentinel.wait(timeout=1.0)


def test_preflight_config_validation_nan_inf_negative():
    """Verifies that StageSupervisorConfig and Phase5Supervisor reject NaN, inf, negative, and conflicting configs."""
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 16000.0
    mock_sampler.get_swap_used_mb.return_value = 0.0
    mock_sampler.get_memory_free_pct.return_value = 65.0

    # 1. NaN timeout
    cfg_nan = Phase5SupervisorConfig(timeout_s=float("nan"))
    sup_nan = Phase5Supervisor(cfg_nan, sampler=mock_sampler)
    with pytest.raises(ValueError, match="must not be NaN or Inf"):
        sup_nan._check_preflight_headroom()

    # 2. Inf headroom
    cfg_inf = Phase5SupervisorConfig(min_headroom_mb=float("inf"))
    sup_inf = Phase5Supervisor(cfg_inf, sampler=mock_sampler)
    with pytest.raises(ValueError, match="must not be NaN or Inf"):
        sup_inf._check_preflight_headroom()

    # 3. Negative timeout
    cfg_neg = Phase5SupervisorConfig(timeout_s=-10.0)
    sup_neg = Phase5Supervisor(cfg_neg, sampler=mock_sampler)
    with pytest.raises(ValueError, match="must be >="):
        sup_neg._check_preflight_headroom()

    # 4. Same ports
    cfg_same_port = Phase5SupervisorConfig(mlx_port=8797, mlx_control_port=8797)
    sup_same_port = Phase5Supervisor(cfg_same_port, sampler=mock_sampler)
    with pytest.raises(ValueError, match="must be distinct"):
        sup_same_port._check_preflight_headroom()


def test_preflight_failure_report_persisted():
    """Verifies that a preflight configuration or telemetry error persists a failure report JSON fail-closed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = os.path.join(tmpdir, "preflight_fail.json")
        cfg_bad = Phase5SupervisorConfig(
            timeout_s=-1.0,
            output_json=out_json,
        )
        mock_sampler = MagicMock(spec=SystemMemorySampler)
        mock_sampler.get_installed_ram_mb.return_value = 32768.0
        mock_sampler.get_memory_headroom_mb.return_value = 16000.0
        mock_sampler.get_swap_used_mb.return_value = 0.0
        mock_sampler.get_memory_free_pct.return_value = 65.0

        supervisor = Phase5Supervisor(cfg_bad, sampler=mock_sampler)
        report = supervisor.run()

        assert report["status"] == "failed"
        assert len(report["errors"]) > 0
        assert "Preflight" in report["errors"][0]
        assert os.path.exists(out_json)
        with open(out_json, "r") as f:
            saved = json.load(f)
            assert saved["status"] == "failed"

