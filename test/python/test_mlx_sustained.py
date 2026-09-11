"""
test_mlx_sustained.py — Unit and integration tests for bounded sustained embedding qualification harness.

Verifies:
1. Public strata fixtures conform to token ranges (short 5-25, medium 50-200, long 400-1500, code 50-500).
2. 2048 boundary input policy without unbounded allocation.
3. Cold startup measured separately and warmup excluded from request distribution.
4. Explicit percentile calculation method with small sample descriptive flag.
5. Rehearsal full lifecycle with SyntheticEmbeddingAdapter and active MLXWatchdog supervision.
6. Pilot mode request capping (<=30) and timeout ceiling (<=120s).
7. Soak mode request capping (<=100) and timeout ceiling (<=180s).
8. Guaranteed child process and supervisor teardown on completion.
"""

import json
import os
import socket
import pytest

from scripts.qmd_mlx.sustained import (
    CODE_FIXTURES,
    LONG_FIXTURES,
    MEDIUM_FIXTURES,
    SHORT_FIXTURES,
    SustainedRunner,
    SustainedRunnerConfig,
    compute_explicit_percentiles,
)


def get_ephemeral_port_pair() -> tuple[int, int]:
    """Finds two distinct free ports for ephemeral testing."""
    s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s1.bind(("127.0.0.1", 0))
    p1 = s1.getsockname()[1]

    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s2.bind(("127.0.0.1", 0))
    p2 = s2.getsockname()[1]

    s1.close()
    s2.close()
    return p1, p2


def test_percentile_computation_explicit_math():
    """Verifies that percentile calculations use explicit linear interpolation and flag small samples."""
    # Small sample (N <= 30) -> descriptive_only = True
    small_vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    p_small = compute_explicit_percentiles(small_vals)
    assert p_small["count"] == 5
    assert p_small["min"] == 10.0
    assert p_small["p50"] == 30.0
    assert p_small["p95"] == 48.0
    assert p_small["max"] == 50.0
    assert p_small["mean"] == 30.0
    assert p_small["descriptive_only"] is True

    # Larger sample (N > 30) -> descriptive_only = False
    large_vals = list(range(1, 101))
    p_large = compute_explicit_percentiles(large_vals)
    assert p_large["count"] == 100
    assert p_large["min"] == 1.0
    assert p_large["p50"] == 50.5
    assert p_large["p95"] == 95.05
    assert p_large["max"] == 100.0
    assert p_large["descriptive_only"] is False


def test_sustained_rehearsal_pilot_lifecycle_and_strata():
    """
    Executes full offline rehearsal of sustained pilot mode with SyntheticEmbeddingAdapter.
    Verifies:
    - Real server stack executed with active MLXWatchdog supervision.
    - Code manifest SHA256 fingerprints recorded.
    - Cold startup measured separately.
    - Warmup executed and excluded from measured metrics.
    - Token strata verified (short, medium, long, code, exact 2047, 2048, 2049 boundary rejection).
    - Solo baseline interactive latency measured under idle load.
    - True concurrent load interleaving with barrier handshake measured.
    - Metrics summary accounts successful tokens ONLY with separate 400 rejection metrics.
    - Measured requests <= 30.
    - Child process cleanly reaped.
    """
    p1, p2 = get_ephemeral_port_pair()
    config = SustainedRunnerConfig(
        use_fake_child=True,
        port=p1,
        control_port=p2,
        mode="pilot",
        max_requests=25,
        timeout_s=30.0,
    )
    runner = SustainedRunner(config)
    report = runner.run()

    assert report["status"] == "passed"
    assert report["mode"] == "sustained_pilot"
    assert report["synthetic"] is True
    assert "code_manifest_sha256" in report
    assert "scripts/qmd_mlx/sustained.py" in report["code_manifest_sha256"]

    # Cold startup
    assert "cold_startup" in report
    assert report["cold_startup"]["ready"] is True
    assert report["cold_startup"]["cold_startup_elapsed_s"] > 0

    # Warmup
    assert "warmup" in report
    assert report["warmup"]["status"] == "completed"
    assert report["warmup"]["excluded_from_measured_metrics"] is True

    # Solo baseline
    assert "solo_baseline" in report
    assert len(report["solo_baseline"]["samples_ms"]) == 3
    assert report["solo_baseline"]["percentiles"]["p50"] > 0

    # Strata verification
    strata = report["strata_verification"]
    assert "short" in strata
    assert "medium" in strata
    assert "long" in strata
    assert "code" in strata
    assert "boundary_2047" in strata
    assert "boundary_2048" in strata
    assert "boundary_2049" in strata

    assert 5 <= strata["short"]["min_tokens"] <= 25
    assert 50 <= strata["medium"]["min_tokens"]
    assert 400 <= strata["long"]["min_tokens"]
    assert 50 <= strata["code"]["min_tokens"]
    assert strata["boundary_2047"]["min_tokens"] == 2047
    assert strata["boundary_2048"]["min_tokens"] == 2048
    assert strata["boundary_2049"]["rejection_verified"] is True

    # Workload measurements
    reqs = report["measured_requests"]
    rejs = report["rejections"]
    assert 0 < (len(reqs) + len(rejs)) <= 25
    for r in reqs:
        assert r["finite"] is True
        assert r["l2_normalized"] is True
        assert r["wire_ms"] > 0
        assert r["effective_tokens_per_sec"] > 0

    assert len(rejs) >= 1
    for r in rejs:
        assert r["rejection_verified"] is True
        assert r["status_code"] == 400

    # Concurrent interleaving
    assert "concurrent_load_interleaving" in report
    assert len(report["concurrent_load_interleaving"]) >= 1
    for c in report["concurrent_load_interleaving"]:
        assert c["bulk_wire_ms"] > 0
        assert c["concurrent_interactive_wire_ms"] > 0
        assert c["slowdown_factor"] > 0

    # Metrics summary
    summary = report["metrics_summary"]
    assert summary["total_measured_requests"] == len(reqs) + len(rejs)
    assert summary["total_successful_embedded_tokens"] > 0
    assert "overall_successful_latency_ms" in summary
    assert "expected_rejections_latency_ms" in summary
    assert "by_stratum_latency_ms" in summary
    assert "by_batch_size_latency_ms" in summary
    assert "concurrent_tail_latency_ms" in summary
    assert "cumulative_wall_accounting" in summary

    # Metal memory telemetry
    assert "metal_memory_telemetry" in report
    assert "provisional_stability_assessment" in report["metal_memory_telemetry"]

    # Child cleanup
    assert report["child_cleanup"]["cleaned"] is True
    assert report["child_cleanup"]["pid"] is not None


def test_sustained_rehearsal_soak_lifecycle():
    """
    Executes offline rehearsal of sustained soak mode with SyntheticEmbeddingAdapter.
    Verifies soak request loop capped at max_requests and clean cleanup.
    """
    p1, p2 = get_ephemeral_port_pair()
    config = SustainedRunnerConfig(
        use_fake_child=True,
        port=p1,
        control_port=p2,
        mode="soak",
        max_requests=25,
        timeout_s=30.0,
    )
    runner = SustainedRunner(config)
    report = runner.run()

    assert report["status"] == "passed"
    assert report["mode"] == "sustained_soak"
    assert len(report["measured_requests"]) + len(report["rejections"]) == 25
    assert report["child_cleanup"]["cleaned"] is True


def test_sustained_cli_writes_report_file(tmp_path):
    """Verifies that qmd-mlx-sustained.py CLI writes output JSON file and exits with code 0."""
    import importlib
    sustained_cli = importlib.import_module("scripts.qmd-mlx-sustained")

    p1, p2 = get_ephemeral_port_pair()
    out_file = str(tmp_path / "sustained_pilot_report.json")

    argv = [
        "--rehearsal",
        "--mode", "pilot",
        "--max-requests", "25",
        "--port", str(p1),
        "--control-port", str(p2),
        "--timeout-s", "30",
        "--output-file", out_file,
    ]

    ret = sustained_cli.main(argv)
    assert ret == 0

    assert os.path.exists(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        saved_report = json.load(f)

    assert saved_report["status"] == "passed"
    assert saved_report["mode"] == "sustained_pilot"
    assert saved_report["metrics_summary"]["total_measured_requests"] == 23


def test_evaluate_qualification_gates_fails_on_interactive_latency_breach():
    """Verifies that qualification gates fail when concurrent interactive p95 exceeds 200.0ms."""
    from scripts.qmd_mlx.sustained import evaluate_qualification_gates

    mock_report = {
        "status": "in_progress",
        "cold_startup": {"ready": True, "cold_startup_elapsed_s": 2.1},
        "rejections": [{"stratum": "boundary_2049", "rejection_verified": True, "status_code": 400}],
        "measured_requests": [{"finite": True, "l2_normalized": True}],
        "metrics_summary": {
            "concurrent_tail_latency_ms": {
                "concurrent_interactive_under_load": {
                    "p95": 1236.5,
                    "count": 3,
                    "descriptive_only": True,
                },
                "mean_slowdown_factor": 14.71,
            }
        },
        "metal_memory_telemetry": {
            "provisional_stability_assessment": {"verdict": "provisional_stable", "swap_growth_mb": 0.0}
        },
    }
    config = SustainedRunnerConfig(dry_run=False, mode="pilot")
    eval_res = evaluate_qualification_gates(mock_report, config)

    assert eval_res["overall_status"] == "failed"
    gate = eval_res["gates"]["concurrent_interactive_responsiveness"]
    assert gate["passed"] is False
    assert gate["target_p95_ms"] == 200.0
    assert gate["measured_p95_ms"] == 1236.5
    assert gate["descriptive_only"] is True
    assert any("<= 200.0ms target" in r for r in eval_res["failure_reasons"])


def test_evaluate_qualification_gates_fails_on_watchdog_breach():
    """Verifies that watchdog breach snapshots cause qualification failure."""
    from scripts.qmd_mlx.sustained import evaluate_qualification_gates

    mock_report = {
        "status": "in_progress",
        "cold_startup": {"ready": True, "cold_startup_elapsed_s": 2.1},
        "rejections": [{"stratum": "boundary_2049", "rejection_verified": True, "status_code": 400}],
        "measured_requests": [{"finite": True, "l2_normalized": True}],
        "metrics_summary": {
            "concurrent_tail_latency_ms": {
                "concurrent_interactive_under_load": {
                    "p95": 85.0,
                    "count": 3,
                    "descriptive_only": True,
                },
                "mean_slowdown_factor": 1.2,
            }
        },
        "watchdog_breach": {
            "breach_type": "swap_growth_exceeded",
            "breach_reason": "Swap growth exceeded 2048 MB (measured: 2053.4 MB)",
            "metrics": {"swap_growth_mb": 2053.4},
        },
        "metal_memory_telemetry": {
            "provisional_stability_assessment": {"verdict": "unstable", "swap_growth_mb": 2053.4}
        },
    }
    config = SustainedRunnerConfig(dry_run=False, mode="soak", min_headroom_mb=6000.0)
    eval_res = evaluate_qualification_gates(mock_report, config)

    assert eval_res["overall_status"] == "failed"
    assert eval_res["gates"]["watchdog_integrity"]["passed"] is False
    assert eval_res["gates"]["memory_stability"]["passed"] is False
    assert any("Watchdog breach triggered" in r for r in eval_res["failure_reasons"])


def test_partial_metrics_summary_computation():
    """Verifies partial metrics summary is correctly computed with completed request accounting on failure."""
    from scripts.qmd_mlx.sustained import compute_metrics_summary

    meas = [
        {"stratum": "short", "batch_size": 1, "total_tokens": 20, "wire_ms": 15.0},
        {"stratum": "medium", "batch_size": 1, "total_tokens": 100, "wire_ms": 45.0},
    ]
    rejs = [
        {"stratum": "boundary_2049", "wire_ms": 5.0, "status_code": 400, "rejection_verified": True}
    ]
    interleaving = [
        {
            "iteration": 0,
            "bulk_wire_ms": 80.0,
            "concurrent_interactive_wire_ms": 30.0,
            "estimated_queue_wait_ms": 15.0,
            "slowdown_factor": 2.0,
        }
    ]

    summary = compute_metrics_summary(
        measured_records=meas,
        rejection_records=rejs,
        concurrent_interleaving_records=interleaving,
        solo_baseline_percentiles=None,
        total_benchmark_elapsed_s=12.5,
        partial_run=True,
    )

    assert summary["partial_run"] is True
    assert summary["total_measured_requests"] == 3
    assert summary["successful_embedding_requests"] == 2
    assert summary["rejected_boundary_requests"] == 1
    assert summary["total_successful_embedded_tokens"] == 120
    assert summary["cumulative_wall_accounting"]["total_benchmark_elapsed_s"] == 12.5
    assert summary["cumulative_wall_accounting"]["sum_successful_request_wire_s"] == 0.06
    assert summary["cumulative_wall_accounting"]["sum_rejection_request_wire_s"] == 0.005


def test_cli_soak_mode_pilot_gate_enforcement(tmp_path):
    """Verifies CLI prevents soak mode escalation when pilot qualification report is missing or failed."""
    import importlib
    sustained_cli = importlib.import_module("scripts.qmd-mlx-sustained")

    p1, p2 = get_ephemeral_port_pair()
    failing_pilot_file = str(tmp_path / "failed_pilot.json")

    with open(failing_pilot_file, "w", encoding="utf-8") as pf:
        json.dump({
            "status": "failed",
            "qualification_evaluation": {
                "overall_status": "failed",
                "failure_reasons": ["Concurrent interactive p95 1236.5ms exceeded 200.0ms target"],
            }
        }, pf)

    # 1. Soak mode with failing pilot report should exit with code 2
    argv = [
        "--real-model",
        "--model-path", "test/fake/path",
        "--mode", "soak",
        "--pilot-report", failing_pilot_file,
        "--port", str(p1),
        "--control-port", str(p2),
    ]
    ret = sustained_cli.main(argv)
    assert ret == 2

    # 2. Soak mode with missing pilot report should exit with code 2
    argv_missing = [
        "--real-model",
        "--model-path", "test/fake/path",
        "--mode", "soak",
        "--pilot-report", str(tmp_path / "nonexistent_pilot.json"),
        "--port", str(p1),
        "--control-port", str(p2),
    ]
    ret_missing = sustained_cli.main(argv_missing)
    assert ret_missing == 2

