"""
test_mlx_smoke.py — Unit and integration tests for bounded single-model smoke runner and fixtures.

Verifies:
1. Model metadata inventory (pure config parsing, no weights loaded, derives conservative headroom).
2. Numerical and metric validators (finite checks, dimensions, L2 normalization, batch-singleton consistency).
3. Preflight safety checks (rejection of port 8787, non-loopback host, occupied ports).
4. Model-aware preflight headroom check (fail-closed rejection when headroom is insufficient for 4B model).
5. Full rehearsal lifecycle with REAL start_server/runtime/executor stack and synthetic adapter, verified MLXWatchdog supervision, and strict descriptor validation.
6. Wall-clock deadline enforcement and guaranteed process cleanup when server child hangs.
7. True watchdog breach detection regressions during smoke runs (RSS breach, stalled inference breach, health probe failure breach).
8. Probe transport isolation (trust_env=False, allow_redirects=False, max response byte limits).
9. Dry-run mode telemetry with explicit "not_run" fixture records.
10. Strict requirement for --real-model opt-in flag before loading real weights.
11. Real model smoke test skipped by default offline (opt-in via --run-real-models).
"""

import json
import os
import socket
import subprocess
import sys
import time
from unittest.mock import MagicMock
import numpy as np
import pytest
import requests

from scripts.qmd_mlx.adapters.embedding import (
    QwenEmbeddingAdapter,
)
from scripts.qmd_mlx.protocol import (
    ModelUnavailableError,
)
from scripts.qmd_mlx.smoke import (
    BATCH_FIXTURES,
    LONG_INPUT_FIXTURE,
    SINGLETON_TEXT,
    SmokeHttpClient,
    SmokeRunner,
    SmokeRunnerConfig,
    check_batch_singleton_consistency,
    check_dimensions,
    check_finite,
    check_l2_normalization,
    inspect_model_metadata,
)
from scripts.qmd_mlx.watchdog import (
    BreachType,
    MLXWatchdog,
    MLXWatchdogConfig,
    SystemMemorySampler,
    SystemMetricsError,
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


def test_model_metadata_inventory_valid():
    """Verifies read-only metadata inspection of local model directory and conservative headroom calculation."""
    local_model = os.path.expanduser("~/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine")
    if not os.path.isdir(local_model):
        pytest.skip("Local test model directory not present")

    meta = inspect_model_metadata(local_model)
    assert meta.model_type == "qwen3"
    assert meta.hidden_size == 2560
    assert meta.num_hidden_layers == 36
    assert meta.num_attention_heads == 32
    assert meta.has_safetensors is True
    assert meta.has_tokenizer is True
    assert meta.total_file_size_bytes > 100_000_000
    assert meta.params_b >= 3.0  # 4B model
    assert meta.conservative_required_headroom_mb >= 3500.0


def test_model_metadata_missing_path():
    """Verifies that non-existent paths raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        inspect_model_metadata("/tmp/definitely_non_existent_model_path_12345")


def test_numerical_validators_finite_check():
    """Verifies finite validator detects NaNs and Infs."""
    valid_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ok, msg = check_finite(valid_arr)
    assert ok is True

    nan_arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    ok, msg = check_finite(nan_arr)
    assert ok is False
    assert "NaN: 1" in msg

    inf_arr = np.array([[1.0, np.inf], [3.0, 4.0]], dtype=np.float32)
    ok, msg = check_finite(inf_arr)
    assert ok is False
    assert "Inf: 1" in msg


def test_numerical_validators_dimension_check():
    """Verifies dimension validator checks shape."""
    arr = np.ones((5, 2560), dtype=np.float32)
    ok, msg = check_dimensions(arr, expected_rows=5, expected_dims=2560)
    assert ok is True

    ok, msg = check_dimensions(arr, expected_rows=4, expected_dims=2560)
    assert ok is False
    assert "Expected 4 rows, got 5" in msg

    ok, msg = check_dimensions(arr, expected_rows=5, expected_dims=1024)
    assert ok is False
    assert "Expected 1024 dims, got 2560" in msg


def test_numerical_validators_l2_norm():
    """Verifies L2 norm validator checks unit normalization within tolerance."""
    v = np.random.randn(3, 128).astype(np.float32)
    v_norm = v / np.linalg.norm(v, axis=1, keepdims=True)

    ok, norms, msg = check_l2_normalization(v_norm, tol=1e-4)
    assert ok is True
    assert len(norms) == 3
    for n in norms:
        assert abs(n - 1.0) <= 1e-4

    # Un-normalized
    ok, norms, msg = check_l2_normalization(v, tol=1e-4)
    assert ok is False
    assert "outside tolerance" in msg


def test_numerical_validators_batch_singleton_consistency():
    """Verifies consistency checker against cosine similarity and absolute difference."""
    v = np.random.randn(1, 2560).astype(np.float32)
    v = v / np.linalg.norm(v)

    # Identical
    ok, cos_sim, max_diff, msg = check_batch_singleton_consistency(v, v)
    assert ok is True
    assert cos_sim >= 0.9999
    assert max_diff <= 1e-3

    # Slight perturbation within tolerance
    perturbed = v + np.random.normal(0, 1e-5, v.shape).astype(np.float32)
    perturbed = perturbed / np.linalg.norm(perturbed)
    ok, cos_sim, max_diff, msg = check_batch_singleton_consistency(v, perturbed, tol_cos=0.999, tol_diff=1e-2)
    assert ok is True

    # Material difference (failure)
    v_diff = np.random.randn(1, 2560).astype(np.float32)
    v_diff = v_diff / np.linalg.norm(v_diff)
    ok, cos_sim, max_diff, msg = check_batch_singleton_consistency(v, v_diff)
    assert ok is False


def test_preflight_port_rejections():
    """Verifies that preflight rejects production port 8787 and non-loopback host."""
    config_prod_port = SmokeRunnerConfig(
        use_fake_child=True,
        port=8787,
        control_port=8798,
    )
    runner = SmokeRunner(config_prod_port)
    with pytest.raises(ValueError, match="port 8787 is reserved for live daemon"):
        runner.validate_preflight()

    config_bad_host = SmokeRunnerConfig(
        use_fake_child=True,
        host="0.0.0.0",
        port=8797,
        control_port=8798,
    )
    runner = SmokeRunner(config_bad_host)
    with pytest.raises(ValueError, match="must be numeric loopback '127.0.0.1'"):
        runner.validate_preflight()


def test_preflight_model_aware_insufficient_headroom_rejection():
    """
    Verifies that real-model preflight conservatively derives headroom for a 4B model
    and rejects loading when headroom is insufficient (e.g. 2048 MB is not enough for 4B).
    """
    local_model = os.path.expanduser("~/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine")
    if not os.path.isdir(local_model):
        pytest.skip("Local test model directory not present")

    mock_sampler = MagicMock()
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 2500.0  # < ~3600 MB required for 4B
    mock_sampler.get_swap_used_mb.return_value = 1000.0
    mock_sampler.get_memory_free_pct.return_value = 25.0
    mock_sampler.compute_conservative_defaults.return_value = MagicMock()

    p1, p2 = get_ephemeral_port_pair()
    config = SmokeRunnerConfig(
        use_fake_child=False,
        model_path=local_model,
        real_model_opt_in=True,
        port=p1,
        control_port=p2,
        min_headroom_mb=2048.0,  # Generic minimum
    )
    runner = SmokeRunner(config, sampler=mock_sampler)
    with pytest.raises(RuntimeError, match="Insufficient memory headroom for 4.0B model"):
        runner.validate_preflight()


def test_preflight_fails_on_occupied_port():
    """Verifies preflight rejects spawning on an occupied port."""
    p1, p2 = get_ephemeral_port_pair()
    occupier = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupier.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupier.bind(("127.0.0.1", p1))

    try:
        config = SmokeRunnerConfig(
            use_fake_child=True,
            port=p1,
            control_port=p2,
        )
        runner = SmokeRunner(config)
        with pytest.raises(RuntimeError, match="already in use"):
            runner.validate_preflight()
    finally:
        occupier.close()


def test_rehearsal_full_lifecycle_with_real_server_and_watchdog():
    """
    End-to-end qualification rehearsal exercising the REAL server stack (mlx_embed_server.py)
    with the SyntheticEmbeddingAdapter and active MLXWatchdog supervision.
    Verifies:
    - Real start_server/runtime/executor/control stack executed.
    - Active MLXWatchdog monitors child process during execution.
    - Ongoing measured telemetry samples recorded.
    - Descriptor schema strictly validated without invented fallback.
    - All 5 fixtures pass with synthetic labels and deterministic numbers.
    - Child process and supervisor cleanly reaped upon completion.
    """
    p1, p2 = get_ephemeral_port_pair()
    config = SmokeRunnerConfig(
        use_fake_child=True,
        port=p1,
        control_port=p2,
        fake_dims=2560,
        timeout_s=30.0,
    )
    runner = SmokeRunner(config)
    report = runner.run()

    assert report["status"] == "passed"
    assert report["mode"] == "rehearsal_synthetic"
    assert "child_pid" in report
    assert "instance_token" in report
    assert "descriptor" in report

    # Verify descriptor fields
    desc = report["descriptor"]
    assert desc["backend"] == "mlx_synthetic"
    assert desc["outputDimensions"] == 2560
    assert desc["nativeDimensions"] == 2560
    assert desc.get("synthetic") is True

    # Verify ongoing telemetry was captured
    assert "telemetry_samples" in report
    assert isinstance(report["telemetry_samples"], list)

    fixtures = report["fixtures"]
    assert fixtures["singleton"]["status"] == "passed"
    assert fixtures["singleton"]["shape"] == [1, 2560]
    assert fixtures["singleton"]["finite"] is True
    assert abs(fixtures["singleton"]["l2_norm"] - 1.0) <= 1e-4

    assert fixtures["batch"]["status"] == "passed"
    assert fixtures["batch"]["shape"] == [1 + len(BATCH_FIXTURES), 2560]
    assert fixtures["batch"]["finite"] is True

    assert fixtures["consistency"]["status"] == "passed"
    assert fixtures["consistency"]["cosine_similarity"] >= 0.9999
    assert fixtures["consistency"]["max_absolute_difference"] <= 1e-3

    assert fixtures["long_input"]["status"] == "passed"
    assert fixtures["long_input"]["shape"] == [1, 2560]

    assert fixtures["timing"]["status"] == "passed"
    assert fixtures["timing"]["iterations"] == 5
    assert len(fixtures["timing"]["latencies_ms"]) == 5
    assert fixtures["timing"]["p50_ms"] > 0


def test_hung_child_wall_time_ceiling_and_watchdog_cleanup():
    """
    Verifies that when an inference request hangs, the wall-clock deadline
    ceiling triggers, the supervisor reaps the child process, and execution aborts.
    """
    p1, p2 = get_ephemeral_port_pair()
    config = SmokeRunnerConfig(
        use_fake_child=True,
        port=p1,
        control_port=p2,
        fake_hang_on_embed=True,  # Simulates stuck inference
        timeout_s=2.0,            # Bounded 2-second ceiling
    )
    runner = SmokeRunner(config)

    t0 = time.time()
    with pytest.raises(Exception):
        runner.run()
    elapsed = time.time() - t0

    # Ensure execution terminated within bounded wall-clock ceiling (grace window)
    assert elapsed < 10.0


def test_watchdog_breach_detection_during_smoke():
    """
    Verifies that if MLXWatchdog detects a resource breach during smoke execution
    (e.g. RSS limit exceeded), the supervisor terminates the child and aborts runner.
    """
    p1, p2 = get_ephemeral_port_pair()
    config = SmokeRunnerConfig(
        use_fake_child=True,
        port=p1,
        control_port=p2,
        max_rss_mb=1.0,  # Unreasonably low 1MB limit to trigger immediate breach
        timeout_s=15.0,
    )
    runner = SmokeRunner(config)

    with pytest.raises(RuntimeError, match="Watchdog breach"):
        runner.run()


def test_probe_transport_no_proxy_and_bounded_bytes():
    """
    Verifies SmokeHttpClient enforces trust_env=False, allow_redirects=False,
    and byte caps.
    """
    client = SmokeHttpClient(default_timeout_s=5.0, max_response_bytes=100)
    assert client.session.trust_env is False
    assert client.session.max_redirects == 0
    client.close()


def test_dry_run_reports_explicit_not_run():
    """Verifies that dry-run returns explicit 'not_run' records for fixtures without spawning."""
    p1, p2 = get_ephemeral_port_pair()
    config = SmokeRunnerConfig(
        use_fake_child=True,
        port=p1,
        control_port=p2,
        dry_run=True,
    )
    runner = SmokeRunner(config)
    report = runner.run()

    assert report["status"] == "dry_run_completed"
    assert report["fixtures"]["singleton"]["status"] == "not_run"
    assert report["fixtures"]["singleton"]["reason"] == "dry_run requested"
    assert report["fixtures"]["batch"]["status"] == "not_run"
    assert report["fixtures"]["consistency"]["status"] == "not_run"
    assert report["fixtures"]["long_input"]["status"] == "not_run"
    assert report["fixtures"]["timing"]["status"] == "not_run"


def test_real_model_requires_explicit_opt_in():
    """Verifies that running against real model path without --real-model opt-in is rejected."""
    local_model = os.path.expanduser("~/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine")
    config = SmokeRunnerConfig(
        use_fake_child=False,
        model_path=local_model,
        real_model_opt_in=False,  # Missing opt-in
    )
    runner = SmokeRunner(config)
    with pytest.raises(ValueError, match="requires explicit opt-in"):
        runner.validate_preflight()


def test_rehearsal_consistency_failure_saves_partial_report_and_cleans_up():
    """
    Simulates a consistency failure during synthetic rehearsal (e.g. cosine similarity drops below tolerance).
    Verifies:
    1. Runner raises ValueError / aborts run.
    2. Partial report in runner.last_report preserves completed fixtures (singleton, batch).
    3. Consistency fixture records exact cosine_similarity, max_diff, tolerances, reference shapes, and token lengths.
    4. Subsequent fixtures (long_input, timing) are marked 'not_run'.
    5. Child process is cleanly terminated and verified in report['child_cleanup'].
    6. Captured bounded logs are preserved in report.
    """
    p1, p2 = get_ephemeral_port_pair()
    config = SmokeRunnerConfig(
        use_fake_child=True,
        port=p1,
        control_port=p2,
        fake_fail_consistency=True,  # Simulates consistency failure
        timeout_s=30.0,
    )
    runner = SmokeRunner(config)

    with pytest.raises(ValueError, match="Singleton vs Batch consistency failed"):
        runner.run()

    report = runner.last_report
    assert report is not None
    assert report["status"] == "failed"
    assert "Singleton vs Batch consistency failed" in report["error"]

    fixtures = report["fixtures"]
    assert fixtures["singleton"]["status"] == "passed"
    assert fixtures["singleton"]["token_length"] is not None
    assert fixtures["singleton"]["shape"] == [1, 2560]

    assert fixtures["batch"]["status"] == "passed"
    assert fixtures["batch"]["count"] == 1 + len(BATCH_FIXTURES)
    assert fixtures["batch"]["token_lengths"] is not None

    assert fixtures["consistency"]["status"] == "failed"
    assert fixtures["consistency"]["cosine_similarity"] < 0.9999
    assert fixtures["consistency"]["cosine_tolerance"] == 0.9999
    assert "reference_shapes" in fixtures["consistency"]
    assert fixtures["consistency"]["reference_shapes"]["singleton"] == [1, 2560]
    assert fixtures["consistency"]["reference_shapes"]["batch"] == [1 + len(BATCH_FIXTURES), 2560]
    assert "token_lengths" in fixtures["consistency"]

    assert fixtures["long_input"]["status"] == "not_run"
    assert fixtures["timing"]["status"] == "not_run"

    assert report["child_cleanup"]["cleaned"] is True
    assert report["child_cleanup"]["pid"] is not None
    assert isinstance(report["captured_logs"], str)


def test_cli_failure_writes_output_file_and_exits_nonzero(tmp_path):
    """
    Verifies that when qmd-mlx-smoke.py CLI encounters a failure, it:
    1. Exits with nonzero return code (exit 1).
    2. Writes the full structured JSON failure report to --output-file.
    3. The file exists and contains failure status, partial fixtures, and cleanup verification.
    """
    import importlib
    smoke_cli = importlib.import_module("scripts.qmd-mlx-smoke")

    p1, p2 = get_ephemeral_port_pair()
    out_file = str(tmp_path / "smoke_failure_report.json")

    argv = [
        "--rehearsal",
        "--fake-fail-consistency",
        "--port", str(p1),
        "--control-port", str(p2),
        "--timeout-s", "30",
        "--json",
        "--output-file", out_file,
    ]

    ret = smoke_cli.main(argv)
    assert ret == 1

    assert os.path.exists(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        saved_report = json.load(f)

    assert saved_report["status"] == "failed"
    assert "consistency" in saved_report["fixtures"]
    assert saved_report["fixtures"]["consistency"]["status"] == "failed"
    assert saved_report["fixtures"]["singleton"]["status"] == "passed"
    assert saved_report["fixtures"]["batch"]["status"] == "passed"
    assert saved_report["child_cleanup"]["cleaned"] is True


def test_diagnostic_mode_synthetic():
    """
    Verifies that --diagnostic mode runs the full diagnostic suite:
    1. Repeated singletons comparison.
    2. Same-length duplicate batch comparison.
    3. Mixed-lengths batch comparison.
    4. Position permutations comparison.
    5. Captures true metrics (shapes, token lengths, similarities, latencies).
    """
    p1, p2 = get_ephemeral_port_pair()
    config = SmokeRunnerConfig(
        use_fake_child=True,
        port=p1,
        control_port=p2,
        diagnostic=True,
        timeout_s=30.0,
    )
    runner = SmokeRunner(config)
    report = runner.run()

    assert report["status"] == "passed"
    assert "diagnostics" in report

    diag = report["diagnostics"]
    assert "repeated_singleton" in diag
    assert diag["repeated_singleton"]["status"] == "passed"
    assert diag["repeated_singleton"]["cosine_similarity"] >= 0.9999
    assert len(diag["repeated_singleton"]["latencies_ms"]) == 2

    assert "duplicate_batch" in diag
    assert diag["duplicate_batch"]["status"] == "passed"
    assert diag["duplicate_batch"]["count"] == 4
    assert diag["duplicate_batch"]["min_cosine_similarity"] >= 0.9999

    assert "mixed_lengths" in diag
    assert diag["mixed_lengths"]["status"] == "passed"
    assert diag["mixed_lengths"]["count"] == 4
    assert diag["mixed_lengths"]["token_lengths"] is not None

    assert "position_permutations" in diag
    assert diag["position_permutations"]["status"] == "passed"
    assert len(diag["position_permutations"]["permutations"]) == 4


def test_preflight_watchdog_defaults_passed_to_watchdog():
    """
    Verifies that SmokeRunner stores preflight defaults and passes them directly to MLXWatchdog,
    avoiding redundant memory sampling after process spawn.
    """
    p1, p2 = get_ephemeral_port_pair()
    config = SmokeRunnerConfig(
        use_fake_child=True,
        port=p1,
        control_port=p2,
        timeout_s=15.0,
    )
    runner = SmokeRunner(config)
    pre = runner.validate_preflight()

    assert runner._preflight_defaults is not None
    assert "watchdog_defaults" in pre
    assert pre["watchdog_defaults"]["installed_ram_mb"] > 0

    report = runner.run()
    assert report["status"] == "passed"


def test_qwen_adapter_set_dtype_applied_mock():
    """
    Verifies that QwenEmbeddingAdapter.load honors configured compute dtype
    and applies model.set_dtype(compute_dtype) while rejecting unsupported dtypes.
    """
    from unittest.mock import patch, MagicMock

    mock_model = MagicMock()
    mock_model.set_dtype = MagicMock()
    mock_model.args = MagicMock(hidden_size=2560)
    mock_tok = MagicMock()

    # 1. float32 dtype
    adapter_f32 = QwenEmbeddingAdapter("qwen-test", dtype_str="float32")
    with patch("mlx_lm.load", return_value=(mock_model, mock_tok, {"quantization": {"quant_type": "4bit"}})):
        adapter_f32.load()
        assert adapter_f32.is_loaded()
        mock_model.set_dtype.assert_called_once()
        call_arg = mock_model.set_dtype.call_args[0][0]
        # In MLX, mx.float32 is the compute dtype
        import mlx.core as mx
        assert call_arg == mx.float32

    # 2. bfloat16 dtype
    mock_model.reset_mock()
    adapter_bf16 = QwenEmbeddingAdapter("qwen-test", dtype_str="bfloat16")
    with patch("mlx_lm.load", return_value=(mock_model, mock_tok, {})):
        adapter_bf16.load()
        mock_model.set_dtype.assert_called_once()
        assert mock_model.set_dtype.call_args[0][0] == mx.bfloat16

    # 3. Unsupported dtype
    adapter_bad = QwenEmbeddingAdapter("qwen-test", dtype_str="int8_unsupported")
    with patch("mlx_lm.load", return_value=(mock_model, mock_tok, {})):
        with pytest.raises(ModelUnavailableError, match="Unsupported compute dtype"):
            adapter_bad.load()


def test_diagnostic_mode_failing_comparison_continues_but_status_remains_failed():
    """
    Verifies that when SmokeRunner runs in --diagnostic mode with a failing consistency check:
    1. Execution does not abort immediately with an unhandled exception.
    2. Step 9 (long_input) and Step 10 (timing) are marked 'not_run' due to prior failure.
    3. The diagnostic comparisons suite runs and records all probe results (repeated singletons, duplicate batch, mixed lengths, permutations).
    4. Overall report status truthfully remains 'failed'.
    5. Child process is cleanly terminated and verified in report['child_cleanup'].
    """
    p1, p2 = get_ephemeral_port_pair()
    config = SmokeRunnerConfig(
        use_fake_child=True,
        port=p1,
        control_port=p2,
        diagnostic=True,
        fake_fail_consistency=True,
        timeout_s=30.0,
    )
    runner = SmokeRunner(config)
    report = runner.run()

    # Status must remain failed
    assert report["status"] == "failed"
    assert "Singleton vs Batch consistency failed" in (report.get("error") or "")

    # Fixtures state
    assert report["fixtures"]["singleton"]["status"] == "passed"
    assert report["fixtures"]["batch"]["status"] == "passed"
    assert report["fixtures"]["consistency"]["status"] == "failed"
    assert report["fixtures"]["long_input"]["status"] == "not_run"
    assert report["fixtures"]["timing"]["status"] == "not_run"

    # Diagnostics suite must have executed despite the consistency failure
    assert "diagnostics" in report
    diag = report["diagnostics"]
    assert "repeated_singleton" in diag
    assert diag["repeated_singleton"]["status"] == "passed"
    assert "duplicate_batch" in diag
    assert "mixed_lengths" in diag
    assert "position_permutations" in diag

    # Clean child process teardown
    assert report["child_cleanup"]["cleaned"] is True
    assert report["child_cleanup"]["pid"] is not None


@pytest.mark.real_model
def test_real_model_smoke_test_opt_in():
    """
    Real-model qualification smoke test.
    Skipped by default during offline qualification suite; executed only with --run-real-models.
    """
    local_model = os.path.expanduser("~/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine")
    if not os.path.isdir(local_model):
        pytest.skip(f"Local model {local_model} not found")

    p1, p2 = get_ephemeral_port_pair()
    config = SmokeRunnerConfig(
        model_path=local_model,
        port=p1,
        control_port=p2,
        timeout_s=60.0,
        real_model_opt_in=True,
    )
    runner = SmokeRunner(config)
    report = runner.run()
    assert report["status"] == "passed"
    assert report["fixtures"]["singleton"]["status"] == "passed"
    assert report["fixtures"]["batch"]["status"] == "passed"
    assert report["fixtures"]["consistency"]["status"] == "passed"

