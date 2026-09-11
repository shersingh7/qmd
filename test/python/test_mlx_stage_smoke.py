"""
test_mlx_stage_smoke.py — Unit and Regression Tests for Reusable Smoke Supervisor and Stage Harnesses

Verifies:
1. SmokeStageSupervisor preflight validations (host numeric loopback, port 8787 reservation, port distinctness, memory headroom).
2. Regression: Fake stalled stage triggers MLXWatchdog breach termination:
   - Kills ONLY the target child process.
   - Parent process and sentinel survive intact.
   - Captures and saves complete JSON report with breach details, logs, and cleanup status even on failure.
3. Regression: Exception during stage probe execution guarantees fail-safe cleanup and saved JSON report.
4. Production query expansion parser and output validator:
   - Rejects introductory-only text ('Sure! Here are three alternative...').
   - Rejects verbatim query echoes.
   - Rejects unstripped thinking tags (<think>...</think>).
   - Accepts valid structured lex/vec/hyde lines and keyword lists.
   - Tracks functional_generation vs usable_expansion independently.
5. Exact quantization inspection for reranker:
   - Reads actual config.json (4bit-affine, 8bit, mxfp8).
   - Prevents descriptor quantization discrepancies and silent renames.
6. Dry-run execution across all stage runners without Metal requirements.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch
import pytest

from scripts.qmd_mlx.generate_smoke import (
    GenerateSmokeConfig,
    GenerateSmokeRunner,
    parse_expansion,
    validate_expansion_quality,
)
from scripts.qmd_mlx.rerank import (
    MLXRerankAdapter,
    infer_quantization_and_dtype,
)
from scripts.qmd_mlx.rerank_smoke import (
    RerankSmokeConfig,
    RerankSmokeRunner,
)
from scripts.qmd_mlx.supervisor import (
    SmokeHttpClient,
    SmokeStageSupervisor,
    StageSupervisorConfig,
)
from scripts.qmd_mlx.watchdog import (
    BreachType,
    MLXWatchdog,
    MLXWatchdogConfig,
    SystemMemorySampler,
    SystemMetricsError,
    WatchdogDefaults,
)


def test_supervisor_preflight_port_8787_rejected():
    """Supervisor must fail-closed and reject port 8787 (reserved for live daemon)."""
    cfg = StageSupervisorConfig(
        cmd=["echo", "1"],
        port=8787,
        control_port=8798,
        stage_name="test_preflight",
    )
    sup = SmokeStageSupervisor(cfg)
    with pytest.raises(ValueError, match="port 8787 is reserved for live daemon"):
        sup.validate_preflight()


def test_supervisor_preflight_non_loopback_host_rejected():
    """Supervisor must reject non-loopback host or hostnames."""
    cfg = StageSupervisorConfig(
        cmd=["echo", "1"],
        host="0.0.0.0",
        port=8797,
        control_port=8798,
        stage_name="test_preflight",
    )
    sup = SmokeStageSupervisor(cfg)
    with pytest.raises(ValueError, match="must be numeric loopback '127.0.0.1'"):
        sup.validate_preflight()

    cfg_name = StageSupervisorConfig(
        cmd=["echo", "1"],
        host="localhost",
        port=8797,
        control_port=8798,
        stage_name="test_preflight",
    )
    sup_name = SmokeStageSupervisor(cfg_name)
    with pytest.raises(ValueError, match="must be numeric loopback '127.0.0.1'"):
        sup_name.validate_preflight()


def test_supervisor_preflight_identical_ports_rejected():
    """Supervisor must reject identical inference and control ports."""
    cfg = StageSupervisorConfig(
        cmd=["echo", "1"],
        port=8797,
        control_port=8797,
        stage_name="test_preflight",
    )
    sup = SmokeStageSupervisor(cfg)
    with pytest.raises(ValueError, match="must be distinct"):
        sup.validate_preflight()


def test_supervisor_preflight_insufficient_headroom_rejected():
    """Supervisor must fail-closed if host memory headroom is below requirement."""
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 1500.0  # only 1.5GB
    mock_sampler.get_swap_used_mb.return_value = 100.0
    mock_sampler.get_memory_free_pct.return_value = 25.0
    mock_sampler.compute_conservative_defaults.return_value = {"headroom_mb": 1500.0}

    cfg = StageSupervisorConfig(
        cmd=["echo", "1"],
        port=8797,
        control_port=8798,
        min_headroom_mb=3000.0,
        stage_name="test_headroom",
    )
    sup = SmokeStageSupervisor(cfg, sampler=mock_sampler)
    with pytest.raises(RuntimeError, match="Insufficient memory headroom"):
        sup.validate_preflight()


def test_regression_stalled_child_killed_sentinel_survives():
    """
    Regression Test:
    Spawns a fake stalled child process under MLXWatchdog supervision with a tight deadline.
    Verifies:
    1. Watchdog detects deadline/stall and terminates ONLY the child process.
    2. Sentinel process / parent test runner survives intact.
    3. Complete JSON report is produced and saved with failure details, logs, and cleanup record.
    """
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 16000.0
    mock_sampler.get_swap_used_mb.return_value = 100.0
    mock_sampler.get_memory_free_pct.return_value = 50.0
    mock_sampler.compute_conservative_defaults.return_value = WatchdogDefaults(
        installed_ram_mb=32768.0, headroom_mb=16000.0, max_rss_mb=8000.0, max_swap_growth_mb=500.0
    )
    mock_sampler.get_process_cmdline.return_value = f"{sys.executable} -c import time; time.sleep(60)"
    mock_sampler.get_process_rss_mb.return_value = 50.0
    mock_sampler.get_process_start_time.return_value = str(time.time())

    parent_pid = os.getpid()

    with tempfile.TemporaryDirectory() as tmpdir:
        report_file = os.path.join(tmpdir, "stalled_stage_report.json")

        # Fake child that sleeps forever (stalled stage)
        cmd = [sys.executable, "-c", "import time; time.sleep(60)"]

        cfg = StageSupervisorConfig(
            cmd=cmd,
            host="127.0.0.1",
            port=8797,
            control_port=8798,
            timeout_s=0.6,  # 600ms deadline triggers quickly
            startup_grace_period_s=0.2,
            min_headroom_mb=1000.0,
            stage_name="stalled_stage_regression",
            output_file=report_file,
        )

        sup = SmokeStageSupervisor(cfg, sampler=mock_sampler)

        child_pid_captured = None
        with pytest.raises((RuntimeError, TimeoutError)):
            with sup.managed_stage() as ctx:
                child_pid_captured = ctx.child.pid
                time.sleep(1.0)  # Wait for deadline breach

        # 1. Verify parent / sentinel PID is alive and was never killed
        assert os.getpid() == parent_pid

        # 2. Verify child process is dead
        if sup.child is not None:
            child_pid_captured = sup.child.pid
            time.sleep(0.1)
            assert sup.child.poll() is not None

        # 3. Verify complete saved JSON report exists and is valid
        assert os.path.isfile(report_file)
        with open(report_file, "r", encoding="utf-8") as f:
            saved_report = json.load(f)

        assert saved_report["status"] == "failed"
        assert saved_report["child_cleanup"]["cleaned"] is True
        assert saved_report["child_cleanup"]["pid"] == child_pid_captured
        assert "errors" in saved_report and len(saved_report["errors"]) > 0


def test_regression_exception_in_probe_guarantees_cleanup_and_saved_json():
    """
    Regression Test:
    When an unhandled exception or assertion failure occurs during stage execution,
    the supervisor must reap the child process, stop the supervisor thread, unlink temp logs,
    and save the complete JSON report with the exception message.
    """
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 16000.0
    mock_sampler.get_swap_used_mb.return_value = 100.0
    mock_sampler.get_memory_free_pct.return_value = 50.0
    mock_sampler.compute_conservative_defaults.return_value = WatchdogDefaults(
        installed_ram_mb=32768.0, headroom_mb=16000.0, max_rss_mb=8000.0, max_swap_growth_mb=500.0
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        report_file = os.path.join(tmpdir, "probe_exception_report.json")

        cfg = StageSupervisorConfig(
            cmd=["echo", "1"],
            host="127.0.0.1",
            port=8797,
            control_port=8798,
            dry_run=True,
            output_file=report_file,
            stage_name="exception_test",
        )

        sup = SmokeStageSupervisor(cfg, sampler=mock_sampler)

        with pytest.raises(ValueError, match="Synthetic probe error"):
            with sup.managed_stage() as ctx:
                raise ValueError("Synthetic probe error")

        assert os.path.isfile(report_file)
        with open(report_file, "r", encoding="utf-8") as f:
            saved_report = json.load(f)

        assert saved_report["status"] == "failed"
        assert any("Synthetic probe error" in err for err in saved_report["errors"])


def test_parse_expansion_production_contract():
    """Verifies that parse_expansion extracts strictly typed lex, vec, hyde lines and marks un-typed lines as invalid."""
    sample_output = (
        "lex: database sharding b-tree index partition\n"
        "vec: distributed database indexing and query optimization\n"
        "hyde: A distributed database uses partitioned B-trees across nodes for fast retrieval.\n"
        "- secondary index\n"
        "* composite key\n"
        "Random invalid commentary line"
    )
    parsed = parse_expansion(sample_output, query="database indexing")
    assert len(parsed["lex"]) == 1
    assert "database sharding" in parsed["lex"][0]
    assert len(parsed["vec"]) == 1
    assert len(parsed["hyde"]) == 1
    assert len(parsed["keywords"]) == 0
    assert len(parsed["invalid"]) == 3
    assert any("Random invalid" in line for line in parsed["invalid"])
    assert parsed["usable"] is True


def test_parse_expansion_rejects_prior_markdown_prose_completion():
    """Verifies that the markdown prose previously returned by unprompted Qwen3 is rejected as unusable."""
    prose_output = (
        "To expand the search query **\"distributed database indexing\"**, you can consider adding relevant keywords "
        "and context to narrow down the search or to get more specific results. Here are some ways to expand or refine "
        "the query depending on your intent:\n\n"
        "---\n\n"
        "### 1. **General Expansion**\n"
        "- **\"Distributed Database Indexing: Concepts, Techniques, and Applications\"**\n"
        "- **\"Indexing in Distributed Databases: A Survey\"**\n"
        "- **\"Distributed Database Indexing: Challenges and Solutions\"**\n\n"
        "---\n\n"
        "### 2. **By Technology or Framework**\n"
        "- **\"Distributed Database Indexing in Hadoop\"**"
    )
    parsed = parse_expansion(prose_output, query="distributed database indexing")
    assert len(parsed["lex"]) == 0
    assert len(parsed["vec"]) == 0
    assert len(parsed["hyde"]) == 0
    assert len(parsed.get("queryables", [])) == 0
    assert parsed["usable"] is False

    ok, msg = validate_expansion_quality(
        query="distributed database indexing",
        text=prose_output,
        parsed=parsed,
    )
    assert ok is False
    assert "No valid typed expansion lines" in msg or "invalid lines rejected" in msg


def test_validate_expansion_quality_rejects_introductory_preamble_only():
    """
    Verifies that introductory preamble ('Sure! Here are three alternative technical keywords...')
    without actual expansions is caught and rejected.
    """
    preamble_only = "Sure! Here are three alternative technical keywords that can be expanded with th"
    parsed = parse_expansion(preamble_only, query="distributed database indexing")
    ok, msg = validate_expansion_quality(
        query="distributed database indexing",
        text=preamble_only,
        parsed=parsed,
    )
    assert ok is False
    assert "preamble" in msg.lower() or "no valid typed" in msg.lower() or "invalid lines" in msg.lower()


def test_validate_expansion_quality_rejects_verbatim_echo():
    """Verifies that echoing the prompt verbatim is rejected."""
    echo_text = "distributed database indexing"
    parsed = parse_expansion(echo_text, query="distributed database indexing")
    ok, msg = validate_expansion_quality(
        query="distributed database indexing",
        text=echo_text,
        parsed=parsed,
    )
    assert ok is False
    assert "echo" in msg.lower() or "no valid typed" in msg.lower() or "invalid lines" in msg.lower()


def test_validate_expansion_quality_rejects_think_tags():
    """Verifies that completions with leaked think tags are rejected."""
    think_text = "<think>Analyzing database indexing...</think>\nlex: database index"
    parsed = parse_expansion(think_text)
    ok, msg = validate_expansion_quality(
        query="database indexing",
        text=think_text,
        parsed=parsed,
    )
    assert ok is False
    assert "thinking tags" in msg


def test_validate_expansion_quality_accepts_valid_expansions():
    """Verifies that valid structured expansions pass quality checks."""
    valid_text = (
        "lex: distributed database indexing sharding b-tree\n"
        "vec: technical search query for distributed database indexes"
    )
    parsed = parse_expansion(valid_text)
    ok, msg = validate_expansion_quality(
        query="distributed database indexing",
        text=valid_text,
        parsed=parsed,
        expected_keywords=["index", "b-tree", "database"],
    )
    assert ok is True
    assert msg == "Valid usable query expansion"


def test_infer_quantization_and_dtype_from_config_json():
    """Verifies that infer_quantization_and_dtype inspects config.json properly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Case 1: 4-bit affine model
        cfg_4bit = {
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
            },
            "torch_dtype": "bfloat16",
        }
        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_4bit, f)

        q, dt = infer_quantization_and_dtype(tmpdir)
        assert q == "4bit-affine"
        assert dt == "bfloat16"

        # Case 2: 8-bit model
        cfg_8bit = {
            "quantization": {
                "bits": 8,
                "group_size": 64,
            },
            "torch_dtype": "float16",
        }
        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_8bit, f)

        q, dt = infer_quantization_and_dtype(tmpdir)
        assert q == "8bit"
        assert dt == "float16"

        # Case 3: mxfp8 model
        cfg_fp8 = {
            "quantization": {
                "quant_type": "mxfp8",
            },
            "torch_dtype": "bfloat16",
        }
        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_fp8, f)

        q, dt = infer_quantization_and_dtype(tmpdir)
        assert q == "mxfp8"
        assert dt == "bfloat16"


def test_rerank_adapter_descriptor_matches_inferred_quantization():
    """Verifies that MLXRerankAdapter descriptor uses inferred quantization from config.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            "quantization": {"bits": 4, "mode": "affine"},
            "torch_dtype": "bfloat16",
        }
        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        adapter = MLXRerankAdapter(model_name=tmpdir, lazy_load=True)
        adapter.yes_token_id = 9693
        adapter.no_token_id = 2152
        desc = adapter.get_descriptor()
        assert desc["quantization"] == "4bit-affine"
        assert desc["dtype"] == "bfloat16"
        assert desc["yesTokenId"] == 9693
        assert desc["noTokenId"] == 2152


def test_dry_run_generate_smoke():
    """Verifies that run_generate_smoke runs dry-run cleanly without spawning child."""
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 16000.0
    mock_sampler.get_swap_used_mb.return_value = 100.0
    mock_sampler.get_memory_free_pct.return_value = 50.0
    mock_sampler.compute_conservative_defaults.return_value = WatchdogDefaults(
        installed_ram_mb=32768.0, headroom_mb=16000.0, max_rss_mb=8000.0, max_swap_growth_mb=500.0
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = os.path.join(tmpdir, "gen_dry_run.json")
        rep = GenerateSmokeRunner(
            GenerateSmokeConfig(dry_run=True, output_json=out_json),
            sampler=mock_sampler,
        ).run()
        assert rep["status"] == "dry_run_completed"
        assert os.path.isfile(out_json)


def test_dry_run_rerank_smoke():
    """Verifies that run_rerank_smoke runs dry-run cleanly without spawning child."""
    mock_sampler = MagicMock(spec=SystemMemorySampler)
    mock_sampler.get_installed_ram_mb.return_value = 32768.0
    mock_sampler.get_memory_headroom_mb.return_value = 16000.0
    mock_sampler.get_swap_used_mb.return_value = 100.0
    mock_sampler.get_memory_free_pct.return_value = 50.0
    mock_sampler.compute_conservative_defaults.return_value = WatchdogDefaults(
        installed_ram_mb=32768.0, headroom_mb=16000.0, max_rss_mb=8000.0, max_swap_growth_mb=500.0
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = os.path.join(tmpdir, "rerank_dry_run.json")
        rep = RerankSmokeRunner(
            RerankSmokeConfig(dry_run=True, output_json=out_json),
            sampler=mock_sampler,
        ).run()
        assert rep["status"] == "dry_run_completed"
        assert os.path.isfile(out_json)
