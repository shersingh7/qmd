"""
test_gguf_benchmark.py — Unit and integration tests for GGUF benchmark runner and harness.
"""

import os
import socket
import pytest

from scripts.qmd_gguf.benchmark import (
    GGUFBenchmarkConfig,
    GGUFBenchmarkRunner,
    compute_code_manifest,
    inspect_gguf_metadata,
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


def test_gguf_code_manifest():
    """Verifies that code manifest captures GGUF benchmark scripts."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    manifest = compute_code_manifest(repo_root)
    assert "scripts/gguf_embed_server.mjs" in manifest
    assert "scripts/qmd_gguf/benchmark.py" in manifest


def test_gguf_metadata_inspection(tmp_path):
    """Verifies metadata extraction from GGUF filename and size."""
    dummy_gguf = tmp_path / "hf_Qwen_Qwen3-Embedding-4B-Q4_K_M.gguf"
    dummy_gguf.write_bytes(b"GGUF" + b"\x00" * 1024)

    meta = inspect_gguf_metadata(str(dummy_gguf))
    assert meta["params_b"] == 4.0
    assert meta["quantization"] == "Q4_K_M"
    assert meta["native_dimensions"] == 2560
    assert meta["conservative_required_headroom_mb"] >= 6000.0


def test_gguf_rehearsal_pilot_lifecycle_and_strata():
    """
    Executes full offline rehearsal of GGUF sustained pilot mode with synthetic Node child server.
    Verifies:
    - Node server child spawned and supervised by Watchdog.
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
    config = GGUFBenchmarkConfig(
        use_fake_child=True,
        port=p1,
        control_port=p2,
        mode="pilot",
        max_requests=25,
        timeout_s=30.0,
    )
    runner = GGUFBenchmarkRunner(config)
    report = runner.run()

    assert report["status"] == "passed"
    assert report["backend"] == "gguf"
    assert report["mode"] == "sustained_pilot"
    assert report["synthetic"] is True
    assert "code_manifest_sha256" in report
    assert "scripts/gguf_embed_server.mjs" in report["code_manifest_sha256"]

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

    # Concurrent load interleaving
    inter = report["concurrent_load_interleaving"]
    assert len(inter) >= 1
    for item in inter:
        assert item["overlap_verified"] is True
        assert item["bulk_wire_ms"] > 0
        assert item["concurrent_interactive_wire_ms"] > 0
        assert item["slowdown_factor"] >= 0.0

    # Summary
    summary = report["metrics_summary"]
    assert summary["total_measured_requests"] == len(reqs) + len(rejs)
    assert summary["successful_embedding_requests"] == len(reqs)
    assert summary["rejected_boundary_requests"] == len(rejs)
    assert summary["total_successful_embedded_tokens"] > 0
    assert summary["overall_effective_tokens_per_sec"] > 0
    assert summary["cumulative_wall_accounting"]["total_benchmark_elapsed_s"] > 0

    # Qualification evaluation
    qual = report["qualification_evaluation"]
    assert qual["overall_status"] == "passed"
    assert len(qual["failure_reasons"]) == 0

    # Child cleanup
    cleanup = report["child_cleanup"]
    assert cleanup["cleaned"] is True
    assert cleanup["pid"] is not None
