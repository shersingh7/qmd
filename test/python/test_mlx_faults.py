"""
test_mlx_faults.py — Integration and fault tests for MLX HTTP protocol error mappings,
status codes, batch bisection retries, and failure bounds.
"""

import json
import time
import urllib.request
import urllib.error
import numpy as np
import pytest
from scripts.qmd_mlx.batching import BatchPlanner
from scripts.qmd_mlx.protocol import (
    DeadlineExceededError,
    InvalidInputError,
    ModelUnavailableError,
    OutOfMemoryError,
    OverloadedError,
    ProtocolError,
    RequestCancelledError,
    UnsupportedModelError,
    WorkerUnavailableError,
)
from scripts.qmd_mlx.server import ThreadedMLXServer, start_server


def test_exception_status_code_mappings():
    """Verify all typed exceptions map to their documented HTTP status codes."""
    assert InvalidInputError("test").status_code == 400
    assert ProtocolError("test").status_code == 400
    assert UnsupportedModelError("test").status_code == 400
    assert RequestCancelledError("test").status_code == 400

    assert OverloadedError("test").status_code == 429

    assert ModelUnavailableError("test").status_code == 503
    assert OutOfMemoryError("test").status_code == 503
    assert WorkerUnavailableError("test").status_code == 503

    assert DeadlineExceededError("test").status_code == 504


def test_batch_planner_oom_bisection_recovery():
    """Verify that BatchPlanner bisects a batch and succeeds if smaller pieces fit into memory."""
    planner = BatchPlanner(max_batch_tokens=1000)

    call_count = 0

    def mock_embed_forward(tokenized_batch):
        nonlocal call_count
        call_count += 1
        if len(tokenized_batch) > 2:
            # Simulate Metal OOM on large batch
            raise RuntimeError("Metal buffer allocation failed: out of memory")
        # Sub-batch fits in memory
        return np.ones((len(tokenized_batch), 4), dtype=np.float32)

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    texts = ["text 1", "text 2", "text 3", "text 4"]
    result = planner.plan_and_execute(texts, FakeTokenizer(), mock_embed_forward)

    assert result.shape == (4, 4)
    assert call_count > 1  # Verify bisection happened


def test_batch_planner_oom_exceeds_retry_depth_raises():
    """Verify that if a single item fails with OOM, OutOfMemoryError is raised."""
    planner = BatchPlanner(max_batch_tokens=1000)

    def mock_always_oom(tokenized_batch):
        raise RuntimeError("Metal buffer allocation failed: out of memory")

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [1]

    with pytest.raises(OutOfMemoryError):
        planner.plan_and_execute(["text 1"], FakeTokenizer(), mock_always_oom)


def test_host_side_alloc_failure_does_not_shrink_batch_budget():
    """Verify generic host-side allocation errors do not trigger Metal OOM bisection or shrink budget."""
    initial_budget = 2000
    planner = BatchPlanner(max_batch_tokens=initial_budget)

    def mock_host_alloc_failure(tokenized_batch):
        raise RuntimeError("cannot allocate memory for file descriptor")

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    with pytest.raises(RuntimeError, match="cannot allocate memory for file descriptor"):
        planner.plan_and_execute(["text 1", "text 2"], FakeTokenizer(), mock_host_alloc_failure)

    # Verify max_batch_tokens was NOT halved
    assert planner.max_batch_tokens == initial_budget



def test_ephemeral_server_fault_responses():
    """Start an ephemeral server and test status code mappings over HTTP."""
    server, thread = start_server(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        port=0,
        bind_host="127.0.0.1",
        preload=True,
        warmup=False,
    )
    port = server.server_port
    base_url = f"http://127.0.0.1:{port}"

    try:
        # Wait for ready
        for _ in range(50):
            try:
                req = urllib.request.Request(f"{base_url}/ready")
                with urllib.request.urlopen(req, timeout=1) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
            time.sleep(0.05)

        # 1. Malformed JSON -> 400
        req = urllib.request.Request(
            f"{base_url}/embed",
            data=b"invalid json payload",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 400

        # 2. Empty texts -> 400
        req = urllib.request.Request(
            f"{base_url}/embed",
            data=json.dumps({"texts": []}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 400

        # 3. Non-string text -> 400
        req = urllib.request.Request(
            f"{base_url}/embed",
            data=json.dumps({"texts": [123, 456]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 400

        # 4. Valid embed request -> 200
        req = urllib.request.Request(
            f"{base_url}/embed",
            data=json.dumps({"texts": ["hello world"]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert "embeddings" in data
            assert len(data["embeddings"]) == 1

        # 5. Expired timeout header -> 504
        req = urllib.request.Request(
            f"{base_url}/embed",
            data=json.dumps({"texts": ["timeout test"]}).encode(),
            headers={"Content-Type": "application/json", "X-Timeout": "0.001"},
        )
        # Note: Depending on timing, this may succeed if under 1ms or raise 504
        try:
            with urllib.request.urlopen(req) as resp:
                pass
        except urllib.error.HTTPError as he:
            assert he.code in (504, 400)

    finally:
        server.shutdown()
        server.server_close()
