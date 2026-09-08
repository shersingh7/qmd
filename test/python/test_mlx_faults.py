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
    """Start an ephemeral server and test status code mappings over HTTP completely offline."""
    from unittest.mock import MagicMock
    from scripts.qmd_mlx.server import ThreadedMLXServer, MLXHTTPRequestHandler, ServerState
    from scripts.qmd_mlx.protocol import DeadlineExceededError

    server = ThreadedMLXServer(("127.0.0.1", 0), MLXHTTPRequestHandler)

    mock_executor = MagicMock()
    mock_executor.is_alive.return_value = True
    server.executor = mock_executor

    mock_runtime = MagicMock()
    mock_runtime.model_name = "mock-embedding-model"
    mock_runtime.native_dims = 384
    mock_runtime.get_descriptor.return_value = {
        "version": 1,
        "backend": "mlx",
        "model": "mock-embedding-model",
        "nativeDimensions": 384,
        "outputDimensions": 384,
        "normalized": True,
    }

    def mock_submit_embed(texts, requested_dims=None, is_query=False, timeout=300.0, cancel_event=None, deadline=None):
        if timeout is not None and timeout <= 0.1:
            raise DeadlineExceededError(f"Request deadline expired (timeout={timeout})")
        if deadline is not None and time.monotonic() > deadline:
            raise DeadlineExceededError("Request deadline exceeded")
        arr = np.ones((len(texts), requested_dims or 384), dtype=np.float32)
        # Normalize
        arr = arr / np.linalg.norm(arr, axis=-1, keepdims=True)
        return arr

    mock_runtime.submit_embed.side_effect = mock_submit_embed
    server.runtime = mock_runtime
    server.state = ServerState.READY

    import threading
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

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

        # 5. Expired / short timeout header -> exact 504 DeadlineExceededError
        req = urllib.request.Request(
            f"{base_url}/embed",
            data=json.dumps({"texts": ["timeout test"]}).encode(),
            headers={"Content-Type": "application/json", "X-Request-Timeout": "0.001"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 504
        body = json.loads(exc_info.value.read().decode())
        assert body.get("type") == "deadline_exceeded"

    finally:
        server.shutdown()
        server.server_close()
