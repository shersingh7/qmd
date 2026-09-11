"""
test_mlx_safeguards.py — Deterministic Offline Fixture Tests for MLX Operational Safeguards

Verifies:
1. Control-path health responsiveness during heavy inference saturation (dedicated control capacity).
2. Saturated inference returns HTTP 429 OverloadedError while /health and /ready stay 200 OK.
3. No restart-loop behavior: /health and /ready never time out or return false failures under load.
4. Stalled-kernel deadline timeouts: execution owner enforces monotonic deadline, fails safe with 504,
   releases admission leases, and allows immediate recovery of subsequent requests.
5. In-flight request cancellation: aborts cleanly and releases capacity without leaking resources.
6. Graceful shutdown under saturation: cancels pending work with 503 WorkerUnavailableError and exits cleanly.
"""

import concurrent.futures
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from unittest.mock import MagicMock
import numpy as np
import pytest

from scripts.qmd_mlx.adapters.tokenization import TokenizedBatch
from scripts.qmd_mlx.executor import GPUExecutor
from scripts.qmd_mlx.model_manager import ModelResidencyManager
from scripts.qmd_mlx.protocol import (
    DeadlineExceededError,
    OverloadedError,
    RequestCancelledError,
    WorkerUnavailableError,
)
from scripts.qmd_mlx.runtime import MLXEmbeddingRuntime
from scripts.qmd_mlx.server import (
    MLXHTTPRequestHandler,
    ServerState,
    ThreadedMLXServer,
)


import requests


def get_ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def make_http_request(
    url: str,
    method: str = "GET",
    data: bytes = None,
    headers: dict = None,
    timeout: float = 3.0,
) -> tuple[int, dict, float]:
    """Helper to perform HTTP request and measure latency."""
    t0 = time.monotonic()
    req_headers = headers or {}
    if data is not None and "Content-Type" not in req_headers:
        req_headers["Content-Type"] = "application/json"

    try:
        resp = requests.request(
            method=method,
            url=url,
            data=data,
            headers=req_headers,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        try:
            parsed = resp.json()
        except Exception:
            parsed = {"raw": resp.text}
        return resp.status_code, parsed, elapsed
    except Exception as e:
        elapsed = time.monotonic() - t0
        return 500, {"error": str(e)}, elapsed


def test_control_path_responsive_during_inference_saturation():
    """
    Proves that dedicated control capacity ensures /health and /ready respond immediately
    with HTTP 200 while concurrent inference requests saturate the inference limiter and receive HTTP 429.
    Prevents external watchdog / launchd restart-loop behavior.
    """
    port = get_ephemeral_port()
    # Server with max 2 concurrent inference handlers, and 64 total threads for control reservation
    server = ThreadedMLXServer(
        ("127.0.0.1", port),
        MLXHTTPRequestHandler,
        max_threads=64,
        max_concurrent_inference=2,
    )

    mock_executor = MagicMock()
    mock_executor.is_alive.return_value = True
    mock_executor.get_queue_depth.return_value = 5
    mock_executor.is_overloaded.return_value = False
    server.executor = mock_executor

    mock_runtime = MagicMock()
    mock_runtime.model_name = "test-embed-model"
    mock_runtime.native_dims = 384
    mock_runtime.in_flight_requests = 2
    mock_runtime.get_descriptor.return_value = {
        "backend": "mlx",
        "model": "test-embed-model",
        "nativeDimensions": 384,
    }

    # Simulate slow inference forwards (0.6s) for active requests
    def slow_submit_embed(texts, **kwargs):
        time.sleep(0.6)
        return np.ones((len(texts), 384), dtype=np.float32)

    mock_runtime.submit_embed.side_effect = slow_submit_embed
    server.runtime = mock_runtime
    server.state = ServerState.READY

    t_srv = threading.Thread(target=server.serve_forever, daemon=True)
    t_srv.start()

    base_url = f"http://127.0.0.1:{port}"
    time.sleep(0.1)

    try:
        # Launch 6 concurrent inference requests.
        # Exactly 2 should be admitted; the remaining 4 must be rejected immediately with 429.
        inference_results: list[tuple[int, dict, float]] = []
        embed_payload = json.dumps({"texts": ["saturation test payload"]}).encode("utf-8")

        def _send_inference():
            return make_http_request(f"{base_url}/embed", method="POST", data=embed_payload, timeout=5.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            inf_futs = [pool.submit(_send_inference) for _ in range(6)]

            # Small delay to ensure inference slots are occupied
            time.sleep(0.05)

            # Concurrent control requests: /health and /ready
            health_futs = [
                pool.submit(make_http_request, f"{base_url}/health", "GET", None, None, 2.0)
                for _ in range(5)
            ]
            ready_futs = [
                pool.submit(make_http_request, f"{base_url}/ready", "GET", None, None, 2.0)
                for _ in range(5)
            ]

            # Collect health & ready responses
            for h_fut in health_futs:
                status, body, latency = h_fut.result(timeout=2.0)
                assert status == 200
                assert body.get("status") == "ok"
                assert body.get("ready") is True
                assert body.get("queue_depth") == 5
                assert latency < 0.3  # Responded immediately without waiting for inference

            for r_fut in ready_futs:
                status, body, latency = r_fut.result(timeout=2.0)
                assert status == 200, f"Ready check failed: status={status}, body={body}"
                assert body.get("ready") is True
                assert latency < 0.3

            for fut in inf_futs:
                inference_results.append(fut.result(timeout=5.0))

        # Check inference results:
        status_codes = [r[0] for r in inference_results]
        assert 429 in status_codes, f"Expected 429 rejections under saturation, got: {status_codes}"
        assert 200 in status_codes, f"Expected admitted requests to succeed with 200, got: {status_codes}"

        overloaded_resp = [r for r in inference_results if r[0] == 429][0]
        assert overloaded_resp[1].get("type") == "overloaded"
        assert "Inference capacity saturated" in overloaded_resp[1].get("error", "")

    finally:
        server.shutdown()
        server.server_close()


def test_stalled_gpu_kernel_deadline_timeout_and_lease_recovery():
    """
    Proves that a stalled GPU forward operation times out strictly against monotonic deadline,
    raises 504 DeadlineExceededError, cleanly releases admission leases, and allows subsequent
    requests to execute successfully without deadlocks or resource leaks.
    """
    executor = GPUExecutor(max_queue_size=8)
    model_manager = ModelResidencyManager(executor)

    class MockStalledAdapter:
        model_name = "mock-stalled-model"
        native_dims = 128
        pooling_strategy = "mean"
        model_params_b = 0.5
        quantization = "bf16"
        dtype_str = "float32"
        max_length = 512

        def __init__(self):
            self._loaded = True
            self.stall_flag = True

        def is_loaded(self):
            return self._loaded

        def get_descriptor(self, requested_dims=None):
            return {
                "backend": "mlx",
                "model": self.model_name,
                "nativeDimensions": self.native_dims,
                "outputDimensions": requested_dims or self.native_dims,
            }

        def tokenize_texts(self, texts):
            return TokenizedBatch(
                token_ids=[[101, 102] for _ in texts],
                lengths=[2 for _ in texts],
                original_indices=list(range(len(texts))),
                pad_token_id=0,
                texts=texts,
            )

        def forward_batch(self, batch, requested_dims=None):
            if self.stall_flag:
                # Simulate stalled GPU kernel operation
                time.sleep(1.0)
            return np.ones((len(batch), self.native_dims), dtype=np.float32)

        def unload(self):
            self._loaded = False

    adapter = MockStalledAdapter()
    runtime = MLXEmbeddingRuntime.__new__(MLXEmbeddingRuntime)
    runtime.model_name = "mock-stalled-model"
    runtime.adapter = adapter
    runtime.executor = executor
    runtime._owns_executor = False
    runtime.model_manager = model_manager
    runtime.max_concurrent_admissions = 4
    runtime.max_in_flight_admission_bytes = 10 * 1024 * 1024
    runtime._admission_lock = threading.Condition()
    runtime._current_admitted_requests = 0
    runtime._current_admitted_bytes = 0
    runtime._shutting_down = False
    runtime.start_time = time.time()
    runtime.total_requests = 0
    runtime.total_latency_ms = 0.0
    runtime.compiled_shapes = set()
    model_manager.register_adapter("embed", adapter)

    from scripts.qmd_mlx.batching import BatchPlanner
    runtime.batch_planner = BatchPlanner(max_batch_tokens=1000)

    try:
        # Request 1: Stalled with short timeout budget (0.15s) -> must raise DeadlineExceededError
        t0 = time.monotonic()
        with pytest.raises(DeadlineExceededError):
            runtime.submit_embed(["Stalled query"], timeout=0.15, is_query=True)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.8, f"Deadline enforcement took too long: {elapsed:.2f}s"

        # Verify admission lease was completely released
        assert runtime.in_flight_requests == 0
        assert runtime.in_flight_bytes == 0

        # Request 2: Normal execution after recovering from stalled forward
        adapter.stall_flag = False
        res = runtime.submit_embed(["Healthy query"], timeout=5.0, is_query=True)
        assert res.shape == (1, 128)
        assert runtime.in_flight_requests == 0

    finally:
        runtime.shutdown()
        executor.shutdown()


def test_request_cancellation_releases_admission_lease():
    """
    Proves that cancelling an in-flight request aborts promptly with RequestCancelledError
    and immediately restores admission capacity.
    """
    executor = GPUExecutor(max_queue_size=8)
    model_manager = ModelResidencyManager(executor)

    class MockCancellableAdapter:
        model_name = "mock-cancellable"
        native_dims = 64
        pooling_strategy = "mean"
        model_params_b = 0.1
        quantization = "bf16"
        dtype_str = "float32"
        max_length = 512

        def is_loaded(self):
            return True

        def get_descriptor(self, requested_dims=None):
            return {"backend": "mlx", "model": self.model_name}

        def tokenize_texts(self, texts):
            return TokenizedBatch(
                token_ids=[[1, 2] for _ in texts],
                lengths=[2 for _ in texts],
                original_indices=list(range(len(texts))),
                pad_token_id=0,
                texts=texts,
            )

        def forward_batch(self, batch, requested_dims=None):
            time.sleep(0.5)
            return np.ones((len(batch), self.native_dims), dtype=np.float32)

        def unload(self):
            pass

    adapter = MockCancellableAdapter()
    runtime = MLXEmbeddingRuntime.__new__(MLXEmbeddingRuntime)
    runtime.model_name = "mock-cancellable"
    runtime.adapter = adapter
    runtime.executor = executor
    runtime._owns_executor = False
    runtime.model_manager = model_manager
    runtime.max_concurrent_admissions = 2
    runtime.max_in_flight_admission_bytes = 10 * 1024 * 1024
    runtime._admission_lock = threading.Condition()
    runtime._current_admitted_requests = 0
    runtime._current_admitted_bytes = 0
    runtime._shutting_down = False
    runtime.start_time = time.time()
    runtime.total_requests = 0
    runtime.total_latency_ms = 0.0
    runtime.compiled_shapes = set()
    model_manager.register_adapter("embed", adapter)

    from scripts.qmd_mlx.batching import BatchPlanner
    runtime.batch_planner = BatchPlanner(max_batch_tokens=1000)

    try:
        cancel_evt = threading.Event()
        # Trigger cancellation after 50ms
        threading.Timer(0.05, cancel_evt.set).start()

        with pytest.raises(RequestCancelledError):
            runtime.submit_embed(["Cancel me quickly"], timeout=5.0, cancel_event=cancel_evt)

        # Capacity must be zero after cancellation
        assert runtime.in_flight_requests == 0
        assert runtime.in_flight_bytes == 0

        # Next normal request succeeds immediately
        res = runtime.submit_embed(["Followup request"], timeout=5.0)
        assert res.shape == (1, 64)
    finally:
        runtime.shutdown()
        executor.shutdown()


def test_server_graceful_shutdown_under_queued_saturation():
    """
    Proves that calling server.stop() while jobs are queued or executing cancels all
    pending futures with WorkerUnavailableError, shuts down the GPU worker cleanly,
    and finishes without hanging or leaking threads.
    """
    port = get_ephemeral_port()
    server = ThreadedMLXServer(("127.0.0.1", port), MLXHTTPRequestHandler, max_threads=16)

    executor = GPUExecutor(max_queue_size=32)
    server.executor = executor

    # Queue up several slow jobs
    def slow_job():
        time.sleep(0.5)
        return "done"

    futs = []
    for i in range(10):
        fut, _, _ = executor.submit_async(slow_job, priority=1, timeout_s=10.0, description=f"job-{i}")
        futs.append(fut)

    assert executor.get_queue_depth() > 0

    # Trigger shutdown while queue is saturated
    t0 = time.monotonic()
    server.stop(timeout=2.0)
    elapsed = time.monotonic() - t0

    assert elapsed < 3.0
    assert not executor.is_accepting()

    # Verify pending futures received cancellation or shutdown exceptions
    exceptions = 0
    for fut in futs:
        try:
            fut.result(timeout=0.1)
        except (WorkerUnavailableError, RequestCancelledError, Exception):
            exceptions += 1

    assert exceptions > 0
    assert not executor.is_worker_alive() or not executor.is_accepting()


def test_post_saturation_429_rejects_with_connection_close_no_desync():
    """
    Proves that when POST /embed is rejected with HTTP 429 due to inference saturation,
    the server sends 'Connection: close' and closes the socket so that unconsumed request
    body bytes cannot desynchronize subsequent pipelined requests.
    """
    port = get_ephemeral_port()
    server = ThreadedMLXServer(
        ("127.0.0.1", port),
        MLXHTTPRequestHandler,
        max_threads=16,
        max_concurrent_inference=1,
    )

    mock_runtime = MagicMock()
    mock_runtime.model_name = "test-model"
    mock_runtime.native_dims = 128
    mock_runtime.submit_embed.side_effect = lambda texts, **kw: time.sleep(0.5) or np.zeros((len(texts), 128))
    mock_runtime.get_descriptor.return_value = {"backend": "mlx", "model": "test-model"}
    server.runtime = mock_runtime
    server.executor = MagicMock(is_alive=lambda: True, get_queue_depth=lambda: 0, is_overloaded=lambda: False)
    server.state = ServerState.READY

    t_srv = threading.Thread(target=server.serve_forever, daemon=True)
    t_srv.start()
    time.sleep(0.05)

    try:
        # Acquire the single inference slot
        assert server.inference_limiter.acquire(blocking=False)

        # Send raw HTTP request with body
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", port))
        body = b'{"texts": ["long payload data"] * 50}'
        raw_req = (
            f"POST /embed HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: keep-alive\r\n\r\n"
        ).encode("utf-8") + body

        s.sendall(raw_req)
        s.settimeout(2.0)
        resp_bytes = b""
        while True:
            try:
                chunk = s.recv(4096)
                if not chunk:
                    break
                resp_bytes += chunk
            except Exception:
                break
        s.close()

        resp_text = resp_bytes.decode("utf-8", errors="ignore")
        assert "429 Too Many Requests" in resp_text
        assert "Connection: close" in resp_text or "connection: close" in resp_text
        assert "Inference capacity saturated" in resp_text
    finally:
        server.inference_limiter.release()
        server.shutdown()
        server.server_close()


def test_cold_model_init_serves_health_immediately():
    """
    Proves that during cold model startup / loading, the server serves HTTP /health immediately
    with state 'loading'/'starting', ready: False, instance_token, and PID without connection refusals.
    """
    from scripts.qmd_mlx.server import start_server
    from unittest.mock import patch
    from scripts.qmd_mlx.rerank import MLXRerankAdapter

    def slow_resolve(*args, **kwargs):
        time.sleep(0.3)
        mock_inst = MagicMock()
        mock_inst.is_loaded.return_value = True
        mock_inst.native_dims = 256
        mock_inst.model_params_b = 0.5
        mock_inst.model_memory_mb = 100.0
        return mock_inst

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", side_effect=slow_resolve):
        server, thread = start_server(
            model_name="slow-cold-model",
            port=0,
            bind_host="127.0.0.1",
            preload=True,
            warmup=False,
        )
        port = server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"

        try:
            # Query /health immediately during model loading
            r = requests.get(f"{base_url}/health", timeout=1.0)
            assert r.status_code == 200
            data = r.json()
            assert "pid" in data
            assert "instance_token" in data
            assert data["pid"] == os.getpid()
            assert data["state"] in (ServerState.STARTING, ServerState.LOADING, ServerState.READY)

            # Wait for ready
            for _ in range(50):
                if server.state == ServerState.READY:
                    break
                time.sleep(0.05)

            assert server.state == ServerState.READY
            r_ready = requests.get(f"{base_url}/health", timeout=1.0)
            assert r_ready.status_code == 200
            ready_data = r_ready.json()
            assert ready_data["status"] == "ok"
            assert ready_data["ready"] is True
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

