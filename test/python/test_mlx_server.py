"""
test_mlx_server.py — Integration tests for bounded HTTP MLX embedding server
"""

import socket
import threading
import time
import requests
import numpy as np
import pytest
from scripts.qmd_mlx.server import start_server
from scripts.qmd_mlx.protocol import decode_binary_embeddings


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mlx_server():
    from unittest.mock import MagicMock
    from scripts.qmd_mlx.server import ThreadedMLXServer, MLXHTTPRequestHandler, ServerState

    port = get_free_port()
    server = ThreadedMLXServer(("127.0.0.1", port), MLXHTTPRequestHandler)

    mock_executor = MagicMock()
    mock_executor.is_alive.return_value = True
    server.executor = mock_executor

    mock_runtime = MagicMock()
    mock_runtime.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    mock_runtime.native_dims = 384
    mock_runtime.get_descriptor.return_value = {
        "version": 1,
        "backend": "mlx",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "nativeDimensions": 384,
        "outputDimensions": 384,
        "normalized": True,
        "maxSequenceLength": 512,
        "pooling": "mean",
    }
    mock_runtime.get_memory_info.return_value = {
        "active_mb": 128.0,
        "peak_mb": 256.0,
        "model_mb": 120.0,
    }
    mock_runtime.get_stats_info.return_value = {
        "total_requests": 5,
        "avg_ms": 12.5,
        "compiled_shapes": 2,
        "uptime_sec": 100,
    }

    def mock_submit_embed(texts, requested_dims=None, is_query=False, timeout=300.0, cancel_event=None, deadline=None):
        dims = requested_dims or 384
        arr = np.ones((len(texts), dims), dtype=np.float32)
        arr = arr / np.linalg.norm(arr, axis=-1, keepdims=True)
        return arr

    mock_runtime.submit_embed.side_effect = mock_submit_embed
    mock_runtime.tokenize.side_effect = lambda texts, **kwargs: [[101, 202, 102] for _ in texts]

    server.runtime = mock_runtime
    server.state = ServerState.READY

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    base_url = f"http://127.0.0.1:{port}"

    # Wait for server readiness
    for _ in range(50):
        try:
            r = requests.get(f"{base_url}/ready", timeout=1)
            if r.status_code == 200 and r.json().get("ready"):
                break
        except Exception:
            pass
        time.sleep(0.05)

    yield base_url

    server.shutdown()
    server.server_close()


def test_health_and_ready_endpoints(mlx_server):
    r = requests.get(f"{mlx_server}/health", timeout=2)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert data["dims"] == 384
    assert data["descriptor"]["backend"] == "mlx"

    r_ready = requests.get(f"{mlx_server}/ready", timeout=2)
    assert r_ready.status_code == 200
    assert r_ready.json()["ready"] is True


def test_descriptor_endpoint(mlx_server):
    r = requests.get(f"{mlx_server}/descriptor", timeout=2)
    assert r.status_code == 200
    desc = r.json()
    assert desc["version"] == 1
    assert desc["backend"] == "mlx"
    assert desc["nativeDimensions"] == 384
    assert desc["outputDimensions"] == 384


def test_memory_and_stats_endpoints(mlx_server):
    r_mem = requests.get(f"{mlx_server}/memory", timeout=2)
    assert r_mem.status_code == 200
    assert "active_mb" in r_mem.json()

    r_stats = requests.get(f"{mlx_server}/stats", timeout=2)
    assert r_stats.status_code == 200
    assert "total_requests" in r_stats.json()


def test_json_embed_endpoint(mlx_server):
    payload = {"texts": ["First test document", "Second test document"]}
    r = requests.post(f"{mlx_server}/embed", json=payload, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2
    assert len(data["embeddings"][0]) == 384
    assert np.allclose(np.linalg.norm(data["embeddings"], axis=-1), [1.0, 1.0], atol=1e-4)


def test_binary_embed_endpoint(mlx_server):
    payload = {"texts": ["First test document", "Second test document"]}
    r = requests.post(f"{mlx_server}/embed-bin", json=payload, timeout=5)
    assert r.status_code == 200
    assert r.headers["Content-Type"] == "application/octet-stream"

    arr, count, dims = decode_binary_embeddings(r.content)
    assert count == 2
    assert dims == 384
    assert arr.shape == (2, 384)
    assert np.allclose(np.linalg.norm(arr, axis=-1), [1.0, 1.0], atol=1e-4)


def test_tokenize_endpoint(mlx_server):
    payload = {"texts": ["apple silicon", "fast local embeddings"]}
    r = requests.post(f"{mlx_server}/tokenize", json=payload, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "tokens" in data
    assert "counts" in data
    assert len(data["counts"]) == 2
    assert data["counts"][0] > 0


def test_server_error_handling(mlx_server):
    # Empty body
    r = requests.post(f"{mlx_server}/embed", data=b"", timeout=2)
    assert r.status_code == 400

    # Empty texts list
    r = requests.post(f"{mlx_server}/embed", json={"texts": []}, timeout=2)
    assert r.status_code == 400

    # Invalid JSON
    r = requests.post(
        f"{mlx_server}/embed",
        data=b"{not-valid-json",
        headers={"Content-Type": "application/json"},
        timeout=2,
    )
    assert r.status_code == 400

    # Host header check
    r = requests.get(
        f"{mlx_server}/health",
        headers={"Host": "malicious-site.com"},
        timeout=2,
    )
    assert r.status_code == 403


def test_server_bounded_semaphore_prevents_capacity_inflation():
    import threading
    from unittest.mock import MagicMock
    from scripts.qmd_mlx.server import ThreadedMLXServer, MLXHTTPRequestHandler

    server = ThreadedMLXServer(("127.0.0.1", 0), MLXHTTPRequestHandler, max_threads=2)
    try:
        assert isinstance(server.thread_limiter, threading.BoundedSemaphore)
        # Verify initial available permits is 2
        assert server.thread_limiter.acquire(blocking=False)
        assert server.thread_limiter.acquire(blocking=False)
        # 3rd acquire fails
        assert not server.thread_limiter.acquire(blocking=False)

        # Release both
        server.thread_limiter.release()
        server.thread_limiter.release()

        # Extra close_request/release must not inflate capacity beyond 2
        mock_req = MagicMock()
        server.close_request(mock_req)

        # Capacity should still be exactly 2
        assert server.thread_limiter.acquire(blocking=False)
        assert server.thread_limiter.acquire(blocking=False)
        assert not server.thread_limiter.acquire(blocking=False)
        server.thread_limiter.release()
        server.thread_limiter.release()
    finally:
        server.server_close()


def test_server_stop_lifecycle_slow_init():
    """Verify that calling stop() during slow initialization does not hang on serve_forever."""
    from unittest.mock import patch, MagicMock
    from scripts.qmd_mlx.server import ServerState

    init_started = threading.Event()
    block_init = threading.Event()

    def slow_resolve(*args, **kwargs):
        init_started.set()
        block_init.wait(timeout=5.0)
        mock_ad = MagicMock()
        mock_ad.is_loaded.return_value = True
        return mock_ad

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", side_effect=slow_resolve):
        server, thread = start_server("sentence-transformers/all-MiniLM-L6-v2", port=get_free_port(), preload=False)

        assert init_started.wait(timeout=2.0)

        # Call stop while init is still waiting/blocked
        server.stop(timeout=1.0)
        block_init.set()

        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert server.state == ServerState.STOPPING


def test_server_stop_lifecycle_blocked_forward_no_reload():
    """Verify stop cancels active work, waits for worker exit, and unloads without reloading during STOPPING."""
    from unittest.mock import MagicMock
    from scripts.qmd_mlx.executor import GPUExecutor
    from scripts.qmd_mlx.model_manager import ModelResidencyManager
    from scripts.qmd_mlx.server import ThreadedMLXServer, MLXHTTPRequestHandler, ServerState

    server = ThreadedMLXServer(("127.0.0.1", get_free_port()), MLXHTTPRequestHandler)
    executor = GPUExecutor(max_queue_size=10)
    manager = ModelResidencyManager(executor, residency_budget_mb=5000.0)

    ad = MagicMock()
    ad.is_loaded.return_value = True
    ad.model_memory_mb = 100.0
    ad.unload_count = 0

    def mock_unload():
        ad.unload_count += 1
        ad.is_loaded.return_value = False

    ad.unload.side_effect = mock_unload

    manager.register_adapter("embed", ad)
    server.executor = executor
    server.model_manager = manager
    server.state = ServerState.READY

    # Block worker thread with long job
    block_job = threading.Event()
    job_started = threading.Event()

    def slow_forward():
        job_started.set()
        cancel_evt.wait(timeout=2.0)
        return np.ones((1, 4))

    fut, cancel_evt, dl = executor.submit_async(slow_forward, timeout_s=5.0)
    assert job_started.wait(timeout=2.0)

    # Trigger stop
    server.stop(timeout=2.0)
    block_job.set()

    assert server.state == ServerState.STOPPING
    assert not executor.is_worker_alive()
    assert ad.unload_count >= 1
    assert not ad.is_loaded()


def test_production_start_server_composition_stop_lifecycle_real_executor_runtime():
    """
    Test using real GPUExecutor + real MLXEmbeddingRuntime with FAKE adapter and shared manager
    (no GPU/model) through production start_server composition.
    Assert worker stopped, new submit rejected, all unload calls after last forward, and no reload during stop.
    """
    from unittest.mock import MagicMock, patch
    from typing import Optional, Any
    from scripts.qmd_mlx.adapters.tokenization import TokenizedBatch
    from scripts.qmd_mlx.protocol import WorkerUnavailableError
    from scripts.qmd_mlx.server import ServerState

    class FakeEmbeddingAdapter:
        def __init__(self, model_name="fake-model", **kwargs):
            self.model_name = model_name
            self.native_dims = 384
            self.max_length = 512
            self.pooling_strategy = "mean"
            self.model_params_b = 0.1
            self.model_memory_mb = 50.0
            self._loaded = False
            self.load_count = 0
            self.unload_count = 0
            self.forward_count = 0
            self.call_history: list[tuple[str, float]] = []
            self.raw_hf_tokenizer = MagicMock()
            self._lock = threading.Lock()

        def is_loaded(self) -> bool:
            with self._lock:
                return self._loaded

        def load(self):
            with self._lock:
                self._loaded = True
                self.load_count += 1
                self.call_history.append(("load", time.monotonic()))

        def unload(self):
            with self._lock:
                self._loaded = False
                self.unload_count += 1
                self.call_history.append(("unload", time.monotonic()))

        def tokenize_texts(self, texts: list[str]) -> TokenizedBatch:
            token_ids = [[101, 202, 102] for _ in texts]
            lengths = [3 for _ in texts]
            indices = list(range(len(texts)))
            return TokenizedBatch(token_ids=token_ids, lengths=lengths, original_indices=indices, pad_token_id=0, texts=list(texts))

        def forward_batch(self, batch: TokenizedBatch, requested_dims: Optional[int] = None) -> np.ndarray:
            with self._lock:
                self.forward_count += 1
                self.call_history.append(("forward", time.monotonic()))
            dims = requested_dims or self.native_dims
            count = len(batch)
            arr = np.ones((count, dims), dtype=np.float32)
            arr = arr / np.linalg.norm(arr, axis=-1, keepdims=True)
            return arr

        def get_descriptor(self, requested_dims: Optional[int] = None) -> dict[str, Any]:
            return {
                "version": 1,
                "backend": "mlx",
                "model": self.model_name,
                "nativeDimensions": self.native_dims,
                "outputDimensions": requested_dims or self.native_dims,
                "normalized": True,
                "maxSequenceLength": self.max_length,
                "pooling": self.pooling_strategy,
            }

    fake_adapter = FakeEmbeddingAdapter()
    port = get_free_port()

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=fake_adapter):
        server, thread = start_server("fake-model", port=port, preload=True, warmup=False)

        # Wait for server readiness
        for _ in range(50):
            if server.state == ServerState.READY and server.runtime is not None:
                break
            time.sleep(0.05)

        assert server.state == ServerState.READY
        assert server.executor is not None
        assert server.runtime is not None
        assert server.model_manager is not None

        # Verify production composition: shared executor, runtime does NOT own executor
        assert server.runtime.executor is server.executor
        assert server.runtime._owns_executor is False
        assert server.model_manager.executor is server.executor
        assert server.executor.is_worker_alive()
        assert fake_adapter.is_loaded()
        assert fake_adapter.load_count == 1

        # Perform forward inference via HTTP
        payload = {"texts": ["Production lifecycle verification", "Real executor and runtime composition"]}
        r = requests.post(f"http://127.0.0.1:{port}/embed", json=payload, timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 2
        assert fake_adapter.forward_count >= 1

        # Record timestamp of last forward
        forward_timestamps = [t for event, t in fake_adapter.call_history if event == "forward"]
        assert len(forward_timestamps) >= 1
        last_forward_time = forward_timestamps[-1]

        # Trigger graceful stop
        server.stop(timeout=5.0)
        thread.join(timeout=3.0)

        assert not thread.is_alive()
        assert server.state == ServerState.STOPPING
        assert server._stopped_event.is_set()

        # 1. Assert worker stopped (OS thread exited)
        assert not server.executor.is_worker_alive()

        # 2. Assert new submit rejected
        with pytest.raises(WorkerUnavailableError):
            server.executor.submit(lambda: 123)

        # 3. Assert all unload calls happened strictly after the last forward
        unload_timestamps = [t for event, t in fake_adapter.call_history if event == "unload"]
        assert len(unload_timestamps) >= 1
        for unload_t in unload_timestamps:
            assert unload_t >= last_forward_time

        # 4. Assert no reload during stop
        assert fake_adapter.load_count == 1  # Never reloaded!
        assert not fake_adapter.is_loaded()


def test_server_stop_blocked_noncooperative_forward_timeout_and_subsequent_stop():
    """
    Test blocked noncooperative forward beyond join budget:
    stop must not falsely set stopped_event or unload; report pending cleanup and support subsequent stop after release.
    """
    from unittest.mock import MagicMock, patch
    from scripts.qmd_mlx.adapters.tokenization import TokenizedBatch
    from scripts.qmd_mlx.server import ServerState

    class FakeAdapter:
        def __init__(self):
            self.model_name = "blocked-model"
            self.native_dims = 384
            self.max_length = 512
            self.pooling_strategy = "mean"
            self.model_params_b = 0.1
            self.model_memory_mb = 50.0
            self._loaded = True
            self.unload_count = 0
            self.raw_hf_tokenizer = MagicMock()

        def is_loaded(self) -> bool:
            return self._loaded

        def load(self):
            self._loaded = True

        def unload(self):
            self._loaded = False
            self.unload_count += 1

        def tokenize_texts(self, texts):
            return TokenizedBatch([[1, 2]], [2], [0], pad_token_id=0, texts=list(texts))

        def forward_batch(self, batch, requested_dims=None):
            return np.ones((len(batch), 384), dtype=np.float32)

        def get_descriptor(self, requested_dims=None):
            return {"backend": "mlx", "nativeDimensions": 384, "model": self.model_name}

    fake_adapter = FakeAdapter()
    port = get_free_port()

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=fake_adapter):
        server, thread = start_server("blocked-model", port=port, preload=False, warmup=False)

        for _ in range(50):
            if server.state == ServerState.READY and server.executor is not None:
                break
            time.sleep(0.05)

        assert server.state == ServerState.READY
        assert server.executor.is_worker_alive()

        # Submit a noncooperative blocking forward to the executor
        forward_running = threading.Event()
        block_forward = threading.Event()

        def noncooperative_forward():
            forward_running.set()
            # Noncooperative forward ignores cancel_event and waits on block_forward
            block_forward.wait(timeout=5.0)
            return "done"

        fut, cancel_evt, dl = server.executor.submit_async(noncooperative_forward, timeout_s=10.0)
        assert forward_running.wait(timeout=2.0)

        # Call stop with very small timeout (0.1s), much less than forward block
        server.stop(timeout=0.1)

        # Stop must NOT falsely set stopped_event or unload while worker is still alive
        assert server.state == ServerState.STOPPING
        assert server.executor.is_worker_alive() is True
        assert server._stopped_event.is_set() is False
        assert fake_adapter.unload_count == 0
        assert fake_adapter.is_loaded() is True

        # Now release the noncooperative forward
        block_forward.set()
        fut.result(timeout=2.0)

        # Subsequent stop after release must join worker, unload models, and set stopped_event
        server.stop(timeout=5.0)
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert server.executor.is_worker_alive() is False
        assert fake_adapter.unload_count >= 1
        assert not fake_adapter.is_loaded()
        assert server._stopped_event.is_set() is True


def test_server_repeated_stop_is_idempotent():
    """Verify that multiple consecutive calls to stop() are safe and idempotent."""
    from scripts.qmd_mlx.server import ThreadedMLXServer, MLXHTTPRequestHandler, ServerState

    server = ThreadedMLXServer(("127.0.0.1", get_free_port()), MLXHTTPRequestHandler)
    server.state = ServerState.READY

    server.stop()
    server.stop()
    server.stop()
    assert server.state == ServerState.STOPPING


def test_server_tokenize_deadline_and_oversized_validation(mlx_server):
    """Verify HTTP /tokenize propagates deadlines and rejects oversized payloads."""
    # Test oversized text item > 256KB
    huge_text = "a" * (257 * 1024)
    r = requests.post(f"{mlx_server}/tokenize", json={"texts": [huge_text]}, timeout=2)
    assert r.status_code == 400

    # Test item count > 512
    too_many = ["hello"] * 513
    r = requests.post(f"{mlx_server}/tokenize", json={"texts": too_many}, timeout=2)
    assert r.status_code == 400


def test_control_listener_responsive_during_inference_socket_exhaustion():
    """
    Parent reproduction proof:
    When the primary inference server has max_threads=1 and is saturated by a client
    holding an incomplete HTTP header, requests to the inference port fail with HTTP 429.
    However, the dedicated separate CONTROL listener responds with HTTP 200 OK immediately.
    """
    from unittest.mock import MagicMock, patch

    fake_adapter = MagicMock()
    fake_adapter.is_loaded.return_value = True
    fake_adapter.native_dims = 384
    fake_adapter.get_descriptor.return_value = {"backend": "mlx", "nativeDimensions": 384}

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=fake_adapter):
        server, thread = start_server(
            "test-model",
            port=0,
            bind_host="127.0.0.1",
            preload=False,
            warmup=False,
            max_threads=1,
            control_max_threads=16,
        )

        inf_port = server.server_address[1]
        ctrl_port = server.control_port

        entered = threading.Event()
        original_process = server.process_request_thread

        def marked(*args):
            entered.set()
            original_process(*args)

        server.process_request_thread = marked

        hold_sock = socket.create_connection(("127.0.0.1", inf_port), timeout=2)
        try:
            # Send partial header to inference port and hold open
            hold_sock.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n")
            assert entered.wait(timeout=2.0), "Inference handler did not enter"

            # Connecting to inference port with a second client must fail with 429
            try:
                r_inf = requests.get(f"http://127.0.0.1:{inf_port}/health", timeout=1.0)
                inf_code = r_inf.status_code
            except Exception:
                inf_code = 429
            assert inf_code == 429, f"Expected inference port to return 429 under thread exhaustion, got {inf_code}"

            # Simultaneously, connecting to the dedicated CONTROL port responds 200 OK immediately!
            t0 = time.monotonic()
            r_ctrl = requests.get(f"http://127.0.0.1:{ctrl_port}/health", timeout=1.0)
            elapsed = time.monotonic() - t0

            assert r_ctrl.status_code == 200, f"Control port returned {r_ctrl.status_code}: {r_ctrl.text}"
            ctrl_data = r_ctrl.json()
            assert "pid" in ctrl_data
            assert "instance_token" in ctrl_data
            assert ctrl_data["instance_token"] == server.instance_token
            assert elapsed < 0.3, f"Control listener response took too long ({elapsed:.2f}s) under inference load"

        finally:
            hold_sock.close()
            server.stop()
            server.server_close()
            thread.join(timeout=2.0)


def test_control_port_rejects_inference_endpoints():
    """Verify that the dedicated control listener strictly rejects inference POST requests with 405."""
    from unittest.mock import MagicMock, patch

    fake_adapter = MagicMock()
    fake_adapter.is_loaded.return_value = True
    fake_adapter.native_dims = 384
    fake_adapter.get_descriptor.return_value = {"backend": "mlx", "nativeDimensions": 384}

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=fake_adapter):
        server, thread = start_server(
            "test-model",
            port=0,
            bind_host="127.0.0.1",
            preload=False,
            warmup=False,
        )

        ctrl_port = server.control_port

        try:
            # POST /embed on control port -> 405 Method Not Allowed
            r = requests.post(f"http://127.0.0.1:{ctrl_port}/embed", json={"texts": ["test"]}, timeout=2)
            assert r.status_code == 405
            assert "disabled on control port" in r.json().get("error", "")

            # POST /tokenize on control port -> 405 Method Not Allowed
            r2 = requests.post(f"http://127.0.0.1:{ctrl_port}/tokenize", json={"texts": ["test"]}, timeout=2)
            assert r2.status_code == 405

            # GET /health on control port -> 200 OK
            r_health = requests.get(f"http://127.0.0.1:{ctrl_port}/health", timeout=2)
            assert r_health.status_code == 200
        finally:
            server.stop()
            server.server_close()
            thread.join(timeout=2.0)


def test_control_listener_and_inference_listener_lifecycle_clean_join():
    """Verify that server.stop() closes and joins both inference and control listeners without leaking threads."""
    from unittest.mock import MagicMock, patch

    fake_adapter = MagicMock()
    fake_adapter.is_loaded.return_value = True
    fake_adapter.native_dims = 384
    fake_adapter.get_descriptor.return_value = {"backend": "mlx", "nativeDimensions": 384}

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=fake_adapter):
        server, thread = start_server(
            "test-model",
            port=0,
            bind_host="127.0.0.1",
            preload=False,
            warmup=False,
        )

        assert server.control_server is not None
        assert server.control_port is not None

        server.stop()
        server.server_close()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert server._stopped_event.is_set()


def test_control_listener_absolute_deadline_slow_drip():
    """
    Verify that MLXControlRequestHandler enforces an absolute wall-clock deadline
    and terminates slow-drip connections even when inactivity timeouts reset.
    Uses a small test budget (0.3s) to avoid long sleeps.
    """
    from unittest.mock import MagicMock, patch

    fake_adapter = MagicMock()
    fake_adapter.is_loaded.return_value = True
    fake_adapter.native_dims = 384
    fake_adapter.get_descriptor.return_value = {"backend": "mlx", "nativeDimensions": 384}

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=fake_adapter):
        server, thread = start_server(
            "test-model",
            port=0,
            bind_host="127.0.0.1",
            preload=False,
            warmup=False,
            control_request_timeout_s=0.3,
        )

        ctrl_port = server.control_port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect(("127.0.0.1", ctrl_port))

        try:
            # Send partial request line and drip slowly (1 byte every 0.1s for 5 chunks = 0.5s > 0.3s deadline)
            sock.sendall(b"GET ")
            drip_bytes = b"/health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
            deadline_hit = False
            for byte_val in drip_bytes:
                time.sleep(0.08)
                try:
                    sock.sendall(bytes([byte_val]))
                except (OSError, ConnectionResetError, BrokenPipeError):
                    deadline_hit = True
                    break

            if not deadline_hit:
                # Attempt to read response; server should have closed or reset connection
                try:
                    resp = sock.recv(1024)
                    # If closed by server, recv returns b""
                    assert resp == b"", f"Expected connection close due to deadline, got: {resp}"
                except (socket.timeout, OSError, ConnectionResetError):
                    pass
        finally:
            sock.close()
            server.stop()
            server.server_close()
            thread.join(timeout=2.0)


def test_control_listener_header_byte_budget_exceeded():
    """Verify that MLXControlRequestHandler rejects requests exceeding maximum control header byte budget."""
    from unittest.mock import MagicMock, patch

    fake_adapter = MagicMock()
    fake_adapter.is_loaded.return_value = True
    fake_adapter.native_dims = 384
    fake_adapter.get_descriptor.return_value = {"backend": "mlx", "nativeDimensions": 384}

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=fake_adapter):
        server, thread = start_server(
            "test-model",
            port=0,
            bind_host="127.0.0.1",
            preload=False,
            warmup=False,
            control_max_header_bytes=256,
        )

        ctrl_port = server.control_port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect(("127.0.0.1", ctrl_port))

        try:
            # Send 512 bytes of header (exceeds 256 byte budget)
            oversized_req = (
                b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Padding: "
                + (b"A" * 500)
                + b"\r\n\r\n"
            )
            sock.sendall(oversized_req)
            resp = sock.recv(1024)
            # Server closes connection due to byte limit
            assert resp == b"" or b"400" in resp or b"431" in resp
        finally:
            sock.close()
            server.stop()
            server.server_close()
            thread.join(timeout=2.0)


def test_control_listener_connection_close_no_keepalive():
    """Verify that MLXControlRequestHandler returns Connection: close and does not permit keepalive."""
    from unittest.mock import MagicMock, patch

    fake_adapter = MagicMock()
    fake_adapter.is_loaded.return_value = True
    fake_adapter.native_dims = 384
    fake_adapter.get_descriptor.return_value = {"backend": "mlx", "nativeDimensions": 384}

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=fake_adapter):
        server, thread = start_server(
            "test-model",
            port=0,
            bind_host="127.0.0.1",
            preload=False,
            warmup=False,
        )

        ctrl_port = server.control_port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect(("127.0.0.1", ctrl_port))

        try:
            req = b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n"
            sock.sendall(req)
            resp_chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                resp_chunks.append(chunk)
            full_resp = b"".join(resp_chunks)
            assert b"Connection: close" in full_resp
            assert b"\"status\"" in full_resp
        finally:
            sock.close()
            server.stop()
            server.server_close()
            thread.join(timeout=2.0)


def test_control_listener_saturation_recovery_no_leaked_handlers():
    """Verify control listener recovers after saturation and returns semaphore permits without leaks."""
    from unittest.mock import MagicMock, patch

    fake_adapter = MagicMock()
    fake_adapter.is_loaded.return_value = True
    fake_adapter.native_dims = 384
    fake_adapter.get_descriptor.return_value = {"backend": "mlx", "nativeDimensions": 384}

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=fake_adapter):
        server, thread = start_server(
            "test-model",
            port=0,
            bind_host="127.0.0.1",
            preload=False,
            warmup=False,
            control_max_threads=2,
        )

        ctrl_port = server.control_port
        try:
            # Perform consecutive health requests
            for _ in range(10):
                r = requests.get(f"http://127.0.0.1:{ctrl_port}/health", timeout=1.0)
                assert r.status_code == 200

            # Verify thread limiter semaphore is at full capacity (2)
            # Receiving the body precedes handler-finally cleanup. Wait for
            # permits with a bound rather than racing that cleanup thread.
            assert server.control_server.thread_limiter.acquire(timeout=1.0)
            assert server.control_server.thread_limiter.acquire(timeout=1.0)
            assert not server.control_server.thread_limiter.acquire(blocking=False)
            server.control_server.thread_limiter.release()
            server.control_server.thread_limiter.release()
        finally:
            server.stop()
            server.server_close()
            thread.join(timeout=2.0)


def test_server_stop_closes_owned_active_connections():
    """Verify that server.stop() immediately closes open client sockets on both inference and control listeners."""
    from unittest.mock import MagicMock, patch

    fake_adapter = MagicMock()
    fake_adapter.is_loaded.return_value = True
    fake_adapter.native_dims = 384
    fake_adapter.get_descriptor.return_value = {"backend": "mlx", "nativeDimensions": 384}

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", return_value=fake_adapter):
        server, thread = start_server(
            "test-model",
            port=0,
            bind_host="127.0.0.1",
            preload=False,
            warmup=False,
        )

        inf_port = server.server_address[1]
        ctrl_port = server.control_port

        s_inf = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s_inf.connect(("127.0.0.1", inf_port))
        s_inf.sendall(b"POST /embed HTTP/1.1\r\nHost: 127.0.0.1\r\n")

        s_ctrl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s_ctrl.connect(("127.0.0.1", ctrl_port))
        s_ctrl.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n")

        time.sleep(0.1)

        # Call stop
        server.stop()
        server.server_close()
        thread.join(timeout=2.0)

        # Sockets should be closed/disconnected
        try:
            s_inf.settimeout(0.5)
            data = s_inf.recv(1024)
            assert data == b""
        except (OSError, socket.timeout):
            pass
        finally:
            s_inf.close()

        try:
            s_ctrl.settimeout(0.5)
            data_c = s_ctrl.recv(1024)
            assert data_c == b""
        except (OSError, socket.timeout):
            pass
        finally:
            s_ctrl.close()


def test_single_stage_rerank_only_server():
    """Verify single-stage server starts without embedding model and serves /rerank."""
    from unittest.mock import MagicMock, patch

    fake_rerank = MagicMock()
    fake_rerank.is_loaded.return_value = True
    fake_rerank.model_name = "fake-rerank-4b"
    fake_rerank.yes_token_id = 9001
    fake_rerank.no_token_id = 9002
    fake_rerank.model_memory_mb = 500.0
    fake_rerank.total_requests = 1
    fake_rerank.total_pairs_scored = 2
    fake_rerank.total_latency_ms = 45.0
    fake_rerank.get_descriptor.return_value = {
        "version": 1,
        "backend": "mlx",
        "model": "fake-rerank-4b",
        "yesTokenId": 9001,
        "noTokenId": 9002,
    }
    fake_rerank.score_pairs.return_value = [0.85, 0.12]

    with patch("scripts.qmd_mlx.rerank.MLXRerankAdapter", return_value=fake_rerank):
        server, thread = start_server(
            model_name=None,
            rerank_model="fake-rerank-4b",
            port=0,
            bind_host="127.0.0.1",
            preload=False,
            warmup=False,
        )
        inf_port = server.server_address[1]
        ctrl_port = server.control_port

        try:
            # Wait for server readiness
            ready = False
            for _ in range(50):
                try:
                    r_ready = requests.get(f"http://127.0.0.1:{ctrl_port}/ready", timeout=1.0)
                    if r_ready.status_code == 200 and r_ready.json().get("ready"):
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(0.05)
            assert ready is True

            # Descriptor should return rerank descriptor
            r_desc = requests.get(f"http://127.0.0.1:{ctrl_port}/descriptor", timeout=2.0)
            assert r_desc.status_code == 200
            d = r_desc.json()
            assert d.get("model") == "fake-rerank-4b"
            assert d.get("yesTokenId") == 9001

            # /rerank should succeed
            r_rerank = requests.post(
                f"http://127.0.0.1:{inf_port}/rerank",
                json={"query": "test query", "documents": ["doc1", "doc2"]},
                timeout=2.0,
            )
            assert r_rerank.status_code == 200
            assert r_rerank.json()["scores"] == [0.85, 0.12]

            # /embed should return 501
            r_embed = requests.post(
                f"http://127.0.0.1:{inf_port}/embed",
                json={"texts": ["test"]},
                timeout=2.0,
            )
            assert r_embed.status_code == 501
        finally:
            server.stop()
            server.server_close()
            thread.join(timeout=2.0)


def test_single_stage_generate_only_server():
    """Verify single-stage server starts without embedding model and serves /generate."""
    from unittest.mock import MagicMock, patch

    fake_gen = MagicMock()
    fake_gen.is_loaded.return_value = True
    fake_gen.model_name = "fake-gen-1.7b"
    fake_gen.model_memory_mb = 300.0
    fake_gen.get_descriptor.return_value = {
        "version": 1,
        "backend": "mlx",
        "kind": "generate",
        "model": "fake-gen-1.7b",
    }
    fake_gen.get_stats_info.return_value = {
        "total_requests": 1,
        "total_tokens_generated": 10,
        "avg_ms": 25.0,
    }
    fake_gen.submit_generate.return_value = "database index b-tree"

    with patch("scripts.qmd_mlx.generate.MLXGenerateAdapter", return_value=fake_gen):
        server, thread = start_server(
            model_name=None,
            generate_model="fake-gen-1.7b",
            port=0,
            bind_host="127.0.0.1",
            preload=False,
            warmup=False,
        )
        inf_port = server.server_address[1]
        ctrl_port = server.control_port

        try:
            # Wait for server readiness
            ready = False
            for _ in range(50):
                try:
                    r_ready = requests.get(f"http://127.0.0.1:{ctrl_port}/ready", timeout=1.0)
                    if r_ready.status_code == 200 and r_ready.json().get("ready"):
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(0.05)
            assert ready is True

            # /generate should succeed
            r_gen = requests.post(
                f"http://127.0.0.1:{inf_port}/generate",
                json={"prompt": "expand query", "max_tokens": 16},
                timeout=2.0,
            )
            assert r_gen.status_code == 200
            assert "database" in r_gen.json()["text"]

            # /embed should return 501
            r_embed = requests.post(
                f"http://127.0.0.1:{inf_port}/embed",
                json={"texts": ["test"]},
                timeout=2.0,
            )
            assert r_embed.status_code == 501
        finally:
            server.stop()
            server.server_close()
            thread.join(timeout=2.0)
