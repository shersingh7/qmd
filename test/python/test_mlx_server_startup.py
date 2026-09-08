"""
test_mlx_server_startup.py — Regression tests for 3-model preload startup registration ordering.
"""

from unittest.mock import MagicMock, patch
import pytest
from scripts.qmd_mlx.executor import GPUExecutor
from scripts.qmd_mlx.model_manager import ModelResidencyManager
from scripts.qmd_mlx.protocol import ModelUnavailableError
from scripts.qmd_mlx.rerank import MLXRerankAdapter
from scripts.qmd_mlx.generate import MLXGenerateAdapter
from scripts.qmd_mlx.server import start_server, ServerState


def test_preload_three_models_registers_before_load():
    """
    Verify that 3-model preload does not call ensure_loaded before registering adapters.
    With the bug: MLXRerankAdapter(lazy_load=False, model_manager=m) calls
    ensure_loaded('rerank') inside __init__, which raises ModelUnavailableError
    because register_adapter('rerank', ...) hasn't happened yet.
    """
    executor = GPUExecutor()
    manager = ModelResidencyManager(executor)

    with patch.object(MLXRerankAdapter, "load") as mock_rerank_load:
        mock_rerank_load.return_value = None
        # Creating adapter with lazy_load=False and model_manager should not fail with ModelUnavailableError
        adapter = MLXRerankAdapter(
            model_name="test-rerank",
            lazy_load=False,
            executor=executor,
            model_manager=manager,
        )
        # Register and ensure loaded
        manager.register_adapter("rerank", adapter)
        manager.ensure_loaded("rerank")
        assert mock_rerank_load.called
    executor.shutdown()


def test_server_startup_three_model_preload_reaches_ready():
    """
    Test start_server with rerank_model and generate_model under preload=True.
    Should reach READY state without ModelUnavailableError.
    """
    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter") as mock_embed, \
         patch.object(MLXRerankAdapter, "load") as mock_rerank_load, \
         patch.object(MLXGenerateAdapter, "load") as mock_gen_load:

        mock_embed_inst = MagicMock()
        mock_embed_inst.is_loaded.return_value = True
        mock_embed_inst.native_dims = 384
        mock_embed_inst.model_params_b = 0.5
        mock_embed_inst.model_memory_mb = 500.0
        mock_embed.return_value = mock_embed_inst

        mock_rerank_load.return_value = None
        mock_gen_load.return_value = None

        server, thread = start_server(
            model_name="test-embed",
            rerank_model="test-rerank",
            generate_model="test-generate",
            port=0,
            bind_host="127.0.0.1",
            preload=True,
            warmup=False,
        )

        try:
            # Wait briefly if still in starting/loading
            import time
            for _ in range(50):
                if server.state in (ServerState.READY, ServerState.FAILED):
                    break
                time.sleep(0.05)
            # Check server state
            assert server.state == ServerState.READY, f"Server state was {server.state}: {server.state_error}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)


def test_server_startup_check_start_boundary_race_managed_loop():
    """
    Deterministic test pausing at check/start boundary:
    asserts init thread joins cleanly, stop is respected, and no requests served after stop.
    """
    import threading
    import requests

    boundary_reached = threading.Event()
    allow_resume = threading.Event()

    def boundary_hook():
        boundary_reached.set()
        allow_resume.wait(timeout=5.0)

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter") as mock_embed:
        mock_embed_inst = MagicMock()
        mock_embed_inst.is_loaded.return_value = True
        mock_embed_inst.native_dims = 384
        mock_embed_inst.model_params_b = 0.5
        mock_embed_inst.model_memory_mb = 100.0
        mock_embed.return_value = mock_embed_inst

        server, thread = start_server(
            model_name="test-embed",
            port=0,
            bind_host="127.0.0.1",
            preload=False,
            warmup=False,
            _before_serve_hook=boundary_hook,
        )

        port = server.server_address[1]

        # Wait until init thread is paused at the exact boundary
        assert boundary_reached.wait(timeout=3.0)

        # Trigger stop while paused at boundary
        server.stop(timeout=2.0)
        assert server.state == ServerState.STOPPING

        # Release init thread to continue
        allow_resume.set()

        # Assert init thread joins cleanly without hanging on serve_forever
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert server._stopped_event.is_set() is True
        assert server.state == ServerState.STOPPING

        # Close server socket
        server.server_close()

        # Assert no requests served after stop
        with pytest.raises(Exception):
            requests.get(f"http://127.0.0.1:{port}/ready", timeout=0.5)


def test_server_startup_failed_preload_serves_diagnostic_endpoints():
    """
    Verify that failed preload enters FAILED state and serves diagnostic control endpoints
    (/health, /ready, /embed rejection) until stop() is called.
    """
    import time
    import requests

    with patch("scripts.qmd_mlx.runtime.resolve_embedding_adapter", side_effect=RuntimeError("Simulated preload failure")):
        server, thread = start_server(
            model_name="failing-model",
            port=0,
            bind_host="127.0.0.1",
            preload=True,
            warmup=False,
        )

        port = server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"

        # Wait for failure state to be reached
        for _ in range(50):
            if server.state == ServerState.FAILED:
                break
            time.sleep(0.05)

        assert server.state == ServerState.FAILED
        assert "Simulated preload failure" in (server.state_error or "")

        # Verify /health responds with degraded/failed diagnostic info
        r_health = requests.get(f"{base_url}/health", timeout=2)
        assert r_health.status_code == 200
        health_data = r_health.json()
        assert health_data["status"] == "degraded"
        assert health_data["state"] == "failed"
        assert health_data["ready"] is False
        assert "Simulated preload failure" in health_data["error"]

        # Verify /ready responds with 503
        r_ready = requests.get(f"{base_url}/ready", timeout=2)
        assert r_ready.status_code == 503
        ready_data = r_ready.json()
        assert ready_data["ready"] is False
        assert ready_data["state"] == "failed"

        # Verify inference /embed is rejected with 503
        r_embed = requests.post(f"{base_url}/embed", json={"texts": ["test"]}, timeout=2)
        assert r_embed.status_code == 503
        assert "failed" in r_embed.json()["error"]

        # Stop server
        server.stop()
        server.server_close()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert server.state == ServerState.STOPPING
        assert server._stopped_event.is_set() is True


def test_launcher_stopped_reporting_logic():
    """
    Verify launcher reports '[mlx-server] Stopped.' only when stopped_event is set
    AND init thread has joined.
    """
    from scripts.mlx_embed_server import main
    from io import StringIO

    # Test case 1: Successful stop and join
    mock_server = MagicMock()
    mock_server._stopped_event.is_set.return_value = True
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True

    def mock_join(*args, **kwargs):
        mock_thread.is_alive.return_value = False

    mock_thread.join.side_effect = mock_join

    with patch("scripts.mlx_embed_server.start_server", return_value=(mock_server, mock_thread)), \
         patch("sys.argv", ["mlx_embed_server.py", "--no-preload"]), \
         patch("time.sleep", side_effect=[None, KeyboardInterrupt]), \
         patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        try:
            main()
        except (KeyboardInterrupt, SystemExit):
            pass

        output = mock_stdout.getvalue()
        assert "[mlx-server] Stopped." in output

    # Test case 2: Join timeout / thread still alive
    mock_server2 = MagicMock()
    mock_server2._stopped_event.is_set.return_value = True
    mock_thread2 = MagicMock()
    mock_thread2.is_alive.return_value = True  # Thread fails to join

    with patch("scripts.mlx_embed_server.start_server", return_value=(mock_server2, mock_thread2)), \
         patch("sys.argv", ["mlx_embed_server.py", "--no-preload"]), \
         patch("time.sleep", side_effect=[None, KeyboardInterrupt]), \
         patch("sys.stdout", new_callable=StringIO) as mock_stdout, \
         patch("sys.stderr", new_callable=StringIO) as mock_stderr:
        with pytest.raises(SystemExit):
            main()

        output = mock_stdout.getvalue()
        err_output = mock_stderr.getvalue()
        assert "[mlx-server] Stopped." not in output
        assert "Shutdown incomplete" in err_output
