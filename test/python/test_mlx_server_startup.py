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
