"""
test_mlx_server.py — Integration tests for bounded HTTP MLX embedding server
"""

import socket
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
    port = get_free_port()
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    server, thread = start_server(
        model_name=model_name,
        port=port,
        bind_host="127.0.0.1",
        preload=True,
        warmup=False,
    )

    base_url = f"http://127.0.0.1:{port}"

    # Wait for server readiness
    for _ in range(50):
        try:
            r = requests.get(f"{base_url}/ready", timeout=1)
            if r.status_code == 200 and r.json().get("ready"):
                break
        except Exception:
            pass
        time.sleep(0.1)

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
