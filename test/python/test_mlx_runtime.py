"""
test_mlx_runtime.py — Tests for MLX embedding runtime, padding invariance, and normalization
"""

import numpy as np
import pytest
from scripts.qmd_mlx.runtime import MLXEmbeddingRuntime, MLXRuntimeError


@pytest.fixture(scope="module")
def mlx_runtime():
    # Use cached small model for real Metal execution
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    runtime = MLXEmbeddingRuntime(
        model_name=model_name,
        quantization="bf16",
        dtype_str="float32",
        max_length=512,
    )
    yield runtime
    runtime.shutdown()


def test_runtime_initialization(mlx_runtime):
    assert mlx_runtime.native_dims == 384
    assert mlx_runtime.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    desc = mlx_runtime.get_descriptor()
    assert desc["backend"] == "mlx"
    assert desc["nativeDimensions"] == 384
    assert desc["outputDimensions"] == 384
    assert desc["normalized"] is True


def test_embedding_normalization_and_finiteness(mlx_runtime):
    texts = ["Fast semantic search on Apple Silicon", "Hybrid retrieval with SQLite and MLX"]
    res = mlx_runtime.embed_direct(texts)

    assert res.shape == (2, 384)
    assert np.all(np.isfinite(res))
    norms = np.linalg.norm(res, axis=-1)
    assert np.allclose(norms, [1.0, 1.0], atol=1e-5)


def test_padding_invariance(mlx_runtime):
    """
    Critical correctness gate: Embedding of text X must be invariant
    to other texts of different lengths in the same batch (no attention leakage across padding).
    """
    text1 = "Short query"
    long_text = (
        "This is an extensive document with numerous tokens designed to test that padding masks "
        "properly isolate shorter sequences in the batch from attention leakage or corruption."
    )

    vec1_alone = mlx_runtime.embed_direct([text1])[0]
    vec1_batched = mlx_runtime.embed_direct([text1, long_text])[0]

    assert np.allclose(vec1_alone, vec1_batched, atol=1e-4)


def test_matryoshka_dimension_reduction(mlx_runtime):
    res_full = mlx_runtime.embed_direct(["test text"])
    assert res_full.shape == (1, 384)

    res_128 = mlx_runtime.embed_direct(["test text"], requested_dims=128)
    assert res_128.shape == (1, 128)
    assert np.all(np.isfinite(res_128))
    norm_128 = np.linalg.norm(res_128, axis=-1)
    assert np.allclose(norm_128, [1.0], atol=1e-5)


def test_tokenize_endpoint(mlx_runtime):
    texts = ["apple silicon", "mlx embedding"]
    tokens = mlx_runtime.tokenize(texts)
    assert len(tokens) == 2
    assert all(isinstance(t, list) for t in tokens)
    assert all(len(t) > 0 for t in tokens)


def test_unsupported_model_raises_error():
    with pytest.raises(MLXRuntimeError):
        MLXEmbeddingRuntime("nonexistent/invalid-model-name-xyz-999")
