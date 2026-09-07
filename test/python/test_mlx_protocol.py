"""
test_mlx_protocol.py — Unit tests for MLX wire protocol binary framing and request validation
"""

import pytest
import numpy as np
from scripts.qmd_mlx.protocol import (
    encode_binary_embeddings,
    decode_binary_embeddings,
    validate_embed_request,
    ProtocolError,
)


def test_roundtrip_binary_encoding():
    arr = np.random.randn(5, 768).astype(np.float32)
    encoded = encode_binary_embeddings(arr, count=5, dims=768)
    assert len(encoded) == 8 + 5 * 768 * 4

    decoded, count, dims = decode_binary_embeddings(encoded)
    assert count == 5
    assert dims == 768
    np.testing.assert_allclose(decoded, arr, rtol=1e-6)


def test_empty_binary_encoding():
    arr = np.empty((0, 0), dtype=np.float32)
    encoded = encode_binary_embeddings(arr, count=0, dims=0)
    assert len(encoded) == 8

    decoded, count, dims = decode_binary_embeddings(encoded)
    assert count == 0
    assert dims == 0
    assert decoded.shape == (0, 0)


def test_reject_nan_in_encoding():
    arr = np.array([[1.0, float("nan")], [0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ProtocolError, match="NaN or Infinite"):
        encode_binary_embeddings(arr, count=2, dims=2)


def test_reject_truncated_buffer():
    arr = np.ones((2, 4), dtype=np.float32)
    encoded = encode_binary_embeddings(arr, 2, 4)
    with pytest.raises(ProtocolError, match="does not match expected"):
        decode_binary_embeddings(encoded[:-4])


def test_validate_embed_request_success():
    payload = {"texts": ["hello", "world"], "dims": 256, "is_query": True}
    texts, dims, is_query = validate_embed_request(payload)
    assert texts == ["hello", "world"]
    assert dims == 256
    assert is_query is True


def test_validate_embed_request_rejects_boolean_dims():
    payload = {"texts": ["hello"], "dims": True}
    with pytest.raises(ProtocolError, match="must be a positive integer"):
        validate_embed_request(payload)


def test_validate_embed_request_rejects_empty():
    with pytest.raises(ProtocolError, match="cannot be empty"):
        validate_embed_request({"texts": []})
