"""
test_mlx_batching.py — Unit tests for batch planning, token-once sorting, and order restoration
"""

import numpy as np
import pytest
from scripts.qmd_mlx.batching import (
    BatchPlanner,
    calculate_default_max_batch_tokens,
    get_system_ram_gb,
)
from scripts.qmd_mlx.protocol import (
    encode_binary_embeddings,
    decode_binary_embeddings,
    validate_embed_request,
    ProtocolError,
)


class MockTokenizer:
    def encode(self, text: str) -> list[int]:
        return [1] * max(1, len(text.split()))


def mock_embed_fn(texts: list[str]) -> np.ndarray:
    dims = 4
    # Unique embedding based on text length and content hash
    result = np.zeros((len(texts), dims), dtype=np.float32)
    for i, t in enumerate(texts):
        val = float(len(t)) + 0.1
        result[i] = [val, val * 2, val * 3, val * 4]
        # Normalize
        result[i] /= np.linalg.norm(result[i])
    return result


def test_system_ram_detection():
    ram_gb = get_system_ram_gb()
    assert isinstance(ram_gb, (int, float))
    assert ram_gb > 0
    tokens = calculate_default_max_batch_tokens()
    assert tokens in (4096, 8192, 12288, 16384, 32768)


def test_batch_planner_order_restoration():
    planner = BatchPlanner(max_batch_tokens=10)
    texts = [
        "short",
        "this is a much longer text with several tokens in it",
        "tiny",
        "medium length phrase here",
    ]
    tok = MockTokenizer()
    embeddings = planner.plan_and_execute(texts, tok, mock_embed_fn)

    assert embeddings.shape == (4, 4)
    # Verify each row corresponds to the original text position
    direct = mock_embed_fn(texts)
    assert np.allclose(embeddings, direct, atol=1e-6)


def test_batch_planner_single_item():
    planner = BatchPlanner()
    texts = ["single text"]
    embeddings = planner.plan_and_execute(texts, MockTokenizer(), mock_embed_fn)
    assert embeddings.shape == (1, 4)


def test_batch_planner_empty():
    planner = BatchPlanner()
    embeddings = planner.plan_and_execute([], MockTokenizer(), mock_embed_fn)
    assert embeddings.shape == (0, 0)


def test_binary_protocol_roundtrip():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    bin_data = encode_binary_embeddings(arr, 2, 3)
    decoded, count, dims = decode_binary_embeddings(bin_data)
    assert count == 2
    assert dims == 3
    assert np.allclose(arr, decoded)


def test_binary_protocol_rejects_nan():
    arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    with pytest.raises(ProtocolError):
        encode_binary_embeddings(arr, 2, 2)


def test_validate_embed_request():
    texts, dims, is_query = validate_embed_request({"texts": ["a", "b"], "dims": 128, "is_query": True})
    assert texts == ["a", "b"]
    assert dims == 128
    assert is_query is True


def test_validate_embed_request_errors():
    with pytest.raises(ProtocolError):
        validate_embed_request({"texts": []})
    with pytest.raises(ProtocolError):
        validate_embed_request({"texts": [123]})
    with pytest.raises(ProtocolError):
        validate_embed_request({"texts": ["ok"], "dims": -5})
    with pytest.raises(ProtocolError):
        validate_embed_request("not-a-dict")
