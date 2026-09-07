"""
test_mlx_rerank.py — Unit tests for MLX rerank adapter
"""

import pytest
import numpy as np


def test_rerank_adapter_imports():
    from scripts.qmd_mlx.rerank import MLXRerankAdapter
    assert MLXRerankAdapter is not None


def test_dynamic_yes_no_token_resolution():
    from scripts.qmd_mlx.rerank import MLXRerankAdapter
    adapter = MLXRerankAdapter(model_name="mlx-community/Qwen3-Reranker-4B-mxfp8")
    assert adapter.yes_token_id is not None
    assert adapter.no_token_id is not None
    assert isinstance(adapter.yes_token_id, int)
    assert isinstance(adapter.no_token_id, int)
    assert adapter.yes_token_id != adapter.no_token_id


def test_rerank_score_ordering_and_determinism():
    from scripts.qmd_mlx.rerank import MLXRerankAdapter
    adapter = MLXRerankAdapter(model_name="mlx-community/Qwen3-Reranker-4B-mxfp8")

    query = "API versioning strategies"
    rel_doc = "This document covers REST API versioning strategies, URI versioning, and HTTP header versioning."
    irrel_doc = "Making chocolate chip cookies requires flour, sugar, butter, baking soda, and chocolate chips."

    scores1 = adapter.score_pairs(query, [rel_doc, irrel_doc])
    scores2 = adapter.score_pairs(query, [rel_doc, irrel_doc])

    assert len(scores1) == 2
    assert len(scores2) == 2

    # Determinism
    np.testing.assert_allclose(scores1, scores2, rtol=1e-5, atol=1e-5)

    # Score ordering: relevant > irrelevant
    assert scores1[0] > scores1[1], f"Expected rel_doc ({scores1[0]}) > irrel_doc ({scores1[1]})"

    # Finite float values between 0.0 and 1.0
    for s in scores1:
        assert 0.0 <= s <= 1.0
        assert np.isfinite(s)


def test_rerank_batch_cardinality_and_order():
    from scripts.qmd_mlx.rerank import MLXRerankAdapter
    adapter = MLXRerankAdapter(model_name="mlx-community/Qwen3-Reranker-4B-mxfp8")

    query = "Distributed consensus algorithm"
    docs = [
        "Chocolate chip cookie recipe.",
        "Raft and Paxos are distributed consensus algorithms for replicated state machines.",
        "Remote work vacation request policy.",
        "API endpoint design guidelines.",
    ]

    scores = adapter.score_pairs(query, docs)
    assert len(scores) == 4

    # Doc at index 1 is the most relevant
    best_idx = int(np.argmax(scores))
    assert best_idx == 1, f"Expected best doc index 1, got {best_idx}"


def test_rerank_invalid_and_empty_inputs():
    from scripts.qmd_mlx.rerank import MLXRerankAdapter, RerankError
    adapter = MLXRerankAdapter(model_name="mlx-community/Qwen3-Reranker-4B-mxfp8")

    # Empty doc list returns empty list
    assert adapter.score_pairs("query", []) == []

    # Empty query or non-string query raises RerankError
    with pytest.raises(RerankError):
        adapter.score_pairs("", ["some document"])

    with pytest.raises(RerankError):
        adapter.score_pairs(None, ["some document"])

    # Non-string doc raises RerankError
    with pytest.raises(RerankError):
        adapter.score_pairs("query", [123])


def test_rerank_truncation_preserves_query():
    from scripts.qmd_mlx.rerank import MLXRerankAdapter
    adapter = MLXRerankAdapter(
        model_name="mlx-community/Qwen3-Reranker-4B-mxfp8",
        max_length=512,
    )

    query = "Critical query terms that must not be truncated"
    huge_doc = "Very repetitive long document content. " * 500

    # Should not raise context overflow error — documents are truncated safely
    scores = adapter.score_pairs(query, [huge_doc])
    assert len(scores) == 1
    assert np.isfinite(scores[0])
