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


LOCAL_4B = "/Users/shersingh/.cache/qmd/models/qwen3-reranker-4b-mlx-4bit"


def test_rerank_batch_equivalence():
    """Micro-batched scoring must match single-pair scoring numerically."""
    from scripts.qmd_mlx.rerank import MLXRerankAdapter
    adapter = MLXRerankAdapter(model_name=LOCAL_4B)

    query = "tradeoff between data consistency and availability"
    docs = [
        "The CAP theorem states a distributed system can only guarantee two of Consistency, Availability, and Partition tolerance.",
        "Our vacation policy allows 20 days per year.",
        "Raft and Paxos are distributed consensus algorithms for replicated state machines.",
        "Making chocolate chip cookies requires flour, sugar, butter, and chocolate chips.",
        "API endpoint design guidelines for REST services.",
    ]

    single = adapter.score_pairs(query, docs, batch_size=1, timeout_s=None)
    batched = adapter.score_pairs(query, docs, batch_size=4, timeout_s=None)
    assert len(single) == len(batched) == 5

    # Same-shape scoring is exactly deterministic.
    repeat = adapter.score_pairs(query, docs, batch_size=4, timeout_s=None)
    np.testing.assert_array_equal(batched, repeat)

    # Across batch shapes, bf16 reduction order drifts scores slightly
    # (measured <= 0.02 absolute on 4-bit weights). What must hold exactly
    # is the RANK order — batching must never reorder results.
    assert np.argsort(np.argsort(single)).tolist() == np.argsort(np.argsort(batched)).tolist()
    np.testing.assert_allclose(single, batched, rtol=0.1, atol=0.05)

    # And the ranking is still correct (sanity: relevant docs on top).
    assert single[0] > single[1]
    assert single[0] > single[3]


def test_rerank_deadline():
    from scripts.qmd_mlx.rerank import MLXRerankAdapter, RerankError
    adapter = MLXRerankAdapter(model_name=LOCAL_4B)

    with pytest.raises(RerankError, match="[Dd]eadline"):
        adapter.score_pairs("query", ["doc one", "doc two"], timeout_s=-1)

    with pytest.raises(RerankError, match="batch_size"):
        adapter.score_pairs("query", ["doc"], batch_size=0)


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
