"""
test_eval_offline.py — Unit tests for offline public-fixture retrieval evaluation harness
"""

import pytest
from scripts.qmd_mlx.eval_harness import (
    IsolatedRetrievalHarness,
    compute_dcg,
    compute_idcg,
    PUBLIC_JUDGED_CORPUS,
    PUBLIC_JUDGED_QUERIES,
)


def test_dcg_idcg_computation():
    relevant = {"doc-1", "doc-2"}
    # Perfect ranking
    ranked_perfect = ["doc-1", "doc-2", "doc-3"]
    dcg = compute_dcg(ranked_perfect, relevant, k=3)
    idcg = compute_idcg(len(relevant), k=3)
    assert dcg > 0.0
    assert abs(dcg - idcg) < 1e-6

    # Imperfect ranking
    ranked_imperfect = ["doc-3", "doc-1", "doc-2"]
    dcg_imp = compute_dcg(ranked_imperfect, relevant, k=3)
    assert dcg_imp < dcg


def test_isolated_retrieval_harness_bm25():
    harness = IsolatedRetrievalHarness()
    try:
        metrics = harness.evaluate_bm25()
        assert metrics.total_queries == len(PUBLIC_JUDGED_QUERIES)
        assert 0.0 <= metrics.hit_at_1 <= 1.0
        assert 0.0 <= metrics.hit_at_3 <= 1.0
        assert 0.0 <= metrics.hit_at_5 <= 1.0
        assert 0.0 <= metrics.mrr <= 1.0
        assert 0.0 <= metrics.ndcg_at_5 <= 1.0
        assert len(metrics.latencies_ms) == len(PUBLIC_JUDGED_QUERIES)
        d = metrics.to_dict()
        assert "hit_at_1" in d
        assert "mrr" in d
    finally:
        harness.close()


def test_isolated_retrieval_harness_with_mock_reranker():
    harness = IsolatedRetrievalHarness()
    try:
        # Define mock scoring function that gives highest score to relevant doc
        def mock_scorer(query: str, docs: list[str]) -> list[float]:
            scores = []
            for doc in docs:
                if "REST API versioning" in doc and "version" in query.lower():
                    scores.append(0.95)
                elif "Raft" in doc and "consensus" in query.lower():
                    scores.append(0.92)
                elif "Write-Ahead" in doc and "sqlite" in query.lower():
                    scores.append(0.90)
                elif "cookie" in doc and "cookie" in query.lower():
                    scores.append(0.88)
                elif "Metal" in doc and "GPU" in query:
                    scores.append(0.85)
                else:
                    scores.append(0.05)
            return scores

        metrics = harness.evaluate_with_reranker(mock_scorer)
        assert metrics.hit_at_1 == 1.0
        assert metrics.hit_at_3 == 1.0
        assert metrics.mrr == 1.0
        assert metrics.ndcg_at_5 == 1.0
    finally:
        harness.close()
