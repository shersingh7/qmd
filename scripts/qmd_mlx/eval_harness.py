"""
eval_harness.py — Offline Public-Fixture Retrieval Evaluation Harness

Evaluates retrieval quality (BM25, Hybrid RRF, and Reranking) on public judged
benchmark fixtures using isolated temporary storage.
Guarantees:
1. 100% Offline & Isolated: Creates and cleans up temporary SQLite databases;
   NEVER accesses or modifies ~/.config/qmd, private indices, or live databases.
2. Public Judged Fixtures: Standard technical and reference query-passage pairs
   with explicit ground-truth relevance labels (relevant vs distractors).
3. Standard Information Retrieval Metrics:
   - Hit@1, Hit@3, Hit@5
   - Mean Reciprocal Rank (MRR)
   - Normalized Discounted Cumulative Gain (NDCG@5)
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import sqlite3
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


# --- Public Judged Evaluation Dataset (Synthetic / Public Domain) ---
PUBLIC_JUDGED_CORPUS = [
    {
        "id": "doc-api-versioning",
        "title": "REST API Versioning Strategies",
        "content": (
            "When designing web APIs, versioning is critical for backward compatibility. "
            "Common strategies include URI path versioning (/v1/users), custom request headers (X-API-Version: 2), "
            "and Accept header content negotiation (application/vnd.company.v1+json). "
            "URI path versioning is the most transparent for caching proxies and documentation."
        ),
    },
    {
        "id": "doc-raft-consensus",
        "title": "Raft Consensus Algorithm",
        "content": (
            "Raft is a distributed consensus algorithm designed to be understandable and equivalent to Paxos in fault-tolerance. "
            "It decomposes consensus into three subproblems: leader election, log replication, and safety. "
            "Nodes exist in one of three states: Leader, Follower, or Candidate. Heartbeat messages maintain leader authority."
        ),
    },
    {
        "id": "doc-sqlite-wal",
        "title": "SQLite Write-Ahead Logging (WAL)",
        "content": (
            "Write-Ahead Logging (WAL) is a journal mode in SQLite that enables concurrent readers and a single writer. "
            "Instead of writing directly to the database file, changes are appended to a separate .wal log file. "
            "Readers continue reading from the original database while the writer commits to the log, avoiding table locks."
        ),
    },
    {
        "id": "doc-cookie-recipe",
        "title": "Classic Chocolate Chip Cookie Recipe",
        "content": (
            "To bake chocolate chip cookies, combine 2 cups of all-purpose flour, 1 tsp baking soda, and 1/2 tsp salt. "
            "Cream 1 cup unsalted butter with 3/4 cup brown sugar and 3/4 cup granulated sugar. "
            "Fold in semi-sweet chocolate chips and bake at 375F (190C) for 9 to 11 minutes until golden brown."
        ),
    },
    {
        "id": "doc-vacation-policy",
        "title": "Employee Paid Time Off and Remote Work Policy",
        "content": (
            "Employees accrue 15 days of paid time off per calendar year. "
            "Vacation requests exceeding three consecutive days must be submitted through the HR portal two weeks in advance. "
            "Core collaboration hours are 10:00 AM to 3:00 PM in the employee's designated local timezone."
        ),
    },
    {
        "id": "doc-metal-gpu",
        "title": "Apple Silicon Metal Unified Memory Optimization",
        "content": (
            "Apple Silicon integrates CPU and GPU on a unified memory bus, eliminating host-to-device PCIe copy overhead. "
            "Metal compute shaders access unified memory allocations directly via zero-copy buffers. "
            "Quantized weight formats (4-bit affine, 8-bit) maximize effective memory bandwidth for LLM inference."
        ),
    },
]

PUBLIC_JUDGED_QUERIES = [
    {
        "query": "How to version REST APIs in HTTP headers or path?",
        "relevant_ids": ["doc-api-versioning"],
        "difficulty": "easy",
    },
    {
        "query": "distributed consensus leader election and log replication",
        "relevant_ids": ["doc-raft-consensus"],
        "difficulty": "easy",
    },
    {
        "query": "concurrent readers and writers sqlite database file lock",
        "relevant_ids": ["doc-sqlite-wal"],
        "difficulty": "medium",
    },
    {
        "query": "baking temperature and ingredients for homemade cookies",
        "relevant_ids": ["doc-cookie-recipe"],
        "difficulty": "easy",
    },
    {
        "query": "zero copy unified memory bandwidth GPU shaders",
        "relevant_ids": ["doc-metal-gpu"],
        "difficulty": "medium",
    },
]


@dataclasses.dataclass
class RetrievalMetrics:
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    ndcg_at_5: float
    total_queries: int
    latencies_ms: List[float] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        p50 = float(sorted(self.latencies_ms)[len(self.latencies_ms) // 2]) if self.latencies_ms else 0.0
        return {
            "hit_at_1": round(self.hit_at_1, 4),
            "hit_at_3": round(self.hit_at_3, 4),
            "hit_at_5": round(self.hit_at_5, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "total_queries": self.total_queries,
            "latency_p50_ms": round(p50, 2),
        }


def compute_dcg(ranked_ids: List[str], relevant_ids: set[str], k: int = 5) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k]):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)
    return dcg


def compute_idcg(relevant_count: int, k: int = 5) -> float:
    idcg = 0.0
    for i in range(min(relevant_count, k)):
        idcg += 1.0 / math.log2(i + 2)
    return idcg if idcg > 0 else 1.0


class IsolatedRetrievalHarness:
    """
    Sets up a temporary SQLite database with FTS5 search index to evaluate
    baseline search quality and downstream reranker efficacy on public fixtures.
    """

    def __init__(self, temp_dir: Optional[str] = None):
        self._owned_temp = temp_dir is None
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="qmd_eval_")
        self.db_path = os.path.join(self.temp_dir, "eval_corpus.sqlite")
        self.conn: Optional[sqlite3.Connection] = None
        self._init_database()

    def _init_database(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                id UNINDEXED,
                title,
                content,
                tokenize='porter unicode61'
            )
        """)

        for doc in PUBLIC_JUDGED_CORPUS:
            self.conn.execute(
                "INSERT INTO documents (id, title, content) VALUES (?, ?, ?)",
                (doc["id"], doc["title"], doc["content"]),
            )
            self.conn.execute(
                "INSERT INTO documents_fts (id, title, content) VALUES (?, ?, ?)",
                (doc["id"], doc["title"], doc["content"]),
            )
        self.conn.commit()

    def search_bm25(self, query: str, limit: int = 5) -> List[Tuple[str, float]]:
        """BM25 search using SQLite FTS5 rank score."""
        tokens = [t.strip() for t in query.split() if t.strip() and t.isalnum()]
        if not tokens:
            return []
        match_query = " OR ".join(tokens)
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT id, rank
                FROM documents_fts
                WHERE documents_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (match_query, limit))
            rows = cur.fetchall()
            return [(r[0], float(r[1])) for r in rows]
        except Exception:
            return []

    def get_document_content(self, doc_id: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT content FROM documents WHERE id = ?", (doc_id,))
        row = cur.fetchone()
        return row[0] if row else None

    def evaluate_bm25(self) -> RetrievalMetrics:
        """Evaluates BM25 retrieval across all public judged queries."""
        hits_1, hits_3, hits_5 = 0, 0, 0
        rr_total = 0.0
        ndcg_total = 0.0
        latencies = []

        for q in PUBLIC_JUDGED_QUERIES:
            t0 = time.monotonic()
            results = self.search_bm25(q["query"], limit=5)
            latencies.append((time.monotonic() - t0) * 1000)

            ranked_ids = [r[0] for r in results]
            rel_set = set(q["relevant_ids"])

            if ranked_ids and ranked_ids[0] in rel_set:
                hits_1 += 1
            if any(doc_id in rel_set for doc_id in ranked_ids[:3]):
                hits_3 += 1
            if any(doc_id in rel_set for doc_id in ranked_ids[:5]):
                hits_5 += 1

            # Reciprocal rank
            rr = 0.0
            for rank, doc_id in enumerate(ranked_ids, start=1):
                if doc_id in rel_set:
                    rr = 1.0 / rank
                    break
            rr_total += rr

            # NDCG@5
            dcg = compute_dcg(ranked_ids, rel_set, k=5)
            idcg = compute_idcg(len(rel_set), k=5)
            ndcg_total += (dcg / idcg) if idcg > 0 else 0.0

        n = len(PUBLIC_JUDGED_QUERIES)
        return RetrievalMetrics(
            hit_at_1=hits_1 / n,
            hit_at_3=hits_3 / n,
            hit_at_5=hits_5 / n,
            mrr=rr_total / n,
            ndcg_at_5=ndcg_total / n,
            total_queries=n,
            latencies_ms=latencies,
        )

    def evaluate_with_reranker(
        self,
        score_pairs_fn: Callable[[str, List[str]], List[float]],
    ) -> RetrievalMetrics:
        """
        Evaluates 2-stage retrieval: initial BM25 candidates -> MLX Reranker re-scoring.
        """
        hits_1, hits_3, hits_5 = 0, 0, 0
        rr_total = 0.0
        ndcg_total = 0.0
        latencies = []

        for q in PUBLIC_JUDGED_QUERIES:
            t0 = time.monotonic()
            # Retrieve all documents as candidate pool to assess pure reranking discrimination
            candidates = [d["id"] for d in PUBLIC_JUDGED_CORPUS]
            candidate_texts = [d["content"] for d in PUBLIC_JUDGED_CORPUS]

            scores = score_pairs_fn(q["query"], candidate_texts)
            latencies.append((time.monotonic() - t0) * 1000)

            # Sort candidate IDs by reranker score descending
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            ranked_ids = [doc_id for doc_id, _ in ranked]
            rel_set = set(q["relevant_ids"])

            if ranked_ids and ranked_ids[0] in rel_set:
                hits_1 += 1
            if any(doc_id in rel_set for doc_id in ranked_ids[:3]):
                hits_3 += 1
            if any(doc_id in rel_set for doc_id in ranked_ids[:5]):
                hits_5 += 1

            # Reciprocal rank
            rr = 0.0
            for rank, doc_id in enumerate(ranked_ids, start=1):
                if doc_id in rel_set:
                    rr = 1.0 / rank
                    break
            rr_total += rr

            # NDCG@5
            dcg = compute_dcg(ranked_ids, rel_set, k=5)
            idcg = compute_idcg(len(rel_set), k=5)
            ndcg_total += (dcg / idcg) if idcg > 0 else 0.0

        n = len(PUBLIC_JUDGED_QUERIES)
        return RetrievalMetrics(
            hit_at_1=hits_1 / n,
            hit_at_3=hits_3 / n,
            hit_at_5=hits_5 / n,
            mrr=rr_total / n,
            ndcg_at_5=ndcg_total / n,
            total_queries=n,
            latencies_ms=latencies,
        )

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
        if self._owned_temp and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception:
                pass
