#!/usr/bin/env python3
"""Retrieval-quality eval: candidate embedding models vs labeled queries.

Builds a local vector index over test/eval-docs with each model, then measures
top-1 Recall on the 25 labeled deep-research queries. Also reports the GGUF
0.6B baseline for reference. No QMD index/database touched — pure in-memory.

Usage: PYTHONPATH=. .venv/bin/python scripts/embedding_recall_eval.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def fmt_query(model_lower: str, q: str) -> str:
    if "qwen" in model_lower:
        return f"Instruct: Retrieve relevant documents for the given query\nQuery: {q}"
    return q


def fmt_doc(model_lower: str, doc: str) -> str:
    # Qwen3-Embedding encodes documents as raw text.
    return doc


def run_model(model_name: str, label: str, evals: list[dict], docs: dict[str, str]):
    from scripts.qmd_mlx.runtime import MLXEmbeddingRuntime

    t0 = time.time()
    rt = MLXEmbeddingRuntime(model_name=model_name, max_length=2048)
    load_s = time.time() - t0

    m_lower = model_name.lower()
    names = list(docs.keys())
    bodies = [fmt_doc(m_lower, docs[n]) for n in names]

    t1 = time.time()
    doc_vecs = rt.submit_embed(bodies, is_query=False, timeout=120.0)
    doc_vecs = np.asarray(doc_vecs, dtype=np.float32)
    doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    doc_unit = doc_vecs / np.maximum(doc_norms, 1e-9)
    index_s = time.time() - t1

    hits = 0
    per_q = []
    t2 = time.time()
    for e in evals:
        q = fmt_query(m_lower, e["query"])
        qv = np.asarray(rt.submit_embed([q], is_query=True, timeout=60.0)[0], dtype=np.float32)
        qv = qv / max(float(np.linalg.norm(qv)), 1e-9)
        sims = doc_unit @ qv
        order = np.argsort(-sims)
        top1 = names[order[0]]
        hit = top1 == e["expected_doc"] or top1.startswith(e["expected_doc"])
        hits += hit
        per_q.append({"query": e["query"][:40], "difficulty": e.get("difficulty", "?"),
                      "expected": e["expected_doc"], "top1": top1, "hit": hit})
    query_s = time.time() - t2

    rt.shutdown()
    import mlx.core as mx
    mx.clear_cache()

    return {
        "label": label, "model": model_name,
        "top1_recall": round(hits / len(evals), 4), "hits": hits, "n": len(evals),
        "load_s": round(load_s, 2), "index_s": round(index_s, 3), "query_s": round(query_s, 2),
        "per_query": per_q,
    }


def main() -> int:
    evals = [json.loads(l) for l in (REPO / "test/eval-deep-research.jsonl").read_text().strip().splitlines()]
    docs_dir = REPO / "test/eval-docs"
    docs = {p.stem: p.read_text() for p in docs_dir.glob("*.md")}

    home = Path.home()
    candidates = [
        ("mlx-0.6B-8bit", "mlx-community/Qwen3-Embedding-0.6B-8bit"),
        ("mlx-4B-affine-4bit", str(home / ".cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine")),
        ("mlx-4B-DWQ", "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"),
    ]

    results = []
    for label, model in candidates:
        try:
            r = run_model(model, label, evals, docs)
        except Exception as exc:
            r = {"label": label, "model": model, "error": str(exc)}
        results.append(r)
        acc = r.get("top1_recall")
        print(f"{label:18s} recall@1 = {acc if acc is None else f'{acc:.0%}'}"
              f"  ({r.get('hits','?')}/{len(evals)})" + ("" if acc is None else f"  [load {r['load_s']}s]"))

    out = REPO / "docs/benchmarks/embedding_recall_eval.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())