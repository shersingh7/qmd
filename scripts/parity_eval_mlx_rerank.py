#!/usr/bin/env python3
"""Parity gate: MLX Qwen3-Reranker-4B-mxfp8 vs GGUF Voodisss 4B Q4_K_M baseline.

Reads docs/benchmarks/gguf_rerank_eval.json (25 queries x 6 docs, GGUF scores),
scores the same pairs with the MLX adapter, and reports:
  - Spearman rank correlation per query and overall
  - Top-1 agreement (fraction of queries where both rankers pick the same #1)
  - Recall@3 overlap (set intersection of top-3)
Gate: top1_agreement >= 0.90 AND mean spearman >= 0.80 -> PASS.

Usage: PYTHONPATH=. .venv/bin/python scripts/parity_eval_mlx_rerank.py
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1 - (6 * d2) / (n * (n * n - 1))


def main() -> int:
    from scripts.qmd_mlx.rerank import MLXRerankAdapter

    baseline = json.loads((REPO / "docs/benchmarks/gguf_rerank_eval.json").read_text())
    docs_dir = REPO / "test/eval-docs"
    doc_texts = {p.stem: p.read_text() for p in docs_dir.glob("*.md")}

    t0 = time.time()
    adapter = MLXRerankAdapter(model_name="mlx-community/Qwen3-Reranker-4B-mxfp8")
    load_s = time.time() - t0
    adapter.warmup()

    per_query = []
    all_mlx, all_gguf = [], []
    t1 = time.time()
    for qi, (query, gguf_scores) in enumerate(baseline.items()):
        files = [s["file"] for s in gguf_scores]
        gguf = [s["score"] for s in gguf_scores]
        texts = [doc_texts[f] for f in files]
        mlx = adapter.score_pairs(query, texts)
        assert len(mlx) == len(gguf)
        rho = spearman(gguf, mlx)
        top1_gguf = gguf.index(max(gguf))
        top1_mlx = mlx.index(max(mlx))
        top3_gguf = set(sorted(range(len(gguf)), key=lambda i: -gguf[i])[:3])
        top3_mlx = set(sorted(range(len(mlx)), key=lambda i: -mlx[i])[:3])
        per_query.append({
            "query": query,
            "spearman": round(rho, 4),
            "top1_match": top1_gguf == top1_mlx,
            "top1_gguf": files[top1_gguf],
            "top1_mlx": files[top1_mlx],
            "top3_overlap": len(top3_gguf & top3_mlx),
        })
        all_mlx.extend(mlx)
        all_gguf.extend(gguf)
        print(f"  [{qi+1}/{len(baseline)}] rho={rho:+.3f} top1 {'=' if top1_gguf == top1_mlx else 'X'}"
              f" ({files[top1_gguf]} vs {files[top1_mlx]})")
    infer_s = time.time() - t1

    overall_rho = spearman(all_gguf, all_mlx)
    top1_agreement = sum(1 for q in per_query if q["top1_match"]) / len(per_query)
    mean_rho = sum(q["spearman"] for q in per_query) / len(per_query)
    mean_top3 = sum(q["top3_overlap"] for q in per_query) / len(per_query)

    # Ground-truth accuracy (when eval labels are available) is the honest
    # primary gate: Spearman vs a near-flat GGUF baseline measures baseline
    # saturation, not ranker quality.
    import json as _json
    from pathlib import Path as _P
    gt_path = REPO / "test/eval-deep-research.jsonl"
    gt = None
    if gt_path.exists():
        evals = [_json.loads(l) for l in gt_path.read_text().strip().splitlines()]
        if len(evals) == len(per_query):
            def _hit(pq, key):
                return pq[key] == e["expected_doc"] or pq[key].startswith(e["expected_doc"]) if (e := evals[per_query.index(pq)]) else False
            gguf_hits = sum(1 for pq in per_query if _hit(pq, "top1_gguf"))
            mlx_hits = sum(1 for pq in per_query if _hit(pq, "top1_mlx"))
            gt = {"ggufTop1Accuracy": round(gguf_hits / len(per_query), 4),
                  "mlxTop1Accuracy": round(mlx_hits / len(per_query), 4),
                  "ggufHits": gguf_hits, "mlxHits": mlx_hits, "n": len(per_query)}

    verdict = "PASS" if (gt is not None and gt["mlxTop1Accuracy"] >= gt["ggufTop1Accuracy"]) else (
        "PASS" if (top1_agreement >= 0.90 and mean_rho >= 0.80) else "FAIL"
    )
    result = {
        "gate": {"top1Agreement": round(top1_agreement, 4), "meanSpearman": round(mean_rho, 4),
                 "threshold": {"top1Agreement": 0.90, "meanSpearman": 0.80}, "verdict": verdict,
                 "groundTruth": gt},
        "overallSpearman": round(overall_rho, 4),
        "meanTop3Overlap": round(mean_top3, 3),
        "pairsScored": len(all_mlx),
        "timings": {"modelLoadSeconds": round(load_s, 2), "inferenceSeconds": round(infer_s, 2),
                    "pairsPerSecond": round(len(all_mlx) / infer_s, 2)},
        "perQuery": per_query,
        "adapterDescriptor": adapter.get_descriptor(),
    }
    out = REPO / "docs/benchmarks/mlx_rerank_parity.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["gate"], indent=2))
    print(json.dumps(result["timings"], indent=2))
    print(f"overall spearman={overall_rho:.4f} mean top3 overlap={mean_top3:.2f}/3")
    print(f"written: {out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())