#!/usr/bin/env python3
"""Round 4: locally-converted 4B 4-bit reranker — discrimination smoke test."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.qmd_mlx.rerank import MLXRerankAdapter

MODEL = str(Path.home() / ".cache/qmd/models/qwen3-reranker-4b-mlx-4bit")

a = MLXRerankAdapter(model_name=MODEL)
docs_dir = REPO / "test/eval-docs"
rel = (docs_dir / "distributed-systems-overview.md").read_text()
irrel = (docs_dir / "remote-work-policy.md").read_text()
cookie = ("Making chocolate chip cookies requires flour, sugar, butter, baking soda, "
          "and chocolate chips. Bake at 190C for 10 minutes.")

q = "tradeoff between data consistency and availability in distributed systems"
q2 = "how employees request vacation days"

print("scores [rel_doc, irrel_doc, cookie]:")
print("  consistency query:", [f"{s:.4f}" for s in a.score_pairs(q, [rel, irrel, cookie])])
print("  vacation query:   ", [f"{s:.4f}" for s in a.score_pairs(q2, [irrel, rel])])