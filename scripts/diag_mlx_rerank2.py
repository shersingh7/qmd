#!/usr/bin/env python3
"""Round 2: manual official Qwen3-Reranker format, both models, sharper negatives."""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import mlx.core as mx
import mlx_lm

INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
SYS = ('Judge whether the Document meets the requirements based on the Query and '
       'the Instruct provided. Note that the answer can only be "yes" or "no."')


def manual_format(query: str, doc: str) -> str:
    return (
        f"<|im_start|>system\n{SYS}<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {INSTRUCT}\n\n<Query>: {query}\n\n<Document>: {doc}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def p_yes(model, tok, prompt, yes_id, no_id):
    tokens = tok.encode(prompt)
    input_ids = mx.array([tokens], dtype=mx.int32)
    logits = model(input_ids)
    last = logits[0, -1, :]
    diff = float(last[yes_id]) - float(last[no_id])
    return 1.0 / (1.0 + float(np.exp(-diff)))


def run(model_name: str, label: str):
    print(f"\n{'='*64}\n{label}: {model_name}")
    model, tok_wrap = mlx_lm.load(model_name)
    tok = getattr(tok_wrap, "_tokenizer", tok_wrap)
    yes_id = int(tok.encode("yes", add_special_tokens=False)[0])
    no_id = int(tok.encode("no", add_special_tokens=False)[0])

    docs_dir = REPO / "test/eval-docs"
    rel = (docs_dir / "distributed-systems-overview.md").read_text()
    irrel = (docs_dir / "remote-work-policy.md").read_text()
    cookie = ("Making chocolate chip cookies requires flour, sugar, butter, baking soda, "
              "and chocolate chips. Bake at 190C for 10 minutes.")

    q = "tradeoff between data consistency and availability in distributed systems"
    q2 = "how employees request vacation days"

    cases = [
        ("rel query / rel doc (full)", manual_format(q, rel)),
        ("rel query / irrel doc (full)", manual_format(q, irrel)),
        ("rel query / cookie doc", manual_format(q, cookie)),
        ("vacation query / vacation doc", manual_format(q2, irrel)),
        ("vacation query / distributed doc", manual_format(q2, rel)),
    ]
    t0 = time.time()
    scores = []
    for name, prompt in cases:
        p = p_yes(model, tok, prompt, yes_id, no_id)
        scores.append(p)
        print(f"  {name}: p_yes = {p:.6f}")
    gap = scores[0] - scores[1]
    print(f"  discrimination gap (rel - irrel): {gap:+.4f} | elapsed {time.time()-t0:.1f}s")
    del model
    mx.clear_cache()
    return scores


if __name__ == "__main__":
    run("mlx-community/Qwen3-Reranker-0.6B-4bit", "0.6B manual format")
    run("mlx-community/Qwen3-Reranker-4B-mxfp8", "4B manual format")