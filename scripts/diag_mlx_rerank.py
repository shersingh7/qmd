#!/usr/bin/env python3
"""Diagnose MLX reranker discrimination failure: adapter bug vs template vs conversion.

Tests 4 hypotheses with minimal GPU time:
  H1 saturation/no-discrimination on full docs
  H2 truncation (2048 cap) cutting relevant content
  H3 template format (single \n vs official \n\n separators)
  H4 broken mxfp8 conversion (cross-check with 0.6B-4bit)
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import mlx.core as mx
import mlx_lm
from transformers import AutoTokenizer


def p_yes_from_logits(model, tok, prompt, yes_id, no_id):
    tokens = tok.encode(prompt)
    input_ids = mx.array([tokens], dtype=mx.int32)
    logits = model(input_ids)
    last = logits[0, -1, :]
    diff = float(last[yes_id]) - float(last[no_id])
    return 1.0 / (1.0 + float(np.exp(-diff)))


def official_format(tok, instruction, query, doc):
    text = [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no.\""},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"},
    ]
    return tok.apply_chat_template(text, tokenize=False, add_generation_prompt=True)


def run(model_name: str, label: str):
    print(f"\n{'='*60}\n{label}: {model_name}")
    model, tok_wrap = mlx_lm.load(model_name)
    tok = getattr(tok_wrap, "_tokenizer", tok_wrap)
    yes_id = int(tok.encode("yes", add_special_tokens=False)[0])
    no_id = int(tok.encode("no", add_special_tokens=False)[0])
    print(f"yes={yes_id} no={no_id}")

    docs_dir = REPO / "test/eval-docs"
    rel = (docs_dir / "distributed-systems-overview.md").read_text()
    irrel = (docs_dir / "remote-work-policy.md").read_text()
    print(f"rel doc: {len(rel)} chars | irrel doc: {len(irrel)} chars")

    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    q = "tradeoff between data consistency and availability in distributed systems"
    q_irrel = "chocolate chip cookie recipe ingredients"

    t0 = time.time()
    cases = [
        ("A: relevant query + relevant doc (FULL)", official_format(tok, instruction, q, rel)),
        ("B: relevant query + irrelevant doc (FULL)", official_format(tok, instruction, q, irrel)),
        ("C: irrelevant query + relevant doc (FULL)", official_format(tok, instruction, q_irrel, rel)),
        ("D: relevant query + relevant doc (SNIPPET 500c)", official_format(tok, instruction, q, rel[:500])),
        ("E: relevant query + irrelevant doc (SNIPPET 500c)", official_format(tok, instruction, q, irrel[:500])),
    ]
    for name, prompt in cases:
        p = p_yes_from_logits(model, tok, prompt, yes_id, no_id)
        print(f"  {name}: p_yes = {p:.6f}")

    # Show the prompt tail so we can see exactly what the model scores
    tail = cases[0][1][-220:].replace("\n", "\\n")
    print(f"  prompt tail: ...{tail}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    del model
    mx.clear_cache()


if __name__ == "__main__":
    run("mlx-community/Qwen3-Reranker-4B-mxfp8", "H1-H3")
    run("mlx-community/Qwen3-Reranker-0.6B-4bit", "H4 cross-check")