#!/usr/bin/env python3
"""Round 3: isolate mxfp8-runtime-on-M2 vs the 4B-mxfp8 conversion specifically."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import mlx.core as mx
import mlx_lm

INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
SYS = ('Judge whether the Document meets the requirements based on the Query and '
       'the Instruct provided. Note that the answer can only be "yes" or "no."')


def manual_format(query, doc):
    return (
        f"<|im_start|>system\n{SYS}<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {INSTRUCT}\n\n<Query>: {query}\n\n<Document>: {doc}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def p_yes(model, tok, prompt, yes_id, no_id):
    tokens = tok.encode(prompt)
    logits = model(mx.array([tokens], dtype=mx.int32))
    last = logits[0, -1, :]
    return 1.0 / (1.0 + float(np.exp(-(float(last[yes_id]) - float(last[no_id])))))


def run(model_name):
    model, tok_wrap = mlx_lm.load(model_name)
    tok = getattr(tok_wrap, "_tokenizer", tok_wrap)
    yes_id = int(tok.encode("yes", add_special_tokens=False)[0])
    no_id = int(tok.encode("no", add_special_tokens=False)[0])
    rel = (REPO / "test/eval-docs/distributed-systems-overview.md").read_text()
    cookie = ("Making chocolate chip cookies requires flour, sugar, butter, baking soda, "
              "and chocolate chips. Bake at 190C for 10 minutes.")
    q = "tradeoff between data consistency and availability in distributed systems"
    p_rel = p_yes(model, tok, manual_format(q, rel), yes_id, no_id)
    p_cookie = p_yes(model, tok, manual_format(q, cookie), yes_id, no_id)
    print(f"{model_name}: rel={p_rel:.6f} cookie={p_cookie:.6f} gap={p_rel-p_cookie:+.4f}")
    del model
    mx.clear_cache()


if __name__ == "__main__":
    run("mlx-community/Qwen3-Reranker-0.6B-mxfp8")