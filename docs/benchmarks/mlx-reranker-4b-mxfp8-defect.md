# MLX Reranker Parity Investigation — Sep 7 2026

## Verdict

**`mlx-community/Qwen3-Reranker-4B-mxfp8` is a defective conversion.** Everything
scores p(yes) ≈ 0.95–0.98 regardless of query-document relevance. It is unusable
as a reranker on any machine, not just M2.

## Evidence chain (all scripts in `scripts/`, eval artifacts in `docs/benchmarks/`)

1. **Parity gate FAILED** (`scripts/parity_eval_mlx_rerank.py →
   `docs/benchmarks/mlx_rerank_parity.json`): top-1 agreement 16%, mean Spearman
   0.097 vs the Voodisss 4B GGUF baseline (25 queries × 6 eval docs). One doc
   (`machine-learning-primer`) won across unrelated queries — systematic, not noise.

2. **Template hypothesis eliminated** (`scripts/diag_mlx_rerank.py`): the
   4B-mxfp8 renders the official Qwen3-Reranker prompt correctly (doc present,
   ends at `<|im_start|>assistant\n`) yet still shows near-zero discrimination:
   relevant 0.985 vs irrelevant 0.966 vs *cookie-recipe-for-a-distributed-systems-query* 0.966.

3. **Adapter + format proven correct by control model** (`scripts/diag_mlx_rerank2.py`):
   same adapter logic, same manual official format, different weights —
   - `Qwen3-Reranker-0.6B-4bit`: relevant 0.788 / irrelevant 0.00007 /
     cookie 0.000026 / wrong-domain 0.000001 — **crisp, correct behavior**
   - `Qwen3-Reranker-4B-mxfp8`: flat "yes" across every case (gap +0.019)

4. **mxfp8 runtime on M2 eliminated** (`scripts/diag_mlx_rerank3.py`):
   `Qwen3-Reranker-0.6B-mxfp8` discriminates perfectly on the same M2 Pro
   (gap +0.76). The emulated mxfp8 path is fine; the 4B mxfp8 *weights* are bad.

## Root cause

Unknown conversion defect in the published `mlx-community/Qwen3-Reranker-4B-mxfp8`
repo (all-yes logit saturation). Not worth reverse-engineering further — the fix
is to not use those weights.

## Fix in progress

Local conversion of the official `Qwen/Qwen3-Reranker-4B` to 4-bit affine (gs 64)
via `mlx_lm convert`, saved to `~/.cache/qmd/models/qwen3-reranker-4b-mlx-4bit`.
Parity gate re-run against this build is the acceptance test before any wiring.

## Side finding

`mlx-community/Qwen3-Reranker-0.6B-4bit` ships a chat template that silently
drops message content when applied via `apply_chat_template` (rendered
`<Query>: <Document>:` empty in diag round 1). The manual official format
works fine with its weights. Adapters must use the manual format or verify
template output.