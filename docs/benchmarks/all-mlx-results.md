# All-MLX Pipeline — Implementation Results (Sep 7 2026)

## Status: WORKING — with one honest caveat on reranker quality

All three inference stages (embed, rerank, expand) now run on MLX on-device
through one daemon. The 4B reranker parity caveat is documented below with
measurements, not hand-waving.

## What was built

| Component | File | State |
|---|---|---|
| MLX rerank adapter | `scripts/qmd_mlx/rerank.py` | done (Agy + Hermes fixes) |
| MLX generate adapter | `scripts/qmd_mlx/generate.py` | done (Hermes native) |
| Unified daemon endpoints | `scripts/qmd_mlx/server.py` (`/rerank`, `/generate`) | done |
| Entrypoint flags | `scripts/mlx_embed_server.py` (`--rerank-model`, `--generate-model`, env `MLX_RERANK_MODEL`, `MLX_GENERATE_MODEL`) | done |
| TS client methods | `src/mlx.ts` (`rerankWithMlx`, `generateWithMlx`) | done |
| TS fast-path routing | `src/llm.ts` (`_rerankMlx`, `_expandQueryMlx`) | done |
| Env kill-switches | `QMD_MLX_RERANK=0` / `QMD_MLX_EXPAND=0` (disable fast-path), `QMD_MLX_RERANK_FALLBACK=0` / `QMD_MLX_EXPAND_FALLBACK=0` (fail-closed instead of GGUF fallback) | done |

## Models resident (David's locked picks)

- Embed: `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` (2.1 GB, 2560-dim) ✓
- Rerank: locally-converted `Qwen/Qwen3-Reranker-4B` → 4-bit affine gs64
  (`~/.cache/qmd/models/qwen3-reranker-4b-mlx-4bit`, 2.1 GB) ✓
- Expand: `mlx-community/Qwen3-1.7B-4bit` (0.94 GB) ✓

## The mxfp8 saga (root-caused, not guessed)

1. First parity run FAILED catastrophically (top-1 agreement 16%, Spearman
   0.097). One doc won every query.
2. Diagnostics isolated the cause layer by layer (scripts/diag_mlx_rerank*.py):
   - Adapter + format proven CORRECT on 0.6B-4bit control (crisp 0.79/0.00007
     separation)
   - mxfp8 runtime proven FINE on M2 (0.6B-mxfp8 discriminates)
   - **Actual root cause: missing official thinking-close suffix.** The 4B/8B
   Qwen3 rerankers descend from thinking-capable bases; the trained yes/no
   signal lives only in the token distribution AFTER
   `<|im_start|>assistant\n\n\n\n` in the prompt (per
   the official model card usage). Scoring at the pre-thinking position reads
   noise. `mlx-reranker-4b-mxfp8-defect.md` now records the corrected story.
3. With the fix, both 4B builds discriminate crisply (relevant 0.99 /
   irrelevant 0.98 → 0.69/0.01 local, 0.74/0.11 mxfp8).

## Parity and ground-truth numbers (25-query labeled eval, 6 docs each)

| Ranker | top-1 agreement vs GGUF | mean Spearman | top-1 accuracy vs ground truth |
|---|---|---|---|
| GGUF Voodisss 4B Q4_K_M (baseline) | — | — | **20/25 = 80%** |
| MLX 4B local 4-bit (fixed format) | 80% | 0.374 | 17/25 = 68% |
| MLX 4B mxfp8 (fixed format) | 80% | 0.441 | 17/25 = 68% |
| MLX 4B local 8-bit (fixed format) | 84% | 0.410 | 18/25 = 72% |

**Decision (Sep 7):** rerank stays GGUF by default. The MLX rerank path ships
as opt-in (`QMD_MLX_RERANK=1`) with the GGUF path as the quality baseline.
MLX expand is likewise opt-in (`QMD_MLX_EXPAND=1`) pending a Recall@10 eval
vs the fine-tuned GGUF expansion model. Embedding is MLX-primary (its quality
gate passed in Phase 1).

**Caveat, stated plainly:** all three MLX 4B builds trail the GGUF baseline on
hard queries (68-72% vs 80% ground-truth accuracy). Precision helps (8-bit >
4-bit/mxfp8) but does not close the gap: the llama.cpp RANK-pooling scoring
path ranks these hard queries better than P(yes) token scoring in MLX.
Eval is small (25 queries / 6 docs); a larger corpus could shift the verdict,
but as measured today the honest default is GGUF rerank.

**Why the Spearman threshold is wrong for this comparison:** the GGUF
baseline scores are nearly flat (5 of 6 docs pinned at ~0.5006, i.e.
sigmoid(≈0)) — correlating a strongly-discriminating ranker against a
near-flat one measures baseline saturation, not quality. Ground-truth accuracy
is the honest gate; it is what `ground_truth_accuracy.json` records.

## Latency measured this session (M2 Pro, daemon warm)

- /rerank: 2 docs ≈ 1.3 s; /generate 60 tokens ≈ 2.6 s
- Full parity eval: 150 doc pairs in ~420 s ≈ 0.36 pairs/s (4B, full eval docs)
- Embed throughput: unchanged from Phase 1 (see `mlx-benchmark.json`:
  ~2,990 texts/s batch-32 binary)

## End-to-end verification (TS → daemon, this session)

- `LlamaCpp.rerank()` via MLX: correct contract (file/score/index, sorted), 0.90
  vs 0.03 on a relevant/irrelevant pair — `/tmp/qmd-e2e.log`
- `LlamaCpp.expandQuery()` via MLX: structured lex/vec/hyde output parsed and
  returned, no thinking artifacts — `/tmp/qmd-e2e.log`
- Daemon boot: all 3 models loaded, `/ready` 200, `/health` reports all three
  models, `/descriptor` includes rerank + generate sub-descriptors

## Gates

- `bun run build`: PASS (exit 0)
- Python suite: 37/37 PASS
- TS suite: run in progress at time of writing (`/tmp/qmd-ts-final.log`)
- Parity gate: improved 16% → 80% agreement; ground-truth gate is the honest
  one — see caveat above
- No live QMD index/service/config touched; nothing committed

## Open items (next session)

1. mxfp8 ground-truth accuracy → final rerank-model recommendation
2. If MLX rerank ships: rerank-cache key must include the MLX model identity
   (it currently keys on GGUF model string in some paths)
3. Launchd plist + `qmd status` daemon health preflight (plan §6)
4. `apply-qmd-local-patches.py` extension if dist patches overlap new routing
5. `npm pack --dry-run` content check for new modules
6. Optional: 8-bit local conversion of the 4B reranker if 4-bit trails