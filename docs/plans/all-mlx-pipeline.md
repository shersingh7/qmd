# All-MLX Pipeline — Implementation Plan (Phase 2)

> **For Hermes:** Execute via Antigravity CLI (`agy`) ONLY after David picks
> models (Section 1). Model IDs are parameters — read them from the decision
> recorded at the top of this file before implementing.

**Goal:** Route 100% of QMD inference (embedding + reranking + query
expansion) through the MLX daemon on Apple Silicon. node-llama-cpp/GGUF
becomes an opt-in emergency fallback (env-gated, default off), not the hot
path. No GGUF model loads during normal `embed`, `query`, `vsearch`, MCP
serve, or nightly cron.

**Architecture:** Extend the existing `scripts/qmd_mlx/` service (Phase 1:
embedding runtime + bounded scheduler + binary protocol) with two new
adapters — rerank (single-token yes/no scoring head) and generate
(query-expansion completions) — behind the same single-GPU execution owner,
admission queue, readiness lifecycle, and telemetry. TypeScript routes all
three operations to the daemon when `embedBackend=mlx`; the embedding-space
contract (Phase 1) extends to rerank/generate descriptors.

**Tech stack:** Unchanged (TS/Bun + SQLite + Python/MLX). New: `mlx-lm`
(generate + rerank scoring) alongside the Phase-1 embedding dependency.
Repo-local ignored venv. No new native deps.

## 0. Decisions (locked Sep 7 2026 by David)

- `EMBED_MODEL = mlx-community/Qwen3-Embedding-4B-4bit-DWQ` (EMB-C)
- `RERANK_MODEL = mlx-community/Qwen3-Reranker-4B-mxfp8` (RR-A)
- `EXPAND_MODEL = mlx-community/Qwen3-1.7B-4bit` (EXP-A)
- Daemon supervision: launchd user agent, loopback-only, autostart (default)

## 1. Model options (verified Sep 7 2026 via HuggingFace API)

All repos are first-party `mlx-community` conversions (full HF repos with
`config.json` + tokenizer — no missing-head risk like the GGUF rerankers).

### Embedding (replaces `Qwen3-Embedding-0.6B-Q8_0` GGUF, 610 MB)

| ID | Repo | Weights | Note |
|---|---|---|---|
| EMB-A (closest) | `mlx-community/Qwen3-Embedding-0.6B-8bit` | ~0.63 GB | Same family, Q8-class like current GGUF. Minimal quality risk. |
| EMB-B (lightest) | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | ~0.34 GB | Half the RAM, small quality cost. |
| EMB-C (better) | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` | ~2.26 GB | Same family, large quality step. Comfortable on 32 GB. |
| EMB-D (best) | `mlx-community/Qwen3-Embedding-8B-4bit-DWQ` | ~4.26 GB | Top of the Qwen3 line (8B topped open MTEB). Heaviest. |
| EMB-E (alt family) | `mlx-community/embeddinggemma-300m-8bit` | ~0.33 GB | Google, tiny/fast, strong per-param. Different instruction format — adapter must use ITS prefix template, not Qwen's. |

"Same size = keep vectors" is FALSE in every case: Q8_0 GGUF vs MLX
8bit/4bit/DWQ/mxfp8 are different arithmetic → different numbers →
`qmd embed -f` required once. The Phase-1 contract check enforces this
(the descriptor includes quantization).

### Reranker (replaces `Voodisss Qwen3-Reranker-4B-Q4_K_M` GGUF, 2.3 GB)

| ID | Repo | Weights | Note |
|---|---|---|---|
| RR-A (same class) | `mlx-community/Qwen3-Reranker-4B-mxfp8` | ~4 GB (est. from param count; HEAD probe blocked, verify on download) | Keeps the crisp separation David chose 4B for. mxfp8 is the ONLY 4B MLX quant available. |
| RR-B (fast) | `mlx-community/Qwen3-Reranker-0.6B-4bit` | ~0.34 GB | ~10× smaller. Quality drop on hard queries (near-tied scores like the old 0.6B GGUF). |
| RR-C (middle) | `mlx-community/Qwen3-Reranker-0.6B-mxfp8` | ~0.61 GB | Same warning as RR-B, slightly better arithmetic. |

No 4bit 4B reranker exists upstream. Fallback: `mlx_lm.convert` the HF
`Qwen/Qwen3-Reranker-4B` locally to 4bit (adds a local-quant step + quality
validation task).

### Query expansion (replaces `tobi/qmd-query-expansion-1.7B-q4_k_m` GGUF, 1.2 GB)

| ID | Repo | Weights |
|---|---|---|
| EXP-A | `mlx-community/Qwen3-1.7B-4bit` | ~0.97 GB |
| EXP-B | `mlx-community/Qwen3-1.7B-8bit` | ~1.83 GB |

Smallest win (expansion is short + cached). EXP-A default.

### Resident-memory sketch (worst realistic combo: EMB-C + RR-A + EXP-A)

~2.3 + ~4 + ~1 ≈ 7–8 GB of 32 GB. Comfortable. EMB-A + RR-A + EXP-A ≈ 6 GB.

## 2. MLX rerank adapter (`scripts/qmd_mlx/rerank.py`)

Qwen3-Reranker scores via chat template + P("yes") vs P("no") on the final
token (same semantics as llama.cpp `pooling_type=RANK`).

1. Tests first (`test/python/test_mlx_rerank.py`): score ordering on a fixed
   synthetic pair set; determinism (same input → same score); batch
   cardinality/order; invalid/empty docs rejected; yes/no token IDs resolved
   from the LOADED tokenizer (never hardcoded); oversized pair truncation
   that keeps the query intact and truncates the document side.
2. Implement: format `query + document` with the model's rerank chat
   template, single forward pass, softmax over the yes/no token logits.
   Share the GPU execution owner with embedding (Section 5).
3. Parity gate: score the labeled retrieval fixture with BOTH the current
   Voodisss 4B GGUF (via node-llama-cpp, offline one-shot) and the MLX
   reranker; report Spearman correlation + top-1 agreement. Ship threshold:
   top-1 agreement ≥ 90% AND no Recall@10 regression, else stop and report.

## 3. MLX generation adapter (`scripts/qmd_mlx/generate.py`)

1. Tests (`test/python/test_mlx_generate.py`): prompt → completion contract
   (text, finish reason, token counts); temperature 0 determinism;
   max-tokens bound enforced; concurrent requests serialize on the GPU owner;
   context-overflow returns a typed error (no silent truncation).
2. Implement thin wrapper over `mlx-lm` streaming generate with bounded
   prefill/decode budgets sized for expansion prompts (short in, short out).
3. Expansion quality gate: run the existing expansion eval
   (`test/eval*.ts` harness or fixture) with GGUF vs MLX expansion;
   downstream Recall@10 must not regress.

## 4. Unified daemon (`scripts/qmd_mlx/server.py`)

1. Endpoints: existing `/embed`, `/embed-bin` + new `/rerank`, `/generate`.
   Shared response envelope + versioned model descriptors in headers.
2. One GPU execution owner for all three adapters (embedding batching stays
   concurrent internally; rerank/generate queue behind it). Rationale: three
   models resident, ONE active at a time — no contention, bounded peak.
3. Admission: separate queue slots per endpoint (embed bulk must not starve
   interactive rerank; cap total). `/ready` = 503 until ALL THREE selected
   models validate + warm. `/health` stays queue-independent.
4. Cold-start budget: measure and record (3 model loads). Preload order:
   embed → rerank → generate. Optional lazy-generate (load on first
   expansion call) if cold start > 60s — decide with measurements.

## 5. TypeScript wiring (`src/`, `src/embedding/`)

1. `rerank()` and `expandQuery()` in `src/store.ts` route to the MLX client
   when backend is `mlx` (mirroring the Phase-1 embed routing). Same
   single-flight init, full-body deadlines, finite-value validation.
2. Contract extension: rerank/generate descriptors (model+revision+quant)
   logged with queries; mismatch vs startup descriptor = loud error.
3. GGUF path: keep code, gate behind `QMD_GGUF_FALLBACK=1` (default: unset
   = fail-closed with an actionable "start the MLX daemon" message).
   Do NOT delete node-llama-cpp in this phase (rollback safety).
4. CLI/MCP surface unchanged. `qmd status` reports all three MLX models +
   daemon health.

## 6. Daemon supervision + packaging

1. `scripts/qmd-mlx-daemon.sh` start/stop/status wrapper (loopback bind,
   log to repo-ignored path or `~/.cache/qmd/`).
2. launchd user agent plist (KeepAlive, RunAtLoad) — install documented,
   NOT auto-installed. Port: keep 8787 (embed) — single port for all three
   endpoints, no new ports.
3. `requirements` updated (`mlx-lm` pinned range validated against the
   chosen models); repo-local venv bootstrap documented in README/CLAUDE.
4. Published package check: `npm pack --dry-run` must include the new
   modules (fix the `files` omission found in Phase 1 if still present).

## 7. Migration (David's live index, 136k vectors)

1. Backup `~/.cache/qmd/index.sqlite` first. Snapshot rollback = restore file.
2. Start daemon, verify `/ready` (all three models).
3. `qmd embed -f` (full re-embed in the new space; ~15–25 min at measured
   throughput, run in background).
4. Recall spot-check: 15–20 real queries from history, `candidateLimit: 15`
   then 40 on second pass; compare top-1/top-3 vs the old GGUF index notes.
   Any systematic miss → keep GGUF backup live, investigate, do NOT delete.
5. Nightly cron: add daemon-health preflight (`/ready` or start it);
   `QMD_EMBED_BACKEND=mlx` (+ new `QMD_MLX_RERANK/GENERATE` flags if added)
   in the cron environment. GGUF fallback stays available via env flag.
6. Local patches script (`apply-qmd-local-patches.py`): extend to the new
   routing if it touches patched regions; verify after every reinstall.

## 8. Benchmarks + acceptance

1. `scripts/bench_mlx.py` extended: rerank pairs/sec (batch 1/8/32, p50/p95),
   expansion ms/completion, cold-start total, 3-model resident RSS, JSON vs
   binary on the new endpoints.
2. End-to-end `qmd query` stage timings (expand / lex / vec / rerank) on 5
   fixed queries, warm + cold, GGUF-baseline vs all-MLX. Publish in
   `docs/benchmarks/all-mlx-results.md` with real numbers only.
3. Gates: `bun run build` 0; full TS suite 0 failed; pytest all pass;
   parity gates in Sections 2–3 met; `git diff --check` clean; no commits
   (leave for review); no live-index mutation by the agent (migration is
   David-run, Section 7).

## 9. Out of scope (explicit)

- Deleting node-llama-cpp dependency or GGUF code paths.
- ANN/vector-DB swap, Rust/Swift rewrite (same rationale as Phase 1).
- Custom local quantization (only if RR-A quality gate fails → then add).
- Changing rerank candidate defaults, RRF, chunking, or collections.
