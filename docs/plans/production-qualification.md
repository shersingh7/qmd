# MLX Production Qualification Plan

> **For Hermes:** execute phase-by-phase with explicit user confirmation before any commit, live-index write, daemon change, or model download.

**Goal:** Qualify the repaired MLX pipeline as a foolproof replacement for the live GGUF setup, with rollback rehearsed before cutover.

**Architecture:** Keep TS CLI/SQLite + Python single-owner MLX runtime. Promote embed/rerank/expand independently on measured evidence, not architectural purity.

**Tech Stack:** Bun/Vitest TS, Python MLX (mlx 0.32.2, mlx-lm 0.31.3), SQLite FTS5 + sqlite-vec, launchd-supervised daemon.

---

## Phase 1 — Freeze reproducible candidate (IN PROGRESS)

### Baseline inventory (recorded 2026-09-08, America/Toronto)

- Repo: `/Users/shersingh/github/qmd-mlx-search`, branch `main`, HEAD `0e0ce47`
- Working tree: 28 modified + 6 untracked files (full repair pass, uncommitted)
- `qmd 2.1.0 (0e0ce47)`
- Live GGUF index: `~/.cache/qmd/index.sqlite` — 958 MB, 11,950 files, 144,698 vectors, 15 collections, updated ~15h ago
- GGUF backup: `index.sqlite.gguf-backup-20260907` (971 MB, Sep 7 17:10) — VERIFY FRESHNESS before relying on it
- Shadow MLX index: `mlx-shadow.sqlite` (321 MB, Sep 7) — stale, do not trust
- Active config: no `QMD_*` env overrides → backend `gguf`, model `hf:ggml-org/embeddinggemma-300M-Q8_0.gguf`
- GGUF model files present: embeddinggemma-300M-Q8_0, Qwen3-Embedding-0.6B-Q8_0, Qwen3-Embedding-4B-Q4_K_M, Qwen3-Reranker-4B-Q4_K_M, qwen3-reranker-0.6b-q8_0, qmd-query-expansion-1.7B-q4_k_m
- Local MLX weights present: qwen3-embedding-4b-mlx-4bit-affine, qwen3-reranker-4b-mlx-4bit, qwen3-reranker-4b-mlx-8bit, qwen3-embedding-4b-hf-fixed
- MLX daemon: running (pid 1446), `/ready` true — LEAVE ALONE during qualification
- Toolchain: Bun 1.3.8, node v22.22.3, Python 3.12.13, mlx 0.32.2, mlx-lm 0.31.3
- Verified gates: `bun run build` exit 0; Vitest 774 passed/72 skipped (23 files); offline pytest 99 passed/10 real-model skipped; `git diff --check` clean

### Step 1a: fresh GGUF backup + restore rehearsal (do FIRST, user runs)

```bash
cp ~/.cache/qmd/index.sqlite ~/.cache/qmd/index.sqlite.gguf-backup-20260908
ls -la ~/.cache/qmd/index.sqlite*
# Restore rehearsal (only if needed):
# cp ~/.cache/qmd/index.sqlite.gguf-backup-20260908 ~/.cache/qmd/index.sqlite
```

### Step 1b: checkpoint commit (needs explicit user confirmation)

```bash
git add -A
git commit -m "qual: MLX reliability repair candidate (verified: build 0, vitest 774+72, pytest 99+10)"
git log --oneline -3
```

Expected: new local commit on `main`, NO push.

## Phase 2 — Operational safeguards (delegate to agy)

1. External resource watchdog (separate process): RSS, memory pressure, swap, responsiveness; kills TEST process only on breach.
2. Control-path health that answers during inference saturation.
3. Fixture tests: stalled kernel + overload → safe failure, resource release, clean recovery.

Pass: deliberately stalled/overloaded fixtures fail safe without Mac impact.

## Phase 3 — Single-model real-model trials (separate port, one model at a time)

Embed → rerank → expand, each: correctness first, then sustained-load resource envelope. NO full-corpus embedding.

## Phase 4 — GGUF comparison (equivalent-model + system-vs-system)

p50/p95 latency, sustained indexing with interactive queries, RSS/swap/responsiveness, held-out retrieval quality (Recall@10, nDCG@10, MRR). Gate: ≥20% practical win, zero material retrieval regression, acceptable daily-driver impact.

## Phase 5 — Isolated MLX index build (subset → batches, interrupt/resume, no overwrite of GGUF index)

## Phase 6 — Cutover with rehearsed rollback (GGUF stack + matching index kept intact)
