# MLX Production Qualification Plan

> **For Hermes:** execute phase-by-phase with explicit user confirmation before any commit, live-index write, daemon change, or model download.

**Goal:** Qualify the repaired MLX pipeline as a foolproof replacement for the live GGUF setup, with rollback rehearsed before cutover.

**Architecture:** Keep TS CLI/SQLite + Python single-owner MLX runtime. Promote embed/rerank/expand independently on measured evidence, not architectural purity.

**Tech Stack:** Bun/Vitest TS, Python MLX (mlx 0.32.2, mlx-lm 0.31.3), SQLite FTS5 + sqlite-vec, launchd-supervised daemon.

---

## Phase 1 — Freeze reproducible candidate (IN PROGRESS)

### Baseline inventory (recorded 2026-09-08, America/Toronto)

- Repo: `/Users/shersingh/github/qmd-mlx-search`, branch `main`, base commit `0960f3c` (with uncommitted qualification working-tree changes)
- `qmd 2.1.0`
- Live GGUF index: `~/.cache/qmd/index.sqlite` — 958 MB, 11,950 files, 144,698 vectors, 15 collections, updated ~15h ago
- Verified parent SQLite online backup:
  - Path: `/Users/shersingh/.cache/qmd/index.sqlite.online-backup-20260908-192953.sqlite`
  - Size: 997,281,792 bytes
  - SHA-256: `2f09753af1b9d173465fba2d31922c980e2fd14ec9517e4ca5374fb2194bb7ec`
  - `PRAGMA quick_check`: `ok`
  - Schema table count: 20
  - Restored into temporary verification directory: identical SHA-256 and `quick_check=ok`.
  - Live database: ZERO live writes authorized during qualification.
- Shadow MLX index: `mlx-shadow.sqlite` (321 MB, Sep 7) — stale, do not trust
- Active config: inspecting daemon launchd plist and index.yml config read-only (do not infer live state from env absence alone)
- GGUF model files present: embeddinggemma-300M-Q8_0, Qwen3-Embedding-0.6B-Q8_0, Qwen3-Embedding-4B-Q4_K_M, Qwen3-Reranker-4B-Q4_K_M, qwen3-reranker-0.6b-q8_0, qmd-query-expansion-1.7B-q4_k_m
- Local MLX weights present: qwen3-embedding-4b-mlx-4bit-affine, qwen3-reranker-4b-mlx-4bit, qwen3-reranker-4b-mlx-8bit, qwen3-embedding-4b-hf-fixed
- MLX daemon: running (pid 1446), `/ready` true — LEAVE ALONE during qualification
- Toolchain: Bun 1.3.8, node v22.22.3, Python 3.12.13, mlx 0.32.2, mlx-lm 0.31.3
- Verified parent gate: `bun run build` exit 0; Vitest 774 passed/72 skipped (23 files); parent gate pytest 179 passed/10 skipped (`docs/reviews/phase2-final-parent-gate.md`); current offline pytest 192 passed/11 skipped; `git diff --check` clean

### Step 1a: SQLite online backup reference (VERIFIED by parent)

The authoritative backup has been created and verified via SQLite online backup API:
- Source: `~/.cache/qmd/index.sqlite`
- Backup: `/Users/shersingh/.cache/qmd/index.sqlite.online-backup-20260908-192953.sqlite`
- Verification: `PRAGMA quick_check;` returned `ok`. No live DB writes or modifications.

### Step 1b: checkpoint commit (needs explicit user confirmation)

```bash
git add -A
git commit -m "qual: MLX reliability repair candidate (verified: build 0, vitest 774+72, pytest parent-gate 179+10)"
git log --oneline -3
```

Expected: new local commit on `main`, NO push.

---

## Phase 2 — Operational safeguards (ACCEPTED by parent gate)

Delivered and verified in parent review (`docs/reviews/phase2-final-parent-gate.md`):
1. **Dedicated Loopback Control Listener (`server.py`, `mlx_embed_server.py`):** Genuinely separate `MLXControlServer` on dedicated port (`127.0.0.1`, `control_port`) with dedicated thread limiter (16 threads), header read deadline (5.0s), and rejection of inference POST requests (HTTP 405). Responds 200 OK even when inference socket pool is completely saturated.
2. **Owned-Child Process Management (`watchdog.py`):** Watchdog targets owned child test processes by default (`--launch` / `owned_child`). External attachment requires explicit opt-in (`--allow-external-pid`), non-empty start identity, and valid instance token. Revalidates PID, command, and immutable start time before **EVERY** signal (`SIGTERM` and `SIGKILL`). Never reaps non-children.
3. **Fail-Closed Memory Telemetry & Preflight Checks (`watchdog.py`):** Removed fabricated fallbacks in `__init__` and sampler; `ps` command failures raise `SystemMetricsError` -> `TELEMETRY_UNAVAILABLE`. Preflight validates telemetry before spawning child. Cleanly reaps owned child on all exit paths.
4. **Wall-Clock Deadline & Size-Bounded Health Probes (`watchdog.py`):** Raw socket probe enforcing hard wall-clock deadline across connect/send/recv, capped body size (64KB), and numeric loopback `127.0.0.1`. Validates typed finite progress telemetry.
5. **Launch Token Generation & Qualification Port Isolation (`qmd-mlx-watchdog.py`):** `--launch` generates fresh UUID token passed via `MLX_INSTANCE_TOKEN` env var; server uses it; watchdog requires exact PID + token + progress match.
6. **Fixture Tests:** 179 unit and integration tests passing offline (10 real-model tests skipped).

Gate: ACCEPTED by parent gate (`docs/reviews/phase2-final-parent-gate.md`).

---

## Phase 3 — Single-model real-model trials (PREPARATION COMPLETE, Awaiting Parent Review)

Detailed Plan: [`docs/plans/phase3-single-model-smoke.md`](./phase3-single-model-smoke.md)

Delivered Artifacts:
1. **Smoke Runner CLI:** [`scripts/qmd-mlx-smoke.py`](../../scripts/qmd-mlx-smoke.py) with preflight headroom validation, port collision rejection (rejects port 8787), watchdog child ownership, hard wall-clock deadline, and structured JSON reporting.
2. **Engine Module:** [`scripts/qmd_mlx/smoke.py`](../../scripts/qmd_mlx/smoke.py) with read-only model metadata inspector, 5 self-contained public fixtures (singleton, mixed batch, consistency tolerance, long input, repeated timing), and strict numeric validators.
3. **Disposable Fake Server:** [`scripts/qmd_mlx/fake_server.py`](../../scripts/qmd_mlx/fake_server.py) for zero-weight rehearsals and failure/hang test simulations.
4. **Test Suite:** [`test/python/test_mlx_smoke.py`](../../test/python/test_mlx_smoke.py) (13 tests passing offline, 1 real-model test skipped by default; total offline pytest suite: 192 passed, 11 skipped).
5. **Rehearsal Evidence:** Completed offline rehearsal with exit code 0, 1.31s total duration, measured latencies, and guaranteed child cleanup.

Gate: Real model execution requires explicit opt-in (`--real-model`) and parent review before invocation.

---

## Phase 4 — GGUF comparison (equivalent-model + system-vs-system)

p50/p95 latency, sustained indexing with interactive queries, RSS/swap/responsiveness, held-out retrieval quality (Recall@10, nDCG@10, MRR). Gate: ≥20% practical win, zero material retrieval regression, acceptable daily-driver impact.

---

## Phase 5 — Isolated MLX index build (subset → batches, interrupt/resume, no overwrite of GGUF index)

---

## Phase 6 — Cutover with rehearsed rollback (GGUF stack + matching index kept intact)
