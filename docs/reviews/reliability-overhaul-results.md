# Reliability Overhaul Implementation & Validation Report

**Date:** 2026-09-07  
**Scope:** Whole-Architecture Reliability Overhaul — Python MLX Engine & TypeScript Storage/Routing  
**Target Repository:** `qmd-mlx-search`  
**Environment:** macOS Darwin arm64, Bun v1.3.8, Python 3.12.13, MLX Metal  

---

## 1. Executive Summary

This report documents the implementation and verification of the whole-architecture reliability overhaul specified in [`docs/reviews/whole-architecture-review-scope.md`](file:///Users/shersingh/github/qmd-mlx-search/docs/reviews/whole-architecture-review-scope.md) and [`docs/plans/reliability-first-overhaul.md`](file:///Users/shersingh/github/qmd-mlx-search/docs/plans/reliability-first-overhaul.md).

All implementation tasks were executed within repository boundaries and temporary test fixtures. No modifications were made to installed production QMD, global configurations, live/shadow databases, launchd service configurations, or running daemons.

All verification builds and test suites execute cleanly with zero failures and exit code 0:
- **TypeScript Build (`bun run build`):** Clean compilation to `dist/`, exit code 0.
- **TypeScript Test Suite (`CI=true bun run test`):** 23 test files passed, 765 tests passed, 72 skipped (837 total), exit code 0.
- **Python MLX Test Suite (`PYTHONPATH=. .venv/bin/pytest test/python/ -v`):** 9 test modules, 71 tests passed, exit code 0.

---

## 2. Whole-Architecture Findings & Implemented Fixes

### A. Python Inference Engine (`scripts/qmd_mlx/`)

| Component / Finding | Description of Fix | Verification Tests |
| :--- | :--- | :--- |
| **Model Residency Registration** | Registered rerank and generate adapters in `ModelResidencyManager` with dedicated memory budgets, single-flight locking, and idle-unload timers. | [`test/python/test_mlx_model_manager.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_model_manager.py), [`test/python/test_mlx_server.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_server.py) |
| **HTTP Security & Headers** | Added strict `Origin` and `Referer` validation alongside `Host` validation to prevent DNS rebinding and cross-site hijacking. Added `Content-Type: application/json` requirement on POST requests with JSON payloads. | [`test/python/test_mlx_server.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_server.py) |
| **Connection & Thread Bounds** | Bounded HTTP handler concurrency with a 64-thread Semaphore returning HTTP 429 when overloaded. Configured 30.0s socket connection read deadlines. | [`test/python/test_mlx_server.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_server.py) |
| **Elimination of Silent Truncation** | Replaced silent truncation in tokenization, prompt generation, and pair reranking with explicit typed errors (`InvalidInputError`, `GenerateError`, `RerankError`) when token limits are exceeded. | [`test/python/test_mlx_rerank.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_rerank.py), [`test/python/test_mlx_generate.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_generate.py), [`test/python/test_mlx_batching.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_batching.py) |
| **Empty Input Validation** | Reject empty or whitespace-only inputs at protocol boundary across embed, rerank, and generate request endpoints. | [`test/python/test_mlx_protocol.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_protocol.py), [`test/python/test_mlx_rerank.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_rerank.py) |
| **Model Tuning & OOM Budget Reduction** | Wired `tune_for_model` on adapter resolution. When an OOM occurs, halved `max_batch_tokens` dynamically to protect subsequent batches. | [`test/python/test_mlx_faults.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_faults.py), [`test/python/test_mlx_batching.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_batching.py) |
| **Transient Logits Optimization** | In Qwen3 reranking adapter, avoided allocating `[batch, seq_len, 152064]` full-vocabulary logits by extracting the last hidden states `[batch, hidden_dim]` and projecting directly onto `(w_yes - w_no)`. | [`test/python/test_mlx_rerank.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_rerank.py) |
| **Executor Health & State Bounding** | Gated `/health` and `/ready` endpoints on `executor.is_alive()`. Bounded `compiled_shapes` cache to 256 entries and deduplicated idle callbacks with a max bound of 32 entries. | [`test/python/test_mlx_server.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_server.py), [`test/python/test_mlx_executor.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_executor.py) |
| **Micro-Batch Activity Touching** | Added `touch("embed")`, `touch("rerank")`, and `touch("generate")` activity touches on every micro-batch execution to maintain accurate idle-unload timers during long multi-batch runs. | [`test/python/test_mlx_runtime.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_runtime.py), [`test/python/test_mlx_rerank.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_rerank.py) |

---

### B. TypeScript Architecture & Storage (`src/`)

| Component / Finding | Description of Fix | Verification Tests |
| :--- | :--- | :--- |
| **Partial-Document Resume Predicate** | Updated `getPendingEmbeddingDocs`, `getHashesNeedingEmbedding`, and `getHashesForEmbedding` to join on `content_chunk_expectations` so documents missing tail chunks are properly resumed rather than skipped. | [`test/indexing-resume.test.ts`](file:///Users/shersingh/github/qmd-mlx-search/test/indexing-resume.test.ts), [`test/store.test.ts`](file:///Users/shersingh/github/qmd-mlx-search/test/store.test.ts) |
| **Resume Re-Chunk Optimization** | Persisted chunk count expectations per `(hash, strategy)` in `content_chunk_expectations` table. On resume, documents with satisfied chunk expectations avoid tokenization re-chunking. Prepared statements are cached and reused across batches. | [`test/indexing-resume.test.ts`](file:///Users/shersingh/github/qmd-mlx-search/test/indexing-resume.test.ts) |
| **Strict All-MLX Mode** | Added `strictMlx` configuration option in `src/llm.ts`. When active, all fallback paths to GGUF (`llama-cpp-embed`, `llama-cpp-rerank`, `llama-cpp-generate`) are fail-closed. | [`test/sdk.test.ts`](file:///Users/shersingh/github/qmd-mlx-search/test/sdk.test.ts) |
| **SDK Durable Job Routing** | Routed SDK `store.embed()` through `runDurableIndexingJob(internal, options)` to ensure all SDK programmatic embedding operations maintain checkpoints and fingerprint safety. | [`test/sdk.test.ts`](file:///Users/shersingh/github/qmd-mlx-search/test/sdk.test.ts) |
| **Bounded Retry & Abort Propagation** | Added strict session validity and `AbortSignal` checks before and during per-chunk fallback iterations to prevent runaway retries upon abort or expired sessions. | [`test/indexing-resume.test.ts`](file:///Users/shersingh/github/qmd-mlx-search/test/indexing-resume.test.ts) |
| **Timeout & Config Precedence** | Aligned `DEFAULT_MLX_TIMEOUT_MS` in `config.ts` to `300_000` ms (matching client/daemon defaults) and enforced caller-first precedence for `mlxBinary`. | [`test/mlx.test.ts`](file:///Users/shersingh/github/qmd-mlx-search/test/mlx.test.ts) |
| **CLI SIGINT Handling** | Attached `AbortController` and `process.on('SIGINT')` in CLI `vectorIndex`, gracefully cancelling the active durable job and cleaning up terminal cursor states. | [`test/cli.test.ts`](file:///Users/shersingh/github/qmd-mlx-search/test/cli.test.ts) |
| **Canonical Space ID Signature** | Unified checkpoint descriptor fingerprinting with canonical `computeEmbeddingSpaceId` from `contract.ts`, including `revision` and `quantization`. | [`test/indexing-resume.test.ts`](file:///Users/shersingh/github/qmd-mlx-search/test/indexing-resume.test.ts) |
| **Atomic Clear & Table Isolation** | Wrapped `clearAllEmbeddings` in an immediate transaction (`BEGIN IMMEDIATE` ... `COMMIT` / `ROLLBACK`) that clears `content_vectors`, drops `vectors_vec`, and deletes `content_chunk_expectations` and `indexing_checkpoints`. | [`test/store.test.ts`](file:///Users/shersingh/github/qmd-mlx-search/test/store.test.ts) |

---

## 3. Surface Inventory

### Python MLX Engine
- [`scripts/qmd_mlx/protocol.py`](file:///Users/shersingh/github/qmd-mlx-search/scripts/qmd_mlx/protocol.py): Typed exception hierarchy, HTTP error status mapping, input validation rejecting empty strings, binary protocol encoding.
- [`scripts/qmd_mlx/executor.py`](file:///Users/shersingh/github/qmd-mlx-search/scripts/qmd_mlx/executor.py): Single GPU execution owner, priority queue, `is_owner_thread()` detection, bounded idle callbacks.
- [`scripts/qmd_mlx/model_manager.py`](file:///Users/shersingh/github/qmd-mlx-search/scripts/qmd_mlx/model_manager.py): Shared conservative residency budget, before-load headroom checking, post-load actual size reconciliation, single-flight locking, LRU eviction, idle unloading.
- [`scripts/qmd_mlx/adapters/tokenization.py`](file:///Users/shersingh/github/qmd-mlx-search/scripts/qmd_mlx/adapters/tokenization.py): `TokenizedBatch` container, explicit oversize error raising instead of silent truncation.
- [`scripts/qmd_mlx/adapters/embedding.py`](file:///Users/shersingh/github/qmd-mlx-search/scripts/qmd_mlx/adapters/embedding.py): Adapter registry exposing model parameters and fail-closed resolution for unsupported models.
- [`scripts/qmd_mlx/batching.py`](file:///Users/shersingh/github/qmd-mlx-search/scripts/qmd_mlx/batching.py): `BatchPlanner` length-bucketed micro-batch planning, memory reduction on OOM.
- [`scripts/qmd_mlx/rerank.py`](file:///Users/shersingh/github/qmd-mlx-search/scripts/qmd_mlx/rerank.py): Qwen3 reranking adapter with direct `(w_yes - w_no)` projection, explicit document size bounds, manager-wired residency.
- [`scripts/qmd_mlx/generate.py`](file:///Users/shersingh/github/qmd-mlx-search/scripts/qmd_mlx/generate.py): Query expansion adapter with explicit prompt bounds and manager-wired residency.
- [`scripts/qmd_mlx/runtime.py`](file:///Users/shersingh/github/qmd-mlx-search/scripts/qmd_mlx/runtime.py): Single monotonic deadline, micro-batch scheduling, `tune_for_model` wiring, bounded compiled shapes.
- [`scripts/qmd_mlx/server.py`](file:///Users/shersingh/github/qmd-mlx-search/scripts/qmd_mlx/server.py): Bound HTTP handler semaphore, read timeouts, Host/Origin/Referer and Content-Type validation, executor-liveness health checks.

### TypeScript Core & CLI
- [`src/store.ts`](file:///Users/shersingh/github/qmd-mlx-search/src/store.ts): `PRAGMA busy_timeout = 5000`, `content_chunk_expectations` table, partial-document resume queries, lazy vector table prepared statements, atomic transaction commits.
- [`src/index.ts`](file:///Users/shersingh/github/qmd-mlx-search/src/index.ts): SDK `embed()` routed via `runDurableIndexingJob`.
- [`src/cli/qmd.ts`](file:///Users/shersingh/github/qmd-mlx-search/src/cli/qmd.ts): SIGINT `AbortController` cancellation in CLI `vectorIndex`, truthful exit codes.
- [`src/llm.ts`](file:///Users/shersingh/github/qmd-mlx-search/src/llm.ts): `strictMlx` configuration option with fail-closed GGUF fallback prevention.
- [`src/embedding/config.ts`](file:///Users/shersingh/github/qmd-mlx-search/src/embedding/config.ts): Aligned timeout constants and caller-first config precedence.
- [`src/indexing/checkpoint.ts`](file:///Users/shersingh/github/qmd-mlx-search/src/indexing/checkpoint.ts): Full canonical `computeEmbeddingSpaceId` integration with `revision` and `quantization`.
- [`src/indexing/job.ts`](file:///Users/shersingh/github/qmd-mlx-search/src/indexing/job.ts): Durable indexing job runner, `validateShadowTarget` path protection, typed `IndexingFingerprintMismatchError`.

---

## 4. Verification Test Results & Actual Command Outputs

### 1. TypeScript Build
```sh
$ bun run build
$ tsc -p tsconfig.build.json && printf '#!/usr/bin/env node\n' | cat - dist/cli/qmd.js > dist/cli/qmd.tmp && mv dist/cli/qmd.tmp dist/cli/qmd.js && chmod +x dist/cli/qmd.js
Exit Code: 0
```

### 2. TypeScript / Vitest Test Suite
```sh
$ CI=true bun run test
 Test Files  23 passed (23)
      Tests  765 passed | 72 skipped (837)
   Duration  70.84s
Exit Code: 0
```

### 3. Python MLX Pytest Suite
```sh
$ PYTHONPATH=. .venv/bin/python -m pytest test/python/ -v
============================= 71 passed in 32.19s ==============================
Exit Code: 0
```

---

## 5. Scope & Safety Compliance

- **No Production / Live State Alteration:** No modifications were made to `~/.cache/qmd/...`, `~/.hermes/...`, live database files, system daemons (`com.qmd.mlxd`), or user documents.
- **Isolated Fixtures:** All tests utilized temporary isolated SQLite databases and ephemeral server ports.
- **No Unsubstantiated Claims:** No performance speed claims or leak-freedom certifications by static inspection.
- **No Unauthorized Git Mutations:** No git commits or pushes were performed.

---

## 6. Fix-All Review Findings Verification (2026-09-07)

A second pass of targeted fixes and regression tests was implemented to resolve all review findings from `docs/plans/fix-all-review-findings.md`:

### Implemented Findings & Enhancements

1. **3-Model Preload Adapter Registration Order (Blocker)**:
   - Fixed startup crash where `ensure_loaded()` was called on `ModelResidencyManager` before registering `rerank` and `generate` adapters.
   - Refactored `RerankAdapter` and `GenerateAdapter` to support `lazy_load=True`, registering them with `ModelResidencyManager` during server initialization prior to running any model preload logic.
   - Added `test/python/test_mlx_server_startup.py` verifying clean startup under all preload combinations (`none`, `embed`, `all`).

2. **Generation Decode Cancellation & Deadlines**:
   - Replaced monolithic synchronous generation with `mlx_lm.stream_generate` yielding per-token decode steps.
   - Checked monotonic deadline and `cancel_event` between decoded tokens to support timely aborts during long sequence generation without blocking the GPU worker loop.
   - Verified in `test/python/test_mlx_generate.py`.

3. **Rerank Yes/No Golden Token ID Resolution**:
   - Implemented context-aware token ID resolution using chat template suffix context (`\n\nIs this relevant (yes/no)?\n\n`) to resolve the exact token ID produced in context rather than bare word tokenization.
   - Added ambiguity detection: if tokenization yields multiple subwords, falls back cleanly or raises a descriptive error.
   - Verified in `test/python/test_mlx_rerank.py`.

4. **Measured Quantization & Revision Tracking**:
   - Replaced default `"none"` and `None` fallbacks with measured extraction from `mlx_lm.load(..., return_config=True)`.
   - Extracted `config["quantization"]` and `config["_commit_hash"]` / `revision`, reporting `"unknown"` when absent.
   - Verified in `test/python/test_mlx_runtime.py`.

5. **Tokenizer Initialization Serialization**:
   - Wrapped tokenizer initialization and lazy-loading in `threading.Lock()` (`_init_lock`) across `BaseEmbeddingAdapter` and subclasses (`tokenize_texts` & `load`).
   - Eliminated race condition under concurrent `tokenize` + `submit_embed` requests.
   - Verified in `test/python/test_mlx_runtime.py`.

6. **Narrow Metal / MLX OOM Exception Matcher**:
   - Restricted OOM error matcher in `scripts/qmd_mlx/batching.py` to strictly match Apple Silicon Metal/MLX memory allocations (`metal buffer allocation failed`, `[metal]`, etc.).
   - Host-side memory errors (`MemoryError`, standard system alloc failures) no longer incorrectly trigger GPU batch token budget reductions.
   - Verified in `test/python/test_mlx_faults.py`.

7. **Session-Ceiling Exemption for Bulk Indexing**:
   - Added `maxDuration?: number` (default `0`, meaning unlimited) to `EmbedOptions` and `IndexingJobOptions`.
   - Exempted bulk indexing jobs from the artificial 120-minute interactive session timeout while preserving per-request socket deadlines.
   - Verified in `test/indexing-resume.test.ts`.

8. **Canonical Checkpoint Identity & Resume Safety**:
   - Updated `computeEmbeddingSpaceId` in `src/embedding/contract.ts` to include `quantization` in the canonical fingerprint signature.
   - Updated `getActiveCheckpoint` in `src/indexing/checkpoint.ts` to query `('in_progress', 'paused', 'cancelled', 'failed')` so killed/failed runs can resume cleanly.
   - Verified in `test/indexing-resume.test.ts`.

9. **Server Semaphore & Executor Worker Loop Protection**:
   - Replaced `threading.Semaphore` with `threading.BoundedSemaphore` in `scripts/qmd_mlx/server.py` to prevent unbounded releases on exception paths.
   - Added top-level error guard in `_worker_loop` (`scripts/qmd_mlx/executor.py`) to prevent unexpected crashes from permanently terminating the background GPU worker thread.
   - Verified in `test/python/test_mlx_server.py` and `test/python/test_mlx_executor.py`.

### Verification Test Summary

| Test Suite | Commands Run | Results | Exit Code |
| :--- | :--- | :--- | :--- |
| **TypeScript Build** | `bun run build` | Clean compilation to `dist/` | `0` |
| **TypeScript Vitest** | `CI=true bun run test` | **23 test files passed**, **768 passed**, 72 skipped (840 total) | `0` |
| **Python MLX Pytest** | `PYTHONPATH=. .venv/bin/python -m pytest test/python/` | **10 test files passed**, **82 passed** in 29.50s | `0` |


