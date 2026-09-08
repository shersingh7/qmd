# Deep Repository Audit & Remediation Results

**Repository:** `qmd-mlx-search`  
**Date:** September 8, 2026  
**Auditor:** Antigravity (Google DeepMind)  
**Verification Environment:** Darwin / Apple Silicon macOS (arm64), Python 3.12.13 (`.venv`), Bun v1.3.8 / Node v22.0.0  
**Working Tree State:** Staged in working directory (uncommitted, no git commits or pushes performed)

---

## 1. Executive Summary & Audit Scope

A rigorous, evidence-driven corrective remediation was conducted across all production execution paths in `qmd-mlx-search`, addressing every issue identified in both `docs/reviews/deep-audit-parent-findings.md` and `docs/reviews/deep-audit-additional-findings.md`.

### Audit Verdict & Scope Distinctions
- **Implementation & Deterministic Test Proof:** All identified race conditions, re-entrancy deadlocks, post-load budget accounting gaps, off-owner execution leaks during active inference, tokenization eviction races, tokenizer identity omissions in TypeScript embedding fingerprints, benchmark batch_size ignoring, and indexing resume identity mismatches have been remediated and verified with deterministic, mock-isolated regression tests in Python and TypeScript.
- **Default Offline & Mock-Only Python Tests:** All default Python test suites run 100% offline with zero external network access (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`) and zero real model downloads. Real MLX/Metal execution tests are explicitly gated behind `@pytest.mark.real_model` and require the `--run-real-models` flag.
- **Performance & Real-Model Claims:** Unit and integration test suites run against mock fixtures and small representative test adapters; performance superiority against GGUF/llama.cpp remains unverified until safe, matched-workload benchmarks are executed on real hardware with production models.
- **Memory Scaling & Jetsam Reality:** The `residency_budget_mb` configuration explicitly bounds resident model weight parameters. Bounding model weights alone does NOT eliminate memory pressure or macOS jetsam risk; runtime activation and request allocations are bounded by early in-flight request and byte admission leasing ($\le 32$ concurrent admission slots, $\le 512$ items, $\le 256\text{ KB}$ per text, $\le 64\text{ MB}$ total in-flight bytes, $\le 10\text{ MB}$ total request char body), executor queue depth caps (max 20), and dynamic micro-batch bisection retry logic.

---

## 2. Item-by-Item Remediation & Technical Details

### [ADD-01] Default Python Test Offline & Mock-Only Enactment
- **Finding:** Default test fixtures instantiated actual models (MiniLM, Qwen, etc.), causing slow downloads and failure in offline environments.
- **Remediation:**
  1. Configured `test/python/conftest.py` with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` and custom `--run-real-models` CLI flag.
  2. Decorated all tests requiring resident Metal weights or model weights with `@pytest.mark.real_model`, skipping them by default unless explicitly opted in.
  3. Replaced real model loads in default test suites with mock adapters and mock HTTP servers.

### [ADD-02] Unloaded Tokenize Self-Deadlock & Qwen Double Load Elimination
- **Finding:** `BaseEmbeddingAdapter.tokenize_texts` held `_init_lock` (non-reentrant `threading.Lock`) and called `self.load()`, which re-acquired `_init_lock`, causing immediate self-deadlock. Furthermore, `QwenEmbeddingAdapter.load()` previously released the lock before assignment, allowing concurrent duplicate loads.
- **Remediation:**
  1. Changed `_init_lock` to `threading.RLock()` in `BaseEmbeddingAdapter`.
  2. Wrapped full loading sequence in `with self._init_lock:` across `QwenEmbeddingAdapter`, `BertEmbeddingAdapter`, and `NomicEmbeddingAdapter`.
- **Regression Tests:** `test_unloaded_tokenize_self_deadlock_regression` and `test_concurrent_tokenize_and_submit_embed_serialized_init` in `test/python/test_mlx_runtime.py`.

### [ADD-03] 4B Model Parameter Scaling & Dynamic Batch Planner Retuning
- **Finding:** `model_params_b` was hardcoded to 0.6 in `BaseEmbeddingAdapter`, causing 4B models (e.g. Qwen3-4B) to use a 0.6B micro-batch budget.
- **Remediation:**
  1. Implemented `infer_model_params_b(model_name, config)` using precise regex `(?:^|[_\-/])(\d+(?:\.\d+)?)[bB](?:[_\-/]|$)`, correctly differentiating parameter counts (4B) from quantization bits (`4bit`/`8bit`).
  2. Updated `BatchPlanner.tune_for_model(model_params_b)` and invoked dynamic retuning on model load inside `MLXEmbeddingRuntime`.
- **Regression Tests:** `test_infer_model_params_b_and_memory_estimation_distinguishes_quantization` in `test/python/test_mlx_runtime.py` and `test_budget_scales_down_with_model_size` in `test/python/test_mlx_batching.py`.

### [ADD-04] Inconsistent Resident Accounting Fallback & 4B vs 4bit Regex Matching
- **Finding:** Substring check `'4b' in name` matched `'4bit'`, and unmeasured models defaulted to ad-hoc 1000MB fallbacks.
- **Remediation:** Implemented `estimate_model_memory_mb(model_name, quantization, params_b)` with proper bit-width arithmetic (4.5 bits for 4-bit, 8.5 for 8-bit, 16 for fp16/bf16, 32 for fp32) and applied it consistently across adapter estimation and residency manager budgeting.
- **Regression Test:** `test_infer_model_params_b_and_memory_estimation_distinguishes_quantization` in `test/python/test_mlx_runtime.py`.

### [ADD-05] Bounded Request/Byte Admission Leasing before CPU Tokenization
- **Finding:** Checking executor queue length alone allowed unbounded concurrent callers to start CPU tokenization and memory allocation simultaneously before queue insertion.
- **Remediation:**
  1. Implemented `AdmissionLease` and condition-variable tracking (`acquire_admission_lease`, `release_admission_lease`) in `MLXEmbeddingRuntime`.
  2. Enforced caps on in-flight requests ($\le 32$ concurrent admissions), item counts ($\le 512$), byte sizes ($\le 64\text{ MB}$ total in-flight bytes, $\le 256\text{ KB}$ per text), and executor queue capacity.
  3. Guaranteed release in `finally:` blocks for both `tokenize()` and `submit_embed()`.
  4. Added public thread-safe executor inspection APIs: `is_accepting()`, `is_worker_alive()`, `is_overloaded()`, `get_queue_depth()`, and `join_worker()`.
- **Regression Tests:** `test_admission_lease_concurrency_and_guaranteed_release` and `test_runtime_bounded_admission_rejection_regression` in `test/python/test_mlx_runtime.py`.

### [ADD-06] Atomic Eviction Reservation Under Lock
- **Finding:** Checking lease status outside lock allowed a race where a concurrent caller acquired a lease on an eviction candidate during eviction execution.
- **Remediation:** `_evict_for_budget_under_owner()` and `_check_idle_unloads()` atomically mark candidate stages as `ModelState.UNLOADING` under `self._lock` prior to initiating eviction.
- **Regression Test:** `test_lifecycle_lease_prevents_idle_and_lru_eviction` in `test/python/test_mlx_model_manager.py`.

### [ADD-07] Post-Load Residency Reconciliation Excluded Newly Loaded Stage (Parent Finding 1)
- **Finding:** `_do_load_internal` calculated resident memory before marking the stage READY, causing post-load reconciliation to omit newly loaded stages.
- **Remediation:** Reconciles `current_other_resident + actual_mb > self.residency_budget_mb`, evicting LRU unpinned models or cleaning up and raising typed `OutOfMemoryError`.
- **Regression Test:** `test_post_load_accounting_underestimated_eviction_regression` in `test/python/test_mlx_model_manager.py`.

### [ADD-08] Owner-Thread Guarantee Bypassed on Unload/Shutdown (Parent Finding 2)
- **Finding:** `unload()` caught submit errors and called `_do_unload()` on caller threads while GPU was active.
- **Remediation:** Verified worker thread liveness with `executor.is_worker_alive()`, joined the worker thread on shutdown before off-owner cleanup, and prevented concurrent off-owner unloading.
- **Regression Tests:** `test_unload_no_off_owner_execution_during_active_inference_regression` in `test/python/test_mlx_model_manager.py` and `test_runtime_shutdown_under_owner_thread` in `test/python/test_mlx_runtime.py`.

### [ADD-09] Tokenization Eviction Race & Tokenizer Preservation (Parent Finding 3)
- **Finding:** Caller-thread tokenization could race model weight eviction.
- **Remediation:** Tokenizer instances (`raw_hf_tokenizer`) are preserved on CPU during model weight eviction (`model = None`), allowing tokenization on CPU without loading GPU weights.
- **Regression Test:** `test_runtime_tokenization_survives_model_weight_eviction_regression` in `test/python/test_mlx_runtime.py`.

### [ADD-10] Default-Model Resumability Fingerprint Inconsistency (Parent Finding 4)
- **Finding:** Indexing job compared `options.model || 'default'` against saved `descriptor.model || targetModel`, causing false mismatch errors on resume with omitted options.
- **Remediation:** Resolved canonical effective model `effectiveModel = options?.model || descriptor?.model || "default"` across all checkpoint comparisons.
- **Regression Test:** `test/indexing-resume.test.ts`.

### [ADD-11] TypeScript Embedding Space ID Tokenizer Identity & Case Sensitivity
- **Finding:** `computeEmbeddingSpaceId` omitted tokenizer identity and lowercased model identifiers, risking collisions on case-sensitive paths.
- **Remediation:** Included `t: descriptor.tokenizer?.trim() || ""` and preserved model case `m: descriptor.model.trim()`.
- **Regression Test:** `test/embedding-contract.test.ts`.

### [ADD-12] Truthful Benchmark Harness & Server Cleanup
- **Finding:** `bench_representative.py` ignored `batch_size` argument in chunking and claimed absent stage breakdowns. `bench_mlx.py` could leak started servers if descriptor fetching threw an error.
- **Remediation:**
  1. Applied `batch_size` to input slicing in `bench_representative.py` and removed misleading claims.
  2. Moved `try ... finally` block in `bench_mlx.py` immediately after `start_server()`.

### [ADD-13] HTTP Fault Status Mapping & X-Request-Timeout exact 504
- **Finding:** `test_ephemeral_server_fault_responses` used real MiniLM and accepted ambiguous 200/504 status codes.
- **Remediation:** Replaced with offline mock server, asserting exact 400 on malformed payloads, 200 on valid payloads, and exact 504 `DeadlineExceededError` on `X-Request-Timeout`.
- **Regression Test:** `test_ephemeral_server_fault_responses` in `test/python/test_mlx_faults.py`.

### [ADD-14] Generation Token Counting & Accurate Stats
- **Finding:** Generation token usage counted raw string characters rather than generated tokens.
- **Remediation:** Accurately counted tokens from stream chunks or tokenizer encoding and updated `total_tokens_generated`, `total_requests`, and `avg_ms`. Added `get_stats_info()`.
- **Regression Test:** `test_generate_token_counting_and_stats` in `test/python/test_mlx_generate.py`.

### [ADD-15] Finite Vector Validation on JSON `/embed`
- **Finding:** JSON `/embed` endpoint did not validate vector finiteness.
- **Remediation:** Added `np.all(np.isfinite(embeddings_arr))` check, raising `ProtocolError` (400) if NaN/Inf is detected.

### [ADD-16] Reranker Tokenization Call-Count Optimization
- **Finding:** `score_pairs_sync` repeatedly tokenized query and document templates $3N$ times.
- **Remediation:** Precomputed query budget once per batch, reducing tokenizer encode calls from $3N$ to $1 + N$.
- **Regression Test:** `test_rerank_tokenization_efficiency_offline` in `test/python/test_mlx_rerank.py`.

### [ADD-17] Graceful Server Teardown & STOPPING State
- **Finding:** Server lacked explicit signal handlers and could accept requests while stopping.
- **Remediation:** Added `ServerState.STOPPING` to reject new requests during shutdown, and added SIGTERM/SIGINT signal handling in `scripts/mlx_embed_server.py`.

#### [ADD-18] Managed Server Serving Loop, Preload Diagnostic Endpoints & Launcher Stop Verification
- **Finding:**
  1. Ad-hoc handshake between `stop_event` check, `_serving_event.set()`, and `serve_forever()` in `_init_and_serve` had a race window where `stop()` called after `stop_event` check but before `_serving_event.set()` (or before `serve_forever()` loop entry) saw `_serving_event` unset, returned without shutting down `HTTPServer`, and the init thread entered `serve_forever()` indefinitely on a stopped executor.
  2. Failed preload previously jumped directly to `finally: server.stop()` without running a serving loop, preventing diagnostic querying of `/health` and `/ready` in `ServerState.FAILED`.
  3. Launcher `scripts/mlx_embed_server.py` unconditionally reported `[mlx-server] Stopped.` even if `thread.join(timeout=10.0)` timed out or `_stopped_event` was not set.
- **Remediation:**
  1. Replaced the ad-hoc `_serving_event` handshake with a lifecycle-safe managed serving loop in `ThreadedMLXServer.serve_forever(poll_interval=0.2)` using `handle_request()` and loop inspection of `_stop_event`. Removed dependency on `BaseServer.shutdown()` to eliminate pre-loop deadlock risks.
  2. Implemented `_before_serve_hook` for deterministic lifecycle synchronization and verified thread join at the check/start boundary.
  3. Restored deliberate FAILED control endpoint serving: on model load or warmup failure, server transitions to `ServerState.FAILED`, records `state_error`, and runs the managed serving loop so that `/health` (200 degraded), `/ready` (503 failed), and `/embed` (503 failed) provide diagnostic context until `server.stop()` is issued.
  4. Updated launcher `scripts/mlx_embed_server.py` to assert `server._stopped_event.is_set()` AND `not thread.is_alive()` before printing `[mlx-server] Stopped.`, logging incomplete shutdowns to stderr and exiting with non-zero status upon join timeout.
- **Regression Tests:**
  - `test_server_startup_check_start_boundary_race_managed_loop` in `test/python/test_mlx_server_startup.py`
  - `test_server_startup_failed_preload_serves_diagnostic_endpoints` in `test/python/test_mlx_server_startup.py`
  - `test_launcher_stopped_reporting_logic` in `test/python/test_mlx_server_startup.py`
  - `test_server_stop_lifecycle_slow_init`, `test_server_stop_lifecycle_blocked_forward_no_reload`, `test_server_repeated_stop_is_idempotent`, `test_server_tokenize_deadline_and_oversized_validation` in `test/python/test_mlx_server.py`.

### [ADD-19] Model Manager `ModelState.EVICTING` Synchronization & Lock Discipline
- **Finding:** When `_check_idle_unloads()` or LRU eviction released `self._lock` before `adapter.unload()`, `acquire_lease()` or `ensure_loaded()` could concurrently acquire a lease on the stage being evicted. Holding `self._lock` across adapter load/unload callbacks risked re-entrancy deadlocks.
- **Remediation:**
  1. Added explicit `ModelState.EVICTING` state and per-stage `threading.Event` synchronization (`_evicting_events`).
  2. `acquire_lease()` and `ensure_loaded()` wait for active eviction events to clear before proceeding.
  3. Released `self._lock` during adapter load and unload callbacks across all manager operations, preventing callback deadlocks.
- **Regression Tests:** `test_evicting_state_acquire_lease_and_ensure_loaded_interleaving`, `test_model_manager_lock_not_held_across_adapter_callbacks` in `test/python/test_mlx_model_manager.py`.

### [ADD-20] Strict Admission Lease Limits, Tokenize Caps & Shutdown Wake
- **Finding:** `acquire_admission_lease` previously only rejected total bytes if `current_requests > 0` and permitted non-positive/excessive item counts. `tokenize()` lacked a total payload byte cap and post-tokenization deadline checks. Threads waiting for admission leases could block indefinitely if shutdown occurred while waiting.
- **Remediation:**
  1. Enforced strict bounds in `acquire_admission_lease`: rejected `total_bytes > max_in_flight_admission_bytes` and invalid item counts ($\le 0$ or $> \text{max}$) immediately even with 0 in-flight requests.
  2. Added 10MB total byte cap and post-tokenization deadline / cancellation checks to `tokenize()`.
  3. Propagated timeout/deadline/cancellation to HTTP `/tokenize`.
  4. Updated `runtime.shutdown()` to set `_shutting_down = True` and notify all waiting admission threads to wake and reject cleanly.
- **Regression Tests:** `test_runtime_admission_lease_oversized_rejection`, `test_runtime_admission_lease_wake_on_shutdown`, `test_runtime_tokenize_caps_and_cancellation` in `test/python/test_mlx_runtime.py`.

### [ADD-21] Dynamic Batch Planner Retuning & Adaptive OOM Halvings Preservation
- **Finding:** Cold/lazy model loads did not retune `BatchPlanner` before micro-batch planning, and repeated `tune_for_model()` calls wiped out adaptive OOM budget reductions (`_oom_halvings`).
- **Remediation:**
  1. Added `_oom_halvings` tracking and `_tuned_model_params_b` in `BatchPlanner`.
  2. Retuned `BatchPlanner` upon cold load before micro-batches are planned in `MLXEmbeddingRuntime`.
  3. Preserved adaptive OOM halvings across model retuning calls.
- **Regression Test:** `test_lazy_load_planner_retuning_and_oom_preservation` in `test/python/test_mlx_runtime.py`.

---

## 3. Sequential Verification Gate Results

### Gate 1: Git Formatting & Whitespace Check
- **Command:** `git diff --check`
- **Result:** `Exit code 0` (Clean diff with zero whitespace errors)

### Gate 2: TypeScript Build (`npm run build`)
- **Command:** `npm run build`
- **Result:** `Exit code 0`
- **Transcript:**
  ```text
  $ tsc -p tsconfig.build.json && printf '#!/usr/bin/env node\n' | cat - dist/cli/qmd.js > dist/cli/qmd.tmp && mv dist/cli/qmd.tmp dist/cli/qmd.js && chmod +x dist/cli/qmd.js
  ```

### Gate 3: Python MLX Test Suite (Pytest - 100% Offline)
- **Command:** `PYTHONPATH=. .venv/bin/pytest test/python/ -v`
- **Result:** `Exit code 0`
- **Summary:** **99 passed, 10 skipped in 7.14s** across 11 test suites (zero downloads, zero external network calls):

### Gate 4: TypeScript Test Suite (Vitest)
- **Command:** `CI=true bun run test` (or `npm test`)
- **Result:** `Exit code 0`
- **Summary:** **774 passed, 72 skipped, 23 files (NOT 846 passed)**:
  - `test/ast-chunking.test.ts`: Passed
  - `test/ast.test.ts`: Passed
  - `test/bench-score.test.ts`: Passed
  - `test/cli.test.ts`: Passed
  - `test/collections-config.test.ts`: Passed
  - `test/embed-outcome.test.ts`: Passed
  - `test/embedding-contract.test.ts`: Passed (including tokenizer identity & case sensitivity)
  - `test/embedding-index.test.ts`: Passed
  - `test/eval-bm25.test.ts`: Passed
  - `test/eval.test.ts`: Passed
  - `test/formatter.test.ts`: Passed
  - `test/indexing-resume.test.ts`: Passed (including Finding 4 regression test)
  - `test/intent.test.ts`: Passed
  - `test/llm.test.ts`: Passed
  - `test/mcp.test.ts`: Passed
  - `test/mlx.test.ts`: Passed (including fail-closed & retry semantics suite)
  - `test/multi-collection-filter.test.ts`: Passed
  - `test/rrf-trace.test.ts`: Passed
  - `test/sdk.test.ts`: Passed
  - `test/store-paths.test.ts`: Passed
  - `test/store.helpers.unit.test.ts`: Passed
  - `test/store.test.ts`: Passed
  - `test/structured-search.test.ts`: Passed

---

## 4. Working Tree Status

```text
Working directory: /Users/shersingh/github/qmd-mlx-search
Modified files:
 M CHANGELOG.md
 M CLAUDE.md
 M scripts/bench_mlx.py
 M scripts/bench_representative.py
 M scripts/mlx_embed_server.py
 M scripts/qmd-mlx-daemon.sh
 M scripts/qmd_mlx/adapters/embedding.py
 M scripts/qmd_mlx/batching.py
 M scripts/qmd_mlx/executor.py
 M scripts/qmd_mlx/generate.py
 M scripts/qmd_mlx/model_manager.py
 M scripts/qmd_mlx/rerank.py
 M scripts/qmd_mlx/runtime.py
 M scripts/qmd_mlx/server.py
 M src/embedding/contract.ts
 M src/indexing/job.ts
 M src/llm.ts
 M test/embedding-contract.test.ts
 M test/indexing-resume.test.ts
 M test/mlx.test.ts
 M test/python/test_mlx_executor.py
 M test/python/test_mlx_faults.py
 M test/python/test_mlx_generate.py
 M test/python/test_mlx_model_manager.py
 M test/python/test_mlx_rerank.py
 M test/python/test_mlx_runtime.py
 M test/python/test_mlx_server.py
 M test/python/test_mlx_server_startup.py
Untracked review/plan artifacts:
 ?? docs/plans/deep-audit-remediation.md
 ?? docs/reviews/deep-audit-additional-findings.md
 ?? docs/reviews/deep-audit-parent-findings.md
 ?? docs/reviews/deep-audit-results.md
 ?? test/python/conftest.py
```

- All changes remain in the working tree without git commits or pushes.
- Production indices (`~/.cache/qmd/`) and system services were untouched.
- All verification gates passed sequentially.
