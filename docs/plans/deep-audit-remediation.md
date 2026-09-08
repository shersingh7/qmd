# Deep Repository Audit & Remediation Implementation Plan (Corrective Pass)

**Date:** 2026-09-08  
**Scope:** Deep Whole-Repo Audit & Actionable Remediation (MLX Model Manager, Runtime, Reranker, TypeScript Job Resumption, Fail-Closed Fallbacks, Protocol & Documentation)  
**Target Repository:** `qmd-mlx-search`  
**Constraints:** Isolated test fixtures only. No modifications to production/global databases (`~/.cache/qmd/...`), live daemons, or system configs. No live model downloads or external network calls during testing.

---

## 1. Executive Summary & Root Cause Analysis

Following parent source review against `0e0ce47`, this corrective pass addresses all four parent findings with deterministic regression tests and remediates memory accounting, concurrency safety, lifecycle leasing, and resume fingerprint consistency.

---

### Parent Findings Summary

1. **[F-01 / Parent Finding 1] Post-Load Accounting Omits Newly Loaded Stage (`scripts/qmd_mlx/model_manager.py`)**:
   - `_do_load_internal` called `_get_resident_model_mb()` before marking the stage `READY`. Because `_get_resident_model_mb()` only sums `READY` stages, the newly loaded stage was excluded from the post-load headroom calculation.
   - If an underestimated model loaded and fit within the budget on its own but caused total residency (existing ready models + new model) to exceed `residency_budget_mb`, no LRU eviction was triggered, violating the budget.
   - **Remediation**: Post-load reconciliation explicitly accounts for the actual size of the newly loaded stage (`actual_mb`), evicts LRU ready models under the owner thread if `resident_other + actual_mb > residency_budget_mb`, and unloads/rejects with `OutOfMemoryError` if the budget cannot be satisfied. Cleanup on failed/partial load frees resources and maintains consistent residency accounting.

2. **[F-02 / Parent Finding 2] Owner-Thread Guarantee Bypassed During Shutdown (`scripts/qmd_mlx/model_manager.py`, `scripts/qmd_mlx/runtime.py`)**:
   - `ModelResidencyManager.unload()` caught executor submission exceptions and fell back to executing `_do_unload()` directly on the caller thread even while the executor worker thread was actively running or forwarding.
   - `MLXEmbeddingRuntime.shutdown()` fell back to `adapter.unload()` on caller thread while executor might still be running.
   - **Remediation**: If the executor is alive, `unload()` must submit exclusively to the owner thread and must NEVER execute `_do_unload()` off-owner when submission fails or times out. External unloading is only permitted when the executor worker thread has completely stopped and joined (`not self.executor.is_alive()`). `MLXEmbeddingRuntime.shutdown()` terminates the executor worker loop first before performing final memory cleanups.

3. **[F-03 / Parent Finding 3] Tokenization Races Eviction & Unbounded Admission (`scripts/qmd_mlx/runtime.py`, `scripts/qmd_mlx/model_manager.py`)**:
   - `submit_embed()` and `tokenize()` loaded the model adapter and then performed tokenization outside the owner thread without an active lifecycle lease. If idle eviction or concurrent stage loading ran between load and tokenize, the model weights were evicted, and `tokenize_texts()` previously attempted direct off-owner `self.load()`.
   - Furthermore, unbounded payloads could cause memory spikes (jetsam) before tokenization or queue admission.
   - **Remediation**:
     - Introduce lifecycle leases (`acquire_lease(stage)` / `release_lease(stage)`) in `ModelResidencyManager`. Idle eviction and LRU eviction skip stages with active in-flight leases.
     - Preserve CPU tokenizer (`raw_hf_tokenizer`) across weight unloads in embedding adapters so CPU tokenization never triggers off-owner GPU loading.
     - Enforce bounded request admission (max 512 texts, max 256KB per item, max total characters) and check executor queue capacity before performing tokenization and allocations.

4. **[F-04 / Parent Finding 4] Default-Model Resumability Fingerprint Mismatch (`src/indexing/job.ts`)**:
   - `runDurableIndexingJob` compared existing checkpoint `fingerprint.model` to `options.model || 'default'`, but saved `descriptor.model || targetModel`.
   - When a job was started with an omitted model option and a real descriptor (e.g. `mlx-community/Qwen3-Embedding-4B-4bit-DWQ`), the initial run saved the descriptor model name. Upon resume with omitted model option, it compared `"mlx-community/..."` to `"default"` and threw a false `IndexingFingerprintMismatchError`.
   - **Remediation**: Resolve a single canonical effective model identity:
     ```ts
     const effectiveModel = options?.model || descriptor?.model || "default";
     ```
     Use `effectiveModel` consistently for checkpoint creation, comparison, and expected fingerprint payloads.

5. **[F-05] Rerank Projection Scaling & Activation Memory Honesty (`scripts/qmd_mlx/rerank.py`)**:
   - Clarify that evaluating `last_logits = lm_head(last_hidden)` on sliced last token states scales as `O(batch * vocab * sizeof(dtype))`.
   - Distinguish resident parameter weight budgets from runtime activation / request buffer memory.

6. **[F-06] Fail-Closed MLX Fallback Error Handling & Retry Safety (`src/llm.ts`)**:
   - When `embedBackend === 'mlx'` and `mlxFallback === false`, errors must throw typed fail-closed errors instead of returning silent `null`.
   - Retries must not be permanently blocked by stale failure latches when fail-closed is active.

---

## 2. Detailed Technical Design & Symbol Map

### 2.1 `scripts/qmd_mlx/model_manager.py`
- Add `self._leases: Dict[str, int] = {}` and `self._evicting_events: Dict[str, threading.Event] = {}` to `ModelResidencyManager`.
- Implement `acquire_lease(stage: str)` and `release_lease(stage: str)`:
  - `acquire_lease` blocks and waits on any stage in `ModelState.EVICTING` before incrementing the lease count.
  - `ensure_loaded` waits for any active eviction on the stage to settle before attempting load.
- Mark stages as `ModelState.EVICTING` with a dedicated event during LRU or idle eviction, releasing `self._lock` while calling `adapter.unload()` to avoid lock contention or adapter re-entrancy deadlock.
- Update `_check_idle_unloads()` and `_evict_for_budget_under_owner()` to respect active leases.
- Update `_do_load_internal()`:
  - Check `actual_mb > self.residency_budget_mb` -> unload and raise `OutOfMemoryError`.
  - Check `_get_resident_model_mb() + actual_mb > self.residency_budget_mb` -> trigger `_evict_for_budget_under_owner(stage, actual_mb)`.
  - Re-verify headroom. If still exceeding budget, unload adapter, clear cache, and raise `OutOfMemoryError`.
  - On any load failure, ensure adapter is cleanly unloaded and state set to `FAILED`.
- Update `unload()`:
  - If `is_owner_thread()`: call `_do_unload()`.
  - Else if `executor.is_alive()`: submit `_do_unload()` to executor. DO NOT catch exception to run `_do_unload()` locally.
  - Else: call `_do_unload()` (safe since worker thread is stopped).

### 2.2 `scripts/qmd_mlx/runtime.py`
- In `acquire_admission_lease()`:
  - Strictly reject requests where `total_bytes > max_in_flight_admission_bytes` or `num_items <= 0` or `num_items > max_items` immediately, even if current admissions are 0.
  - Wake and reject waiting admission threads immediately upon `runtime.shutdown()` via `_shutting_down` flag and `_admission_lock.notify_all()`.
- In `submit_embed()`:
  - Validate request text counts (<= 512), item lengths (<= 256KB), and total char budget.
  - Check executor alive state and queue capacity before tokenizing.
  - Retune `BatchPlanner` dynamically upon cold model load before micro-batches are planned.
  - Wrap tokenization and micro-batch execution in `self.model_manager.acquire_lease("embed")` / `finally: release_lease("embed")`.
- In `tokenize()`:
  - Enforce 10MB total byte cap and item length limits.
  - Wrap in lifecycle lease and check timeout/cancellation post-tokenization.
- In `shutdown()`:
  - Terminate executor first (`self.executor.shutdown()`), then clean up adapter memory safely without off-owner racing.

### 2.3 `scripts/qmd_mlx/batching.py`
- In `BatchPlanner`:
  - Maintain `_oom_halvings` count and track `_tuned_model_params_b`.
  - Retune dynamically when `tune_for_model()` is called while preserving adaptive OOM budget halvings across cold/lazy reloads.

### 2.4 `scripts/qmd_mlx/server.py` & `scripts/mlx_embed_server.py`
- Implement coordinated stop lifecycle:
  - `_serving_event`, `_stop_event`, `_stopped_event`, explicit `stop()`, `shutdown()`, and `server_close()`.
  - Guard `BaseServer.shutdown()` behind `_serving_event.is_set()` to prevent hang before `serve_forever()`.
  - In `_init_and_serve`: stop admission, cancel active/queued work first, wait for owner thread exit safely, then unload models.
  - Prevent admitted requests or background loops from reloading during `STOPPING`.
  - In launcher, join server thread before printing `[mlx-server] Stopped.`.

### 2.5 `scripts/qmd_mlx/adapters/embedding.py`
- In `BaseEmbeddingAdapter.unload()`:
  - Delete `self.model` and reset `self.model_memory_mb = 0.0`.
  - Keep `self.raw_hf_tokenizer` in memory so CPU tokenization is safe and does not trigger off-owner loading.

### 2.6 `src/indexing/job.ts`
- Resolve `const effectiveModel = options?.model || descriptor?.model || "default";`.
- Use `effectiveModel` for mismatch check (`existingModel !== effectiveModel`) and checkpoint saving.

### 2.7 `src/llm.ts`
- In `_embedMlx` and `_embedBatchMlx`:
  - When `!this.mlxFallback || this.failClosed`, throw the error rather than returning silent `null` or setting permanent `mlxFailed = true`.
- In `embed` and `embedBatch`:
  - Propagate typed fail-closed errors when MLX is configured and fallback is disabled.

---

## 3. Deterministic Regression Test Plan

### Python Test Suite (`test/python/`)
1. **`test/python/test_mlx_model_manager.py`**:
   - `test_residency_post_load_accounting_eviction`: Two adapters where adapter 1 is loaded (1000MB), adapter 2 is estimated at 1000MB but actually loads as 1800MB (budget 2500MB). Asserts that adapter 1 is evicted, adapter 2 is READY, and total resident memory is 1800MB.
   - `test_residency_post_load_over_budget_rejection`: Single adapter whose actual loaded size exceeds total budget is unloaded, state marked FAILED, and raises `OutOfMemoryError`.
   - `test_unload_no_off_owner_execution_during_active_inference`: Forward job blocks worker thread; external `unload()` times out and does NOT unload the adapter while forward is running.
   - `test_lifecycle_lease_prevents_idle_and_lru_eviction`: Active lease prevents idle evictions.
   - `test_evicting_state_acquire_lease_and_ensure_loaded_interleaving`: Concurrently acquiring lease or loading waits on `ModelState.EVICTING` without racing.
   - `test_model_manager_lock_not_held_across_adapter_callbacks`: Verifies manager lock is released during adapter callbacks to prevent deadlock.

2. **`test/python/test_mlx_runtime.py`**:
   - `test_runtime_tokenization_survives_model_weight_eviction`: Model weights unloaded while retaining tokenizer; `tokenize()` succeeds on CPU, and `submit_embed()` safely reloads weights on the owner thread before forward pass.
   - `test_runtime_early_admission_rejection`: Oversized requests (> 512 texts or full queue) fail immediately before tokenization.
   - `test_runtime_admission_lease_oversized_rejection`: Requests exceeding total bytes or with invalid counts reject immediately even with 0 in flight.
   - `test_runtime_admission_lease_wake_on_shutdown`: Waiting admission threads wake and reject cleanly on runtime shutdown.
   - `test_runtime_tokenize_caps_and_cancellation`: Enforces 10MB total byte cap and post-tokenization deadline/cancel propagation.
   - `test_lazy_load_planner_retuning_and_oom_preservation`: Retunes planner dynamically on cold load while preserving adaptive OOM halvings.

3. **`test/python/test_mlx_server.py`**:
   - `test_server_stop_lifecycle_slow_init`: Server stop during slow preloading aborts promptly without hang.
   - `test_server_stop_lifecycle_blocked_forward_no_reload`: Server stop with active forward pass cancels, stops admissions, and does not reload during STOPPING.
   - `test_server_repeated_stop_is_idempotent`: Multiple consecutive `stop()` calls behave idempotently.
   - `test_server_tokenize_deadline_and_oversized_validation`: HTTP /tokenize validates bounds and timeout/deadline headers.

4. **`test/python/test_mlx_rerank.py`**:
   - `test_rerank_quantized_linear_head_evaluation`: QuantizedLinear head execution with `lm_head(last_hidden)` and output verification.

5. **`test/python/test_mlx_executor.py`**:
   - `test_executor_submit_cancellation_on_exception`: Cancellation event set on wait aborts.

### TypeScript Test Suite (`test/`)
1. **`test/indexing-resume.test.ts`**:
   - `test_resume_with_omitted_model_and_real_descriptor`: Initial run with omitted `options.model` and real descriptor (`mlx-community/Qwen3-Embedding-4B-4bit-DWQ`) interrupted; resume run with omitted `options.model` matches fingerprint and resumes without mismatch error.
2. **`test/mlx.test.ts`**:
   - `test_mlx_fail_closed_throws_typed_error`: Unreachable MLX backend throws descriptive error on `embed()` and `embedBatch()` when `mlxFallback: false`, and retries attempt connection again rather than returning permanent silent null.

---

## 4. Verification Gates Sequence
1. `git diff --check`
2. `npm run build`
3. `PYTHONPATH=. .venv/bin/pytest test/python/ -v`
4. `npm test` (or `bun run test`)
5. Update `docs/reviews/deep-audit-results.md` in place with genuine test counts and accurate facts.
