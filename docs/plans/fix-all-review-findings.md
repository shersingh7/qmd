# Fix-all review findings — implementation plan

> **For Hermes:** Use default coding delegation (Antigravity CLI via `coding-delegate.sh`) to implement this plan task-by-task.

**Goal:** Fix all open findings from the whole-tree review so the MLX fork is startup-correct in strict all-MLX mode, cancellable, honestly identified, and safe to take into staged load validation.

**Architecture:** Keep the single-GPU-owner + residency-manager design. Fix startup ordering, decode cancellation, tokenizer/rerank identity, OOM classification, session-ceiling story, and small hardiness gaps. No rewrites, no new frameworks, no performance promises.

**Tech Stack:** Bun/TypeScript/Vitest, Python/MLX/pytest, SQLite. Repo `.venv`. Temp fixtures + ephemeral ports only.

**Scope (hard):** Repository changes only. Do NOT touch installed QMD, live/shadow DBs, launchd, private corpus, global config. No model downloads. No commits or pushes. No production benchmarks. Record real command outputs + exit codes; never fabricate transcripts.

---

### Task 0: Reproduce the preload blocker with a failing test

**Objective:** Prove the 3-model preload startup bug before fixing it.

**Files:**
- Test: `test/python/test_mlx_server_startup.py` (new)

**Step 1: Write failing test**

```python
def test_preload_three_models_registers_before_load():
    # Fake adapters: record register_adapter vs ensure_loaded call order
    # on ModelResidencyManager; assert register happens before any load.
    # Also assert start_server with rerank+generate under preload=True
    # reaches READY with fakes (no real weights).
    assert False  # replace with real test
```

**Step 2: Run test to verify failure**

Run: `PYTHONPATH=. .venv/bin/python -m pytest test/python/test_mlx_server_startup.py -q`
Expected: FAIL — `ModelUnavailableError: No adapter registered for stage 'rerank'`

**Step 3: Stop — do not fix in this task.**

---

### Task 1: Fix adapter registration order

**Objective:** 3-model preload reaches READY.

**Files:**
- Modify: `scripts/qmd_mlx/server.py:425-448` (construct adapters `lazy_load=True`, `register_adapter`, then `ensure_loaded`)
- Modify: `scripts/qmd_mlx/rerank.py:82-86`, `scripts/qmd_mlx/generate.py:66-70` (never `ensure_loaded` an unregistered stage from `__init__`; accept manager + explicit `load_via_manager()` or lazy default)

**Step 1:** Make Task 0 test pass with minimal ordering change only.
**Step 2:** Rerun Task 0 test. Expected: PASS.
**Step 3:** Run `PYTHONPATH=. .venv/bin/python -m pytest test/python/test_mlx_server.py test/python/test_mlx_model_manager.py -q`. Expected: PASS, exit 0.

---

### Task 2: Bound generation decode with deadline/cancel

**Objective:** Long expansions can't wedge the owner thread silently.

**Files:**
- Modify: `scripts/qmd_mlx/generate.py:_generate_sync`, `submit_generate`
- Test: `test/python/test_mlx_generate.py`

**Step 1: Write failing test** — fake `mlx_lm.generate` that sleeps in slices; set `cancel_event` mid-decode; assert `RequestCancelledError` within bounded time (not full decode).
**Step 2:** Implement per-slice deadline/cancel checks (token-callback if `mlx_lm` supports it, else chunked `max_tokens` loop with deadline checks between chunks, preserving exact output concatenation).
**Step 3:** If chunked decode changes output bytes, document + golden-test it. Lower interactive default `max_tokens` only if plan-approved; otherwise keep value, add bound.
**Step 4:** Run generate tests. Expected: PASS.

---

### Task 3: Pin rerank yes/no token IDs

**Objective:** Scores can't silently use wrong token IDs.

**Files:**
- Modify: `scripts/qmd_mlx/rerank.py:121-129`
- Test: `test/python/test_mlx_rerank.py`

**Step 1: Write failing test** — resolve IDs for `"yes"`, `" yes"`, `"no"`, `" no"`; assert chosen IDs equal the IDs the official chat template actually emits (encode the full suffix context, not the bare word).
**Step 2:** Fix resolution (prefer template-context encoding; explicit error if ambiguous).
**Step 3:** Run rerank tests. Expected: PASS.

---

### Task 4: Thread real quantization/revision into descriptors

**Objective:** Space-ID safety rests on measured identity, not labels.

**Files:**
- Modify: `scripts/qmd_mlx/adapters/embedding.py:get_descriptor`, `load` methods; `scripts/qmd_mlx/runtime.py` (pass-through)
- Test: `test/python/test_mlx_runtime.py`

**Step 1: Write failing test** — fake loaded model exposing config quantization/revision; assert `get_descriptor()` reports measured values, not constructor defaults.
**Step 2:** Read actual values from loaded model config where available; fall back to explicit `"unknown"` (never silently `"bf16"`/`""`).
**Step 3:** Run runtime tests. Expected: PASS.

---

### Task 5: Serialize tokenizer init under owner

**Objective:** No concurrent first-use padding/tokenizer race.

**Files:**
- Modify: `scripts/qmd_mlx/adapters/embedding.py:load` (padding-side set), `scripts/qmd_mlx/runtime.py:tokenize`
- Test: `test/python/test_mlx_runtime.py`

**Step 1: Write failing test** — concurrent `tokenize` + `submit_embed` on cold adapter with fakes; assert single padding-side assignment, single load.
**Step 2:** Route tokenizer init through manager/owner or a dedicated init lock.
**Step 3:** Run tests. Expected: PASS.

---

### Task 6: Narrow OOM classification

**Objective:** Host-side alloc failures must not shrink GPU batch budget.

**Files:**
- Modify: `scripts/qmd_mlx/batching.py:230-248`
- Test: `test/python/test_mlx_faults.py`

**Step 1: Write failing test** — raise generic `"cannot allocate"` from host-side fake; assert budget unchanged + error propagates (not `OutOfMemoryError` + halved budget).
**Step 2:** Restrict match to Metal/MLX allocation signatures; only those halve `max_batch_tokens` + bisect.
**Step 3:** Run fault tests. Expected: PASS.

---

### Task 7: Settle the session-ceiling story

**Objective:** Code and docs agree on multi-hour job behavior.

**Files:**
- Inspect: `src/store.ts` session `maxDuration`, `src/cli/qmd.ts` wiring
- Modify or document: whichever is smaller and truthful

**Step 1:** Read actual ceiling value + path (bulk job only).
**Step 2:** Either (a) exempt checkpoint-driven bulk path from the cap with auto-resume-safe abort, or (b) keep cap + implement auto-restart across ceilings. Do not ship cap + "deadline-less" claim together.
**Step 3:** Add/adjust resume test proving kill-mid-job → rerun completes exact missing chunks.
**Step 4:** Run `CI=true bun run test test/indexing-resume.test.ts`. Expected: PASS.

---

### Task 8: Verify canonical checkpoint identity end-to-end

**Objective:** Revision/quantization changes can't resume silently.

**Files:**
- Inspect: `src/indexing/checkpoint.ts`, `src/indexing/job.ts:156`
- Test: `test/indexing-resume.test.ts`

**Step 1:** Confirm `computeDescriptorSignature` delegates to `computeEmbeddingSpaceId` (or replace the call site).
**Step 2: Write failing test** — seed checkpoint + vectors, change only `revision`; assert typed mismatch error, zero inference, rows byte-equivalent.
**Step 3:** Run resume tests. Expected: PASS.

---

### Task 9: Bounded semaphore + executor loop guard

**Objective:** No silent capacity inflation; owner thread can't die quietly.

**Files:**
- Modify: `scripts/qmd_mlx/server.py:50-85`, `scripts/qmd_mlx/executor.py:_worker_loop`
- Test: `test/python/test_mlx_executor.py`, `test/python/test_mlx_server.py`

**Step 1:** Replace raw `Semaphore` with bounded accounting (assert-balanced acquire/release in tests) or guard release to only after successful acquire.
**Step 2:** Add top-level guard in `_worker_loop` around queue bookkeeping (job fns already caught) that logs + keeps serving.
**Step 3:** Run executor + server tests. Expected: PASS.

---

### Task 10: Full verification + honest report

**Objective:** Prove the tree state with real outputs.

**Files:**
- Modify: `docs/reviews/reliability-overhaul-results.md` (append fix-all section)

**Step 1:** Run `bun run build`. Record exit code.
**Step 2:** Run `CI=true bun run test`. Record files passed, tests passed/skipped, exit code.
**Step 3:** Run `PYTHONPATH=. .venv/bin/python -m pytest test/python/ -q`. Record passed count, exit code.
**Step 4:** Write results with exact counts + remaining gaps. No performance claims, no "production-ready" unless every task above genuinely passes.
