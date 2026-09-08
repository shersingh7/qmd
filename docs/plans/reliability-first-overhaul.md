# Reliability-first Apple Silicon overhaul — implementation plan

**Goal:** Make the experimental MLX path correct, recoverable, bounded in resource usage and measurable before reconsidering deployment.

**Architecture:** Retain TypeScript CLI/MCP, SQLite FTS5, sqlite-vec and existing search composition. Replace inference lifecycle and bulk-job plumbing incrementally: one daemon execution owner, explicit model adapters, tokenized batches, deadline-aware scheduling, transactional progress. No wholesale rewrite or promised speed multiplier.

**Tech stack:** Existing Bun/TypeScript/Vitest, Python/MLX/pytest and SQLite. Use repository .venv. Follow AGY default coding delegation rather than spawning additional implementation agents.

## Scope and options

1. Minimal patches: fix HTTP deadline and CLI exit code. Necessary but insufficient; leaves GPU concurrency, resume and correctness issues.
2. **Recommended: targeted overhaul below.** Preserve proven retrieval/database components and replace the unreliable inference/job boundaries.
3. Full rewrite in Swift/Rust: defer. No profile establishes language overhead as the bottleneck; major correctness and integration cost.

Authorization is repository development and tests only. Do not change installed QMD, launchd, global config, live/shadow databases, existing daemons, OS memory settings, or model defaults. Do not run production update/embed. Use temporary test databases and ephemeral test ports. Do not upload personal corpus or download more weights. Do not commit/push without renewed request. Existing source is baseline 0c4a154; historical benchmark reports are evidence, not acceptance.

## Evidence from fresh source review

- HIGH: scripts/qmd_mlx/server.py:267–272 explicitly passes timeout=60.0 despite runtime.py:399 default 300s. Prior alignment claim was incorrect.
- HIGH: runtime.py:413 waits on Future with no timeout cancellation; worker checks cancellation only before the whole job (241), not between micro-batches. Timed-out requests can continue consuming GPU.
- HIGH: runtime.py:246–247 reload is outside try; an exception terminates worker while readiness remains static. shutdown enqueues into possibly full queue and only joins two seconds.
- HIGH: server.py:225 executes reranker directly from HTTP threads; generation owns a separate unbounded queue (generate.py:59). No daemon-wide GPU ownership or residency budget.
- HIGH: src/cli/qmd.ts:1778–1782 prints 100%/Done even with errors; store.ts:1559–1565 marks unattempted work as failures. Counts do not distinguish attempted, skipped and committed.
- HIGH: store.ts:1575–1594 increments counters inside transaction before COMMIT; rollback enters generic inference retry. DB errors and inference failures are conflated.
- HIGH correctness risk: runtime.py:279–309 uses tokenizer-dependent padding, calls causal backbone without padding mask, and pools sum(mask)-1. This assumes right padding; verify mixed-length left/right padding and singleton equivalence before changing masks. Model pooling is guessed from name (150–154).
- MEDIUM: batching.py:105 tokenizes to count, then runtime.py:280 tokenizes again. Approximate fallback hides tokenizer failure. Claimed token-once implementation is not token-once.
- MEDIUM: server state is handler class-global; preload=False marks READY without runtime; health does not track dead owner. HTTP threads/body reads lack explicit bounded admission/read deadlines; JSON top-level arrays reach payload.get.
- MEDIUM: batching budget is a heuristic parameter estimate and token count, not measured resource headroom. Broad retry on any 'metal' error can retry non-OOM defects.
- Documentation: CLAUDE.md still claims 2–5x faster. Remove unsupported current claims; keep historical results clearly labeled.

These are verified source observations or explicitly labeled risks, not a completed exhaustive audit. First task extends review across remaining surfaces and records exact evidence.

## Task 1 — Full surface inventory and regression baseline

Files: docs/reviews/reliability-overhaul-audit.md (new); existing src/{llm,mlx,store}.ts, src/embedding/*, src/cli/qmd.ts, src/mcp/*, scripts/qmd_mlx/*, scripts/qmd-mlx-daemon.sh, scripts/apply-fork-dist-patches.py, tests and docs.

1. Map config → descriptor → tokenization → request → forward → vector insertion → retrieval → cache provenance.
2. Inspect actual installed library signatures and attention implementation read-only. Do not assert flash attention absent without evidence.
3. Audit embed fallback (actual branches, not stale comments), rerank/expansion cache provenance on fallback, descriptor revision/quantization/pooling identity, dimension compatibility, query/document formatting and chunk completeness.
4. Run baseline build, Vitest and Python tests with actual exit status and logs. Separate native ABI/environment failures from code failures. Tests must not call production endpoints; isolate HOME/cache/config where appropriate and inspect integration fixtures first.
5. Record HIGH/MEDIUM findings with exact symbols; refine tasks based on source evidence.

## Task 2 — Truthful bulk outcomes and atomic counters (first implementation slice)

Files: src/store.ts, src/cli/qmd.ts, test/embedding-batching.test.ts (locate actual existing test), new test/embed-outcome.test.ts.

1. Add failing tests: partial failure must not produce complete outcome; fatal error/abort must exit nonzero; rollback must not advance committed counters; skipped work must not be reported as attempted failure.
2. Define structured outcome with status complete/partial/failed/cancelled, attempted/committed/failed/skipped/pending where known, processed vs selected documents, first failure reason. Unknown totals must be null/unknown, never fabricated.
3. Stage counter deltas and apply after COMMIT only. Separate DB errors from retryable inference errors. Do not retry a failed DB transaction by issuing new GPU calls.
4. CLI prints Done/100% only for verified completion. Set process.exitCode nonzero for partial/failed/cancelled while preserving graceful cleanup.
5. Add CLI subprocess tests with temporary database and mock server, asserting actual exit code, summary and persisted rows.

## Task 3 — One deadline and typed errors

Files: src/mlx.ts, scripts/qmd_mlx/{protocol,server,runtime,batching}.py; test/mlx.test.ts, test/python/test_mlx_{server,runtime,batching}.py.

1. Reproduce explicit 60s override using injected clocks/timeouts, not minute-long sleeps.
2. Introduce validated bounded request timeout converted once into local monotonic absolute deadline. Budget includes queue wait, load, all micro-batches and response consumption; do not compare monotonic clocks across processes.
3. On timeout/cancellation mark job cancelled and skip remaining micro-batches; settle futures exactly once. GPU kernel already submitted may complete; don't promise hard preemption.
4. Typed errors: invalid input, overloaded, deadline, cancelled, unsupported model, OOM, unavailable worker, identity mismatch. HTTP mapping 400/429 or 503/504 as appropriate. No catch-all inference retry storms.
5. Restore sensible interactive session defaults instead of the global 120-minute override; bulk lifecycle belongs to resumable jobs, not an ever-larger session deadline.
6. Test queue-expired job never forwards, timeout after first micro-batch skips second, repeated deadline failures leave subsequent request usable, response body timeout stays active.

## Task 4 — Single execution owner and model residency manager

Create scripts/qmd_mlx/executor.py and model_manager.py; modify runtime.py, rerank.py, generate.py, server.py.

1. Add deterministic fake-model concurrency tests proving no overlap across embed/rerank/generate, single-flight loading, no unload during work, expiry and recovery after reload failure.
2. Executor owns all MLX model load/eval/unload/cache operations; adapters become synchronous model-specific computation called only by owner.
3. Bounded queue with interactive/bulk priority and starvation protection. Yield bulk at micro-batch boundaries; do not preempt running kernels.
4. Explicit unloaded/loading/ready/busy/failed/stopping states, derived health from owner liveness, per-server instance state not handler class state.
5. Shared conservative residency budget; evict idle models first. All three stages support lazy load and idle unload. Use monotonic completion time for inactivity.
6. Bounded shutdown and settle pending futures. Reload failure fails job without silently killing owner. No global memory tuning.
7. HTTP admission/read deadlines and Host/Origin validation; malformed JSON type rejects cleanly; overload cannot spawn unlimited inference work.

## Task 5 — Explicit embedding adapters and token-once batches

Create scripts/qmd_mlx/adapters/embedding.py and tokenization.py; modify runtime.py, batching.py, descriptor contracts and tests.

1. Write mixed-length/singleton, empty text, left/right padding, over-context, requested-dimension and ordering tests.
2. Model adapter declares supported architecture, trained pooling, tokenizer revision, context, query/document instruction recipe, normalization and dimensional reduction support. Unknown generic causal model fails closed, not guessed from filename.
3. Tokenize once into TokenizedBatch: ids, lengths, mask, original indices, special-token policy. Planner buckets token arrays; forward consumes arrays without second text tokenization.
4. For supported Qwen adapter verify installed backbone mask API and correct final non-pad pooling; right padding may avoid causal padding contamination but must be explicitly enforced/tested. Do not arbitrarily change model semantics.
5. No silent truncation. Explicit oversize error or chunker split with verified full-content coverage, including instruction/special-token budget.
6. Descriptor/cache identity includes semantic recipe and real model/config revision/quantization. No embedding backend fallback into another space. Legacy DB migration must be explicit and isolated.

## Task 6 — Resource-aware batch execution

Files: batching.py, executor.py, model_manager.py; new test/python/test_mlx_resource_policy.py.

1. Test conservative policies for 8/16/24/32/64 GB with mocked RAM/Metal limits and memory pressure.
2. Bound padded tokens, sequence length, batch rows, queued bytes and residency separately; preserve explicit operator budget. Avoid assuming physical RAM is free RAM.
3. Known recoverable allocation error halves batch with strict retry/deadline bound, releases intermediates and reduces future budget. Other Metal errors fail rather than recursive retry.
4. Expose throughput vs interactive resource profiles without changing model size or retrieval quality silently. No hardware-specific custom kernels before profiling.

## Task 7 — Durable indexing jobs and safe shadow workflow

Create src/indexing/job.ts and checkpoint.ts; modify store.ts/CLI; new test/indexing-resume.test.ts.

1. Inspect schema first: prove missing chunks in partially embedded documents are selected on resume (not only documents with zero vectors).
2. Persist fingerprint of content/chunker/embedding descriptor plus chunk identity and committed progress in temporary/shadow DB. Idempotent writes, bounded transaction batches.
3. Interrupt after first batch, restart, finish exact missing chunks; no duplicates or skipped tails. Test changed document, stale fingerprint and failed transaction.
4. Bulk jobs use bounded requests and explicit pause/cancel/resume, not a two-hour timer masking a multi-day task.
5. Shadow tooling must reject default/live target and require explicit path. No cutover implementation required now; document SQLite backup API/checkpoint/quiescence requirements and separate future approval.

## Task 8 — Honest observability and benchmark harness (no full performance campaign yet)

Create scripts/bench_representative.py and docs/benchmarks/acceptance-protocol.md; revise bench_mlx.py and documentation.

1. Instrument tokenize/queue/load/forward/pool/serialize/write durations, real input and padded tokens, actual backend/model, timeout/OOM/retry counts, completed chunks, RSS and Metal active/peak separately.
2. Bench manifest captures commit, dependencies, device, model identity, verified lengths, cold/warm/cache states, batch size and seed. Synchronize MLX before timing.
3. Harness strata short/medium/production/long, mixed languages/code, variable length. Public fixtures default; private corpus sampling only by future approval and local-only with no raw texts in reports.
4. Same-model/size/quantization comparisons labeled separately from model upgrades. Compare full end-to-end CLI/MCP workloads, not inference-only speed presented as indexing speed.
5. Quality protocol: fixed held-out labels, Recall@10/nDCG@10/MRR, rerank accuracy and expansion downstream retrieval; same candidates/cache state. Do not tune on holdout.
6. Tests validate benchmark count reconciliation, percentile calculations, failed runs never labeled success and absent data never fabricated. Run short synthetic correctness smoke only at this stage.

## Acceptance and release gates

- All modified behavior has regressions; clean build + complete Vitest/Python results with exit codes. Test skip counts and integration exclusions explicit.
- Mock fault tests cover dead worker/reload, queue full, timeout, malformed body, OOM, DB rollback, interrupted resume, descriptor mismatch and all-stage unloading.
- Small isolated integration proves real request routing, no leftover processes and no live filesystem modifications.
- Only after correctness gates: controlled representative performance evaluation, same-model baselines, warm interactive p50/p95 plus throughput and peak RSS/swap impact. Proposed promotion target is >=20% practical speed improvement with no material retrieval regression; this is a decision threshold, NOT a promised result.
- No deployment based solely on unit tests or short-text throughput. Full-corpus trial and production installation require separate approval.

## Execution and report

Implement in dependency order, starting Tasks 1–3. Continue through remaining tasks when validated; do not pretend broad refactoring is complete if only first slice passes. Report exact changed files, executed tests and failures, architectural deviations, unimplemented tasks and risks in docs/reviews/reliability-overhaul-results.md. Do not run unrelated package patchers against installed production. Do not commit/push.
