# Apple Silicon Search and Inference Overhaul — Implementation Plan

> **For Hermes:** Execute through the explicitly requested Antigravity CLI (`agy`), then independently review and verify. This routing overrides the generic subagent-driven-development workflow.

**Goal:** Make QMD's MLX path correct, bounded, measurable, and fast on M-series MacBooks without corrupting existing vector indexes or degrading retrieval quality silently.

**Architecture:** Keep TypeScript/Bun, SQLite FTS5/sqlite-vec, CLI/SDK/MCP, and the GGUF expansion/reranking implementation. Replace the ad-hoc MLX server with a model-aware embedding runtime behind a bounded single GPU execution owner. Introduce an explicit embedding-space contract and backend-aware tokenization; optimize batching and retrieval only with correctness and benchmark gates.

**Tech stack:** Existing TypeScript/Bun and SQLite stack; Python 3.11+ isolated venv; MLX and a verified embedding-specific implementation (prefer mlx-embeddings if the installed API/model support is suitable), NumPy, local loopback HTTP. No mandatory PyTorch/SentenceTransformers fallback.

## Scope and safety

- Repository: `/Users/shersingh/github/qmd-mlx-search`; baseline commit `39d9b26`; initially clean git status.
- Actual verification host: Apple M2 Pro, arm64, 32 GiB RAM. Metal reports recommended working set 26800603136 bytes. This is an upper device limit, not permission to consume it all.
- Implement repository code, tests, docs, and isolated benchmark artifacts. Do not publish, push, globally install/link, edit launchd/Hermes configuration, restart live QMD, or modify personal indexes/model caches.
- Never run collection add/update/embed against the live index. Tests may create their own temporary databases via the existing test APIs; do not use operational CLI indexing commands automatically.
- Keep `bin/qmd` a shell wrapper. Never use `bun build --compile`.
- Use Bun for JS commands. New Python dependencies belong in a repo-local ignored venv; do not change global packages.
- Use public/synthetic fixture text only. Prefer existing local model files for live validation; download a small supported embedding model only if necessary for the authorized implementation, with no credentials or private corpus uploads.
- Preserve existing changes; do not reset or delete unrelated files. Leave implementation uncommitted for inspection.

## Review findings (baseline evidence)

| Priority | Evidence | Finding and impact |
|---|---|---|
| P0 | `src/mlx.ts:66`; `bun run build` | Build fails with TS2304: `BodyInit` not found. |
| P0 | `scripts/mlx_embed_server.py:_load_model`, `_get_compiled_embed` (114–149, 203–221) | Generic causal LM loader/output is treated as embedding hidden states. `model(input_ids)` can return vocabulary logits; blindly mean-pooling is not a valid universal embedding implementation. Mask is applied to pooling but not to the forward pass. Qwen and encoder families require different semantics. |
| P0 | `_load_model:119–136`; live installed `mlx_lm.utils.load` signature | Quantization kwargs are unsupported by the locally installed loader. Exception triggers an undisclosed SentenceTransformers backend with `trust_remote_code=True`; not guaranteed MLX. |
| P0 | `src/llm.ts:498,1079–1144`; `src/store.ts:1402,1422,1520`; `src/mlx.ts:146` | Automatic fallback may change embedding spaces, model labels are inconsistent (`mlx`, HF ID, GGUF default), and index insert uses requested label rather than returned identity. Matching dimension does not mean compatible vectors. |
| P1 | `src/store.ts:2216,2236`; `src/llm.ts:908–913` | Chunking uses the global GGUF tokenizer/context even when the store uses MLX. Loads unnecessary weights and may tokenize with a different model. |
| P1 | server defaults line 59 vs store chunking defaults | Server silently truncates to 512 tokens while QMD targets roughly 900-token chunks. Content can disappear from semantic recall. |
| P1 | server `ThreadedHTTPServer`, model globals, `_load_model` | Unbounded HTTP threads enter shared lazy model loading and MLX execution without single-flight initialization or GPU ownership. Concurrent load/forward passes can inflate memory and destabilize latency. |
| P1 | `_auto_max_batch_tokens:172`; `embed_for_binary:329–352`; `embed_for_json` | Hard-coded 20 GB budget is unsafe for 8/16 GB Macs; sum of raw lengths ignores padded/attention cost; JSON bypasses adaptive splitting entirely; binary path tokenizes repeatedly. |
| P1 | `src/mlx.ts:_fetch`, `_decodeBinary`, `embedBatchConcurrent` | Timeout ends after headers, not body; no response shape/finite-value/length validation; invalid batch size/concurrency can hang or crash; class drops binary preference. |
| P1 | server `do_POST`, readiness setup | No body-size/read-deadline or robust type/dimension limits; lazy startup declares ready before model validation; `is_query` parsed but unused; remote code enabled implicitly. |
| P1 | `src/store.ts:3015–3047` | Global KNN limited to `limit*3` before collection filtering can miss valid scoped matches. Must preserve documented sqlite-vec no-JOIN KNN constraint. |
| P2 | `generateEmbeddings:1478–1520`; `_embedBatchMlx:1131` | First chunk embedded twice; indexing batches always 32, so >64 concurrent path is not reached by normal indexing. Per-chunk insertion deserves transaction profiling. |
| P2 | server warmup 362–364; README 15–26,64–91 | Warmup caps both 4 and 16 at 4; unconditional 2–5x/zero-copy/AMX speed claims are unsupported. Float boxing and byte copies still occur; llama.cpp also uses Metal/unified memory. CLI flags and environment docs disagree with implementation. |

## Target modules

```
src/embedding/contract.ts       # backend/model/tokenizer/pooling/normalization/dimension fingerprint
src/embedding/config.ts         # validated env/config resolution and precedence
src/mlx.ts                      # compatibility exports + bounded HTTP client
src/embedding/protocol.ts       # binary/JSON validation and metadata
scripts/mlx_embed_server.py      # stable command entrypoint
scripts/qmd_mlx/runtime.py       # model adapter + one inference owner + lifecycle
scripts/qmd_mlx/batching.py      # token-once, bounded length buckets, order restoration
scripts/qmd_mlx/server.py        # HTTP validation, admission, deadlines, readiness
scripts/qmd_mlx/protocol.py      # little-endian framing and shared response schema
scripts/bench_mlx.py             # reproducible actual inference/transport benchmarks
src/bench/                      # end-to-end retrieval quality and stage timings
```

Use fewer modules if that keeps boundaries clear; do not rewrite unrelated 4k-line store logic solely to achieve this layout. Extract focused interfaces, not a framework.

## Task 1 — Establish reproducible baseline and build gate

**Files:** `src/mlx.ts`, `package.json`, `docs/benchmarks/apple-silicon.md`; new `test/mlx.test.ts`.
1. Preserve build failure and existing full test output (`/tmp/qmd-mlx-baseline-tests.log`) as baseline evidence. Inspect test setup/CI skips before choosing commands.
2. Add a mock HTTP smoke test importing the client under the build tsconfig. Fix `BodyInit` with a request body type appropriate to actual JSON strings, not broad DOM libs added just to suppress the error.
3. Run `bun run build` and `CI=true bun run test`. Record full exit codes, failure names and skipped tests, not a fabricated green summary.
4. Distinguish baseline environment failures from regressions. Make default non-model gates reliable offline.

## Task 2 — Embedding contract and safe backend selection

**Files:** new `src/embedding/contract.ts`, `src/embedding/config.ts`; `src/llm.ts`, `src/collections.ts`, `src/index.ts`, `src/cli/qmd.ts`; test `test/embedding-contract.test.ts`.
1. Write failing tests: same dimensions/different model must be rejected; revision, pooling, query prefix, normalization, truncation limit and reduced dimensions affect identity; config/env precedence; MLX unavailable must not initialize GGUF silently.
2. Resolve one explicit descriptor before formatting or indexing: backend, actual model/revision, tokenizer, pooling, query/doc formatting, native/output dimensions, max input tokens, dtype/quantization when relevant. Hash canonical versioned JSON; never use plain `mlx` as identity.
3. Single-flight initialization. MLX default is fail-closed. Legacy `mlxFallback=true` may only permit a verified compatible space, otherwise actionable error. Never fallback mid-index to a different model.
4. Make all CLI/SDK entrypoints use the same resolver; remove ineffective dtype control or validate server-reported value. Preserve existing GGUF behavior.
5. Run targeted tests and build.

## Task 3 — Correct model-aware MLX runtime

**Files:** `scripts/qmd_mlx/runtime.py`, entrypoint, requirements; `test/python/test_mlx_runtime.py`.
1. Inspect actual embedding library and supported model APIs before implementation. Pin a tested compatible dependency range. Generic `mlx_lm` logits are forbidden as an embedding substitute.
2. Start with a genuinely supported small embedding family; fail clearly on unsupported architecture. Do not advertise universal Nomic/Qwen support without tests. Preserve trained model pooling, mask, query/doc prefix and normalization semantics.
3. Default remote code off; no silent CPU/PyTorch fallback. Optional fallback must be explicit and identify itself truthfully.
4. Tests: padding invariance, single-vs-batch parity, query/doc formatting once only, correct dimensions, finite normalized vectors, invalid dims, valid Matryoshka reductions only for models that support them.
5. Verify dtype handling including real bfloat16 vs float16; use official library conversion/loading APIs for quantized checkpoints. No invented kwargs.
6. Exercise one real supported model on Metal using public fixture strings. Compare with authoritative same-model implementation where available; record tolerance and actual results. Mock tests do not satisfy this gate.

## Task 4 — Bounded scheduling and correct readiness

**Files:** runtime and `scripts/qmd_mlx/server.py`; `test/python/test_mlx_server.py`.
1. Tests: concurrent first calls load once, max active GPU jobs = 1, bounded queue rejects overload, health responds during inference, failed preload never ready, shutdown joins owner, malformed/oversized/slow requests rejected.
2. All MLX model creation/eval/disposal on one dedicated execution owner. HTTP may accept concurrently with bounded admission; health must not wait behind the GPU queue. Bound HTTP connection resources too, not just inference futures.
3. Explicit starting/loading/ready/failed/stopping lifecycle. `/ready` returns 503 unless model validation/warmup succeeded; `/health` distinguishes process liveness.
4. Body bytes, input count, per-input bytes, dims, queue slots, socket read time and inference admission timeout are validated. Reject booleans where integers expected. Disconnect/cancel before execution discards queued jobs; do not claim arbitrary mid-kernel cancellation.
5. Bind loopback, reject foreign Host/Origin, do not log private input bodies. Add malformed Content-Length/chunked-body and localhost browser-origin tests.

## Task 5 — Token-once batching and memory policy

**Files:** `scripts/qmd_mlx/batching.py`, runtime; `test/python/test_mlx_batching.py`.
1. Tests: exact input order after length sorting; JSON/binary parity; mixed short/long documents; bounded padded tokens; a single oversize input; OOM retry preserves order/cardinality; no duplicate tokenization inside embedding scheduler.
2. Tokenize once, store IDs/masks, bounded buckets by token length and batch count, budget on padded batch size and conservative attention cost. Share executor for JSON and binary.
3. Detect hardware RAM/recommended working set. Default conservatively across 8/16/24/32/64+ GiB, reserve system headroom, support explicit validated memory/token caps; no sysctl changes.
4. Warm real supported shapes (including actual batch 16 if budget permits); bound compilation shape set. Make compile optional and retain only when measured faster/correct; do not assume decoder compilation helps every encoder.
5. Shrink microbatch on genuine memory-allocation failure only; bounded retries, no partial duplicate output. Report tokenization/queue/inference/serialization time, batch tokens, memory peak and backend.

## Task 6 — Robust efficient wire client

**Files:** `src/embedding/protocol.ts`, `src/mlx.ts`; `test/mlx.test.ts`.
1. Tests: stalled response body aborts, malformed/truncated/oversized binary, wrong count/dims, NaN/Infinity, HTTP errors, negative/zero/noninteger batch parameters, partial failures and tail batch, preference retention.
2. Deadline covers fetch plus bounded body consumption. Versioned model metadata in headers/JSON must match negotiated descriptor. Explicit little-endian header and exact byte count, size caps before allocation.
3. Keep Float32Array views internally where compatible; retain public `number[]` SDK shape at boundary rather than breaking API. Accurately document unavoidable socket/serialization copies.
4. Bounded client concurrency, default conservative; preserve exact output cardinality/order on failures. No accidental null assertion crashes.

## Task 7 — Index identity and backend-aware chunking

**Files:** `src/store.ts`, `src/llm.ts`, `src/embedding/contract.ts`; tests `test/embedding-index.test.ts`, existing chunking suites.
1. Tests on temporary databases: space mismatch refuses reads/writes before mutation; same-size different model rejected; legacy GGUF remains usable; old MLX/unidentified space requires explicit rebuild, never inferred from size; force preflight must not clear an index before backend validation.
2. Store versioned embedding metadata transactionally. Never mix spaces. Use actual result identity instead of options label; validate every batch.
3. Inject active store tokenizer into chunking; provide bounded batch tokenization/counting via MLX runtime without loading GGUF weights. Keep token positions consistent and preserve public compatibility.
4. Bound chunks to negotiated context minus title/prefix/special-token overhead. No silent 512-token truncation of 900-token content. Test multilingual text, long tokens, headings/AST and final re-split guarantees.
5. Reuse first embedding probe result. Batch inserts in bounded transactions, preserving resumability and accurate progress/error reporting. Test partial failure does not mark missing chunks complete. If existing pending-chunk logic needs adjustment, include regression coverage.

## Task 8 — Scoped retrieval correctness and hot-path work

**Files:** `src/store.ts:searchVec`, hybrid/structured search; existing `test/multi-collection-filter.test.ts`, new scoped KNN tests.
1. Fixture: many globally closer excluded vectors hide the requested collection beyond 3x cutoff. Add tests for duplicates/chunks, inactive documents and multi-collection query.
2. Prefer a tested sqlite-vec filter-supported strategy without forbidden KNN JOIN. If unavailable, use bounded progressive overfetch with clear exhaustion semantics and parameter limits; do not falsely promise exact scoped recall if a budget caps it.
3. Fetch narrow candidate metadata before whole bodies where compatible; avoid repeated document/context hydration. Profile before adding ANN dependencies or GPU index duplication.
4. Preserve RRF and existing default quality. Expose opt-in fast/balanced/deep candidate budgets only if consistent across SDK/CLI/MCP and benchmarked. Do not call smaller candidate pools an inference speedup. Keep expansion/reranking architecture unless profiles demonstrate a replacement is necessary.

## Task 9 — Real benchmarks and quality gate

**Files:** `scripts/bench_mlx.py`, public fixture under `test/fixtures/`, `docs/benchmarks/apple-silicon.md`; optional retrieval harness additions.
1. Reproducible JSON output: commit, hardware/OS, versions, exact model/revision, dtype, dims, input lengths, warmup, samples, errors, RSS/Metal memory, cold startup, warm p50/p95, texts/sec, tokens/sec, per-stage time. Synchronize MLX eval before timing.
2. Benchmark single queries, batches 1/8/32, mixed lengths, concurrent interactive requests during bulk work, JSON vs binary and compile off/on; batch 64 only if safe. Separate model loading and transport-only numbers from inference.
3. Compare old and new only when old generates valid equivalent embeddings. If baseline cannot load/produces invalid vectors, say speedup unavailable; still report real new runtime numbers. Never repair baseline by inventing results.
4. Use a small labeled public retrieval fixture; report Recall@k/MRR or nDCG and scoped recall. Numerical optimizations preserve single/batch embeddings within justified tolerance. Quality changes require separate evidence.
5. Proposed objectives (not promises): no hidden GGUF load in MLX indexing, one resident embedding model, bounded overload, no swap-driven budget overshoot, and no warm latency regression on equivalent workloads. Publish actual measurements rather than an invented multiplier.

## Task 10 — Packaging, docs and final acceptance

**Files:** `README.md`, `CLAUDE.md`, `CHANGELOG.md`, `package.json`, `.github/workflows/ci.yml`, requirements, docs.
1. Ensure published package actually contains MLX entrypoint/modules/requirements or clearly documents source-only installation. Current `files` list omits scripts. Verify package contents via dry-run packing, no publish.
2. Replace unverified 2–5x, zero-copy end-to-end, AMX and precision claims; correct flags/env and supported model examples. Document isolated venv, startup/readiness, overload, descriptor migration/rebuild instructions and rollback. Never automatically rebuild live user data.
3. Run build, full offline TS gate, all Python tests, shell launcher test if applicable, actual local Metal smoke, benchmark and package-content check. Ensure Python tests run in CI without requiring Metal by separating model integration tests explicitly.
4. Inspect `git diff --check`, `git diff --stat`, `git status --short`; review every new file. Capture baseline failures/skips separately; no `|| true` on acceptance gates.
5. Write `docs/benchmarks/implementation-results.md`: task completion matrix, exact commands/results, model and benchmark artifact paths, remaining blockers, compatibility/migration details. Do not mark deferred work complete.

## Architectural decisions

- No blanket Rust/Swift rewrite: bridge overhead is not yet demonstrated to dominate model inference.
- No universal ANN/vector DB swap: preserve SQLite durability and small deployment; only add approximate indexing after corpus-scale profiling and recall validation.
- No simultaneous unrestricted GPU workers: throughput comes first from correct batching and warm model reuse, not multiplying contention.
- No silent model downgrade or truncated retrieval for headline latency wins.
- MLX embedding improvements cannot by themselves eliminate the cost of a large GGUF reranker. Report end-to-end stages separately.
