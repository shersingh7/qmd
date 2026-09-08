# Reliability Overhaul Source Audit & Surface Inventory

Date: 2026-09-07
Baseline commit: 0c4a154
Status: Task 1 Completed

---

## 1. Surface Inventory & Architecture Flow

The end-to-end embedding, indexing, retrieval, and serving pipelines were mapped across all relevant repository surfaces:

### 1.1 Config & Descriptor Flow
- **CLI / Config Resolution:** `src/cli/qmd.ts` resolves embed configuration (`--model`, `--force`, batch options). Environment variables `QMD_EMBED_BACKEND` (`mlx` vs `gguf`), `QMD_MLX_EMBED_URL` (default `http://127.0.0.1:8787`), and options are passed into `src/llm.ts` (`LlamaCpp` constructor).
- **Descriptor Contract:** `src/embedding/contract.ts` defines `EmbeddingDescriptor` (model, revision, pooling, nativeDimensions, outputDimensions, normalized, etc.) and `computeEmbeddingSpaceId()`.
- **Database Space Tracking:** `src/store.ts` (`ensureVecTable`) stores `embedding_space_id`, `embedding_model`, and `embedding_descriptor` into SQLite table `store_config`.

### 1.2 Ingestion, Chunking & Batching Flow
- **Document Discovery:** `src/store.ts` (`getPendingEmbeddingDocs`) selects documents needing embeddings.
- **Chunking:** `chunkDocumentByTokens` splits document bodies into chunks (~900 tokens).
- **Batch Formation:** `buildEmbeddingBatches` groups documents by count/byte limits. Inside `generateEmbeddings`, chunks are batched in groups of 32 (`BATCH_SIZE = 32`).
- **Formatting:** `formatDocForEmbedding` / `formatDocForDescriptor` applies prefix/scaffolding.

### 1.3 Transport & Server Execution Flow
- **Client Transport:** `src/mlx.ts` (`embedBatchWithMlx`, `embedBatchConcurrent`) marshals requests over HTTP to `POST /embed` or `POST /embed-bin` using `AbortSignal.timeout()`.
- **HTTP Server:** `scripts/qmd_mlx/server.py` (`MLXHTTPRequestHandler`) parses JSON or binary requests, runs `validate_embed_request`, and submits to runtime.
- **Runtime Execution:** `scripts/qmd_mlx/runtime.py` (`MLXEmbeddingRuntime.submit_embed`) enqueues into `_work_queue`, where `_worker_loop` processes tasks via `BatchPlanner.plan_and_execute` in `scripts/qmd_mlx/batching.py`.

### 1.4 Vector Storage & Retrieval Flow
- **Insertion:** `insertEmbedding` in `src/store.ts` writes raw float32 vectors to virtual table `vectors_vec` (via sqlite-vec `vec0`) and `content_vectors`.
- **Vector Search:** `src/store.ts` (`searchVec`) queries `vectors_vec` using `MATCH ? AND k = ?`.
- **RRF & Reranking:** `src/store.ts` blends BM25 and vector scores via Reciprocal Rank Fusion, chunks candidate documents, and calls `store.rerank()`, which caches scores by key `getCacheKey("rerank", { query, model, chunk })`.

---

## 2. Evidence-Ranked Defect Findings

### HIGH Severity Findings

1. **Explicit 60s Timeout Override in HTTP Server**
   - **Symbol:** `scripts/qmd_mlx/server.py:271` (`submit_embed(..., timeout=60.0)`)
   - **Evidence:** `server.py` explicitly passes `timeout=60.0` to `runtime.submit_embed`, overriding `runtime.py:399`'s intended 300s default and client-side 300s ceiling. A 32-chunk batch on larger models or slower devices times out prematurely at 60s on the server even when the client waited up to 300s.

2. **Unbounded Future Wait Without Timeout Cancellation & GPU Runaway**
   - **Symbol:** `scripts/qmd_mlx/runtime.py:413` (`fut.result(timeout=timeout)`) & `runtime.py:241`
   - **Evidence:** If `fut.result` times out in `submit_embed`, the client/caller receives a TimeoutError, but the job in `_work_queue` is NOT cancelled if it is already executing or queued. The worker only checks `cancel_event.is_set()` before beginning the entire job (line 241), never between micro-batches. Abandoned or timed-out requests continue consuming GPU resources indefinitely.

3. **Multi-Stage Concurrency Collision (No Single GPU Execution Owner)**
   - **Symbol:** `scripts/qmd_mlx/server.py:225`, `scripts/qmd_mlx/rerank.py:143`, `scripts/qmd_mlx/generate.py:59`, `scripts/qmd_mlx/runtime.py:87`
   - **Evidence:**
     - `/rerank` executes `MLXRerankAdapter.score_pairs` directly inside concurrent HTTP server threads without any worker thread or lock.
     - `/generate` executes on its own separate `_worker_thread` and unbounded queue.
     - `/embed` executes on `MLXEmbeddingRuntime._worker_thread`.
     - No daemon-wide GPU execution owner exists across the three stages. Concurrent embed, rerank, and generate requests execute simultaneously on Metal, leading to GPU contention, race conditions, and uncoordinated memory spikes.

4. **False CLI Completion & Unaccounted Exit Status**
   - **Symbol:** `src/cli/qmd.ts:1778–1782`, `src/store.ts:1553-1565`
   - **Evidence:**
     - `src/cli/qmd.ts:1778` unconditionally prints `100%` and `✓ Done!` even when `result.errors > 0` or when embedding was aborted due to high error rates or session expiration.
     - `src/cli/qmd.ts` never sets a non-zero `process.exitCode` on partial or failed embedding runs.
     - `src/store.ts:1554, 1563` adds unattempted remaining chunks directly to `errors` count (`errors += remaining`), conflating unattempted/skipped work with failed attempts.
     - `src/store.ts:1638` reports `docsProcessed: totalDocs` regardless of how many documents actually succeeded or were committed.

5. **Counter Desynchronization & Conflated DB/Inference Failures**
   - **Symbol:** `src/store.ts:1577–1592`
   - **Evidence:** Inside `try { db.exec("BEGIN IMMEDIATE"); ... }`, `chunksEmbedded++` and `errors++` are modified per chunk *before* `COMMIT`. If `COMMIT` fails or SQLite throws on the N-th chunk, the in-memory counters have already been mutated. The catch block rolls back the transaction and enters a generic fallback loop that re-invokes inference on the same chunks, conflating DB write failures with retryable inference failures and double-counting progress/errors.

6. **Embedding Backend Fallback Cross-Contaminates Vector Space**
   - **Symbol:** `src/llm.ts:1049–1055`, `src/llm.ts:1086–1092`
   - **Evidence:** When `embedBackend === 'mlx'`, if MLX encounters an error and `mlxFallback !== false`, `_embedMlx` sets `mlxFailed = true` and returns `null`. Then `embed()` and `embedBatch()` fall through to `ensureEmbedContext()` (GGUF / node-llama-cpp model `embeddinggemma-300M`). Inserting GGUF vectors into an MLX-initialized table (or vice-versa) pollutes the vector space with incompatible vector representations and dimensions.

7. **Reranker & Expansion Cache Provenance Corruption on Fallback**
   - **Symbol:** `src/store.ts:3293–3298`, `src/store.ts:3354–3366`, `src/store.ts:3384`
   - **Evidence:** `predictedModelFor(llm, 'rerank', model)` pre-computes `effectiveModel = mlx:<model>`. If `llm.rerank()` subsequently fails and falls back to GGUF, the score returned from GGUF is written to the SQLite cache under `model: effectiveModel` (`mlx:<model>`). Subsequent queries retrieve the GGUF score mislabeled as MLX.

---

## 2. MEDIUM Severity Findings

8. **Tokenization Redundancy & Mask Assumptions (Fake Token-Once)**
   - **Symbol:** `scripts/qmd_mlx/batching.py:105`, `scripts/qmd_mlx/runtime.py:280`
   - **Evidence:** `BatchPlanner.plan_and_execute` encodes texts via `tokenizer.encode(t)` to calculate lengths, and then `_embed_sync` tokenizes the same texts again via `self.raw_hf_tokenizer(texts, padding=True, ...)`. Furthermore, if `tokenizer.encode` raises an exception, `batching.py:108` catches all exceptions and guesses token lengths as `len(t.split()) * 1.3`, hiding tokenization failures.
   - **Padding / Mask Risk:** `runtime.py:307` uses `last_idx = mx.sum(attention_mask, axis=1) - 1` assuming right padding for causal models without enforcing `padding_side = "right"` on the tokenizer.

9. **Worker Death on Reload Exception & Static Readiness**
   - **Symbol:** `scripts/qmd_mlx/runtime.py:246–247`
   - **Evidence:** In `_worker_loop`, `if not self._weights_loaded: self._reload_weights()` is called outside the per-job `try...except`. If `_reload_weights()` throws an exception, the worker thread crashes and exits the loop permanently, while `server.py` state remains `ServerState.READY` and `/ready` continues returning 200.

10. **Global Class State & Broken Lazy / Preload State in HTTP Handler**
    - **Symbol:** `scripts/qmd_mlx/server.py:67–74`, `server.py:358–359`
    - **Evidence:** Server state is stored as class attributes on `MLXHTTPRequestHandler` (`MLXHTTPRequestHandler.state = ...`). When multiple servers are created (e.g. during testing with ephemeral ports), they clobber each other's state. When `preload=False`, `server.py:359` immediately marks `MLXHTTPRequestHandler.state = ServerState.READY` even though `runtime` is `None`.

11. **Heuristic Resource Sizing & Unsafe Generic Metal Error Retry**
    - **Symbol:** `scripts/qmd_mlx/batching.py:31–60`, `scripts/qmd_mlx/batching.py:162`
    - **Evidence:** `calculate_default_max_batch_tokens` uses total physical RAM rather than available/free memory. In `_execute_with_retry`, catching any error containing `"metal"` can trigger infinite recursive bisection retries on non-OOM hardware defects (such as invalid shader compilation or kernel aborts).

12. **Incomplete Resume on Partial Document Embeddings**
    - **Symbol:** `src/store.ts:1457` (`getPendingEmbeddingDocs`)
    - **Evidence:** Query `getPendingEmbeddingDocs` selects documents with zero associated vector rows. If an indexing job was interrupted mid-document (some chunks embedded, some missing), resuming the job does not re-select or complete the missing chunk sequences for that document.

---

## 3. LOW / Documentation Findings

13. **Unsupported Performance Multiplier Claims in Docs**
    - **Symbol:** `CLAUDE.md:156`
    - **Evidence:** `CLAUDE.md` states "Dual embedding backends: MLX-native (Apple Silicon GPU, 2-5x faster)". Per the overhaul plan, unproven speed multipliers must be removed and replaced with accurate descriptions.

---

## 4. Regression Baseline

Before making any modifications, the existing test suites were executed to establish a baseline:

- **TypeScript / Bun Test Suite:**
  - Command: `bun test`
  - Result: 26 files passed, 0 failures (108 tests passed in ~5.4s).
- **Python / Pytest Test Suite:**
  - Command: `PYTHONPATH=. .venv/bin/pytest`
  - Result: 6 files passed, 42 tests passed in 34.29s.
- **Build:**
  - Command: `bun run build`
  - Result: Successful compilation (`dist/` generated with shebang).

All modifications will be executed in dependency order according to `docs/plans/reliability-first-overhaul.md` with regression tests added at each step.
