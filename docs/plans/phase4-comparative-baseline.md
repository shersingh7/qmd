# Phase 4 Plan: Fair GGUF Comparative Baseline Benchmark

**Target Investigation**: Empirical Comparative Baseline of GGUF (`node-llama-cpp`) vs MLX for Embedding Models  
**Status**: DRAFT / EXECUTION READY  
**Primary Artifact Reference**: `docs/reviews/artifacts/phase3-parent-priority-pilot.json` (MLX 4B failed 200ms target: solo p95 126.03ms, concurrent p95 642.06ms)  
**Acceptance Standard**: Apple Silicon MLX Benchmark & Acceptance Protocol (`docs/benchmarks/acceptance-protocol.md`)

---

## 1. Executive Summary & Objective

In Phase 3, sustained qualification testing of `qwen3-embedding-4b-mlx-4bit-affine` revealed a critical latency bottleneck:
- **Solo Interactive Baseline (Idle)**: p50 = 123.57ms, p95 = 126.03ms.
- **Concurrent Interactive Query (Under Bulk Load)**: p50 = 641.48ms, p95 = 642.06ms (5.16x slowdown vs solo baseline).
- **Gate Outcome**: **FAILED** (violated the strict $\le 200.0\text{ms}$ concurrent interactive responsiveness target).

Rather than engaging in ungrounded MLX queue tuning or relaxing the acceptance criteria, Phase 4 establishes an **isolated, fair, reproducible GGUF baseline benchmark** using the existing `node-llama-cpp` implementation and locally cached weights.

The goal is to answer the fundamental architectural questions with empirical data:
1. **4B Parity**: How does `hf_Qwen_Qwen3-Embedding-4B-Q4_K_M.gguf` perform under identical workload strata, batching, and concurrent interleaving compared to `qwen3-embedding-4b-mlx-4bit-affine`?
2. **Current Live Baseline**: How does the actual active production embedding model (`hf_Qwen_Qwen3-Embedding-0.6B-Q8_0.gguf` identified in `~/.config/qmd/index.yml`) perform under the same harness?
3. **Architectural Recommendation**: Determine whether 4B embedding is fundamentally suitable for real-time interactive search on Apple Silicon, or whether off-peak/background indexing or a smaller model tier (0.6B / 1.5B) is required.

---

## 2. Model & Configuration Inventory (Read-Only Audit)

### 2.1 Cached Models in `~/.cache/qmd/models/`

| Model File / Directory | Size (Bytes) | Format / Quantization | Purpose / Role |
|---|---|---|---|
| `hf_Qwen_Qwen3-Embedding-4B-Q4_K_M.gguf` | 2,496,703,776 | GGUF / `Q4_K_M` (k-quants) | Equivalent 4B GGUF comparison target |
| `hf_Qwen_Qwen3-Embedding-0.6B-Q8_0.gguf` | 639,150,592 | GGUF / `Q8_0` (8-bit linear) | **ACTUAL current live embedding baseline** |
| `hf_ggml-org_embeddinggemma-300M-Q8_0.gguf` | 333,590,944 | GGUF / `Q8_0` | Upstream default configuration fallback |
| `qwen3-embedding-4b-mlx-4bit-affine` | 2,274,123,111 | MLX / 4-bit affine (group 64) | Reference MLX 4B implementation |

### 2.2 Quantization Scheme Disclosure

The quantization schemes are **not mathematically identical**:
- **MLX 4-bit Affine**: `bits=4, group_size=64, mode=affine` with per-group FP16 scale and FP16 bias, computed with FP32 activations and last-token hidden-state pooling.
- **GGUF Q4_K_M**: llama.cpp k-quants format utilizing 4-bit quantized weights with block-level scales and minimum values, executed via Metal shaders in `node-llama-cpp`.
- **GGUF Q8_0**: 8-bit quantized weights with per-block 32-element scaling.

This discrepancy is explicitly disclosed in all manifests and reports. Model comparisons must not claim numerical identity across quantization formats, but rather evaluate operational speed, memory footprint, and responsiveness on identical input corpora.

### 2.3 Live System Configuration Source Audit

A read-only audit of `~/.config/qmd/index.yml` confirms:
```yaml
models:
  embed: /Users/shersingh/.cache/qmd/models/hf_Qwen_Qwen3-Embedding-0.6B-Q8_0.gguf
  generate: hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf
  rerank: /Users/shersingh/.cache/qmd/models/hf_Voodisss_Qwen3-Reranker-4B-Q4_K_M.gguf
```
> [!IMPORTANT]
> The live production embedding space is `hf_Qwen_Qwen3-Embedding-0.6B-Q8_0.gguf` (0.6B parameters, 1024 dimensions), **not** the upstream default `embeddinggemma-300M`. The comparative analysis will evaluate both the 4B equivalent GGUF and the actual live 0.6B GGUF.

---

## 3. Experimental Controls & Equivalence Protocol

To ensure a strictly fair, apples-to-apples benchmark against MLX Phase 3 results:

1. **Exact Corpus Fixtures & Ordering**:
   - `SHORT_FIXTURES` (5 items, 12–13 tokens): Short interactive queries.
   - `MEDIUM_FIXTURES` (3 items, 61–75 tokens): Medium documentation paragraphs.
   - `LONG_FIXTURES` (2 items, 706–941 tokens): Long technical specifications.
   - `CODE_FIXTURES` (2 items, 135–178 tokens): Python & TypeScript syntax blocks.
   - `boundary_2047` (exact 2047 tokens): Boundary passing singleton.
   - `boundary_2048` (exact 2048 tokens): Boundary passing singleton.
   - `boundary_2049` (exact 2049 tokens): Boundary rejection test.

2. **Role & Prompt Formatting Policy**:
   - For Qwen3 embedding models:
     - Queries: `Instruct: Retrieve relevant documents for the given query\nQuery: {query}`
     - Documents: Raw unadorned text `{text}` (or `{title}\n{text}`).
   - Exact compliance with `src/llm.ts` `formatQueryForEmbedding` and `formatDocForEmbedding`.

3. **Backend Tokenizer Independence**:
   - Tokenizer token counts are measured directly from the active backend (`node-llama-cpp` tokenizer vs MLX HuggingFace tokenizer).
   - Counts are reported as-is without normalizing by guessed or foreign tokenizers.

4. **Measurement Protocol & Wall-Clock Accounting**:
   - **Cold Startup**: Measured from process launch to first `/health` readiness response.
   - **GPU Warmup**: 1 query + 1 batch-4 request executed post-startup; strictly excluded from all reported latency and throughput metrics.
   - **Solo Baseline**: 3 interactive query requests measured under idle system conditions.
   - **Workload Strata**:
     - Stage A: Singletons (batch size 1 across short, medium, long, code).
     - Stage B: Boundary tests (2047 pass, 2048 pass, 2049 reject).
     - Stage C: Batch size 2 progression (short, medium, code).
     - Stage D: Batch size 4 progression (short, medium).
     - Stage E: Concurrent interleaving (3 iterations of 1 batch-4 bulk request + 1 concurrent interactive query with execution handshake).
   - **End-to-End Accounting**: Latency measured via HTTP socket wire protocol to maintain exact structural comparability with the MLX runner, capturing serialization, buffer conversion, and compute.

5. **Numerical Correctness & Validation**:
   - Every embedding vector must be verified for:
     - Finite values (no NaN, no $\pm\infty$).
     - Correct dimensionality (2560 for 4B, 1024 for 0.6B).
     - Unit L2 normalization ($\|v\|_2 = 1.0 \pm 10^{-4}$).

6. **External Watchdog & Memory Supervision**:
   - Continuous background supervisor sampling RSS, system swap usage, swap growth, and available RAM headroom every 250ms.
   - Strict process ownership of the Node child process with guaranteed SIGTERM/SIGKILL teardown.
   - Absolute wall-clock timeout of 120.0s and maximum 30 measured requests for the pilot.

---

## 4. Safety & System Isolation Guardrails

1. **Sequential Execution (Never Concurrent MLX + GGUF)**:
   - MLX and GGUF qualification processes must **never** run concurrently on the GPU.
   - Headroom must be verified ($\ge 6000\text{ MB}$ available) before starting any run.
2. **Live Daemon Protection**:
   - The existing live MLX daemon (PID 1446 on port 8787) is left completely untouched.
   - GGUF benchmark server binds exclusively to isolated qualification ports (e.g., `8795` / `8796`).
3. **No Database or Index Writes**:
   - Zero modifications to `~/.config/qmd/`, SQLite databases, or local file collections.
4. **No Downloads or Package Installations**:
   - Harness executes strictly offline using existing `node-llama-cpp` and local weights.

---

## 5. Harness Architecture & Component Design

```
+-------------------------------------------------------------------------------+
| Python Qualification Harness (scripts/qmd_gguf_benchmark.py)                  |
|                                                                               |
|  +---------------------+   +---------------------+   +---------------------+  |
|  | Preflight & Memory  |   | MLXWatchdog /       |   | Fixture Execution & |  |
|  | Headroom Guard      |   | Supervisor Thread   |   | Correctness Gates   |  |
|  +---------------------+   +---------------------+   +---------------------+  |
|             |                         |                         |             |
|             v                         v                         v             |
|  +-------------------------------------------------------------------------+  |
|  | Owned Child Subprocess: Node.js GGUF Server                             |  |
|  | (scripts/gguf_embed_server.mjs on http://127.0.0.1:8795)               |  |
|  |                                                                         |  |
|  |  [node-llama-cpp] -> Metal Context -> Qwen3 GGUF (4B Q4_K_M / 0.6B Q8) |  |
|  |  Endpoints: /embed, /tokenize, /health, /descriptor, /memory            |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
```

### Component Roles:
1. `scripts/gguf_embed_server.mjs`: Lightweight HTTP server implementing the QMD embedding protocol on top of `node-llama-cpp`, supporting batching, tokenization, context pooling, and memory introspection.
2. `scripts/qmd_gguf_benchmark.py`: Python benchmark runner sharing the proven telemetry, watchdog, and gate evaluation logic from `scripts/qmd_mlx/sustained.py`.
3. `test/python/test_gguf_benchmark.py`: Unit and mock tests ensuring harness correctness, parameter parsing, and gate enforcement without invoking Metal.

---

## 6. Execution Milestones & Deliverables

| Step | Milestone | Criteria |
|---|---|---|
| **M1** | Offline Test Suite Verification | Python pytest (212 pass / 11 skip) + Vitest (775 passed, 72 skipped / 847 total) + `bun run build` exit 0 |
| **M2** | GGUF Benchmark Implementation | `scripts/gguf_embed_server.mjs`, `scripts/qmd_gguf_benchmark.py`, unit tests |
| **M3** | Harness Offline Verification | Offline tests passing with mock/dry-run support |
| **M4** | Preflight Headroom Check | Host memory headroom $\ge 6000\text{ MB}$, ports 8795/8796 free |
| **M5** | Stage 1 Pilot: GGUF 4B Q4_K_M | Bounded pilot ($\le 30$ requests, $\le 120\text{s}$) on 4B GGUF |
| **M6** | Stage 2 Pilot: GGUF 0.6B Q8_0 | Bounded pilot on actual live baseline model for reference |
| **M7** | Comparative Analysis & Report | Structured JSON artifacts + comprehensive evaluation report |
