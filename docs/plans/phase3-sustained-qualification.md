# Phase 3 Sustained Embedding Qualification Plan

## 1. Executive Summary & Objective

**Objective:** Execute bounded sustained qualification of the primary Apple Silicon MLX embedding model (`qwen3-embedding-4b-mlx-4bit-affine`, 2560d) under strict resource isolation, active watchdog supervision, and rigorous measurement standards defined in `docs/benchmarks/acceptance-protocol.md`.

**State & Provenance:**
- **Parent Test Gates:** **198 passed, 11 skipped** (offline Python test suite).
- **TypeScript / Vitest Gates:** **846 passed** (23 test files).
- **Core Numerical Fix:** Parent resolved batch-vs-singleton numerical divergence in `QwenEmbeddingAdapter.load` by configuring `model.set_dtype(configured dtype)` (float32 compute), preserving packed 4-bit integer weights while eliminating batch-dependent kernel precision loss. Real smoke + diagnostic passed (`/tmp/qmd-phase3-parent-fixed.json`, cosine similarity = 1.000000, max absolute difference = 3.87e-7).
- **Numerical Semantics Contract:** Because compute precision changed, existing vector indexes generated with prior representations cannot silently be reused. Embedding descriptors and fingerprint space IDs incorporate compute-policy (`dtype`) identity to enforce explicit versioned compatibility (`version: 1`, no silent mixing, no live migration).

---

## 2. Safety Invariants & Owned Process Architecture

| Safety Principle | Enforcement Mechanism | Verification Gate |
| :--- | :--- | :--- |
| **Owned Process Composition** | Reuses `SmokeRunner` / `MLXWatchdog` architecture. Spawns `scripts/mlx_embed_server.py` with unique `MLX_INSTANCE_TOKEN`. Never uses unmonitored `Popen` or mock servers. | Child PID strictly verified and reaped by watchdog supervisor. |
| **Active Watchdog Supervision** | Background supervisor samples RSS, system swap growth, system memory pressure, HTTP `/health`, and GPU worker progress every 250ms. | Automated fail-closed termination on threshold breaches or stuck worker. |
| **Strict Resource Ceilings** | Memory preflight requires $\ge 6,000\text{ MB}$ available headroom (exceeding conservative model budget of $\ge 3,500\text{ MB}$). Refuses startup if headroom inadequate. | Preflight validation check fails closed before spawn. |
| **Strict Port & Endpoint Isolation** | Dedicated loopback endpoints on `127.0.0.1:8797` (inference) and `127.0.0.1:8798` (control). Port `8787` is strictly prohibited. | Live production daemon (PID 1446 on 8787) is read-only and untouched. |
| **100% Offline Airgap** | Injects `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`. Local weights directory path only. | Zero remote network requests or downloads. |
| **Outer Wall-Clock Ceiling** | Strict monotonic wall-clock deadlines on harness, supervisor, and `SmokeHttpClient` (`trust_env=False`, `allow_redirects=False`, 10MB streaming limit). | Prevents unbounded hangs under slow-drip or delayed responses. |
| **Guaranteed Teardown** | Multi-layer `finally` blocks guarantee supervisor termination, graceful `SIGTERM` followed by `SIGKILL` if needed, process reap, and temp log cleanup. | Verified zero orphaned processes or GPU memory leaks. |
| **Strictly Sequential GPU Work** | Never overlap models, tests, or builds. Only one model loaded in Metal memory at any time. | Pre-run verification confirms zero other active GPU test processes. |

---

## 3. Workload Strata & Public Fixtures

In accordance with `docs/benchmarks/acceptance-protocol.md` §3, all inputs are drawn from public, deterministic fixtures verified via `/tokenize` to conform to token strata:

| Stratum | Description | Typical Tokens | Purpose & Characteristics |
| :--- | :--- | :--- | :--- |
| **Short Queries** | Single-line search queries | 5 – 25 | Interactive query latency (p50, p95); prioritized execution. |
| **Medium Passages** | Standard document paragraphs | 50 – 200 | Average semantic search corpus unit. |
| **Long Documents** | Multi-paragraph technical documentation / RFCs | 400 – 1,500 | Attention scaling, memory pressure, micro-batching. |
| **Code Snippets** | TypeScript / Python code blocks | 50 – 500 | Variable syntax-dense token distribution. |

### 3.1 Long-Input Policy & 2048 Boundary Verification
- The harness validates input behavior at the 2048 token boundary (`maxTokens: 2048`).
- Token lengths are measured via tokenization, not character counts.
- Sequences exceeding 2048 tokens are strictly bounded by the tokenizer (`max_length=2048`, `truncation=True`) to prevent unbounded allocation or GPU out-of-memory faults.

---

## 4. Measurement Methodology & Metrics Specification

The harness generates structured JSON manifests capturing truthful, non-fabricated metrics:

1. **Cold Startup & Warmup Isolation:**
   - Cold startup latency recorded separately.
   - Warmup passes (1 singleton, 1 batch of 4) executed and strictly excluded from measured request statistics.
2. **Latency Distributions (Explicit Percentile Method):**
   - Latency distributions reported with Min, p50, p95, Max, and Mean.
   - For small sample sizes ($N \le 30$), percentiles are explicitly noted as descriptive.
   - Calculations use standard linear interpolation: `np.percentile(latencies, q, method='linear')`.
3. **Wire End-to-End vs. Forward Latency Distinction:**
   - `total_ms`: Wall-clock duration from client request initiation to response receipt.
   - `forward_ms`: GPU Metal backbone compute + pooling + normalization time.
   - `tokenize_ms`: CPU tokenization time.
4. **Effective Token Throughput:**
   - Calculated as: $\text{Effective Tokens/s} = \frac{\sum \text{True Input Tokens (non-pad)}}{\text{Total Wall-Clock Time (s)}}$.
5. **Padding Overhead:**
   - Calculated as: $\text{Pad Overhead \%} = \frac{\text{Padded Tokens} - \text{True Tokens}}{\text{True Tokens}} \times 100$ when micro-batch padding is measured, else marked `"unavailable"`.
6. **Unified Memory Telemetry:**
   - Process RSS (`ps -o rss=`) sampled continuously.
   - System swap usage & swap growth (`sysctl vm.swapusage`).
   - Metal unified memory metrics from `/memory` control endpoint (`active_mb`, `peak_mb`, `model_mb`).
7. **Interleaved Workload Execution:**
   - Conservative batch progression: Batch 1 $\to$ Batch 2 $\to$ Batch 4.
   - Interleaving: Interactive short query requests (priority 0) interleaved with bounded indexing-like batches (priority 1) to measure real tail latency under multi-tenant queueing.

---

## 5. Two-Stage Bounded Execution Protocol

```
+---------------------------------------------------------------------------------+
|                       Two-Stage Sustained Qualification                         |
+---------------------------------------------------------------------------------+
|  [Step 1: Offline Fixture Gates]                                                |
|     - Run full offline pytest suite (198 passed, 0 failures).                   |
|     - Verify all synthetic adapter rehearsals pass with 0 GPU load.             |
|                                                                                 |
|  [Step 2: Preflight Memory Telemetry Check]                                     |
|     - Verify available headroom >= 6000 MB.                                     |
|     - Verify ports 8797 / 8798 free; verify live PID 1446 untouched.            |
|                                                                                 |
|  [Step 3: Stage 1 — Bounded Pilot]                                              |
|     - Max 120s wall time, <= 30 measured requests.                              |
|     - Workload: Strata validation + Batch 1, 2, 4 + Interleaved.                |
|     - Output: docs/reviews/artifacts/phase3-pilot-report.json                   |
|     - Gate: Check zero errors, stable RSS, finite vectors, cleanup verified.   |
|                                                                                 |
|  [Step 4: Stage 2 — Bounded Soak] (Authorized ONLY upon Pilot PASS)             |
|     - Max 180s wall time, capped at 100 iterations.                             |
|     - Sustained mixed batch + interactive stream.                               |
|     - Output: docs/reviews/artifacts/phase3-soak-report.json                    |
|     - Gate: Verify zero memory growth across iterations, zero breaches.         |
|                                                                                 |
|  [Step 5: Final Review & Artifact Report]                                       |
|     - Document provenance: Git HEAD, diff hash, model dtype, quantization.      |
|     - Re-run parent test gates to confirm 100% integrity.                       |
+---------------------------------------------------------------------------------+
```

---

## 6. Release Gates & Success Criteria

1. **Gate 1 (Offline Integrity):** All unit/integration tests pass with 0 regressions.
2. **Gate 2 (Strata Verification):** Token counts verified for short (5–25), medium (50–200), long (400–1500), and code (50–500).
3. **Gate 3 (Numerical Correctness):** Cosine similarity $\ge 0.9999$ between singleton and batch representations; all embeddings finite and unit-normalized ($\|\mathbf{v}\|_2 = 1.0 \pm 10^{-4}$).
4. **Gate 4 (Memory Stability):** Zero memory pressure alerts, zero swap growth, and Metal active memory stable across all measured iterations.
5. **Gate 5 (Lifecycle & Isolation):** Watchdog supervisor and server child cleanly reaped on completion with zero orphaned processes and zero live daemon interference.
