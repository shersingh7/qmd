# Phase 3 Sustained Embedding Qualification Report (v2)

**Target**: `qwen3-embedding-4b-mlx-4bit-affine`  
**Host Architecture**: Apple Silicon (Darwin arm64, MLX Backend)  
**Evaluation Date**: 2026-09-09 / 2026-09-10  
**Status**: **STAGE 1 PILOT PASSED | STAGE 2 SOAK PARTIAL (WATCHDOG SWAP BREACH AT REQ 99)**

---

## 1. Provenance & Execution Context

| Metric | Value |
|---|---|
| **Git Commit (HEAD)** | `0960f3c4c8fc2c9432ad9249bd808955c04a0a39` |
| **Model Path** | `/Users/shersingh/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine` |
| **Architecture** | `Qwen3ForCausalLM` (36 layers, 2560 hidden dimension, 4.0B parameters) |
| **Quantization** | 4-bit affine (`bits=4, group_size=64, mode=affine`) |
| **Compute Dtype** | `float32` (via `model.set_dtype(configured_dtype)` preserving 4-bit integer weights) |
| **Environment** | Python 3.12.13, Pytest 9.1.1, Bun 1.2+ (Offline: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) |
| **Dedicated Endpoints** | Inference `http://127.0.0.1:8797` \| Control `http://127.0.0.1:8798` |
| **Live Daemon Isolation** | Port 8787 untouched (PID 1446 verified healthy and uninterrupted) |

### Exact Source Code SHA256 Fingerprint Manifest
| Script / Component | SHA256 Fingerprint |
|---|---|
| `scripts/qmd-mlx-sustained.py` | `4b362144916062044ad7778dfc5049ce6edb1e2a99b74660a9345098a0900a04` |
| `scripts/qmd_mlx/sustained.py` | `a39d89264627eaecbbdc48e7e163b4f6974794e6dc63e408ecbbcf8f70067a9a` |
| `scripts/qmd_mlx/server.py` | `743f026845b74aae29c4f87c2b86959c80a2de90a456188e63e52f7d5f803730` |
| `scripts/qmd_mlx/runtime.py` | `4b3283bd61dff944f7fa6847824758af3253d7edb4355aea0dc3e73a26649595` |
| `scripts/qmd_mlx/watchdog.py` | `6be04e4cf74a66ef9a73881a12ded4eaf02e92ae3f2021a7e630a545e85a0887` |
| `scripts/qmd_mlx/executor.py` | `2c926f275707053b5b4a99383e98c28a22772704b6579da1f9f2a2c61613285c` |
| `scripts/qmd_mlx/batching.py` | `1212cfca3b50d59a6b5727e212922df0f19f5d1156b211d5ab13a654a5cb96f9` |
| `scripts/qmd_mlx/model_manager.py` | `2823fb35ca5635f9f01dd3fef59d4d8e723d8d4b50f9cd01974e53b404f80f6d` |
| `scripts/qmd_mlx/adapters/embedding.py` | `f356c5b72b7c5f10af1171f432b38e6ca4725fd759f2cab2961b21be0dde3662` |
| `scripts/qmd_mlx/adapters/tokenization.py` | `56ba3f16dd87cdee78f1b8e4c323dcb4fc442a8e7442bca51d272365e20e5bee` |
| `test/python/test_mlx_sustained.py` | `6c10156d4001b9be8095adcb07b03650ea2fe5d3a54d6fc2c253457d07914b1c` |

---

## 2. Test Gate Summary

- **Offline Unit & Integration Suite (`pytest test/python/`)**: **204 passed, 11 skipped in 28.88s** (0 failures, 215 total items).
- **TypeScript / Bun Vitest Suite (`bun test` / `bun run test`)**: **775 passed, 72 skipped (847 total across 23 test files) in 70.03s** (0 failures).
- **TypeScript Build (`bun run build`)**: Exit code 0.

> [!NOTE]
> **Prior Artifact Clarification**: Initial artifacts `phase3-pilot-report.json` and `phase3-soak-report.json` represented preliminary baseline measurements using sequential interleaving and preliminary length fixtures. They remain immutable as historical records. The v2 artifacts below represent the complete Phase 3 qualification with true concurrent load measurement, exact 2047/2048/2049 boundaries, and separate rejection accounting.

---

## 3. Embedding Space ID & Compute-Policy Audit

The embedding contract computes canonical space IDs to ensure cache validity and prevent mixing incompatible embeddings:
1. `src/embedding/contract.ts` incorporates `dt: descriptor.dtype?.trim() || ""` into canonical geometry hashing.
2. `test/embedding-contract.test.ts` verifies that varying `dtype` generates distinct embedding space IDs.
3. No silent migrations or implicit fallbacks are permitted; index recreation is required across distinct compute policy signatures.

---

## 4. Stage 1 Bounded Pilot (v2) Results

- **Artifact**: `docs/reviews/artifacts/phase3-pilot-v2-report.json`
- **Execution Mode**: `pilot` (Real Model, `synthetic: false`)
- **Duration**: 29.99s (timeout: 120.0s)
- **Preflight Headroom**: 14,880.8 MB ($\ge 6,000.0\text{ MB}$ required)
- **Cold Startup Latency**: 3.400s
- **GPU Warmup Latency**: 266.43ms (excluded from measured workload metrics)
- **Status**: **PASSED**

### Workload & Boundary Strata Metrics
- **Total Measured Requests**: 23 / 30
  - **Successful Embed Requests**: 22
  - **Expected Boundary Rejections (400)**: 1
- **Successful Embedded Tokens**: 9,005 tokens (rejections strictly excluded)
- **True Token Throughput**: 317.5 tokens/sec
- **Cumulative Wall Accounting**:
  - Total Benchmark Elapsed: 28.73s
  - Successful Wire Time Sum: 28.363s
  - Expected Rejection Wire Time: 0.004s (3.55ms)
- **Latency Distribution (Successful Requests)**:
  - `min`: 80.92ms
  - `p50`: 602.42ms
  - `p95`: 4604.39ms
  - `max`: 4775.22ms
  - `mean`: 1289.24ms (descriptive only: N=22)
- **Strata Latency Profile**:
  - `short` (12–13 tokens, N=8): `p50` = 168.12ms, `p95` = 1232.10ms, `mean` = 525.09ms
  - `medium` (61–75 tokens, N=4): `p50` = 280.94ms, `p95` = 352.17ms, `mean` = 282.62ms
  - `long` (706–941 tokens, N=2): `p50` = 1781.97ms, `p95` = 2012.59ms, `mean` = 1781.97ms
  - `code` (135–178 tokens, N=3): `p50` = 417.25ms, `p95` = 750.56ms, `mean` = 520.09ms
  - `boundary_2047` (exact 2047 tokens, N=1): `p50` = 4775.22ms (PASSED, unit norm $\|v\|_2 = 1.0$)
  - `boundary_2048` (exact 2048 tokens, N=1): `p50` = 4693.45ms (PASSED, unit norm $\|v\|_2 = 1.0$)
  - `boundary_2049` (exact 2049 tokens, N=1): Verified HTTP 400 explicit rejection in 3.55ms (`InvalidInputError: Text at index 0 token length (2049) exceeds max_length (2048)`)
  - `mixed` (batch-4 bulk, N=3): `p50` = 2795.47ms, `p95` = 2900.57ms, `mean` = 2813.05ms

### True Concurrent Load Interleaving (Tail Latency & Queue Wait)
Executed using two concurrent client threads with barrier/telemetry synchronization (Interactive query arriving while bulk indexing batch is actively executing on GPU):
- **Solo Baseline Interactive Query (Idle GPU)**: `p50` = 82.01ms, `p95` = 84.29ms (N=3)
- **Concurrent Interactive Query Under Load**: `p50` = 1220.65ms, `p95` = 1236.50ms (N=3)
- **Concurrent Bulk Indexing Batch**: `p50` = 2795.47ms, `p95` = 2900.57ms (N=3)
- **Estimated Queue Wait Delta**: `p50` = 1138.64ms, `max` = 1156.25ms
- **Slowdown Factor Under Load**: 14.71x mean slowdown
- **Request Overlap**: 100% verified concurrent execution window.

### Unified Memory & Metal Telemetry
- Peak Metal Active: 4017.8 MB
- Active Metal Delta: 238.4 MB
- Process Cleanup: PID 27585 cleanly terminated on SIGTERM (Exit code: 0, no SIGKILL).

---

## 5. Stage 2 Bounded Soak (v2) Results

- **Artifact**: `docs/reviews/artifacts/phase3-soak-v2-report.json`
- **Execution Mode**: `soak` (Real Model, `synthetic: false`)
- **Duration**: 50.75s (timeout: 180.0s)
- **Preflight Headroom**: 18,509.0 MB ($\ge 6,000.0\text{ MB}$ required)
- **Cold Startup Latency**: 4.097s
- **GPU Warmup Latency**: 264.62ms
- **Status**: **PARTIAL / WATCHDOG INTERVENTION AT REQ 99 (SWAP GROWTH BREACH)**

### Workload & Watchdog Telemetry
- **Completed Requests**: 98 / 100 requests completed successfully (14,600+ tokens).
- **Watchdog Breach Event**: At request 98/99, system swap growth reached 2053.4 MB (exceeding conservative 2048.0 MB threshold from baseline 5272.4 MB).
- **Watchdog Intervention**: `MLXWatchdog` immediately triggered SIGTERM on target PID 28086, reaped process cleanly (Exit 0, no hang), and persisted partial failure artifact.
- **Provisional Stability Assessment**: **Unstable under sustained 100-request soak** due to macOS system swap escalation during multi-batch allocation cycles.

---

## 6. Output Embeddings Mathematical Validity

Across all successful measured requests in Pilot v2:
1. **Finitude**: 100% of generated embeddings contain strictly finite real numbers (no `NaN`, `Inf`, or `-Inf`).
2. **L2 Normalization**: All vectors verified unit Euclidean norm ($\|v\|_2 = 1.000000 \pm 10^{-6}$).
3. **Dimensionality**: Exactly 2560 dimensions matching `qwen3` embedding geometry.
4. **Boundary Verification**: Exact 2047 and 2048 token inputs pass with valid unit embeddings; exact 2049 token inputs strictly rejected with HTTP 400 (`InvalidInputError`).

---

## 7. Phase 3 Sustained Qualification Verdict

| Criterion | Requirement | Result | Verdict |
|---|---|---|---|
| **Python Offline Suite** | $\ge 200$ passing, 0 failures | 204 passed, 11 skipped (0 failed) | **PASS** |
| **TypeScript Suite** | 100% passing | 775 passed, 72 skipped (847 total) | **PASS** |
| **Space ID Identity** | Compute dtype included in hash | Verified in contract and unit tests | **PASS** |
| **Cold Startup Latency** | $< 10.0\text{s}$ | 3.40s (Pilot v2) / 4.10s (Soak v2) | **PASS** |
| **Warmup Exclusion** | Excluded from measured stats | Excluded (266.4ms / 264.6ms) | **PASS** |
| **2047/2048 Boundary Policy** | Pass with unit embeddings | 2047 & 2048 verified unit norm | **PASS** |
| **2049 Over-Length Policy** | Explicit HTTP 400 rejection | HTTP 400 verified in 3.55ms | **PASS** |
| **True Concurrent Tail Latency** | Measure queue wait under load | Solo: 82.0ms p50 \| Concurrent: 1220.6ms p50 | **PASS** |
| **Stage 1 Bounded Pilot** | Complete $\le 30$ reqs, $\le 120\text{s}$ | 23 requests in 29.99s, 0 violations | **PASS** |
| **Stage 2 Bounded Soak** | Complete 100 reqs with stable swap | Swap growth breach (2053.4 MB) at req 99 | **FAIL / PARTIAL** |
| **Watchdog Teardown** | Clean SIGTERM on breach | PID 28086 cleanly reaped on SIGTERM | **PASS** |
| **Live Daemon Isolation** | Port 8787 unaffected | PID 1446 healthy | **PASS** |

**Summary**: Stage 1 Bounded Pilot (v2) successfully completed all functional, boundary (2047/2048/2049), and concurrent load qualification gates. Stage 2 Bounded Soak completed 98 requests before the active `MLXWatchdog` intervened on system swap growth (2053.4 MB), proving watchdog enforcement while highlighting host swap sensitivity under sustained multi-batch execution.
