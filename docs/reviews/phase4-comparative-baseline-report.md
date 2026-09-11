# Phase 4 Comparative Baseline Report: GGUF vs MLX Embedding Qualification

**Target Investigation**: Empirical Comparative Baseline of GGUF (`node-llama-cpp`) vs MLX for 4B & 0.6B Embedding Models  
**Host Architecture**: Apple Silicon (Darwin arm64, Metal Unified Memory)  
**Evaluation Date**: 2026-09-09 / 2026-09-10  
**Status**: **COMPARATIVE BASELINE EVALUATED (PROVISIONAL) | 4B MODELS UNDER BULK LOAD EXCEED CONCURRENT 200ms TARGET | ARCHITECTURAL TIERING & VECTOR SPACE ISOLATION DELIVERED**

---

## 1. Provenance & Execution Context

| Metric / Parameter | Value |
|---|---|
| **Git Commit (HEAD)** | `0960f3c4c8fc2c9432ad9249bd808955c04a0a39` |
| **Hardware Platform** | Apple Silicon (Darwin arm64, 32 GB Unified RAM) |
| **Runtimes** | Node.js v22.22.3, Bun 1.3.8, Python 3.12.13 (`node-llama-cpp` 3.18.1, `mlx` 0.32.2, `mlx-lm` 0.31.3) |
| **Offline Test Status** | Python: **212 passed, 11 skipped in 30.96s** \| Vitest: **775 passed, 72 skipped (847 total)** \| Build: Exit code 0 |
| **Live Daemon Protection** | Port 8787 untouched (PID 1446 verified healthy and uninterrupted) |
| **GGUF Qualification Ports** | Inference `http://127.0.0.1:8795` \| Control `http://127.0.0.1:8796` |
| **MLX Qualification Ports** | Inference `http://127.0.0.1:8797` \| Control `http://127.0.0.1:8798` |

### Exact Source Code SHA256 Fingerprints
| Script / Component | SHA256 Fingerprint |
|---|---|
| `scripts/gguf_embed_server.mjs` | `3aa899479b4a45053fb2a8f89e4cbfd8e4fcaeec69c6cf5f42f74136611e9a3b` |
| `scripts/qmd-gguf-benchmark.py` | `04d7c5a0ec7b9e7610fa658d533dc2720235ca2cb449be1a7b4f53528b17ee72` |
| `scripts/qmd_gguf/benchmark.py` | `43594d4d6211ae431b816616e1eeb0953a5518b5ea014f3da3b578c7dbd67a9a` |
| `scripts/qmd-mlx-sustained.py` | `b7a9aa95c2aedc2dbe56e10de11dba1fee525e9e9f83f4840417a3d036528274` |
| `scripts/qmd_mlx/sustained.py` | `282fc4cd97cae8ddb7a3f914427bd32349888ca410744f8158d7a988a5a61aff` |
| `scripts/qmd_mlx/watchdog.py` | `6be04e4cf74a66ef9a73881a12ded4eaf02e92ae3f2021a7e630a545e85a0887` |
| `test/python/test_gguf_benchmark.py` | `5c9f5647565780d6b63c7b746a2a074094feec74d320b925b412e69fa03e8717` |
| `test/python/test_mlx_sustained.py` | `8d7c7775d6d062e5290e4470680bebcff14917c1a7e22078819e36990460c4af` |

> [!NOTE]
> All existing raw measurement JSON artifacts (`phase3-parent-priority-pilot.json`, `phase4-gguf-4b-pilot.json`, `phase4-gguf-06b-live-pilot.json`) remain strictly immutable.

---

## 2. Model & Live Configuration Audit

### 2.1 Model Inventory in `~/.cache/qmd/models/`
1. `hf_Qwen_Qwen3-Embedding-4B-Q4_K_M.gguf` (2,496,703,776 bytes): Equivalent 4B GGUF comparison target.
2. `hf_Qwen_Qwen3-Embedding-0.6B-Q8_0.gguf` (639,150,592 bytes): **ACTUAL current live embedding baseline**.
3. `hf_ggml-org_embeddinggemma-300M-Q8_0.gguf` (333,590,944 bytes): Upstream default fallback.
4. `qwen3-embedding-4b-mlx-4bit-affine/` (2,274,123,111 bytes): Reference MLX 4B model.

### 2.2 Live QMD Configuration (`~/.config/qmd/index.yml`)
```yaml
models:
  # FORK (qmd-mlx-search, Sep 2026): MLX embed REVERTED after production-chunk
  # measurement (MLX 4B = 1.94s/chunk vs GGUF 0.6B = 0.155s/chunk; full
  # re-embed would take ~77h). GGUF 0.6B restored as the live space.
  embed: /Users/shersingh/.cache/qmd/models/hf_Qwen_Qwen3-Embedding-0.6B-Q8_0.gguf
  generate: hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf
  rerank: /Users/shersingh/.cache/qmd/models/hf_Voodisss_Qwen3-Reranker-4B-Q4_K_M.gguf
```
> [!NOTE]
> The live system actively runs the 0.6B Q8_0 GGUF model (`1024d`), **not** a 4B model or default 300M model.

### 2.3 Quantization Disclosure
- **MLX 4-bit Affine**: `bits=4, group_size=64, mode=affine` with per-group FP16 scale and bias, evaluated with FP32 activations in Metal shaders.
- **GGUF Q4_K_M**: llama.cpp k-quants format with block-wise quantization scales and minimum offsets.
- **GGUF Q8_0**: 8-bit linear quantization with 32-element blocks.
These schemes are **not mathematically identical**, and performance is compared on identical input corpora under exact protocol matching.

---

## 3. Empirical Results: Head-to-Head Comparative Matrix (Provisional Pilot Data)

> [!NOTE]
> The benchmark figures below represent small sample runs across specific context roles and batches. They provide provisional comparative pilot measurements under tested harness conditions and should not be construed as universal claims of architectural superiority.

All tests executed with exact identical fixtures (`SHORT_FIXTURES`, `MEDIUM_FIXTURES`, `LONG_FIXTURES`, `CODE_FIXTURES`, `boundary_2047`, `boundary_2048`, `boundary_2049`), identical role prefixes (`Instruct: ...\nQuery: ...`), exact same batch progressions (1, 2, 4), and identical concurrent interleaving with execution barrier handshakes.

| Benchmark Dimension | MLX 4B 4-bit Affine (Priority Pilot) | GGUF 4B Q4_K_M (`node-llama-cpp`) | GGUF 0.6B Q8_0 (Live QMD Baseline) |
|---|---|---|---|
| **Artifact File** | `phase3-parent-priority-pilot.json` | `phase4-gguf-4b-pilot.json` | `phase4-gguf-06b-live-pilot.json` |
| **Model Parameters** | 4.0B (2560d) | 4.0B (2560d) | 0.6B (1024d) |
| **Model Weight Size** | 2,274 MB | 2,496 MB | 639 MB |
| **Preflight Headroom** | 15,186.4 MB | 13,871.8 MB | 14,274.7 MB |
| **Cold Startup Time** | 3.694s | **0.761s** | 0.988s |
| **GPU Warmup Time** | 395.02ms | 445.96ms | **92.75ms** |
| **Solo Baseline Interactive p50** | 123.57ms | 91.41ms | **20.22ms** |
| **Solo Baseline Interactive p95** | 126.03ms | 91.72ms | **20.42ms** |
| **Concurrent Interactive p50 (under bulk load)** | 641.48ms | 2477.15ms | **388.94ms** |
| **Concurrent Interactive p95 (under bulk load)** | **642.06ms** | 2477.28ms | **391.09ms** |
| **Concurrent Queue Slowdown Factor** | **5.16x** | 27.06x | 19.24x |
| **Concurrent Bulk Batch p50 (4 items)** | 3626.84ms | 2405.45ms | **393.00ms** |
| **Singleton Medium (68–75 tokens) p50** | **324.01ms** | 339.06ms | **56.62ms** |
| **Singleton Long (706–941 tokens) p50** | 2791.88ms | **1988.45ms** | **322.43ms** |
| **Boundary 2047 Tokens (Pass)** | **7049.59ms** | 9573.83ms | **1695.38ms** |
| **Boundary 2048 Tokens (Pass)** | **6764.42ms** | 9591.73ms | **1696.62ms** |
| **Boundary 2049 Rejection (HTTP 400)** | Verified (4.42ms) | Verified (4.69ms) | Verified (4.41ms) |
| **Effective Token Throughput** | 242.4 tok/s (pilot) / 317.5 (v2) | 216.3 tok/s | **1278.9 tok/s** |
| **Peak Active Memory** | 4017.8 MB | 3225.5 MB | **1382.8 MB** |
| **Swap Growth / Stability** | 0.0 MB (Stable) | 0.0 MB (Stable) | 0.0 MB (Stable) |
| **Qualification Gate (Target $\le 200\text{ms}$)** | **FAILED** (642.06ms) | **FAILED** (2477.28ms) | **FAILED** (391.09ms) |

---

## 4. Empirical Comparative Findings

### 4.1 Concurrent 4B Latency Under Single-Stream GPU Scheduling
The empirical comparison between `qwen3-embedding-4b-mlx-4bit-affine` and `hf_Qwen_Qwen3-Embedding-4B-Q4_K_M.gguf` indicates that under single-stream GPU batch execution where bulk matrix multiplications occupy the Metal command queue for ~500ms, preemption is bounded by the in-flight sub-batch duration:
1. **GGUF / node-llama-cpp**: Evaluates bulk batches as atomic compute commands on Metal. When a 4-item bulk batch (~700 tokens) is executing (taking ~2.4s), a concurrent interactive query is blocked in queue, resulting in **2477.28ms p95 latency (27.06x slowdown)**.
2. **MLX Priority Pipeline**: Yields between sub-batches and prioritizes interactive queries over bulk requests. This reduces concurrent tail latency from 2477ms (GGUF) down to **642.06ms (5.16x slowdown)**.
3. **Observed Boundary**: Because the interactive query must wait for the currently executing Metal command buffer to complete, concurrent interactive p95 latency remains above the $\le 200\text{ms}$ qualification gate for 4B models when uncoordinated bulk batches are active.

### 4.2 Raw Throughput & Long Sequence Scaling (MLX vs GGUF)
- For **short solo queries (12 tokens)**: GGUF 4B is slightly faster on idle GPU (91.72ms vs 126.03ms) due to lightweight C++ Metal dispatch overhead in node-llama-cpp.
- For **long context boundaries (2048 tokens)**: MLX completed in 6.76s vs 9.59s for GGUF (a ~29.5% reduction in execution latency, or a 1.42x speed ratio), demonstrating the scaling behavior of MLX attention and affine dequantization shaders at long sequence lengths.
- For **overall token throughput**: MLX achieves 242.4–317.5 tokens/sec vs 216.3 tokens/sec in GGUF.

### 4.3 Baseline Characteristics of 0.6B Q8_0 GGUF
The benchmark of the live baseline (`hf_Qwen_Qwen3-Embedding-0.6B-Q8_0.gguf`) documents the operational characteristics of the current live search configuration:
1. **Solo Query Latency**: **20.42ms** (6.2x lower latency than MLX 4B, 4.5x lower than GGUF 4B).
2. **Indexing Throughput**: **1278.9 tokens/sec** (~4–5x higher token throughput than 4B models on this hardware).
3. **Memory Footprint**: **1382.8 MB** (vs 4017.8 MB for MLX 4B and 3225.5 MB for GGUF 4B).
4. **Full-Corpus Re-indexing Feasibility**: Re-indexing sizing and duration must be measured through dedicated shadow indexing rather than unverified extrapolations.

---

## 5. Architectural Recommendations & Vector Space Isolation

```
+-------------------------------------------------------------------------------+
|                        RECOMMENDED ARCHITECTURE TIERING                       |
|                                                                               |
|  [ Interactive Search Mode ]        ---->  0.6B / 1.5B Embedding Tier        |
|  - Real-time CLI / MCP search               - 20ms solo p95 latency           |
|  - Sub-400ms worst-case queue               - 1.3 GB memory footprint         |
|                                                                               |
|  [ Asynchronous / Off-Peak Indexing ] ----> 4B Embedding Tier (MLX)          |
|  - Scheduled background bulk indexing       - High-capacity representation    |
|  - Zero concurrent search interference      - Fast 2048-token context scaling |
|  - Pause bulk indexing on search request                                      |
+-------------------------------------------------------------------------------+
```

### 5.1 Critical Rule: Strict Vector Space & Index Isolation
- **NEVER query 4B indexed vectors with 0.6B query embeddings (or vice versa)**. Vector spaces generated by different models and dimensions (e.g., 2560d vs 1024d) are mathematically non-interoperable and semantically incompatible.
- **Dedicated Index per Space**: Each embedding model requires its own matching model, compute policy, and separate index store.
- **Dimensionality vs Recall**: Higher vector dimensionality (e.g. 2560d vs 1024d) does not inherently guarantee higher recall. Recall depends on representation quality, training distribution, and index compatibility.
- **Safe Cutover**: Any transition to a new embedding model requires full offline index creation in an isolated shadow database before any live switchover.

### 5.2 Qualification Gate Maintenance
- Keep the $\le 200.0\text{ms}$ concurrent responsiveness gate intact without threshold lowering.
- Maintain the current live `Qwen3-Embedding-0.6B-Q8_0` as the active live search baseline.
- Future model promotions to live search require passing all qualification gates with verified empirical benchmark artifacts.
