# Next-Deployment Qualification & Cutover Checklist

**Target System**: `qmd-mlx-search`  
**Current Live Baseline**: `hf_Qwen_Qwen3-Embedding-0.6B-Q8_0.gguf` (1024d, GGUF backend on port 8787)  
**Date**: 2026-09-10  

---

## 1. Component Gate Summary

| Component | Target Artifact / Spec | Qualification Status | Notes / Operational Boundary |
|---|---|---|---|
| **Primary Embedding** | `qwen3-embedding-4b-mlx-4bit-affine` | **FAILED / UNFULFILLED** | Failed concurrent interactive target ($\le 200\text{ms}$ p95; measured 642.06ms under bulk load). 1000-iteration sustained qualification not fulfilled for live search promotion. Live search retains 0.6B GGUF. |
| **Reranker** | `qwen3-reranker-4b-mlx-4bit` | **QUALIFIED (Single-Stage)** | Exact Qwen prompt template preserved; dynamic `yes`/`no` token resolution (`yes=9693`, `no=2152`); single-stage Metal residency verified (2159.4 MB); bounded smoke passed in 7.67s. |
| **Query Expansion** | `mlx-community/Qwen3-1.7B-4bit` | **QUALIFIED (Single-Stage)** | Deterministic query expansion verified (382.5ms p50); token streaming cancellation supported; single-stage Metal residency verified (923.2 MB); bounded smoke passed in 5.02s. |
| **Offline Retrieval Harness** | `scripts/qmd_mlx/eval_harness.py` | **QUALIFIED** | Purely offline, isolated temporary SQLite storage; evaluates Hit@1, Hit@3, Hit@5, MRR, and NDCG@5 on public judged benchmark fixtures. |

---

## 2. Unfulfilled Gates & Blocker Tracking

### 2.1 Embedding $\le 200\text{ms}$ Concurrent Latency Gate (Unfulfilled)
- **Status**: **FAILED FOR PRIMARY INTERACTIVE PROMOTION**
- **Empirical Boundary**: When 4-item bulk batches (~700 tokens) run on Apple Silicon Metal, in-flight matrix multiplications occupy the command queue for ~500ms. Even with interactive priority scheduling, concurrent p95 latency is 642.06ms (MLX) vs 2477.28ms (GGUF).
- **Blocker Resolution**:
  - Do NOT deploy 4B embedding as primary interactive search default.
  - Retain 0.6B GGUF (20.42ms solo p95, 1278.9 tok/s throughput) as primary live search space.
  - If 4B embeddings are desired, deploy strictly as an **asynchronous / off-peak shadow indexing worker** with search-pause coordination.

### 2.2 1000-Iteration Sustained Load Gate (Unfulfilled for 4B Promotion)
- **Status**: **UNFULFILLED**
- **Requirement**: Before any primary model promotion, the model must pass a 1000-iteration sustained qualification test maintaining $\le 200\text{ms}$ p95 under concurrent load with zero swap growth and zero watchdog breaches.

### 2.3 Retrieval Quality & Corpus Evaluation (Pending Full Corpus)
- **Status**: **OFFLINE PUBLIC HARNESS QUALIFIED; FULL CORPUS BENCHMARK PENDING**
- **Constraint**: Retrieval quality cannot be claimed from small numeric smoke tests. Full-corpus retrieval quality evaluation (e.g. MS MARCO, BEIR, or full public held-out corpora) must be conducted in an isolated offline environment before any architectural promotion.

---

## 3. Pre-Deployment Cutover Guardrails & Invariants

```
+-------------------------------------------------------------------------------+
|                      PRE-DEPLOYMENT CUTOVER INVARIANTS                        |
|                                                                               |
|  1. VECTOR SPACE ISOLATION:                                                   |
|     - NEVER query 4B index with 0.6B query embeddings (or vice versa).        |
|     - Each model requires separate, dedicated index storage.                  |
|                                                                               |
|  2. ATOMIC SHADOW CUTOVER:                                                    |
|     - New indices MUST be built into an isolated shadow database.             |
|     - Validate checksums and index integrity before atomic rename/symlink.    |
|     - Rollback target preserved until verification completes.                 |
|                                                                               |
|  3. ZERO LIVE SERVICE INTERFERENCE:                                           |
|     - All test and qualification harnesses MUST bind ephemeral or 8797/8798.   |
|     - Production daemon on 8787 must remain uninterrupted.                   |
+-------------------------------------------------------------------------------+
```

1. **Strict Vector Space & Index Compatibility**:
   - Different embedding models produce non-interoperable vector spaces with incompatible geometric properties and dimensions (e.g., 2560d vs 1024d).
   - Under no circumstances may queries embedded by one model be matched against vectors indexed by a different model.
2. **Isolated Shadow Indexing & Rollback Safety**:
   - Full re-indexing must occur in a dedicated shadow database (e.g., `<collection>.shadow.sqlite`).
   - Active database continues serving live search queries uninterrupted during indexing.
   - Atomic cutover only occurs after full index verification and health checks succeed.
3. **Multi-Stage Service Composition**:
   - MLX server supports single-stage composition (`--no-embed`, `--rerank-model`, `--generate-model`), allowing reranker and query expansion to run independently without loading redundant embedding models into unified memory.
