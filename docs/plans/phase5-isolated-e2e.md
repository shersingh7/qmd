# Phase 5 Plan: Isolated End-to-End Public Retrieval & Recovery Qualification

**Target Repository**: `qmd-mlx-search` (`/Users/shersingh/github/qmd-mlx-search`)  
**Status**: ACTIVE / EXECUTION READY  
**Prerequisites**:
- Parent Phase 4 Reranker qualification verified (`docs/reviews/artifacts/phase4-parent-rerank-supervised.json`)
- Parent Phase 4 Generation qualification verified (`docs/reviews/artifacts/phase4-parent-generation-final.json`)
- Verified test baselines: Vitest 862 passed / 24 files; Python 233 passed / 11 skipped
- Active production live baseline retained: GGUF `hf_Qwen_Qwen3-Embedding-0.6B-Q8_0.gguf` (1024d) on production port 8787 (PID 1446 untouched)
- Rollout status: **BLOCKED** from live promotion (sustained 4B concurrent latency 642.06ms > 200ms target; 1000-batch sustained test unfulfilled; full-corpus evaluation pending)

---

## 1. Executive Summary & Qualification Scope

Phase 5 transitions from isolated single-component unit/smoke qualification to **isolated end-to-end production pipeline qualification**. It tests the full lifecycle of QMD—from filesystem document scanning, AST/smart chunking, vector embedding generation, durable checkpointing, interrupted-state recovery, vector storage, BM25/Vector/Hybrid/Rerank retrieval, SQLite online-backup restoration, to MCP Streamable HTTP transport probing—using **100% disposable shadow databases and curated public technical fixtures**.

### Strict Operational Principles:
1. **Zero Live Interference**: Zero interaction with `~/.cache/qmd/index.sqlite`, `~/.config/qmd/`, user document collections, or live production ports (`8181` / `8787`).
2. **Real Production Paths**: All indexing, vector storage (`sqlite-vec`), retrieval (FTS5 + vector + RRF + rerank), checkpointing, and recovery run through the actual production TypeScript SDK and Python MLX bridge—not mock or in-memory stand-ins.
3. **Strict Vector Space Isolation**: Baseline GGUF 0.6B (1024d) and Candidate MLX 4B (1024d/2560d) are indexed into **separate, dedicated shadow databases**. Query spaces are never mixed.
4. **Residency & Memory Safety**: Sequential execution with single GPU ownership. Reranker/generator models are never loaded concurrently with heavy embedding workloads on Metal. Headroom is verified ($\ge 6000\text{ MB}$) before every stage.
5. **No Premature Promotion**: Results from small public fixture corpora are labeled as **Preliminary Release Smoke**. Gates for concurrent interactive latency ($\le 200\text{ms}$) and 1000-batch memory stability remain tracked as unfulfilled blockers.

---

## 2. Public Curated Fixture Corpus & Held-Out Judgments

### 2.1 Curated Public Technical Documents (16 Documents)
The evaluation corpus contains 16 self-contained, technical markdown documents across distinct computer science domains, designed to test lexical, semantic, and hybrid retrieval:

| Doc ID | Title | Topic / Domain | Key Technical Concepts |
|---|---|---|---|
| `doc-api-versioning` | REST API Versioning Strategies | Web Architecture | URI versioning, custom headers, content negotiation (`Accept`), backward compatibility |
| `doc-raft-consensus` | Raft Distributed Consensus | Distributed Systems | Leader election, log replication, safety, heartbeats, term numbers, split votes |
| `doc-sqlite-wal` | SQLite Write-Ahead Logging | Database Internals | WAL journal mode, reader/writer concurrency, checkpointing, table locks, SHM file |
| `doc-apple-metal-mem` | Apple Silicon Unified Memory | OS & Hardware | Unified memory architecture, zero-copy buffers, Metal shaders, PCIe elimination |
| `doc-ann-vector-index` | Approximate Nearest Neighbor Vector Search | Information Retrieval | HNSW graphs, IVFPQ quantization, cosine similarity, recall vs latency trade-offs |
| `doc-cache-invalidation` | Cache Invalidation and Consistency | Distributed Caching | Write-through, write-back, cache-aside, TTL expiration, stampede mitigation |
| `doc-b-tree-indexing` | B-Tree and LSM-Tree Storage Engines | Database Storage | Read amplification, write amplification, SSTables, compaction, range queries |
| `doc-jwt-auth` | JSON Web Token (JWT) Security Patterns | Application Security | Asymmetric RSA/ECDSA signing, refresh token rotation, revocation lists, claims |
| `doc-compiler-jit` | Just-In-Time (JIT) Compilation Techniques | Language Runtimes | Bytecode interpretation, profiling tier, trace optimization, deoptimization |
| `doc-linux-epoll` | Linux Epoll I/O Multiplexing | Operating Systems | Edge-triggered vs level-triggered, `O(1)` event polling, file descriptors, reactor pattern |
| `doc-tls-handshake` | TLS 1.3 Cryptographic Handshake | Cryptography & Networks | 1-RTT handshake, Diffie-Hellman key exchange, forward secrecy, 0-RTT resumption |
| `doc-garbage-collection` | Generational Garbage Collection | Memory Management | Young/old generation, card tables, mark-sweep-compact, pause times, write barriers |
| `doc-cookie-recipe` | Classic Chocolate Chip Cookies (Distractor) | Culinary / General | Flour, baking soda, brown sugar, butter, baking temperature |
| `doc-vacation-policy` | Company Vacation and PTO Policy (Distractor) | Corporate Policy | Paid time off accrual, core hours, HR submission window, local timezones |
| `doc-startup-fundraising` | Startup Seed and Series A Pitch Memo | Business & Finance | Runway calculation, SAFEs, valuation cap, dilution, ARR metrics |
| `doc-phoenix-launch` | Project Phoenix Post-Mortem | Retrospective | Migration rollback, staging fidelity, beta user feedback, launch incident |

### 2.2 Pre-Defined Held-Out Judged Queries (10 Queries)
Relevance judgments are defined and frozen **prior to benchmark execution**:

| Query ID | Query String | Expected Relevant Doc IDs | Category / Difficulty | Evaluation Purpose |
|---|---|---|---|---|
| `Q01` | "HTTP header versus URI path versioning for web APIs" | `["doc-api-versioning"]` | Easy / Keyword | Tests direct lexical match on HTTP API concepts |
| `Q02` | "how do distributed nodes elect a leader after heartbeat timeout" | `["doc-raft-consensus"]` | Medium / Semantic | Tests semantic understanding of consensus algorithms |
| `Q03` | "concurrent readers non-blocking during database write commits" | `["doc-sqlite-wal"]` | Medium / Semantic | Tests SQLite WAL concurrency retrieval |
| `Q04` | "eliminating CPU to GPU buffer copy overhead on Apple chips" | `["doc-apple-metal-mem"]` | Medium / Conceptual | Tests unified memory concepts without exact keyword overlap |
| `Q05` | "hierarchical graph index for fast cosine similarity search" | `["doc-ann-vector-index"]` | Medium / Semantic | Tests vector search ANN indexing concepts |
| `Q06` | "preventing thundering herd when cached entries expire" | `["doc-cache-invalidation"]` | Hard / Technical | Tests cache stampede mitigation terminology |
| `Q07` | "tradeoffs between write amplification in LSM trees vs B-trees" | `["doc-b-tree-indexing"]` | Hard / Technical | Tests database storage engine trade-offs |
| `Q08` | "cryptographic key exchange with forward secrecy and 1-RTT latency" | `["doc-tls-handshake"]` | Hard / Technical | Tests network security protocol details |
| `Q09` | "ingredients and oven temperature for homemade chocolate cookies" | `["doc-cookie-recipe"]` | Easy / Distractor | Tests negative domain separation against technical docs |
| `Q10` | "employee paid time off accrual rules and request notice window" | `["doc-vacation-policy"]` | Easy / Distractor | Tests non-technical policy query precision |

---

## 3. Production Indexing & Durable Recovery Qualification

### 3.1 Isolated Shadow Database Workflow
```
+-------------------------------------------------------------------------------+
|                       ISOLATED SHADOW INDEXING PIPELINE                       |
|                                                                               |
|  [Curated Public Docs] ──► FastGlob Scanner ──► Content-Addressed Hash        |
|                                                      │                        |
|                                                      ▼                        |
|  [Checkpoint SQLite] ◄── runDurableIndexingJob ◄── Smart / AST Chunker        |
|           │                                          │                        |
|           ▼                                          ▼                        |
|   Active Checkpoint ──► Model Embeddings ──► sqlite-vec (Float32Array)        |
+-------------------------------------------------------------------------------+
```

1. **Target Validation**: All shadow paths must pass `validateShadowTarget()`:
   - Must not equal realpath of `~/.cache/qmd/index.sqlite`.
   - Must not match dev/inode of any existing live database.
   - Must contain isolation markers (`shadow`, `test`, `tmp`, or `scratch`).
2. **Dual Backend Indexing**:
   - **GGUF 0.6B Shadow Index**: Built with `hf_Qwen_Qwen3-Embedding-0.6B-Q8_0.gguf` (1024d) via `node-llama-cpp`.
   - **MLX 4B Shadow Index**: Built with `qwen3-embedding-4b-mlx-4bit-affine` / `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` (1024d/2560d) via MLX server.
   - Separate SQLite files created in disposable temporary directories.

### 3.2 Durable Indexing Recovery & Interruption Lifecycle
To qualify the production recovery path:
1. **Mid-Batch Cancellation**:
   - Initiate durable indexing on multi-chunk document corpus with small batch size (`maxDocsPerBatch: 1`).
   - Trigger `AbortSignal.abort()` mid-run after initial chunks are committed.
   - Assert job returns status `"cancelled"`, active checkpoint is persisted with `status: "in_progress"` or `"cancelled"`, and committed chunk vectors are intact in `content_vectors`.
2. **Exact Identity Resumption**:
   - Spawn fresh indexing process targeting the same shadow database with identical model and chunking strategy.
   - Assert indexing resumes from the recorded checkpoint, only embeds remaining missing chunks, and reconciles all chunks without duplicates.
   - Assert `content_vectors` has consecutive sequence IDs $0 \dots N-1$ with zero duplicates or gaps.
   - Assert checkpoint status transitions to `"completed"`.
3. **Fingerprint Mismatch Protection**:
   - Attempt to resume an active checkpoint using a mismatched model name, chunk strategy, or descriptor signature without `--force`.
   - Assert execution is immediately refused with typed `IndexingFingerprintMismatchError`.
   - Assert the database remains 100% byte-equivalent with zero inference calls executed.
4. **SQLite Online-Backup & Temp Restore**:
   - Perform online backup via SQLite `VACUUM INTO 'backup.sqlite'`.
   - Open restored backup with a fresh `QMDStore` instance.
   - Execute test queries and verify identical ranked retrieval results and vector counts as the primary shadow database.

---

## 4. End-to-End Retrieval Evaluation Protocol

### 4.1 Retrieval Pipeline Stages
For each backend index (GGUF 0.6B and MLX 4B):
1. **Stage 1: Lexical Search (BM25)**: SQLite FTS5 index over document titles and text chunks.
2. **Stage 2: Vector Search**: Semantic similarity matching using `sqlite-vec` cosine distance.
3. **Stage 3: Hybrid Retrieval (RRF)**: Reciprocal Rank Fusion ($k=60$) combining BM25 and Vector search results.
4. **Stage 4: Reranked Hybrid Retrieval**: RRF top candidates rescored with single-stage reranker (`qwen3-reranker-4b-mlx-4bit` or GGUF reranker).

### 4.2 Standard Information Retrieval Metrics
Evaluated across all 10 held-out judged queries:
- **Recall@5 (Hit@5)**: Proportion of queries where at least one ground-truth document appears in the top-5 results:
  $$\text{Recall@5} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{Top5}(q) \cap \text{Rel}(q) \neq \emptyset)$$
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of the first relevant document:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_1(q)}$$
- **nDCG@5 (Normalized Discounted Cumulative Gain)**:
  $$\text{DCG@5} = \sum_{i=1}^5 \frac{\text{rel}_i}{\log_2(i+1)}, \quad \text{nDCG@5} = \frac{\text{DCG@5}}{\text{IDCG@5}}$$

---

## 5. Integrated MCP Transport Retrieval Probe

### 5.1 Protocol Verification
- Launch QMD MCP Streamable HTTP server on ephemeral port (OS-assigned port `0` or isolated test port `8795`).
- Bind store explicitly to the public fixture shadow database.
- Execute JSON-RPC tool calls via HTTP `POST /mcp`:
  - `initialize`: Protocol negotiation (`2025-03-26`), verify server capability registration.
  - `tools/list`: Assert `search`, `vector_search`, `get`, and `multi_get` tools are exposed.
  - `tools/call search`: Execute judged query `"Raft consensus leader election"` and assert structured results contain `doc-raft-consensus` with docid `#...`.
  - `tools/call get`: Retrieve document content by docid or path.
- Verify complete server teardown with socket closure and store disconnection.

---

## 6. Execution Safeguards & Telemetry Supervision

1. **Watchdog Supervision**:
   - `MLXWatchdog` active across all native stages sampling RSS, swap usage, swap growth, and memory percentage every 250ms.
   - Preflight memory check: $\ge 6000\text{ MB}$ available RAM headroom.
   - Memory breach thresholds: $\text{max\_rss\_mb} = 8192\text{ MB}$, $\text{max\_swap\_growth\_mb} = 2048\text{ MB}$, $\text{min\_free\_memory\_pct} = 12.0\%$.
2. **Timeouts & Boundedness**:
   - Absolute wall-clock timeout $\le 120.0\text{s}$ per stage.
   - Child process teardown with SIGTERM and verified SIGKILL fallback.
3. **Sequential Execution**:
   - Phase A: Offline public fixture validation & recovery tests (TypeScript / Node).
   - Phase B: GGUF 0.6B production indexing & retrieval evaluation.
   - Phase C: MLX 4B production indexing & retrieval evaluation (supervised).
   - Phase D: MCP transport retrieval probe.
   - Phase E: Full test suite verification (`bun run test`, `pytest`).

---

## 7. Deliverables & Success Criteria

1. `docs/plans/phase5-isolated-e2e.md`: This comprehensive execution plan.
2. `scripts/qmd_mlx/phase5_e2e_qualification.py` / TypeScript test harness: Verifiable qualification script.
3. `docs/reviews/artifacts/phase5-isolated-e2e.json`: Complete execution telemetry, numerical counts, latency profiles, and IR metric comparisons.
4. `docs/reviews/phase5-isolated-e2e-acceptance.md`: Concise phase acceptance document noting preliminary smoke status and documenting unfulfilled gates.
5. `docs/plans/next-deployment-checklist.md`: Updated checklist maintaining blocked rollout status.
