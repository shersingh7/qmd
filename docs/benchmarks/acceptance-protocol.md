# Apple Silicon MLX Benchmark & Acceptance Protocol

This document defines the official, non-negotiable benchmark methodology, measurement standards, and acceptance criteria for all MLX-accelerated operations (embedding, reranking, generation) in `qmd-mlx-search`.

---

## 1. Principles & Antipattern Prohibitions

1. **No Untested / Unmeasured Speedup Claims**: Blanket claims (such as "2-5x faster" or "10x speedup") are strictly forbidden unless accompanied by a complete benchmark manifest and reproducible run log on specified Apple Silicon hardware.
2. **End-to-End Accounting**: Benchmarks must clearly distinguish between:
   - Raw Metal matrix-multiplication forward pass latency.
   - Host CPU tokenization latency.
   - Wire protocol serialization / deserialization overhead.
   - SQLite `COMMIT` / index disk write overhead.
   Inference-only latency must never be presented as overall indexing throughput.
3. **Apples-to-Apples Comparisons**: Model performance comparisons must hold architecture, parameter size, and quantization constant (e.g., comparing MLX Qwen2.5-Coder-1.5B 4-bit against GGUF Qwen2.5-Coder-1.5B Q4_K_M). Comparing a 0.5B model against a 7B model and presenting the speedup as an MLX backend enhancement is invalid.
4. **Metal Queue Synchronization**: Metal compute queues on Apple Silicon execute asynchronously. All benchmark timers must explicitly synchronize (`mx.eval()` and `mx.synchronize()`) before start and after completion of measured intervals.

---

## 2. Benchmark Manifest Specification

Every benchmark run must generate an immutable JSON manifest capturing:

```json
{
  "timestamp": "2026-09-07T23:00:00Z",
  "git_commit": "abcdef123456...",
  "platform": "macOS-15.3-arm64-arm-64bit",
  "chip": "Apple M2 Pro",
  "python_version": "3.12.13",
  "mlx_version": "0.22.0",
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "quantization": "bf16",
  "batch_size": 16,
  "warmup_runs": 2,
  "measured_runs": 5
}
```

---

## 3. Workload Strata & Corpus Fixtures

Benchmarks must report across four standard representative strata:

| Stratum | Description | Typical Tokens / Item | Purpose |
| :--- | :--- | :--- | :--- |
| **Short Queries** | Single-line search queries | 5 – 25 | Interactive query latency (p50, p95, p99) |
| **Medium Passages** | Standard document paragraphs | 50 – 200 | Average semantic search corpus unit |
| **Long Documents** | Multi-paragraph articles / RFCs | 400 – 1500 | Attention scaling, memory pressure, micro-batching |
| **Code Snippets** | TypeScript / Python code blocks | 50 – 500 | Variable length, syntax-heavy token distribution |

Public fixtures must be used for automated CI/regression tests. Private or operator corpora may only be sampled with explicit authorization, with all raw text redacted from reports.

---

## 4. Metrics & Instrumentation Requirements

For each stratum, the harness must report:

1. **Tokenization Latency (`tokenize_ms_p50`)**: Host CPU time spent tokenizing raw text into token IDs.
2. **GPU Forward Latency (`forward_ms_p50`, `forward_ms_p95`, `forward_ms_p99`)**: Metal execution time for tensor preparation, backbone forward pass, and pooling.
3. **Total Latency (`total_ms_p50`)**: Wall-clock duration from request receipt to embedding delivery.
4. **Effective Token Throughput (`tokens_per_sec`)**: True input tokens processed per second (excluding padding tokens).
5. **Padding Overhead (`padding_overhead_pct`)**: Percentage of compute wasted on pad tokens (`(padded_tokens - raw_tokens) / raw_tokens * 100`).
6. **Memory Footprint**:
   - `metal_active_mb`: Currently allocated Metal unified memory buffer.
   - `metal_peak_mb`: High-water mark of Metal unified memory buffer during run.
   - `process_rss_mb`: Operating system Resident Set Size.

---

## 5. Decision Thresholds & Release Gates

To qualify for promotion from experimental/shadow status to production default:

1. **Practical Speedup**: Must demonstrate $\ge 20\%$ lower p50/p95 latency or $\ge 20\%$ higher sustained token throughput compared to the baseline on the same hardware and equivalent model configuration.
2. **Zero Semantic Degradation**: Embedding vectors must maintain $\ge 0.999$ cosine similarity against reference MLX forward implementations.
3. **Retrieval Equivalence**: On standard evaluation benchmark queries, hybrid search must maintain equal or higher Recall@10, nDCG@10, and MRR.
4. **Memory Stability**: No Metal memory growth across 1,000 consecutive batch iterations (zero memory leaks).
5. **Durable Recovery**: Interrupted indexing runs must resume cleanly with zero duplicate vector insertions and exact count reconciliation.
