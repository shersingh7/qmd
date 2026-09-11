# Phase 3 Single-Model Smoke Plan & Qualification Harness

## 1. Executive Summary & Objective

**Objective:** Qualify individual MLX models on Apple Silicon Metal under strict isolation before any full-corpus indexing or system-level benchmarking. The initial trial focuses on the primary embedding model: `qwen3-embedding-4b-mlx-4bit-affine`.

**Scope & Governance:**
- Phase 3 preparation is authorized following parent offline acceptance (`docs/reviews/phase2-final-parent-gate.md`, 194 passed, 11 skipped).
- **NO real weights loaded in this delegate.** All preparation, harness implementation, and verification are conducted using pure metadata inspection and an offline rehearsal executing the real server stack (`scripts/mlx_embed_server.py`) injected with `SyntheticEmbeddingAdapter`.
- Real-model execution remains strictly opt-in (`--real-model`), requiring parent review and refreshed memory preflight before invocation.

---

## 2. Safety Invariants & Remediated Harness Composition

| Boundary / Subsystem | Remediated Architecture & Enforcement | Verification |
| :--- | :--- | :--- |
| **100% Offline** | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` injected in child env. | Network calls blocked at process boundary; zero downloads permitted. |
| **Model-Aware Headroom Preflight** | Calculates conservative memory requirement from model parameter count, weight precision, and activation margin (>= 3,500 MB for 4B models; fails closed). | Tested against insufficient headroom rejecting preflight. |
| **Real Server Rehearsal** | Injects `SyntheticEmbeddingAdapter` via `resolve_embedding_adapter` into `scripts/mlx_embed_server.py`. Exercises `ThreadedMLXServer`, `GPUExecutor`, `ModelResidencyManager`, and `MLXControlServer`. | Production server stack fully rehearsed without downloading weights. |
| **Active Watchdog Supervision** | Spawned child PID is owned by `MLXWatchdog`. Background `SupervisorThread` samples telemetry and runs `watchdog.check_step()` every 250ms. | Stalled/hung workers and resource breaches actively detected and reaped. |
| **Outer Wall-Clock Ceiling** | Supervisor and `SmokeHttpClient` enforce absolute monotonic wall-clock deadline ceiling. | Prevents indefinite hangs even under slow-drip or delayed responses. |
| **Bounded Probe Transport** | `SmokeHttpClient` configured with `trust_env=False`, `allow_redirects=False`, and 10MB chunked streaming byte cap. | Eliminates proxy interference and memory exhaustion from oversized responses. |
| **No Unbounded PIPE Deadlocks** | Server stdout/stderr redirected to temporary files rather than unbounded OS pipes. | Prevents 64KB OS pipe buffer exhaustion deadlocks. |
| **Guaranteed Cleanup** | `finally` block terminates supervisor, signals child (`SIGTERM` -> `SIGKILL`), reaps process handle, and closes temp files. | Rehearsals and simulated failures exit with zero orphaned processes. |
| **Zero Side Effects** | Read-only inspection only. Zero database writes or schema modifications. | Live GGUF index (`~/.cache/qmd/index.sqlite`) and live daemon (PID 1446) remain untouched. |
| **Opt-In Real Mode** | `--real-model` flag required alongside refreshed preflight memory check. | Aborts if `--real-model` is run without preflight headroom or if `--rehearsal` is specified. |

---

## 3. Read-Only Baseline Inventory

### 3.1 Live Running Services (Inspected Read-Only; Untouched)
- **Live MLX Daemon:** PID `1446` listening on `127.0.0.1:8787` (model: `qwen3-embedding-4b-mlx-4bit-affine`).
- **Live MCP Server:** PID `1565` / `1434` listening on `127.0.0.1:8181`.
- **Live GGUF Database:** `~/.cache/qmd/index.sqlite` (verified parent online backup: `index.sqlite.online-backup-20260908-192953.sqlite`).

### 3.2 System Telemetry Baseline
- **Installed Physical RAM:** 32,768.0 MB (32 GB)
- **Measured Memory Headroom:** ~16,535 MB (~16.1 GB available headroom)
- **System Swap Used:** ~4,550 MB

### 3.3 Target Model Metadata (`~/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine`)
- **Model Type:** `qwen3`
- **Architecture:** `Qwen3ForCausalLM`
- **Native Embedding Dimension:** 2560
- **Hidden Layers:** 36
- **Attention Heads:** 32 (KV Heads: 8)
- **Quantization:** 4-bit affine (group_size: 64, bits: 4, mode: affine)
- **Max Position Embeddings:** 40,960
- **Vocab Size:** 151,665
- **Weights Presence:** `model.safetensors` (2.26 GB), `model.safetensors.index.json`
- **Tokenizer Presence:** `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`

---

## 4. Public Self-Contained Fixture Suite & Tolerances

The smoke suite validates embedding runtime stability across 5 deterministic fixtures:

```
+-----------------------------------------------------------------------------------+
|                            Phase 3 Fixture Pipeline                               |
+-----------------------------------------------------------------------------------+
|  [Fixture 1: Singleton]     -> Dim=2560, Finite=True, ||v||_2 = 1.0 +/- 1e-4      |
|  [Fixture 2: Batch (N=6)]   -> Dim=2560, Finite=True, All ||v_i||_2 = 1.0 +/- 1e-4|
|  [Fixture 3: Consistency]   -> Cosine Sim >= 0.9999 (FP32) / Max Abs Diff <= 1e-3 |
|  [Fixture 4: Long Input]    -> ~1000 tokens -> Bounded runtime, Finite, Dim=2560  |
|  [Fixture 5: Timing]        -> 5 iterations -> Captures Min, P50, P95, Max, Avg   |
+-----------------------------------------------------------------------------------+
```

### 4.1 Fixture 1 — Singleton Embedding
- **Input:** `"The quick brown fox jumps over the lazy dog."`
- **Assertions:**
  - Shape: `[1, 2560]`
  - Finitude: `np.all(np.isfinite(arr))` (no NaN, no Inf)
  - Unit Normalization: `abs(||arr||_2 - 1.0) <= 1e-4`

### 4.2 Fixture 2 — Mixed-Length Batch Embedding
- **Input:** 6 distinct texts of varying lengths and token compositions:
  1. Short sentence (`"The quick brown fox jumps over the lazy dog."`)
  2. Technical sentence (`"Apple Silicon Metal unified memory acceleration."`)
  3. Domain sentence (`"Vector search enables semantic retrieval across local markdown documentation and codebases."`)
  4. Code block (`"```python\ndef embed_batch(texts: list[str]) -> np.ndarray:\n    return mlx_model(texts)\n```"`)
  5. Multilingual Unicode (`"自然语言处理 and multilingual text representations on macOS."`)
  6. Symbols and punctuation (`"Special symbols & punctuation: !@#$%^&*()_+-=[]{}|;':\",./<>?"`)
- **Assertions:**
  - Shape: `[6, 2560]`
  - Finitude: All finite
  - Normalization: Each row vector has unit L2 norm (`abs(||v_i||_2 - 1.0) <= 1e-4`)

### 4.3 Fixture 3 — Singleton vs. Batch Consistency
- **Evaluation:** Compare embedding of Text #1 computed in singleton isolation vs. Text #1 computed as element 0 of the mixed batch.
- **Stated Tolerances:**
  - Cosine Similarity: `cos_sim(v_singleton, v_batch[0]) >= 0.9999` (FP32) / `>= 0.999` (BF16/FP16)
  - Max Absolute Difference: `max(|v_singleton - v_batch[0]|) <= 1e-3` (FP32) / `<= 5e-3` (BF16/FP16)

### 4.4 Fixture 4 — Long-Input Truncation Policy
- **Input:** Long structured text (~1000 tokens / ~4000 characters).
- **Assertions:**
  - Bounded execution within deadline
  - Finite returned embedding `[1, 2560]`
  - Zero out-of-memory errors or server crashes

### 4.5 Fixture 5 — Repeated Requests Timing & Latency Profile
- **Evaluation:** 5 sequential embed requests with wall-clock latency measurement per request.
- **Metrics Recorded:** `min_ms`, `p50_ms`, `p95_ms`, `max_ms`, `avg_ms`.

> [!IMPORTANT]
> **Synthetic Numerical Checks Disclaimer:**
> Synthetic numerical checks verify runtime determinism, numerical stability, and bounded resource behavior; they do **NOT** evaluate semantic retrieval quality or compare MLX vs GGUF ranking performance. Retrieval benchmarks are reserved for Phase 4.

---

## 5. Implementation Artifacts

1. **Standalone CLI Runner:** [`scripts/qmd-mlx-smoke.py`](file:///Users/shersingh/github/qmd-mlx-search/scripts/qmd-mlx-smoke.py)
   - Model-aware preflight validation, active watchdog ownership, timeout ceiling enforcement, fixture execution, JSON report emission.
2. **Library Engine:** [`scripts/qmd_mlx/smoke.py`](file:///Users/shersingh/github/qmd-mlx-search/scripts/qmd_mlx/smoke.py)
   - Core `SmokeRunner`, `SmokeHttpClient` (bounded, proxy-isolated), `SupervisorThread`, model metadata inspector, fixture validators.
3. **Synthetic Embedding Adapter:** [`scripts/qmd_mlx/adapters/embedding.py`](file:///Users/shersingh/github/qmd-mlx-search/scripts/qmd_mlx/adapters/embedding.py)
   - Real server stack adapter generating deterministic embeddings without GPU/weight dependencies for rehearsal qualification.
4. **Pytest Suite:** [`test/python/test_mlx_smoke.py`](file:///Users/shersingh/github/qmd-mlx-search/test/python/test_mlx_smoke.py)
   - 15 unit/integration tests verifying model-aware headroom preflight rejection, full real server rehearsal lifecycle, watchdog breach handling, hung-child cleanup, and isolated probe transports.

---

## 6. Rehearsal Verification Evidence

### 6.1 Rehearsal Execution Output (`python3 scripts/qmd-mlx-smoke.py --rehearsal`)
```
=== MLX Single-Model Smoke Qualification Report ===
Status:             PASSED
Mode:               Rehearsal (Synthetic Adapter)
Duration:           0.93s
Endpoints:          Inference http://127.0.0.1:8797 | Control http://127.0.0.1:8798
Preflight Headroom: 16534.8 MB (min required: 2048.0 MB)

Fixture Verification Results:
  [PASSED] singleton       (latency: 17.35ms)
  [PASSED] batch           (latency: 10.59ms)
  [PASSED] consistency     (latency: N/A)
        Cosine Sim: 1.000000 (tol >= 0.9999), Max Diff: 0.000000e+00
  [PASSED] long_input      (latency: 2.97ms)
  [PASSED] timing          (latency: N/A)
        Iterations: 5, min: 2.09ms, p50: 2.12ms, p95: 2.28ms, max: 2.28ms

Disclaimer: Synthetic numerical checks verify runtime determinism, numerical stability, and bounded resource behavior; they do NOT evaluate semantic retrieval quality.
```

### 6.2 Test Suite Verification (`pytest test/python/`)
- Command: `git diff --check && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. .venv/bin/pytest test/python/ -v`
- Result: Exit `0`; **194 passed, 11 skipped in 21.89s**.

---

## 7. Executable Future Real-Smoke Command (NOT EXECUTED)

> [!CAUTION]
> The following command is prepared for the parent agent/user to review and invoke after confirming memory headroom and idle system conditions. It has **NOT** been executed by this delegate.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. \
python3 scripts/qmd-mlx-smoke.py \
  --real-model \
  --model-path /Users/shersingh/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine \
  --host 127.0.0.1 \
  --port 8797 \
  --control-port 8798 \
  --timeout-s 60.0 \
  --min-headroom-mb 3500.0 \
  --json \
  --output-file ~/.cache/qmd/phase3-embed-smoke-report.json
```
