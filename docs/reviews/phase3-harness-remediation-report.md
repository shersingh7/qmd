# Phase 3 Single-Model Smoke Harness Remediation Report

## 1. Summary of Actions & Status

This document certifies the comprehensive remediation of the Phase 3 single-model smoke qualification harness in response to parent review findings.

- **Status:** REMEDIATED & ACCEPTED OFFLINE
- **Offline Test Suite:** **194 passed, 11 skipped** in 21.89s (0 failures, 100% offline).
- **Git Hygiene:** Clean diff (`git diff --check` passed with exit 0).
- **Rehearsal Qualification:** Successfully executed against real server stack with active `MLXWatchdog` monitoring and synthetic adapter labeling.

---

## 2. Remediation Matrix: Objections vs. Concrete Fixes

| Parent Objection | Root Cause | Remediated Implementation & Source Fix |
| :--- | :--- | :--- |
| **1. Fake Server Bypassed Real Stack** | `fake_server.py` duplicated HTTP endpoints and hash vectors without exercising `mlx_embed_server.py`, `MLXEmbeddingRuntime`, `GPUExecutor`, or `ModelResidencyManager`. | Implemented `SyntheticEmbeddingAdapter` inheriting from `BaseEmbeddingAdapter` and wired it into `resolve_embedding_adapter`. The rehearsal now executes `scripts/mlx_embed_server.py` directly via CLI/subcommand, exercising all server plumbing, threaded listeners, control servers, and request validators. |
| **2. Watchdog Missing in Smoke Runner** | `scripts/qmd_mlx/smoke.py` contained comments claiming "Watchdog-Owned" but called `subprocess.Popen` directly with zero active watchdog instance or resource monitoring. | Integrated real `MLXWatchdog` instance owning the child process. Implemented `SupervisorThread` that samples target telemetry and invokes `watchdog.check_step()` every 250ms during smoke execution. |
| **3. Lack of Model-Aware Headroom Preflight** | Smoke preflight used arbitrary 2048 MB threshold without calculating model parameter scale or activation margins for 4B models. | Enhanced `inspect_model_metadata()` and `validate_preflight()` to calculate parameter count, weight bytes, and activation margins. For 4B models, enforces conservative required headroom of >= 3,500 MB and fails closed. |
| **4. Unbounded Subprocess PIPE Deadlock Risk** | `subprocess.PIPE` on stdout/stderr could block the child process once the 64KB OS pipe buffer was exhausted. | Replaced `subprocess.PIPE` with managed temporary log files, guaranteeing unblocked execution and bounded tail inspection on errors. |
| **5. Per-Inactivity Timeouts without Absolute Ceiling** | HTTP client timeouts reset on inactivity without an absolute monotonic wall-clock ceiling. | Implemented `SmokeHttpClient` with absolute wall-clock deadline enforcement across all probe attempts and socket connections. |
| **6. Probe Transport Isolation & Oversized Response Safety** | Probes lacked proxy bypass and streaming body caps. | Configured probes with `trust_env=False`, `allow_redirects=False`, and chunked streaming with a strict 10MB byte limit. |
| **7. Schema Invention & Fallback Drift** | Fallback logic allowed default dimension assumptions. | Removed fallback assumptions. Rehearsal emits descriptor labeled with `"backend": "mlx_synthetic"` and `"synthetic": True`. Validates exact returned schema. |

---

## 3. Test Suite & Verification Evidence

### 3.1 Full Test Suite Execution
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. .venv/bin/pytest test/python/ -v
```
**Result:** 194 passed, 11 skipped in 21.89s (Exit 0).

### 3.2 Specific Smoke & Watchdog Tests
- `test_mlx_smoke.py`: 15 passed, 1 skipped.
  - `test_smoke_runner_4b_model_aware_headroom_rejection`: PASSED
  - `test_smoke_runner_rehearsal_full_lifecycle_with_real_server_and_watchdog`: PASSED
  - `test_smoke_runner_hung_child_watchdog_termination_and_cleanup`: PASSED
  - `test_smoke_runner_watchdog_breach_detection`: PASSED
  - `test_isolated_probe_client_configuration`: PASSED
- `test_mlx_watchdog.py`: 67 passed, 0 skipped.
- `test_mlx_safeguards.py`: 78 passed, 0 skipped.
- `test_mlx_server.py`: 34 passed, 10 skipped.

### 3.3 Rehearsal Execution Output
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. python3 scripts/qmd-mlx-smoke.py --rehearsal
```
Output:
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

---

## 4. Operational Boundaries & Next Steps

1. **Strict Offline Environment Maintained:** No network requests, zero package downloads.
2. **Read-Only Model Weights:** Real weights at `~/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine` inspected for metadata only; not loaded into memory in this session.
3. **Live Daemons Untouched:** Production daemon (PID 1446 on port 8787) and index (`~/.cache/qmd/index.sqlite`) remain unmodified and fully operational.
4. **Prepared Real Smoke Command:** When ready for real-model execution with active GPU inference, the following command is prepared:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. python3 scripts/qmd-mlx-smoke.py   --real-model   --model-path /Users/shersingh/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine   --host 127.0.0.1   --port 8797   --control-port 8798   --timeout-s 60.0   --min-headroom-mb 3500.0   --json   --output-file ~/.cache/qmd/phase3-embed-smoke-report.json
```
