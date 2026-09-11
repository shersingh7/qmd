# Phase 5 Parent Review & Offline Corrections Report

**Date**: 2026-09-10  
**Status**: Offline Corrections Completed & Verified  
**Scope**: Narrow foreground offline corrections in `/Users/shersingh/github/qmd-mlx-search` Phase 5 partial files.

---

## 1. Executive Summary

This review documents the narrow offline corrections applied to the Phase 5 qualification pipeline in `scripts/qmd_mlx/phase5_e2e_qualification.py`, `scripts/phase5_e2e_runner.ts`, `test/python/test_phase5_qualification.py`, and `test/phase5-recovery.test.ts`.

All corrections were implemented strictly offline in foreground without invoking real models, full benchmarking suites, daemon modifications, background tasks, or network downloads.

> [!NOTE]
> **Live Daemon Context**: The existing live daemon process (PID 1722) running `mlx_embed_server.py` on port 8787 was left completely untouched. The qualification pipeline enforces strict endpoint isolation: independent inference port `8797` and control port `8798`, strictly refusing port `8787`.

---

## 2. Issues Identified & Applied Corrections

| Issue Identified by Parent | Root Cause | Code Correction | Offline Test Evidence |
| :--- | :--- | :--- | :--- |
| **1. Python Test Collection Failure** | `phase5_e2e_qualification.py` imported nonexistent `WatchdogThresholds` from `scripts.qmd_mlx.watchdog`. | Removed nonexistent import; refactored to compose `SmokeStageSupervisor` and `SystemMemorySampler` from existing `scripts/qmd_mlx/supervisor.py` and `scripts/qmd_mlx/watchdog.py`. | `.venv/bin/pytest test/python/test_phase5_qualification.py` collects and passes all 8 unit tests in 0.22s. |
| **2. Fake Telemetry & Unmanaged Child** | `Phase5Supervisor` used fake defaults (32768MB / 50%), unmanaged `Popen` with undrained pipes, and lacked `MLXWatchdog` integration. | Replaced `Phase5Supervisor` with composition of existing `SmokeStageSupervisor`, providing active `MLXWatchdog` thread monitoring, safe log files, and bounded deadlines. | `test_phase5_supervisor_preflight_sufficient_headroom`, `test_phase5_supervisor_preflight_insufficient_headroom_fails_closed`, `test_phase5_supervisor_preflight_telemetry_error_fails_closed`. |
| **3. Async Promise Executor & Swallowed Errors** | `evaluateRetrievalStage` used `new Promise(async (resolve) => ...)` anti-pattern and swallowed query errors with `console.error`. | Refactored `evaluateRetrievalStage` to native `async` function, capturing structured per-query error details and propagating exceptions rather than claiming nominal passed. | `evaluateRetrievalStage` in `scripts/phase5_e2e_runner.ts` and `test/phase5-recovery.test.ts`. |
| **4. Expansion Model Invocation during Hybrid Search** | Calling `store.search({ query: ... })` triggered `hybridQuery` which loads the query expansion model. | Updated `evaluateRetrievalStage` hybrid mode to pass typed queries: `queries: [{ type: "lex", text: q.query }, { type: "vec", text: q.query }]`, executing purely embedding + BM25 search without expansion. | `evaluateRetrievalStage` in `scripts/phase5_e2e_runner.ts`. |
| **5. False Gating on Skipped Stages** | `checks.gguf06bRetrievalComplete` and `checks.mlx4bRetrievalComplete` had ambiguous semantics when stages were skipped, leading to false gates. | Marked GGUF stage explicitly `status: "blocked"` pending supervised integration; gated `passed` check strictly on whether non-skipped stages completed successfully. | `Phase5Supervisor.run` and `runPhase5Qualification` report structures. |
| **6. Overstated Recovery Claims** | Report did not distinguish between in-process checkpoint recovery and OS process crash recovery. | Explicitly annotated recovery report with `mode: "in_process_checkpoint"` and `freshProcessRecovery: "pending"`. | `recovery` object in `Phase5ExecutionReport`. |
| **7. Potential Model Autodownloads** | Fallbacks to remote HuggingFace hub strings (`mlx-community/...`) could trigger network downloads. | Enforced explicit cached local path assertions (`existsSync` / `os.path.exists`) without fallback strings, failing closed with clear errors if weights are absent. | `test_phase5_supervisor_missing_model_fails_closed_without_autodownload`. |
| **8. Unmanaged Direct TS Execution** | `scripts/phase5_e2e_runner.ts` CLI entrypoint could be executed unmanaged without supervisor approval. | Added launch token assertion (`--launch-token` / `QMD_PHASE5_LAUNCH_TOKEN`); rejects direct CLI calls while safe import tests remain unaffected. | `test_phase5_ts_runner_launch_token_assertion`. |

---

## 3. Verification Test Evidence

### Python Offline Test Suite
Command: `PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/pytest test/python/test_phase5_qualification.py`
```text
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/shersingh/github/qmd-mlx-search
plugins: anyio-4.15.1
collected 8 items

test/python/test_phase5_qualification.py ........                        [100%]

============================== 8 passed in 0.22s ===============================
```

### TypeScript Offline Test Suite
Command: `CI=true bun run test -- test/phase5-recovery.test.ts`
```text
 Test Files  25 passed (25)
      Tests  795 passed | 72 skipped (867)
   Start at  17:22:39
   Duration  64.33s (transform 825ms, setup 0ms, collect 8.42s, tests 68.42s, environment 2ms, prepare 1.25s)
```

---

## 4. Outstanding Release Gates (Unfulfilled)

1. **Supervised GGUF Integration**: GGUF baseline qualification remains explicitly BLOCKED pending supervised lifecycle integration.
2. **Fresh-Process Crash Recovery**: Full OS process crash and recovery across process boundaries is marked `pending`; only in-process checkpoint resumption is qualified.
3. **High-Concurrency Interactive Latency**: MLX 4B interactive latency under concurrent load remains to be proven under target SLAs (<=200ms).
4. **1000-Batch Sustained Memory Stress Test**: Long-running sustained stress test pending before live production promotion.
5. **Full-Corpus Evaluation**: Full MS MARCO / BEIR evaluation pending beyond curated preliminary public fixtures.
