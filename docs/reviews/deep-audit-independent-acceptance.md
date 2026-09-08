# Independent acceptance record

## Verdict
Repository remediation has passed the current offline correctness gates. Preserve the TypeScript CLI/SQLite plus Python single-owner runtime architecture; the lifecycle, admission, model metadata and identity boundaries needed substantial targeted repairs rather than a wholesale rewrite.

This is NOT a proof of a bug-free repository, production readiness, MLX/GGUF quality parity, performance superiority, or full-corpus stability. No production cutover is approved by this record.

## Parent-executed gates
- `git diff --check`: exit 0 on final working tree.
- `bun run build`: exit 0 on the final TypeScript changes; subsequent fixes touched Python/docs only.
- `CI=true bun run test`: exit 0, 23 files passed, 774 tests passed, 72 skipped. Raw parent log: `/tmp/qmd-deep-audit-parent-vitest.log`.
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. .venv/bin/pytest test/python/ -q`: exit 0 on final Python tree, 99 passed, 10 real-model tests skipped. Raw parent log: `/tmp/qmd-deep-audit-parent-pytest-final.log`.

## Principal inspected repairs
- Shared executor shutdown before unloading; physical thread-liveness checks and pending cleanup for noncooperative work.
- Managed HTTP serving loop; stop during startup, failed-preload diagnostic serving, truthful launcher stop reporting.
- Admission request/byte reservations before embedding/tokenization, strict caps, shutdown wake and timeout checks.
- Explicit eviction state and lifecycle synchronization.
- Correct post-load combined residency accounting and consistent model-size estimates.
- Adapter lock deadlock/double-load repairs; separate 4B parameter identity from 4-bit quantization; measured metadata and batching retune preserving OOM backoff.
- Quantized reranker head handling and duplicate tokenization reduction.
- Fail-closed MLX embeddings and recovery, checkpoint default-model identity, tokenizer-sensitive/case-preserving embedding fingerprints.
- Finite JSON vectors, generation token accounting, benchmark batch-size behavior and accurate scope labels.
- Offline default Python gate with explicit opt-in real-model tests; effective timeout regression.

## Remaining acceptance limits
- Actual production-model inference, sustained mixed workloads, p95/p99 latency, peak RSS/swap and retrieval-quality comparisons have NOT been independently exercised in this remediation phase.
- Input/admission/weight budgets are safeguards, not a hard bound on native activation allocations or a guarantee against macOS memory pressure. Cooperative deadlines cannot preempt an arbitrary native model load/kernel already executing.
- HTTP health probes still share the bounded connection pool and may return overload during saturation; dedicated control-path capacity is not implemented.
- Benchmark harness is a synthetic embedding microbenchmark, not an end-to-end GGUF comparison or full-corpus retrieval acceptance campaign.
- Fingerprint semantics changed: existing indexes may require explicit isolated rebuild; no live vectors were migrated.
- Independent broad TypeScript subagent timed out without a review conclusion; parent source/diff review and test gates are the evidence, not that unfinished review.

## Scope
All repository edits remain uncommitted. No deployment or push was performed. Production index/config/daemon changes and model downloads were not authorized or performed by the parent. Early delegate Python runs used existing small-model fixtures; those fixtures were subsequently replaced/gated, so do not describe the entire historical task as mock-only. Final independent default gate is offline with real-model tests skipped.
