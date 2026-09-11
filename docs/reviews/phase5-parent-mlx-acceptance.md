# Phase 5A parent acceptance — bounded MLX-only release smoke

Artifact: `artifacts/phase5-parent-mlx-trial-v4.json`. Earlier v1–v3 are failed attempts, retained unchanged.

## Observed
- Real production SDK indexing, vector search, typed lex+vec fusion, and HTTP MCP query/get on disposable public fixture SQLite databases.
- 16 docs; 10 authored technical evaluation queries (small easy development corpus, not an independent generalization benchmark).
- Vector and hybrid each Recall@5=1, MRR=1, nDCG@5=1; BM25 each 0.1. Ten query result records per mode verified programmatically. No GGUF comparison performed.
- Cooperative in-process interruption left 2 chunks; resume completed 16 vectors. Consecutive sequences checked. Mismatched synthetic checkpoint was rejected. This is NOT a killed-process recovery test or proof DB remains byte-identical on rejection.
- VACUUM INTO restored 16 documents and 16 vectors. Restored lexical Raft lookup succeeded. Artifact field `restored_retrieval_identical` overstates the test: full pre/post ranked-vector equality has NOT been tested.
- HTTP MCP reported 4 tools; query returned one Raft document; get yielded 621 characters.
- 15.791s supervised trial. Owned model child PID10914 reaped exit0, no SIGKILL, supervision errors empty. Raw supervisor report preserved in aggregate artifact.

## Parent fixes made after delegate
- Explicit isolated MLX backend/URL and fallback=false for recovery, restore and MCP stores rather than inheriting ambient backend.
- Read descriptor from inference listener.
- Corrected SDK typed query payload `text` → `query` and actual MCP tool `query` with `searches`, rerank=false.
- Fixed production validateSemanticQuery falsely classifying internal word hyphens as exclusions (`non-blocking`, `B-tree`); six targeted regressions pass, true lexical -term exclusions still rejected.
- Preserve supervisor cleanup/errors alongside aggregate report.

## Independently executed final gates
- git diff --check: exit0.
- bun run build: exit0.
- CI=true bun run test: 801 passed,72 skipped,26 files; /tmp/qmd-phase5-final-ts.log.
- HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. .venv/bin/pytest test/python/ -q:241passed,11skipped; /tmp/qmd-phase5-final-python.log.

## Not accepted for cutover
Fresh-process checkpoint recovery, full restored retrieval equivalence, persisted DB invariance on mismatch, supervised matching GGUF comparator, held-out quality at representative scale, sustained-memory gate, concurrent latency target and rollback remain open. No live service/index/config changed. Artifact `passed` means this MLX-only smoke, NOT all Phase5 or production acceptance. GGUF blocked is not a passed comparison. No performance superiority claim.
