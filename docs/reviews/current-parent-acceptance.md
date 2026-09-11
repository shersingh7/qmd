# Current parent acceptance — not approved for production cutover

## Verified code/component gates
- Independent final diff check exit 0.
- Independent configured Vitest runner exit 0: log /tmp/qmd-final-ts-verified.log.
- Independent offline Python: 233 passed, 11 skipped in 32.91 seconds, exit 0.
- TypeScript build passed before final test-only update.
- Supervised reranker smoke: phase4-parent-rerank-supervised.json, passed ranking discrimination, consistency and input validation; exact 4bit-affine descriptor; child cleanup verified.
- Supervised generation with shared production TS prompt/parser: phase4-parent-generation-final.json, passed typed output contract on four public prompts; full outputs and child cleanup retained. This does NOT establish retrieval improvement. Expansions remain generic in some cases.
- CI-only expansion test failures corrected to exercise disposable fixture HTTP path instead of early CI model prohibition. Production CI protection unchanged.
- Prompt improved to request concrete terms and hypothetical answer statements rather than questions. Tests updated to verify that contract.

## Outstanding release gates (not bugs claimed fixed)
1. Real concurrent 4B pilot fails <=200ms p95 (latest parent 642.06ms). Do not silently relax target or run soak as if accepted.
2. Sustained resource acceptance incomplete; earlier soak failed under system swap growth. Attribution to MLX leak unproven.
3. No held-out retrieval equivalence established against live GGUF0.6B. Passing typed generation and tiny ranked pairs is not quality equivalence.
4. No isolated full production indexing/recovery reconciliation or end-to-end restored QMD rollback trial accepted.
5. Integrated all-stage residency, sustained performance and 1000-batch memory gate incomplete.
6. Changes remain uncommitted; current code not deployed. Do not reuse old numerical embedding space silently. Query and document embedding spaces must match.

## Deployment decision
Retain current live GGUF system. Candidate is suitable for further isolated evaluation, not a verified faster/reliable replacement. Do not describe the work as all issues fixed or all phases complete. Scope expansion for downloads, live indexes/config/services, commits or push is not inferred from repo repair authorization.
