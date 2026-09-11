# Parent gate: bounded qualification preparation

## Verified
- HEAD: 0960f3c; subsequent corrections remain working-tree changes.
- Parent command: `git diff --check && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. .venv/bin/pytest test/python/ -q`
- Exit 0; 179 passed, 10 skipped in 16.65s.
- Parent inspected CLI source: pure configuration validation now occurs before bind, sampling and Popen. Actual PID is finalized after spawn. Qualification requires explicit inference/control ports and IPv4 loopback. Child port conflicts are checked.
- Read-only watchdog print-defaults succeeded: measured headroom 16491.671875 MB, max RSS 8192 MB, swap-growth allowance 2048 MB. These are transient measurements, not a future launch authorization or native GPU hard limit.

## Acceptance scope
Accept the repaired offline qualification harness for Phase 3 preparation. This is NOT deployment acceptance, proof of full safety, or model-performance acceptance. Independent subagent reviews timed out and contribute no approval. Source review and actual parent test output are the evidence.

## Next gate
Prepare a bounded, local-path-only real-model smoke harness with a no-model dry-run rehearsal. Before actual weights load: inventory existing inference services read-only; ensure no competing heavy job; refresh headroom; identify exact weights and tokenizer; choose distinct free loopback ports; invoke watchdog-owned child; enforce total run deadline and cleanup. No downloads, existing index writes, live daemon changes or commits. Stop on insufficient headroom rather than changing live service state.

Run embedding first: singleton and mixed-length batch finite outputs, dimension/norm and consistency, long-input policy, repeated request measurements. Do not call this retrieval-quality verification or MLX-vs-GGUF comparison. Rerank/expand and full-index work remain later gates.
