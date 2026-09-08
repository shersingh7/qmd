# Additional independent review inputs — not accepted completion

Parent directly verified these source issues after independent Python review:

- `BaseEmbeddingAdapter.tokenize_texts` acquires non-reentrant `_init_lock`, then calls subclass `load()` which acquires same lock. Unloaded tokenization self-deadlocks. Qwen.load also releases its lock before real loading/assignment, allowing direct concurrent loads. Test deterministic unloaded tokenize and concurrent loaders with fake loader; enforce coherent lifecycle ownership rather than papering over just one lock.
- Base `model_params_b=0.6` is never overridden by Qwen adapter, yet runtime tunes batching before load. The 4B embedding default therefore uses 0.6B planning. Derive trustworthy parameter/memory metadata, distinguish parameter count from quant bits, and retune after actual metadata loads. Test 0.6B-4bit versus 4B-4bit.
- Embedding adapter `model_memory_mb` initialized zero but not updated on load. Manager accounting uses 1000 fallback, admission uses different heuristic; `'4b' in name` matches `'4bit'`. Use consistent conservative pre-load estimate and measured post-load storage (including quantization metadata), not inconsistent fallback sums.
- `test/python/test_mlx_faults.py::test_ephemeral_server_fault_responses` currently loads real MiniLM with preload=True. Step 5 sends wrong `X-Timeout` header, silently accepts HTTP 200. Replace with fake runtime and deterministic deadline delay/cancel and exact status assertion using actual header. No model downloads/loads for default correctness gate.

Additional child-reported candidates MUST verify before fixing or declaring defects:
- Other default Python test fixtures instantiate actual models and hardcode local user paths; separate opt-in real-model integration tests from offline default gate and enforce HF offline/no download in default tests.
- HTTP handler capacity shared by health and inference can make overload probes return 429; consider bounded control-path capacity, not unlimited threads.
- SIGTERM launcher lacks graceful shutdown; teardown can unload before admitted request microbatches stop then reload during STOPPING; drain/cancel + reject new work + owner-safe cleanup + join lifecycle.
- Rerank repeated query/doc tokenization and default singleton pair batches; preserve scoring semantics while reducing duplicate tokenization; exact tokenizer call-count tests.
- /embed JSON should reject nonfinite outputs as binary path does.
- generate reported token usage must count tokens, not characters; stats must update accurately.
- bench_mlx descriptor fetch outside try/finally can leak started server on descriptor error; fix and test cleanup.
- Verify residency OOM vs activation OOM catch boundary before asserting batch-budget poisoning; source child may be wrong because ensure_loaded may be outside retry closure. Keep only demonstrable defects.

The broad TypeScript child timed out without summary. Do not describe it as completed independent approval. Parent will review TypeScript paths directly.

No green unit-suite claim establishes real-model parity, throughput superiority, stable p95 latency or macOS safety. Record unmet acceptance honestly. Do not run existing full Python suite until real-model fixture behavior is controlled.
