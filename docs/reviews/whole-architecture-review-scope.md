# Whole-architecture review — current user scope

User requests architecture-first review and fixes, strict MLX target for all inference stages, leak-resistant and efficient production-scale design. Defer test execution, benchmarks, inference and load testing until after architecture recommendations. Static inspection and repository edits authorized; no deployment, private corpus access, installed config/daemon changes, commits/pushes or downloads.

Do not certify leak-proof or production performance by inspection. Final report must distinguish source defects, fixed-but-unexecuted changes, design risks and later load acceptance. Regression code can be authored without running suites; syntax/build checks only if separately justified, not performance work.

Independent review split: Python inference/resource/security; TypeScript routing/index/storage. Parent reviews packaging/supervision/config/docs and reconciles concrete findings before AGY implementation. Preserve current uncommitted work.

## Parent source findings for reconciliation

1. src/indexing/checkpoint.ts computeDescriptorSignature:33–40 hashes backend, model, output dimensions and pooling only, concatenated without field framing. Omits revision, quantization, normalization, token limit/recipe identity. This is weaker than a complete vector-space identity contract. Reuse canonical descriptor encoding/hash rather than inventing a second partial identity.
2. getActiveCheckpoint:82–88 selects only in_progress/paused. Failed/cancelled runs are not included although they retain partial vectors; completed prior recipe also absent. Trace persisted recipe protection independent of job status so any next run cannot reuse stale sequence numbers after recipe/model changes. Do not auto-delete vectors.
3. scripts/qmd-mlx-daemon.sh:19–29 still defines embed-only defaults, empty rerank/generate, stale claim no idle-unload story, and short-text model recommendation. Strict all-MLX must be an explicit coherent profile with required three model descriptors and fail-closed routes, not independent opt-ins silently falling back. Preserve existing production installation; do not run wrapper.
4. Daemon wrapper start writes plist, bootouts current label then probes registration but only warns before success. stop suppresses all bootout errors and unconditionally prints stopped. Environment overrides are consumed by parent shell but launchd environment persistence must be traced into template; do not assume MLX_PORT/model vars reach launched process. Separate install/config generation from lifecycle, verify exact instance identity and readiness in tooling design.
5. scripts/mlx_embed_server.py handles only KeyboardInterrupt; inspect SIGTERM launchd shutdown path and cleanup ownership. Explicit dtype/quantization CLI help claims compute changes; verify actual loader honors or rejects rather than merely relabels descriptor.
6. scripts/mlx_server_requirements.txt unbounded minimums do not reproduce the reviewed MLX stack. Separate runtime/dev deps and add checked-in constraints/version compatibility policy based on installed metadata (no downloads), not newly invented versions.
7. src/embedding/config.ts DEFAULT_MLX_TIMEOUT_MS=60000 while client has 300000. Trace actual usage before judging effective timeout; remove dead resolution paths or centralize configuration. Explicit mlxBinary currently can be overridden by environment despite documented caller-first precedence. Other invalid env values silently choose defaults; strict profile should surface configuration errors.

Pending full review results before final implementation work order. No performance or leak-freedom claims.
