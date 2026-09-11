# Phase 2 parent review — acceptance withheld pending corrections

## Verified source findings

1. HIGH — watchdog.py TargetProcessValidator.validate_target uses broad `(python|mlx|qmd|server)` regex; matching command alone does not establish an owned test process or immutable process identity. SIGKILL escalation does not revalidate identity. The previous claim "only test PID" was overstated.
2. HIGH — MLXWatchdog._default_health_probe accepts HTTP 200 without worker progress or instance identity. A stuck worker with responsive control HTTP remains undetected.
3. HIGH — ThreadedMLXServer's inference semaphore leaves shared 128 connection slots vulnerable to idle keepalive/partial-header saturation. It is not dedicated control capacity. Inference rejection before consuming body keeps the connection alive, risking protocol desynchronization.
4. HIGH — watchdog memory telemetry fails open: missing vm_stat falls back to installed RAM fractions, unavailable swap becomes zero, minimum RSS budget can exceed low headroom, and purgeable pages are added to other page queues without disjointness proof.
5. HIGH — previous rollback instructions copied the main SQLite file with active WAL and did not establish a consistent current backup. File-size equality was not content or restore validation.

## Backup correction executed by parent

Using the user's existing backup authorization, created an SQLite online backup from a read-only source connection. No live database overwrite or service restart. The first verification open failed; opening the new backup normally and changing ONLY its journal mode to DELETE made it standalone and verifiable.

- Path: `/Users/shersingh/.cache/qmd/index.sqlite.online-backup-20260908-192953.sqlite`
- Size: 997281792 bytes
- `PRAGMA quick_check`: `ok`
- Schema table count: 20
- SHA-256: `2f09753af1b9d173465fba2d31922c980e2fd14ec9517e4ca5374fb2194bb7ec`
- Restored into a disposable temporary directory: identical SHA-256 and `quick_check=ok`; temporary copy removed after verification.
- This proves backup-copy integrity/SQLite consistency, not model compatibility or end-to-end QMD query restoration. That gate remains separate.
- Previous `index.sqlite.gguf-backup-20260908` raw copy remains untouched and must not be relied on as the validated backup.

## Status

Corrective implementation delegated with plan-before-code, offline regressions, real disposable subprocess watchdog tests and raw socket saturation tests. No Phase 3 real-model run started. No new commit/push or deployment authorized by this review. The existing GGUF baseline configuration is not proven by absence of QMD environment variables; resolve the live daemon's executable/config/index identities before comparative measurements.
