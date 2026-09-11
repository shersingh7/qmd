# Phase 2 Safety Corrections Implementation Plan

**Target Repository:** `/Users/shersingh/github/qmd-mlx-search`  
**Date:** 2026-09-08  
**Status:** IN PROGRESS  

---

## 1. Context & Review Findings

Parent review identified critical safety regressions in the initial Phase 2 candidate:
1. **Control Path Starvation:** Inference socket exhaustion (e.g. partial HTTP headers or keepalive connections) blocked `/health` on the shared HTTP listener, causing false 429 rejections (proven by `/tmp/qmd_phase2_control_repro.py`).
2. **Process Targeting Safety:** `allow_external_pid` was declared but never read; regex matching was overly broad; non-children were targeted without immutable identity validation before `SIGKILL`.
3. **Fabricated Telemetry Fallback:** `SystemMemorySampler` and `MLXWatchdog.__init__` caught `SystemMetricsError` and fabricated baseline swap (0.0) and headroom budgets (16GB RAM / 4GB headroom / 2600MB RSS ceiling). `ps` failures were conflated with `TARGET_EXITED`.
4. **Health Probe Inactivity vs Wall-Clock Deadlines:** Health probing relied on socket inactivity timeouts and unbounded `read()`, vulnerable to slow-drip headers/bodies and proxy/redirect hijacking.
5. **Instance Token & Launch Isolation:** Server tokens were optional in validation; qualification tests lacked explicit port separation and clean token generation.
6. **Documentation & Backup Integrity:** Inappropriate author attribution; premature "COMPLETE" claims; unvalidated file-copy backup instructions instead of parent's verified SQLite online backup (`/Users/shersingh/.cache/qmd/index.sqlite.online-backup-20260908-192953.sqlite`).

---

## 2. Concrete Architecture & Implementation Tasks

### Task 1: Dedicated Bounded Loopback Control Listener (`server.py`, `mlx_embed_server.py`)
- **`MLXControlServer` / `MLXControlRequestHandler`:**
  - Independent `HTTPServer` bound strictly to `127.0.0.1` on `control_port`.
  - Independent thread limiter (`control_max_threads=16`), completely decoupled from inference concurrency limits.
  - Serves GET `/health`, `/ready`, `/descriptor`, `/memory`, `/stats`.
  - Rejects POST/inference requests (`/embed`, `/embed-bin`, `/tokenize`, `/rerank`, `/generate`) with 405 Method Not Allowed / 404 Not Found.
  - Enforces strict socket/header timeout.
- **`start_server()` & Lifecycle:**
  - Accepts `control_port: Optional[int] = None` (defaults to `port + 1` if `port > 0` else ephemeral `0`).
  - Spawns dedicated serving threads for both inference server and control server.
  - Shares single PID, instance token, executor, runtime, and progress telemetry.
  - `server.stop()` stops both listeners, waits for in-flight requests, shuts down executor, unloads models when safe, and cleanly joins both HTTP threads.
  - Preserves GET `/health` on inference port for backward compatibility.
- **CLI Wiring (`mlx_embed_server.py`):**
  - Adds `--control-port` flag (default from `MLX_CONTROL_PORT` or auto `port + 1`).
  - Reads `MLX_INSTANCE_TOKEN` from environment if present.

### Task 2: Owned-Child-Only Process Management by Default (`watchdog.py`, `qmd-mlx-watchdog.py`)
- **Default Owned-Child Mode:**
  - Enforce `owned_child` requirement by default.
  - If `owned_child is None`, require `allow_external_pid=True`, non-empty `instance_token`, and non-empty `expected_start_time`. Otherwise reject with `TargetValidationError`.
  - Validate that `owned_child.pid == config.pid`.
- **Pre-Signal Revalidation:**
  - Before **every** signal (`SIGTERM` and `SIGKILL`), revalidate PID liveness, command line, and start time.
  - Reject recycled PIDs, mismatched handles, and unrelated processes.
  - Never call `os.waitpid` on non-children.
  - Support clean cleanup and reaping of owned child on all exit paths.

### Task 3: Fail-Closed Memory Telemetry & Preflight Checks (`watchdog.py`, `qmd-mlx-watchdog.py`)
- **Remove Fabricated Fallbacks:**
  - `MLXWatchdog.__init__` must NOT catch `SystemMetricsError` to substitute fake 16GB/4GB/2600MB values.
  - If initial telemetry sampling fails, fail closed immediately.
- **Differentiate `ps` Failure from Process Exit:**
  - `get_process_rss_mb(pid)`: Command failures or timeouts raise `SystemMetricsError` -> `BreachType.TELEMETRY_UNAVAILABLE`.
  - `BreachType.TARGET_EXITED` is emitted ONLY when verified by `TargetProcessValidator.is_pid_alive(pid) == False`.
- **Preflight Checks:**
  - In `qmd-mlx-watchdog.py`, run preflight checks (`get_installed_ram_mb`, `get_memory_headroom_mb`, `get_swap_used_mb`) before launching subprocess.
- **Document Dry-Run Semantics:**
  - Explicitly document that `--dry-run` monitors and returns breach status without delivering termination signals.

### Task 4: Wall-Clock Deadline & Size-Bounded Health Probing (`watchdog.py`)
- **`_bounded_health_probe(host, port, timeout_s, max_bytes=65536)`:**
  - Numeric loopback `127.0.0.1` only; bypass proxies (`ProxyHandler({})`) and redirects.
  - Computes hard wall-clock deadline `t_deadline = time.monotonic() + timeout_s`.
  - Sets socket timeouts to remaining time before each I/O call.
  - Caps received response body to `max_bytes` (64KB); rejects oversized responses.
  - Parses and validates JSON payload structure and typed finite fields:
    - `pid` (int matching `config.pid`)
    - `instance_token` (str matching `config.instance_token`)
    - `worker_idle` (bool)
    - `worker_alive` (bool)
    - `active_job_age_s` (None or finite float >= 0)
    - `completed_sequence` (int >= 0)
  - Closes sockets promptly on all paths without thread or connection leaks.

### Task 5: Launch Token Generation & Qualification Port Isolation (`qmd-mlx-watchdog.py`)
- **Token Generation:**
  - `--launch` generates a fresh UUID instance token and passes it via `MLX_INSTANCE_TOKEN` env var to the child process.
  - Watchdog sets `config.instance_token` to this token and strictly enforces match on every probe.
- **Port Isolation:**
  - Add `--control-port` to watchdog CLI.
  - Target control port for health probes when configured.

### Task 6: Documentation & Parent SQLite Backup References
- **`docs/reviews/phase2-safety-corrections.md`:**
  - Remove personal author attribution.
  - Update status to IN PROGRESS.
  - Record each requirement, test, and known limitation.
- **`docs/plans/production-qualification.md`:**
  - Update Phase 1 backup reference to parent's verified SQLite online backup:
    `/Users/shersingh/.cache/qmd/index.sqlite.online-backup-20260908-192953.sqlite` (SHA-256 `2f09753af1b9d173465fba2d31922c980e2fd14ec9517e4ca5374fb2194bb7ec`).
  - Reiterate zero live DB writes.

### Task 7: Comprehensive Offline Regressions & Verification
- **New Tests:**
  1. `test_control_listener_responsive_during_inference_socket_exhaustion`: Saturated inference socket pool (100% busy with held header) while control listener responds 200 OK.
  2. `test_control_port_rejects_inference_requests`: POST `/embed` on control port returns 405/404.
  3. `test_bounded_health_probe_slow_drip_headers_and_bodies`: Proves hard wall-clock timeout aborts slow-drip streams.
  4. `test_bounded_health_probe_oversized_and_malformed`: Proves oversized bodies (>64KB) or malformed typed fields trigger probe failure without worker leaks.
  5. `test_watchdog_fails_closed_on_initial_telemetry_failure`: Proves no fabricated defaults in `__init__`.
  6. `test_external_pid_requires_opt_in_and_token`: Proves `allow_external_pid` enforcement.
  7. `test_watchdog_cli_launch_with_token_and_disposable_server`: Full CLI subprocess test with token passing and cleanup.
- **Validation Gates:**
  - Full offline `pytest test/python/` green.
  - `git diff --check` clean.

---

## 8. Specific Parent Safety Corrections Implementation & Acceptance Criteria

### 8.1 MLXControlRequestHandler Absolute Deadlines & Bounded Header Lifetime
- **Implementation:**
  - Replace inactivity-based `socket.settimeout(5.0)` with a wall-clock monotonic deadline reader and byte-bounded request parser (`_BoundedControlReader` / per-read timeout adjustment).
  - Enforce maximum control header byte budget (16 KB) and request deadline (default 5.0s, configurable in tests to smaller budgets like 0.2-0.5s to avoid long sleeps).
  - Enforce `Connection: close` and `self.close_connection = True` unconditionally on all responses from `MLXControlRequestHandler` (no keepalive connections on control plane).
  - Ensure deadline cleanup and socket closing operate strictly on the owned request socket without background timer threads that could touch recycled file descriptors.
  - Server shutdown lifecycle: `server.stop()` closes owned active connection sockets on both inference and control listeners, attempts to join serving threads (`t_http`, `t_ctrl`), and truthfully reports pending cleanup if native work remains stuck without unloading models.
- **Acceptance Criteria:**
  - Slow-drip HTTP client sending 1 byte/interval beyond total deadline is terminated by control server.
  - Control listener recovers after saturation without leaking handler threads or resources.
  - Saturated inference pool does not degrade control listener responsiveness.
  - `server.stop()` with blocked native worker does not unload live kernel or set false stopped status.

### 8.2 Watchdog Host & Port Restriction
- **Implementation:**
  - `MLXWatchdogConfig.__post_init__` strictly enforces numeric loopback IP (e.g. `127.0.0.1`, `127.x.y.z`, `::1`); rejects hostnames (`localhost`, `example.com`) and non-loopback addresses (`0.0.0.0`, LAN IPs).
  - Direct socket probing uses numeric IP directly without DNS resolution paths.
  - Require explicit, distinct `port` and `control_port` (`port != control_port`) for owned qualification (`--launch`) before spawn to eliminate accidental probing of default production port 8787.
  - Preflight socket check verifies `port` and `control_port` are not already occupied before spawning child process (fail closed if occupied).
- **Acceptance Criteria:**
  - Config rejects hostnames and non-loopback IPs at init.
  - Config and CLI reject identical `port` and `control_port`.
  - CLI preflight rejects occupied ports before subprocess spawn.
  - Test launches harmless fake server with explicit distinct ports and proves health check and cleanup.

### 8.3 Watchdog CLI Invariants, Prelaunch Headroom Snapshot & Lifecycle Safety
- **Implementation:**
  - Correct CLI documentation and help text: all watchdog options (`--port`, `--control-port`, `--once`, `--json`, etc.) must appear before `--launch` (`argparse.REMAINDER`).
  - Prelaunch validation: snapshot baseline swap (`baseline_swap_mb`) and headroom defaults BEFORE spawning child; verify headroom >= 256MB before spawn; reuse prelaunch baseline in watchdog (no recomputation after child allocation).
  - Mutual exclusion: reject `--pid` + `--launch` together before spawn.
  - Permission checks: reject `--pid` without `--allow-external-pid` and non-empty `--instance-token` before attempting attachment.
  - Safe lifecycle cleanup: wrap spawn and watchdog initialization in `try...finally` to ensure owned child process is always cleanly terminated, killed if unresponsive, and reaped (`wait()`) with bounded timeouts upon errors, exceptions, or `KeyboardInterrupt`.
  - Accurately document `--dry-run` semantics: prevents breach-triggered signals during checks, but owned children spawned by watchdog are reaped upon watchdog process exit (no orphan policy).
- **Acceptance Criteria:**
  - CLI usage help shows options before `--launch`.
  - Passing `--pid` and `--launch` simultaneously fails with clear error.
  - Passing `--pid` without required permissions fails before spawn.
  - Prelaunch swap and memory snapshot are passed into watchdog instance.
  - Exceptions during watcher init cleanly reap spawned child without orphans.

---

## 9. Plan Addendum: Strict Pre-Spawn Startup Order & Endpoint Validation Invariants

### 9.1 Pure Pre-Spawn Configuration & Option Validation
- **Requirement:** All PID-independent configuration fields, port ranges, host policy, threshold values, expected command regex patterns, non-empty `--launch` commands, and child CLI conflicts must be validated strictly BEFORE port binding checks, telemetry sampling, or `subprocess.Popen` execution.
- **Pure Validation Design:** Implement `validate_config_parameters(...)` in `scripts/qmd_mlx/watchdog.py` and call it upfront in `scripts/qmd-mlx-watchdog.py`. Avoid dummy PID validation side effects; instantiate `MLXWatchdogConfig` with the actual child PID only after process spawn (or when `--pid` is verified).
- **Failure Behavior:** If any configuration field is invalid (e.g. NaN, inf, negative, out-of-range port, equal ports, invalid regex, host mismatch, empty launch command), exit immediately with return code 2 before calling `socket.bind`, `sampler.get_installed_ram_mb`, or `subprocess.Popen`.

### 9.2 Host Policy Restriction to `127.0.0.1`
- **Requirement:** Eliminate socket family mismatch where `::1` (IPv6) is accepted by parser/config but rejected at runtime by `AF_INET` socket binds and the underlying server.
- **Implementation:** Explicitly restrict qualification host to `127.0.0.1`. Update `is_numeric_loopback(host)` and `validate_config_parameters` to reject `::1`, hostnames (`localhost`), and non-loopback addresses.

### 9.3 Child CLI Argument Conflict Detection
- **Requirement:** When launching `mlx_embed_server.py` or any child via `--launch`, ensure explicit child arguments (e.g. `--port <val>`, `--control-port <val>`, `--host <val>`) do not conflict with or silently override the watchdog's configured endpoints.
- **Implementation:** Inspect child command tokens before spawn. If conflicting explicit port/control-port/host values are detected, reject with a clear error before spawn.

### 9.4 Verification & Parametrized CLI Regressions
- **Regressions:** Add parametrized unit tests using a monkeypatched `Popen` spy asserting that `subprocess.Popen` is NEVER called when given invalid parameters (NaN, inf, negative thresholds, out-of-range ports, equal ports, missing ports, empty launch commands, invalid regex, invalid hosts, conflicting child ports).


