# Phase 2 Safety Corrections Review & Safeguard Verification

**Date:** 2026-09-08  
**Status:** IN PROGRESS (Phase 2 Safety Corrections & Startup-Order Regressions Complete — Awaiting Parent Acceptance)  
**Gate:** Do NOT start Phase 3 model trial until parent review independently accepts this safeguard implementation.

---

## 1. Executive Summary

Following parent source review rejecting the initial Phase 2 candidate and confirming specific safety correction areas, the implementation has been completed and verified with deterministic offline unit and integration tests.

| Safety Requirement | Parent Finding | Implemented Correction & Regression |
|---|---|---|
| **1. Control Header Deadlines & Socket Boundedness** | `MLXControlRequestHandler.setup` claimed absolute deadline but used `socket.settimeout(5.0)` (inactivity timeout); slow drip indefinitely resets inactivity; keepalive connections hung control capacity. | Implemented `_BoundedDeadlineRfile` enforcing monotonic wall-clock deadline across all reads and 16 KB header byte budget. `MLXControlRequestHandler` unconditionally sends `Connection: close` and sets `close_connection = True`. Active sockets are tracked and closed on `stop()`. Verified with `test_control_listener_absolute_deadline_slow_drip`, `test_control_listener_header_byte_budget_exceeded`, `test_control_listener_connection_close_no_keepalive`, and `test_control_listener_saturation_recovery_no_leaked_handlers`. |
| **2. Watchdog Numeric Loopback & Distinct Ports** | `MLXWatchdogConfig.__post_init__` did not restrict host, permitting hostnames/DNS paths. `control_port` being optional risked probing production port 8787 by accident during qualification. | `is_numeric_loopback` strictly restricts `host` to numeric IPv4 loopback IP (`127.0.0.1`); hostnames (`localhost`, etc.), IPv6 (`::1`), and non-loopback IPs are rejected to prevent socket AF_INET / IPv6 mismatches. `port != control_port` is strictly enforced. Qualification CLI (`--launch`) requires explicit, distinct `--port` and `--control-port` options before `--launch`, and runs preflight socket bind checks to fail closed before spawn if endpoints are occupied. |
| **3. Pre-Spawn Startup Order & Pure Config Validation** | In `qmd-mlx-watchdog.py`, `subprocess.Popen` and socket binding occurred before `MLXWatchdogConfig` initialization, allowing child processes to spawn before threshold/port/regex arguments were validated. | Implemented pure `validate_config_parameters(...)` called upfront in `qmd-mlx-watchdog.py` BEFORE any socket binding, telemetry sampling, or `subprocess.Popen`. Validates finite positive thresholds, port ranges (1-65535), distinct ports, host policy (`127.0.0.1`), regex compilation, and non-empty launch command. Finalizes actual child PID only after process spawn without dummy production PID side effects. |
| **4. Child CLI Argument Conflict Detection** | Child commands launched via `--launch` could specify explicit conflicting arguments (e.g. `--port 9999`) that silently override environment-passed ports. | Implemented `check_child_cli_conflicts(...)` inspecting child command arguments before spawn. Rejects conflicting explicit `--port`, `--control-port`, or `--host` arguments with return code 2 before spawn. Verified with `test_watchdog_cli_child_cli_matching_args_accepted` and parametrized regressions. |
| **5. Lifecycle & Live Kernel Safety** | Stop lifecycle needed active connection closing and truthful pending state when native work is stuck. | `ThreadedMLXServer` and `MLXControlServer` track active sockets and close them during `stop()`. If executor worker thread is still alive after join timeout, `stop()` truthfully defers model unload and reports pending cleanup rather than unloading models while native execution is ongoing. |

---

## 2. Detailed Technical Corrections

### 2.1 Dedicated Loopback Control Listener & Absolute Deadlines (`scripts/qmd_mlx/server.py`)
- **Monotonic Wall-Clock Deadline Reader (`_BoundedDeadlineRfile`):** Wraps raw socket `rfile` to calculate remaining time (`rem = deadline - time.monotonic()`) before every single `recv()`. If `rem <= 0`, immediately raises `socket.timeout`. Prevents slow-drip HTTP clients from indefinitely keeping handler threads occupied.
- **Bounded Header Byte Budget:** Tracks cumulative bytes read during request/header parsing; if total bytes exceed `max_header_bytes` (16 KB default), raises `ValueError` and terminates connection.
- **Connection Close Invariant:** `MLXControlRequestHandler._send_json` unconditionally sends `Connection: close` and sets `self.close_connection = True`. No keepalive connections are allowed on the control listener.
- **Active Socket Tracking & Safe Stop:** `ThreadedMLXServer` and `MLXControlServer` track open client sockets under lock. `server.stop()` closes all active client sockets, joins `t_ctrl` and `t_http`, cancels pending work, and if native work is stuck beyond join budget, avoids unloading live models and reports truthful pending status.

### 2.2 Watchdog Host & Port Restriction (`scripts/qmd_mlx/watchdog.py`)
- **Strict IPv4 Numeric Loopback Enforcement:** `is_numeric_loopback()` validates that `host` is strictly `127.0.0.1` (or IPv4 `127.0.0.0/8`). Hostnames (`localhost`, `example.com`), IPv6 addresses (`::1`), and non-loopback addresses (`0.0.0.0`, LAN IPs) are rejected in `validate_config_parameters` and `MLXWatchdogConfig.__post_init__` to ensure socket operations (`socket.AF_INET`) are consistent across the entire qualification path.
- **Port Distinctness & Range Validation:** `validate_config_parameters` validates that `port` and `control_port` are in `1..65535` and `port != control_port`.
- **Direct IP Probing:** `_default_health_probe` connects directly to numeric loopback without any DNS lookup or proxy resolution path.

### 2.3 Strict Pre-Spawn Startup Order & Pure Config Validation (`scripts/qmd-mlx-watchdog.py`)
- **Pure Parameter Validation Upfront:** `validate_config_parameters` checks all PID-independent configuration fields (thresholds, ranges, host policy, regex pattern, non-empty launch command) BEFORE any socket binding, telemetry sampling, or child process spawning.
- **Child CLI Conflict Prevention:** `check_child_cli_conflicts` inspects `cmd` tokens for explicit `--port`, `--control-port`, `--host` flags and rejects conflicting child endpoint specifications before spawn.
- **Prelaunch Validation & Headroom Snapshot:** Before spawning the child process:
  1. Validates mutual exclusion (`--pid` vs `--launch`).
  2. Validates external attach permissions (`--allow-external-pid` + `--instance-token`).
  3. Validates that `--port` and `--control-port` are distinct and currently unoccupied.
  4. Samples preflight physical RAM, swap usage baseline (`baseline_swap_mb`), and available headroom (`headroom_mb`).
  5. Verifies headroom >= 256 MB.
  6. Passes prelaunch snapshot (`baseline_swap_mb`, `defaults`) into `MLXWatchdog` to avoid re-sampling after model starts allocating memory.
- **Safe Lifecycle & Dry-Run Documentation:** Subprocess lifecycle is wrapped in `try...finally` to ensure spawned child processes are always reaped (`terminate()` -> `wait(2.0)` -> `kill()` -> `wait(1.0)`). Documented that `--dry-run` prevents breach-triggered signals during checks, but owned children are always cleaned up on process exit (no orphan policy).

---

## 3. Verification Results

### 3.1 Python Safeguards & Runtime Test Suite (`PYTHONPATH=. .venv/bin/pytest`)
```
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/shersingh/github/qmd-mlx-search
plugins: anyio-4.15.1
collected 189 items

test/python/test_mlx_batching.py ..........                              [  5%]
test/python/test_mlx_executor.py ...........                             [ 11%]
test/python/test_mlx_faults.py .....                                     [ 13%]
test/python/test_mlx_generate.py ss....                                  [ 16%]
test/python/test_mlx_model_manager.py ................                   [ 25%]
test/python/test_mlx_protocol.py .......                                 [ 29%]
test/python/test_mlx_rerank.py .ss.ss..s.s.                              [ 35%]
test/python/test_mlx_runtime.py s..s...................                  [ 47%]
test/python/test_mlx_safeguards.py ......                                [ 50%]
test/python/test_mlx_server.py ......................                    [ 62%]
test/python/test_mlx_server_startup.py .....                             [ 65%]
test/python/test_mlx_watchdog.py ....................................... [ 85%]
...........................                                              [100%]

======================= 179 passed, 10 skipped in 16.70s =======================
```
*(10 skipped tests correspond to real-model weight download tests, skipped intentionally in offline test mode.)*

### 3.2 Specific Verified Watchdog & Startup-Order Regressions
- `test_pure_validate_config_parameters_function`: Proves pure parameter validation rejects invalid host (`::1`, `localhost`), out-of-range ports, equal ports, invalid thresholds (NaN, inf, negative), and invalid regex without requiring PID.
- `test_watchdog_cli_child_cli_matching_args_accepted`: Proves matching explicit child arguments (`--port 8787 --control-port 8788 --host 127.0.0.1` and `--port=8787`) pass pre-spawn conflict checks.
- `test_watchdog_cli_prespawn_validation_popen_never_called` (37 parametrized cases): Proves monkeypatched `subprocess.Popen` is NEVER called when CLI is given:
  - NaN thresholds (`--health-timeout-s nan`, `--min-free-memory-pct nan`, `--max-rss-mb nan`, `--max-swap-growth-mb nan`, `--stalled-timeout-s nan`)
  - Inf thresholds (`--health-timeout-s inf`, `--grace-period-s inf`, `--startup-grace-period-s inf`)
  - Negative/zero thresholds (`--health-timeout-s -1.0`, `--health-timeout-s 0.0`, `--check-interval-s 0`, `--check-interval-s -0.5`, `--max-rss-mb -50`, `--max-swap-growth-mb -10`, `--min-free-memory-pct 0.0`, `--min-free-memory-pct 150.0`, `--consecutive-failures 0`)
  - Out of range ports (`--port 0`, `--port 70000`, `--control-port 0`, `--control-port 65536`)
  - Equal/missing ports (`--port 8787 --control-port 8787`, missing `--control-port`, missing `--port`)
  - Empty `--launch` (`--launch`, `--launch ""`, `--launch "   "`)
  - Invalid regex (`--expected-cmd "["`, `--expected-cmd "(+invalid"`)
  - Invalid host (`--host ::1`, `--host localhost`, `--host 0.0.0.0`)
  - Conflicting child CLI arguments (`--port 9999`, `--control-port 9998`, `--port=9999`, `--control-port=9998`, `--host 192.168.1.5`)

### 3.3 Git Diff Quality Check
- `git diff --check`: **exit 0** (clean, zero trailing whitespace or formatting defects)

---

## 4. Honest Unresolved Limitations & Operational Notes

1. **Apple Silicon Unified Memory vs Process RSS:**
   - Metal GPU allocations and unified memory buffers on macOS Apple Silicon reside partially in wired / `IOAccelerator` system pages rather than standard process RSS (`ps -o rss=`).
   - The watchdog monitors both target process RSS and system-wide memory pressure (`memory_pressure -Q` and swap growth) to detect aggregate pressure breaches even if Metal buffer allocations do not fully reflect in RSS.
2. **Launchd Daemon vs Qualification Test Server:**
   - The qualification watchdog is designed to supervise test instances running on separate ephemeral ports (e.g. 8787/8788), strictly isolated from any live production daemon.
   - The live production daemon (`com.qmd.mlxd.plist`) remains managed independently by launchd.
3. **Offline Test Policy:**
   - All tests run completely offline using mock architectures and synthetic fixtures. No real weights, live databases, or network calls were initiated.
4. **SQLite Database Backup Status:**
   - Verified online backup by parent: `/Users/shersingh/.cache/qmd/index.sqlite.online-backup-20260908-192953.sqlite` (SHA-256 `2f09753af1b9d173465fba2d31922c980e2fd14ec9517e4ca5374fb2194bb7ec`, `PRAGMA quick_check=ok`, schema table count=20).
   - Zero live database writes during Phase 2.

---

## 5. Next Steps

- **Hold:** Phase 3 single-model real-model trials remain paused pending explicit parent review and acceptance of this safeguard implementation. No production-ready claim is made.
