#!/usr/bin/env python3
"""
qmd-mlx-watchdog.py — Standalone External Resource Watchdog CLI for MLX Production Qualification

Monitors:
- Target MLX server PID RSS vs available memory headroom
- macOS System Memory Pressure (memory_pressure / vm_stat)
- System Swap Growth (sysctl vm.swapusage)
- HTTP /health Endpoint Responsiveness (on dedicated control port or inference port)
- Instance Token & PID Strict Binding Validation
- Worker Progress & Stuck Inference Detection

Guarantees:
- Owned-child-only management by default; external attach requires explicit opt-in, non-empty start identity, and instance token.
- Safe Target Isolation: Revalidates start time before EVERY signal (SIGTERM & SIGKILL); never signals recycled PIDs or unrelated sentinels.
- Conservative Defaults: Derived from available memory headroom, not installed RAM; no purgeable double-counting.
- Fail Closed: Telemetry failures and preflight checks trigger fail-safe alerts rather than hallucinating safety.
- Non-blocking: Probing stays responsive with strict wall-clock timeouts, numeric 127.0.0.1 loopback, no proxy interference, and bounded response sizes.
- Clean Lifecycle: Reaps owned test child process on all exit paths (--once, breach, error, interrupt).

Usage:
    python3 scripts/qmd-mlx-watchdog.py --port 8787 --control-port 8788 --launch python3 scripts/mlx_embed_server.py --port 8787 --control-port 8788
    python3 scripts/qmd-mlx-watchdog.py --port 8787 --control-port 8788 --once --json --launch python3 scripts/mlx_embed_server.py --port 8787 --control-port 8788
    python3 scripts/qmd-mlx-watchdog.py --pid <PID> --allow-external-pid --instance-token <TOKEN> --port 8787 --control-port 8788
    python3 scripts/qmd-mlx-watchdog.py --print-defaults
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import uuid
from typing import Optional

# Ensure repository root and scripts directory are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from scripts.qmd_mlx.watchdog import (
    MLXWatchdog,
    MLXWatchdogConfig,
    SystemMemorySampler,
    SystemMetricsError,
    TargetValidationError,
    is_numeric_loopback,
    validate_config_parameters,
)


def check_child_cli_conflicts(
    cmd: list[str],
    expected_port: int,
    expected_control_port: int,
    expected_host: str,
) -> Optional[str]:
    """
    Inspects child command arguments before spawn to ensure explicit CLI arguments
    do not conflict with or override the watchdog's configured endpoints.
    """
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        # Check --port
        if tok == "--port" and i + 1 < len(cmd):
            val_str = cmd[i + 1]
            try:
                if int(val_str) != expected_port:
                    return (
                        f"Explicit child CLI argument '--port {val_str}' conflicts with "
                        f"watchdog '--port {expected_port}'. Conflicting child endpoint arguments are rejected before spawn."
                    )
            except ValueError:
                return f"Explicit child CLI argument '--port {val_str}' is not a valid integer."
            i += 2
            continue
        elif tok.startswith("--port="):
            val_str = tok.split("=", 1)[1]
            try:
                if int(val_str) != expected_port:
                    return (
                        f"Explicit child CLI argument '{tok}' conflicts with "
                        f"watchdog '--port {expected_port}'. Conflicting child endpoint arguments are rejected before spawn."
                    )
            except ValueError:
                return f"Explicit child CLI argument '{tok}' is not a valid integer."
            i += 1
            continue

        # Check --control-port
        if tok == "--control-port" and i + 1 < len(cmd):
            val_str = cmd[i + 1]
            try:
                if int(val_str) != expected_control_port:
                    return (
                        f"Explicit child CLI argument '--control-port {val_str}' conflicts with "
                        f"watchdog '--control-port {expected_control_port}'. Conflicting child endpoint arguments are rejected before spawn."
                    )
            except ValueError:
                return f"Explicit child CLI argument '--control-port {val_str}' is not a valid integer."
            i += 2
            continue
        elif tok.startswith("--control-port="):
            val_str = tok.split("=", 1)[1]
            try:
                if int(val_str) != expected_control_port:
                    return (
                        f"Explicit child CLI argument '{tok}' conflicts with "
                        f"watchdog '--control-port {expected_control_port}'. Conflicting child endpoint arguments are rejected before spawn."
                    )
            except ValueError:
                return f"Explicit child CLI argument '{tok}' is not a valid integer."
            i += 1
            continue

        # Check --host
        if tok == "--host" and i + 1 < len(cmd):
            val_str = cmd[i + 1]
            if val_str != expected_host:
                return (
                    f"Explicit child CLI argument '--host {val_str}' conflicts with "
                    f"watchdog '--host {expected_host}'. Conflicting child endpoint arguments are rejected before spawn."
                )
            i += 2
            continue
        elif tok.startswith("--host="):
            val_str = tok.split("=", 1)[1]
            if val_str != expected_host:
                return (
                    f"Explicit child CLI argument '{tok}' conflicts with "
                    f"watchdog '--host {expected_host}'. Conflicting child endpoint arguments are rejected before spawn."
                )
            i += 1
            continue

        i += 1

    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="External Resource Watchdog for MLX Production Qualification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pid", type=int, default=None, help="Target process PID to monitor")
    parser.add_argument("--host", default="127.0.0.1", help="Target server numeric loopback host")
    parser.add_argument("--port", type=int, default=None, help="Target server HTTP port (required for --launch)")
    parser.add_argument("--control-port", type=int, default=None, help="Dedicated loopback control port for health probes (required for --launch)")
    parser.add_argument(
        "--instance-token",
        default=None,
        help="Expected unique server instance token to verify against HTTP /health response",
    )
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        default=None,
        help="Maximum allowable target process RSS in MB (default: auto from headroom)",
    )
    parser.add_argument(
        "--max-swap-growth-mb",
        type=float,
        default=None,
        help="Maximum allowable system swap growth in MB from baseline (default: auto from headroom)",
    )
    parser.add_argument(
        "--min-free-memory-pct",
        type=float,
        default=12.0,
        help="Minimum system memory free percentage before triggering breach",
    )
    parser.add_argument(
        "--health-timeout-s",
        type=float,
        default=3.0,
        help="Timeout in seconds for HTTP /health probe",
    )
    parser.add_argument(
        "--consecutive-failures",
        type=int,
        default=3,
        help="Number of consecutive failed health probes before triggering breach",
    )
    parser.add_argument(
        "--check-interval-s",
        type=float,
        default=1.0,
        help="Polling interval in seconds between monitoring checks",
    )
    parser.add_argument(
        "--grace-period-s",
        type=float,
        default=3.0,
        help="Grace period in seconds between SIGTERM and SIGKILL escalation",
    )
    parser.add_argument(
        "--startup-grace-period-s",
        type=float,
        default=30.0,
        help="Grace period in seconds allowing cold model loading before enforcing health availability",
    )
    parser.add_argument(
        "--stalled-timeout-s",
        type=float,
        default=60.0,
        help="Maximum duration in seconds an in-flight execution job may run before being deemed stalled",
    )
    parser.add_argument(
        "--expected-cmd",
        default=r"(python|mlx|qmd|server)",
        help="Regex pattern that target command-line must match for safety verification",
    )
    parser.add_argument(
        "--allow-external-pid",
        action="store_true",
        help="Explicitly permit attaching to an externally spawned PID (requires --instance-token)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log and report breach events without sending SIGTERM/SIGKILL signals. Note: Owned child processes spawned via --launch are still reaped upon watchdog exit to enforce no orphan processes.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Perform a single check cycle and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Format status output as JSON",
    )
    parser.add_argument(
        "--print-defaults",
        action="store_true",
        help="Print computed conservative defaults for this machine and exit",
    )
    parser.add_argument(
        "--launch",
        nargs=argparse.REMAINDER,
        help="Launch and own child process with arguments. NOTE: All watchdog options must precede --launch.",
    )

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2

    # 1. Print defaults check (if requested, sample and exit)
    if args.print_defaults:
        sampler = SystemMemorySampler()
        try:
            defaults = sampler.compute_conservative_defaults()
        except SystemMetricsError as e:
            print(f"Error sampling memory statistics: {e}", file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps(defaults.__dict__, indent=2))
        else:
            print("Conservative Watchdog Defaults (derived from measured headroom):")
            print(f"  Installed RAM:          {defaults.installed_ram_mb:.1f} MB")
            print(f"  Available Headroom:     {defaults.headroom_mb:.1f} MB")
            print(f"  Max Process RSS:        {defaults.max_rss_mb:.1f} MB (<= 65% of headroom)")
            print(f"  Max Swap Growth:        {defaults.max_swap_growth_mb:.1f} MB")
            print(f"  Min Free Memory:        {defaults.min_free_memory_pct:.1f}%")
            print(f"  Health Timeout:         {defaults.health_timeout_s:.1f}s")
            print(f"  Max Failed Probes:      {defaults.consecutive_health_failures}")
            print(f"  Startup Grace Period:   {defaults.startup_grace_period_s:.1f}s")
            print(f"  Stalled Job Timeout:    {defaults.stalled_inference_timeout_s:.1f}s")
        return 0

    # 2. Early validation: Mutual exclusion
    if args.pid is not None and args.launch is not None:
        print(
            "Error: Cannot specify both --pid and --launch. "
            "Choose --pid to attach to an existing process or --launch to spawn an owned child.",
            file=sys.stderr,
        )
        return 2

    if args.pid is None and args.launch is None:
        print(
            "Error: Either --pid <PID> or --launch <CMD...> is required (or use --print-defaults).",
            file=sys.stderr,
        )
        return 2

    # 3. Early validation: Host restriction (must be strictly numeric IPv4 loopback 127.0.0.1)
    if not is_numeric_loopback(args.host) or args.host != "127.0.0.1":
        print(
            f"Error: Host '{args.host}' must be a numeric IPv4 loopback IP ('127.0.0.1'). "
            "Hostnames, IPv6 addresses (::1), and non-loopback addresses are rejected.",
            file=sys.stderr,
        )
        return 2

    # 4. Early validation: External PID permissions & identity
    if args.pid is not None:
        if not (isinstance(args.pid, int) and args.pid > 1):
            print(
                f"Error: Invalid pid {args.pid}: must be an integer > 1",
                file=sys.stderr,
            )
            return 2
        if not args.allow_external_pid:
            print(
                f"Error: Attaching to external PID {args.pid} is disabled by default. "
                "Pass --allow-external-pid and --instance-token to opt in.",
                file=sys.stderr,
            )
            return 2
        if not args.instance_token:
            print(
                "Error: --instance-token is required when attaching to external PID.",
                file=sys.stderr,
            )
            return 2

    # 5. Early validation: Port requirements, ranges, and launch command
    effective_port = args.port
    effective_control_port = args.control_port

    if args.launch is not None:
        if effective_port is None or effective_control_port is None:
            print(
                "Error: Owned qualification (--launch) requires explicit, distinct --port and --control-port "
                "options specified BEFORE --launch to prevent accidental production probing.",
                file=sys.stderr,
            )
            return 2

        # Validate clean launch command BEFORE Popen or binding
        launch_cmd = list(args.launch)
        if launch_cmd and launch_cmd[0] == "--":
            launch_cmd = launch_cmd[1:]
        if not launch_cmd or all(not c.strip() for c in launch_cmd):
            print(
                "Error: --launch command cannot be empty.",
                file=sys.stderr,
            )
            return 2

        # Check child CLI conflicts
        conflict_err = check_child_cli_conflicts(
            cmd=launch_cmd,
            expected_port=effective_port,
            expected_control_port=effective_control_port,
            expected_host=args.host,
        )
        if conflict_err:
            print(f"Error: {conflict_err}", file=sys.stderr)
            return 2
    else:
        if effective_port is None:
            effective_port = 8787

    # 6. Early validation: Pure config parameters (all thresholds, ranges, regex)
    try:
        validate_config_parameters(
            host=args.host,
            port=effective_port,
            control_port=effective_control_port,
            max_rss_mb=args.max_rss_mb,
            max_swap_growth_mb=args.max_swap_growth_mb,
            min_free_memory_pct=args.min_free_memory_pct,
            health_timeout_s=args.health_timeout_s,
            consecutive_health_failures=args.consecutive_failures,
            check_interval_s=args.check_interval_s,
            grace_period_s=args.grace_period_s,
            startup_grace_period_s=args.startup_grace_period_s,
            stalled_inference_timeout_s=args.stalled_timeout_s,
            expected_cmd_pattern=args.expected_cmd,
        )
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        return 2

    # 7. Preflight port occupancy checks (for --launch)
    if args.launch is not None:
        for p_name, p_val in [("Inference port", effective_port), ("Control port", effective_control_port)]:
            try:
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                test_sock.bind((args.host, p_val))
                test_sock.close()
            except OSError as e:
                print(
                    f"Error: {p_name} {p_val} on {args.host} is already in use ({e}). "
                    f"Refusing to spawn child on an occupied endpoint.",
                    file=sys.stderr,
                )
                return 2

    # 8. Preflight telemetry & headroom validation: SNAPSHOT baseline BEFORE spawn
    sampler = SystemMemorySampler()
    try:
        sampler.get_installed_ram_mb()
        headroom = sampler.get_memory_headroom_mb()
        baseline_swap = sampler.get_swap_used_mb()
        sampler.get_memory_free_pct()
        defaults = sampler.compute_conservative_defaults(headroom_mb=headroom)
    except SystemMetricsError as e:
        print(f"[watchdog] Preflight telemetry check failed: {e}", file=sys.stderr)
        return 2

    if headroom < 256.0:
        print(f"[watchdog] Preflight check failed: critically low memory headroom ({headroom:.1f} MB < 256 MB)", file=sys.stderr)
        return 2

    owned_child: Optional[subprocess.Popen] = None
    target_pid = args.pid
    instance_token = args.instance_token

    try:
        if args.launch is not None:
            cmd = list(args.launch)
            if cmd and cmd[0] == "--":
                cmd = cmd[1:]
            if not instance_token:
                instance_token = uuid.uuid4().hex

            child_env = os.environ.copy()
            child_env["MLX_INSTANCE_TOKEN"] = instance_token
            child_env["MLX_EMBED_PORT"] = str(effective_port)
            child_env["MLX_CONTROL_PORT"] = str(effective_control_port)

            print(
                f"[watchdog] Launching owned child process: {' '.join(cmd)} (instance_token={instance_token})",
                file=sys.stderr if args.json else sys.stdout,
            )
            try:
                owned_child = subprocess.Popen(cmd, env=child_env)
                target_pid = owned_child.pid
            except Exception as e:
                print(f"[watchdog] Failed to spawn child process: {e}", file=sys.stderr)
                return 2

        # Construct MLXWatchdogConfig with finalized target_pid
        try:
            config = MLXWatchdogConfig(
                pid=target_pid,
                host=args.host,
                port=effective_port,
                control_port=effective_control_port,
                max_rss_mb=args.max_rss_mb,
                max_swap_growth_mb=args.max_swap_growth_mb,
                min_free_memory_pct=args.min_free_memory_pct,
                health_timeout_s=args.health_timeout_s,
                consecutive_health_failures=args.consecutive_failures,
                check_interval_s=args.check_interval_s,
                grace_period_s=args.grace_period_s,
                startup_grace_period_s=args.startup_grace_period_s,
                stalled_inference_timeout_s=args.stalled_timeout_s,
                expected_cmd_pattern=args.expected_cmd,
                instance_token=instance_token,
                allow_external_pid=args.allow_external_pid or (owned_child is not None),
                dry_run=args.dry_run,
            )
        except ValueError as e:
            print(f"[watchdog] Invalid configuration: {e}", file=sys.stderr)
            return 2

        try:
            watchdog = MLXWatchdog(
                config=config,
                sampler=sampler,
                owned_child=owned_child,
                baseline_swap_mb=baseline_swap,
                defaults=defaults,
            )
        except (TargetValidationError, SystemMetricsError) as e:
            print(f"[watchdog] Target process safety validation failed: {e}", file=sys.stderr)
            return 2

        if args.once:
            result = watchdog.check_step()
            if args.json:
                print(json.dumps({
                    "timestamp": result.timestamp,
                    "healthy": result.healthy,
                    "breach_type": result.breach_type.value if result.breach_type else None,
                    "breach_reason": result.breach_reason,
                    "metrics": result.metrics,
                    "terminated_pid": result.terminated_pid,
                    "sigkill_used": result.sigkill_used,
                }, indent=2))
            else:
                if result.healthy:
                    print(f"[watchdog] OK: Target PID {target_pid} healthy. Metrics: {result.metrics}")
                else:
                    print(f"[watchdog] BREACH: {result.breach_reason}")
            return 0 if result.healthy else 1

        return watchdog.run_loop()

    except KeyboardInterrupt:
        print("\n[watchdog] Interrupted by user; stopping watchdog.", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"[watchdog] Unexpected watchdog error: {e}", file=sys.stderr)
        return 2
    finally:
        # Guarantee cleanup and reaping of owned test child on every exit path
        if owned_child is not None and owned_child.poll() is None:
            try:
                owned_child.terminate()
                owned_child.wait(timeout=2.0)
            except Exception:
                try:
                    owned_child.kill()
                    owned_child.wait(timeout=1.0)
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
