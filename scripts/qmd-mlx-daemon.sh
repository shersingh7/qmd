#!/usr/bin/env bash
# qmd-mlx-daemon.sh — control the unified QMD MLX inference daemon.
#
#   qmd-mlx-daemon.sh run      # foreground (for launchd) — never backgrounds
#   qmd-mlx-daemon.sh start    # install plist + bootstrap via launchd
#   qmd-mlx-daemon.sh stop     # bootout via launchd
#   qmd-mlx-daemon.sh status   # launchd state + /ready probe
#
# Models are David's locked picks (all-mlx-pipeline.md §0). Override via env:
#   MLX_EMBED_MODEL, MLX_RERANK_MODEL, MLX_GENERATE_MODEL, MLX_PORT
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.qmd.mlxd"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PORT="${MLX_PORT:-8787}"
VENV_PY="$REPO/.venv/bin/python"

# Production = embed-only daemon. Rerank/expansion stay in-process GGUF
# (measured: GGUF ranks hard queries better). Set MLX_RERANK_MODEL /
# MLX_GENERATE_MODEL explicitly to opt into those adapters; there is
# deliberately no default (an idle-unload story for those adapters does not
# exist yet — see docs/benchmarks/all-mlx-results.md).
# Default: local affine 4-bit build (scripts/convert-qwen3-embedding-4b.sh) —
# measured better recall (84% vs 80%) AND faster kernels (22.1 vs 18.7 t/s)
# than the community DWQ build. Override with MLX_EMBED_MODEL.
EMBED_MODEL="${MLX_EMBED_MODEL:-$HOME/.cache/qmd/models/qwen3-embedding-4b-mlx-4bit-affine}"
RERANK_MODEL="${MLX_RERANK_MODEL:-}"
GENERATE_MODEL="${MLX_GENERATE_MODEL:-}"

cmd_run() {
  # Foreground exec — launchd supervises the process directly.
  exec "$VENV_PY" "$REPO/scripts/mlx_embed_server.py" \
    --model "$EMBED_MODEL" \
    --rerank-model "$RERANK_MODEL" \
    --generate-model "$GENERATE_MODEL" \
    --port "$PORT" \
    --host 127.0.0.1
}

cmd_start() {
  if [ ! -x "$VENV_PY" ]; then
    echo "error: venv python missing at $VENV_PY (create the repo venv first)" >&2
    exit 1
  fi
  mkdir -p "$HOME/Library/LaunchAgents"
  sed -e "s|__REPO__|$REPO|g" "$REPO/scripts/launchd/com.qmd.mlxd.plist" > "$PLIST"
  # Bootout first: makes start idempotent (no silent no-op when a stale
  # registration exists, no silent failure when bootstrap errors).
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  # bootout is async — an immediate re-bootstrap races it ("Input/output
  # error"). Wait for the port to actually free first.
  for _ in $(seq 1 15); do
    if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then break; fi
    sleep 1
  done
  local attempt=1
  while [ $attempt -le 3 ]; do
    if launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>&1; then break; fi
    if [ $attempt -eq 3 ]; then
      echo "error: launchd bootstrap failed for $LABEL after 3 attempts" >&2
      return 1
    fi
    sleep 2
    attempt=$((attempt + 1))
  done
  if ! launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -q "state = running"; then
    echo "warning: $LABEL bootstrapped but not running yet — check ~/.cache/qmd/mlx-daemon.log" >&2
  fi
  echo "started $LABEL (port $PORT)"
}

cmd_stop() {
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  echo "stopped $LABEL"
}

cmd_status() {
  launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -E "state|pid" | head -3 || echo "not bootstrapped"
  curl -s --max-time 5 "http://127.0.0.1:$PORT/ready" || echo " (daemon not responding)"
  echo
}

case "${1:-status}" in
  run) cmd_run ;;
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  *) echo "usage: $0 {run|start|stop|status}" >&2; exit 1 ;;
esac
