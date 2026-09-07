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

EMBED_MODEL="${MLX_EMBED_MODEL:-mlx-community/Qwen3-Embedding-4B-4bit-DWQ}"
RERANK_MODEL="${MLX_RERANK_MODEL:-$HOME/.cache/qmd/models/qwen3-reranker-4b-mlx-4bit}"
GENERATE_MODEL="${MLX_GENERATE_MODEL:-mlx-community/Qwen3-1.7B-4bit}"

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
  launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || launchctl kickstart -k "gui/$(id -u)/$LABEL"
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
