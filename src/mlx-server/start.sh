#!/usr/bin/env bash
# =============================================================================
# MLX Server for QMD — Startup Script
# =============================================================================
# Starts the FastAPI MLX server with embedding, reranking & generation.
#
# Models loaded (lazy where possible):
#   Embedding:  Qwen3-Embedding-8B-4bit-DWQ  (loads immediately, ~4GB)
#   Reranker:   Qwen3-Reranker-8B-mxfp8       (lazy, first /v1/rerank call, ~7.8GB)
#   Generation: Qwen3-8B-MLX-4bit             (lazy, first /v1/generate call, ~4.3GB)
#
# Usage:
#   ./start.sh                       # Default: port 8080
#   MLX_PORT=9000 ./start.sh         # Custom port
#   MLX_PORT=8080 ./start.sh &       # Background
#
# Health check:
#   curl http://127.0.0.1:8080/health
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${MLX_VENV_DIR:-${SCRIPT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
STAMP_FILE="${VENV_DIR}/.requirements.sha256"
LOG_FILE="${SCRIPT_DIR}/server.log"
PID_FILE="${SCRIPT_DIR}/server.pid"

# Required for macOS with multiple OpenMP libraries loaded
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'  # No Color

info()  { echo -e "${GREEN}[MLX]${NC} $*"; }
warn()  { echo -e "${YELLOW}[MLX]${NC} $*"; }
error() { echo -e "${RED}[MLX]${NC} $*" >&2; }

# ─── Check if server is already running ──────────────────────────────────────

check_running() {
    local port="${MLX_PORT:-8080}"
    if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
        info "Server already running on port ${port}"
        HEALTH=$(curl -sf "http://127.0.0.1:${port}/health" 2>/dev/null)
        info "Health: ${HEALTH}"
        return 0
    fi
    return 1
}

# ─── Stop existing server ──────────────────────────────────────────────────────

stop_server() {
    if [ -f "${PID_FILE}" ]; then
        OLD_PID=$(cat "${PID_FILE}")
        if kill -0 "${OLD_PID}" 2>/dev/null; then
            info "Stopping existing server (PID ${OLD_PID})..."
            kill "${OLD_PID}" 2>/dev/null || true
            sleep 2
            # Force kill if still running
            if kill -0 "${OLD_PID}" 2>/dev/null; then
                warn "Force killing ${OLD_PID}..."
                kill -9 "${OLD_PID}" 2>/dev/null || true
                sleep 1
            fi
        fi
        rm -f "${PID_FILE}"
    fi
}

# ─── Main ──────────────────────────────────────────────────────────────────────

if check_running; then
    warn "Server is already running. Use 'stop_server' or kill the process first."
    echo ""
    echo "  To stop:  kill \$(cat ${PID_FILE})"
    echo "  To restart: ${BASH_SOURCE[0]} --restart"
    echo ""
    exit 0
fi

if [ "${1:-}" = "--restart" ] || [ "${1:-}" = "-r" ]; then
    stop_server
fi

# ─── Setup venv ────────────────────────────────────────────────────────────────

if [ ! -d "${VENV_DIR}" ]; then
    info "Creating Python virtual environment at ${VENV_DIR}..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

if [ ! -f "${REQ_FILE}" ]; then
    warn "requirements.txt not found, skipping pip install"
else
    REQ_HASH="$(shasum -a 256 "${REQ_FILE}" | awk '{print $1}')"
    INSTALLED_HASH=""
    if [ -f "${STAMP_FILE}" ]; then
        INSTALLED_HASH="$(cat "${STAMP_FILE}")"
    fi

    if [ "${REQ_HASH}" != "${INSTALLED_HASH}" ]; then
        info "Installing/updating Python dependencies..."
        python -m pip install --upgrade pip -q
        python -m pip install -r "${REQ_FILE}" -q
        printf '%s' "${REQ_HASH}" > "${STAMP_FILE}"
    fi
fi

# ─── Preflight checks ─────────────────────────────────────────────────────────

info "Preflight checks..."

# Check MLX is available
if ! python -c "import mlx; print(f'  MLX version: {mlx.__version__}')" 2>/dev/null; then
    error "MLX not installed. Run: pip install mlx mlx-lm"
    exit 1
fi

# Check model cache
MODEL_NAME="${MLX_MODEL_PATH:-mlx-community/Qwen3-Embedding-8B-4bit-DWQ}"
CACHE_STATUS=$(_find_cached_snapshot "${MODEL_NAME}" 2>/dev/null || echo "not cached")
info "Embedding model: ${MODEL_NAME}"

# ─── Start server ─────────────────────────────────────────────────────────────

PORT="${MLX_PORT:-8080}"
HOST="${MLX_HOST:-127.0.0.1}"

info "Starting MLX Server on http://${HOST}:${PORT}"
info "  Embedding model:  ${MLX_MODEL_PATH:-mlx-community/Qwen3-Embedding-8B-4bit-DWQ}"
info "  Reranker model:   ${MLX_RERANK_MODEL:-mlx-community/Qwen3-Reranker-8B-mxfp8}"
info "  Generation model: ${MLX_GENERATE_MODEL:-Qwen/Qwen3-8B-MLX-4bit}"
info "  Log file: ${LOG_FILE}"
echo ""

# Start the server — foreground mode (Ctrl+C to stop)
# For background: ./start.sh &
python "${SCRIPT_DIR}/server.py" 2>&1 | tee -a "${LOG_FILE}" &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"

# Wait for server to be ready (up to 120 seconds for model loading)
info "Waiting for server to be ready (models take ~10-30s to load)..."
for i in $(seq 1 60); do
    if curl -sf "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        error "Server process died unexpectedly. Check ${LOG_FILE}"
        exit 1
    fi
    sleep 2
done

if curl -sf "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
    HEALTH=$(curl -sf "http://${HOST}:${PORT}/health" 2>/dev/null)
    info "Server is ready!"
    info "Health: ${HEALTH}"
    echo ""
    info "Endpoints:"
    info "  POST http://${HOST}:${PORT}/v1/embeddings"
    info "  POST http://${HOST}:${PORT}/v1/rerank"
    info "  POST http://${HOST}:${PORT}/v1/generate"
    info "  POST http://${HOST}:${PORT}/v1/chat/completions"
    info "   GET http://${HOST}:${PORT}/health"
    echo ""
    info "Server PID: ${SERVER_PID} (saved to ${PID_FILE})"
    info "To stop: kill ${SERVER_PID}"
else
    warn "Server hasn't responded yet. Models may still be loading."
    warn "Check health with: curl http://${HOST}:${PORT}/health"
    warn "Check logs: tail -f ${LOG_FILE}"
fi

# If running in foreground, keep the script alive and wait for the server
wait ${SERVER_PID}