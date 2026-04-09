# =============================================================================
# MLX Embedding Server - Startup Script
# =============================================================================
# Starts the FastAPI MLX server for QMD Qwen3-Embedding-8B-4bit-DWQ embeddings.
#
# Usage:
#   ./start.sh              # Default: port 8080
#   MLX_PORT=9000 ./start.sh  # Custom port
#
# Environment variables:
#   MLX_PORT         Server port (default: 8080)
#   MLX_HOST         Server host (default: 127.0.0.1)
#   MLX_MODEL_PATH   Override model path (default: use HF cache)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${MLX_VENV_DIR:-${SCRIPT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
STAMP_FILE="${VENV_DIR}/.requirements.sha256"

# Required for macOS with multiple OpenMP libraries loaded
export KMP_DUPLICATE_LIB_OK=TRUE

export OMP_NUM_THREADS=4  # Limit threads to avoid contention

if [ ! -d "${VENV_DIR}" ]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

if [ ! -f "${REQ_FILE}" ]; then
    echo "[start.sh] WARNING: requirements.txt not found, skipping pip install"
else
    REQ_HASH="$(shasum -a 256 "${REQ_FILE}" | awk '{print $1}')"
    INSTALLED_HASH=""
    if [ -f "${STAMP_FILE}" ]; then
        INSTALLED_HASH="$(cat "${STAMP_FILE}")"
    fi

    if [ "${REQ_HASH}" != "${INSTALLED_HASH}" ]; then
        python -m pip install --upgrade pip
        python -m pip install -r "${REQ_FILE}"
        printf '%s' "${REQ_HASH}" > "${STAMP_FILE}"
    fi
fi

echo "[start.sh] Starting MLX Embedding Server..."
echo "[start.sh] Model: ${MLX_MODEL_PATH:-mlx-community/Qwen3-Embedding-8B-4bit-DWQ}"
echo "[start.sh] URL:   http://${MLX_HOST:-127.0.0.1}:${MLX_PORT:-8080}"
echo ""

exec python "${SCRIPT_DIR}/server.py"
