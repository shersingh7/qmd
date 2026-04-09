#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${MLX_VENV_DIR:-${SCRIPT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
STAMP_FILE="${VENV_DIR}/.requirements.sha256"

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

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

exec python "${SCRIPT_DIR}/server.py"
