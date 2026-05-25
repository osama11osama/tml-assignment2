#!/bin/bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/home/atml_team044/tml26_task2}"
PY="/opt/conda/bin/python"
REQ="${ROOT}/requirements.txt"

export PYTHONUSERBASE="${ROOT}/.pyuser"
export PIP_CACHE_DIR="${ROOT}/.cache/pip"

mkdir -p "${PYTHONUSERBASE}" "${PIP_CACHE_DIR}"

cd "${ROOT}"
if [[ ! -f "${REQ}" ]]; then
  echo "Missing requirements file: ${REQ}" >&2
  exit 1
fi

"${PY}" -m pip install --user -r "${REQ}"
"${PY}" - <<'PY'
import safetensors  # noqa: F401
import sklearn  # noqa: F401
print("python env OK: safetensors + sklearn import")
PY
