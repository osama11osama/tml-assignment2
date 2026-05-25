#!/bin/bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/home/atml_team044/tml26_task2}"
PY="/opt/conda/bin/python"
SID="${1:?suspect id required}"

export PYTHONUSERBASE="${ROOT}/.pyuser"
PYVER="$("${PY}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
USER_SITE="${ROOT}/.pyuser/lib/python${PYVER}/site-packages"
export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
exec "${PY}" scripts/master_extract.py --suspects "${SID}" --device cuda --batch-size 128 --no-claim --worker-name "condor-${SID}"
