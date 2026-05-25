#!/bin/bash
set -euo pipefail

ROOT="/home/atml_team044/tml26_task2"
PY="/opt/conda/bin/python"

export PYTHONUSERBASE="${ROOT}/.pyuser"
PYVER="$("${PY}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
USER_SITE="${ROOT}/.pyuser/lib/python${PYVER}/site-packages"
export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
exec "${PY}" scripts/master_score_variants.py
