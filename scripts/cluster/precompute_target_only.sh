#!/bin/bash
# Target cache only — run ONCE per machine before any workerN_extract.sh
#   bash scripts/cluster/precompute_target_only.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
exec "$PY" scripts/master_precompute_target.py --device "${DEVICE:-cuda}" --import-legacy-40k --batch-size "${BATCH_SIZE:-128}"
