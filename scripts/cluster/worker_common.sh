#!/bin/bash
# Shared helpers for worker0–worker3 extract scripts.
# Sourced by workerN_extract.sh — do not run directly.

set -euo pipefail

WORKER_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$WORKER_ROOT"

PY="${PYTHON:-python}"
BATCH="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"

_run_target_if_missing() {
  if [[ -f results/cache_master/config.json ]]; then
    return 0
  fi
  echo "No results/cache_master/config.json — running target precompute first..."
  "$PY" scripts/master_precompute_target.py \
    --device "$DEVICE" \
    --import-legacy-40k \
    --batch-size "$BATCH"
}

_run_extract_worker() {
  local wi="$1"
  local wname="${2:-worker${wi}}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

  echo "=============================================="
  echo " Master cache extract — worker ${wi} / ${NUM_WORKERS}"
  echo " Suspect ids: i % ${NUM_WORKERS} == ${wi}"
  echo " Machine: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES}  device=${DEVICE}"
  echo "=============================================="

  _run_target_if_missing

  "$PY" scripts/master_extract.py \
    --device "$DEVICE" \
    --batch-size "$BATCH" \
    --worker-index "$wi" \
    --num-workers "$NUM_WORKERS" \
    --worker-name "$wname"

  "$PY" scripts/master_status.py --worker-index "$wi" --num-workers "$NUM_WORKERS"
  echo "Worker ${wi} finished."
}
