#!/bin/bash
# Worker 2 of 4 — suspects 2, 6, 10, …, 358 (90 suspects)
#
# Run on YOUR PC or RunPod:
#   bash scripts/cluster/worker2_extract.sh
#
# On RunPod (GPU 1 of 3):
#   CUDA_VISIBLE_DEVICES=1 WORKER_NAME=runpod-w2 bash scripts/cluster/worker2_extract.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/worker_common.sh"

_run_extract_worker 2 "${WORKER_NAME:-worker2}"
