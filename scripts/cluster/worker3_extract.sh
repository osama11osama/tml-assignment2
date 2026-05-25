#!/bin/bash
# Worker 3 of 4 — suspects 3, 7, 11, …, 359 (90 suspects)
#
# Run on YOUR PC or RunPod:
#   bash scripts/cluster/worker3_extract.sh
#
# On RunPod (GPU 2 of 3):
#   CUDA_VISIBLE_DEVICES=2 WORKER_NAME=runpod-w3 bash scripts/cluster/worker3_extract.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/worker_common.sh"

_run_extract_worker 3 "${WORKER_NAME:-worker3}"
