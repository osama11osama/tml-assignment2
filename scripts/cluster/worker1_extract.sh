#!/bin/bash
# Worker 1 of 4 — suspects 1, 5, 9, …, 357 (90 suspects)
#
# Run on YOUR PC or RunPod (use GPU 0 if 3 GPUs on one pod):
#   bash scripts/cluster/worker1_extract.sh
#
# On RunPod with 3 parallel jobs, prefer:
#   CUDA_VISIBLE_DEVICES=0 WORKER_NAME=runpod-w1 bash scripts/cluster/worker1_extract.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/worker_common.sh"

_run_extract_worker 1 "${WORKER_NAME:-worker1}"
