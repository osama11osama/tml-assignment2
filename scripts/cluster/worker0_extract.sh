#!/bin/bash
# Worker 0 of 4 — suspects 0, 4, 8, …, 356 (90 suspects)
#
# Run on YOUR PC or any machine with 1 GPU:
#   cd /path/to/tml-assignment2
#   chmod +x scripts/cluster/worker0_extract.sh
#   bash scripts/cluster/worker0_extract.sh
#
# Optional env:
#   CUDA_VISIBLE_DEVICES=0  DEVICE=cuda  BATCH_SIZE=128
#   WORKER_NAME=local-5060

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=worker_common.sh
source "$SCRIPT_DIR/worker_common.sh"

_run_extract_worker 0 "${WORKER_NAME:-local-worker0}"
