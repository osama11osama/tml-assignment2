#!/bin/bash
# RunPod: start workers 1, 2, 3 in background (one GPU each).
# PC should run worker0 separately.
#
#   tmux new -s a2split
#   bash scripts/cluster/runpod_workers_1_2_3.sh
#   tail -f runlogs/worker{1,2,3}.log

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p runlogs
chmod +x scripts/cluster/worker*_extract.sh

for wi in 1 2 3; do
  gpu=$((wi - 1))
  log="runlogs/worker${wi}.log"
  echo "Starting worker ${wi} on GPU ${gpu} -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  WORKER_NAME="runpod-w${wi}" \
  nohup bash scripts/cluster/worker${wi}_extract.sh >> "${log}" 2>&1 &
done

echo "Workers 1,2,3 launched. tail -f runlogs/worker1.log runlogs/worker2.log runlogs/worker3.log"
