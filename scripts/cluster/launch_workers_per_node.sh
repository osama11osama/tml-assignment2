#!/bin/bash
# Launch N parallel master_extract workers on ONE node (one GPU each).
# Use on Instant Cluster nodes with 8 GPUs — uses all GPUs on that machine.
#
#   NUM_LOCAL_WORKERS=8 bash scripts/cluster/launch_workers_per_node.sh
#
# WORKER_INDEX = NODE_RANK * NUM_LOCAL_WORKERS + local_gpu_id
# NUM_WORKERS  = NUM_NODES * NUM_LOCAL_WORKERS  (set NUM_WORKERS env before launch)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

NUM_LOCAL="${NUM_LOCAL_WORKERS:-${NUM_TRAINERS:-1}}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-1}"
NUM_WORKERS="${NUM_WORKERS:-$((NUM_NODES * NUM_LOCAL))}"

mkdir -p runlogs

for local_id in $(seq 0 $((NUM_LOCAL - 1))); do
  global_id=$((NODE_RANK * NUM_LOCAL + local_id))
  logfile="runlogs/extract_w${global_id}.log"
  echo "Starting worker ${global_id}/${NUM_WORKERS} on GPU ${local_id} -> ${logfile}"
  CUDA_VISIBLE_DEVICES="${local_id}" \
  WORKER_INDEX="${global_id}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  WORKER_INDEX_SET=1 \
  STEP=extract \
  WORKER_NAME="node${NODE_RANK}-gpu${local_id}" \
  nohup bash scripts/cluster/runpod_instant_cluster_master.sh >> "${logfile}" 2>&1 &
done

echo "Launched ${NUM_LOCAL} workers on node ${NODE_RANK}. tail -f runlogs/extract_w*.log"
