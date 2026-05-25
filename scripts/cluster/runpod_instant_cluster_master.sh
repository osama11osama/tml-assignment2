#!/bin/bash
# Master cache v2 on RunPod Instant Cluster OR any multi-GPU / multi-node setup.
#
# Requires shared filesystem (Network Volume) mounted at REPO root, e.g. /workspace/tml-assignment2
#
# Usage on PRIMARY node (NODE_RANK=0) once:
#   STEP=target bash scripts/cluster/runpod_instant_cluster_master.sh
#
# Usage on EVERY GPU worker (parallel):
#   STEP=extract bash scripts/cluster/runpod_instant_cluster_master.sh
#
# After all workers finish (NODE_RANK=0):
#   STEP=score bash scripts/cluster/runpod_instant_cluster_master.sh
#
# Env (auto on Instant Cluster, or set manually):
#   NODE_RANK, NUM_NODES, NUM_TRAINERS (GPUs per node)
#   WORKER_INDEX — optional override (0 .. NUM_WORKERS-1)
#   NUM_WORKERS  — optional override (default NUM_NODES * NUM_TRAINERS)
#   CUDA_VISIBLE_DEVICES — set per process when launching multiple workers per node

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
BATCH="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda}"
STEP="${STEP:-extract}"
WORKER_NAME="${WORKER_NAME:-runpod-$(hostname)-${NODE_RANK:-0}}"

# RunPod Instant Cluster env (see docs.runpod.io/instant-clusters)
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-1}"
NUM_TRAINERS="${NUM_TRAINERS:-1}"

NUM_WORKERS="${NUM_WORKERS:-$((NUM_NODES * NUM_TRAINERS))}"
WORKER_INDEX="${WORKER_INDEX:-$NODE_RANK}"

if [[ "$NUM_WORKERS" -gt 1 && "$NUM_NODES" -gt 1 ]]; then
  # If you launch one process per node only, shard by NODE_RANK:
  if [[ -z "${WORKER_INDEX_SET:-}" ]]; then
    WORKER_INDEX="${NODE_RANK}"
    NUM_WORKERS="${NUM_NODES}"
  fi
fi

mkdir -p runlogs results/cache_master/suspects results/cache_master/claims

log() { echo "[$(date -Iseconds)] [rank=${WORKER_INDEX}/${NUM_WORKERS}] $*"; }

case "$STEP" in
  target)
    log "STEP=target — precompute target cache (run on ONE node only)"
    "$PY" scripts/master_precompute_target.py \
      --device "$DEVICE" \
      --import-legacy-40k \
      --batch-size "$BATCH"
    ;;

  extract)
    log "STEP=extract — suspect master cache"
    if [[ ! -f results/cache_master/config.json ]]; then
      log "ERROR: run STEP=target on NODE_RANK=0 first"
      exit 1
    fi
    "$PY" scripts/master_extract.py \
      --device "$DEVICE" \
      --batch-size "$BATCH" \
      --worker-index "$WORKER_INDEX" \
      --num-workers "$NUM_WORKERS" \
      --worker-name "$WORKER_NAME"
    ;;

  score)
    log "STEP=score — CPU variants (no GPU)"
    "$PY" scripts/master_score_variants.py
    "$PY" submission.py --validate-only \
      results/submissions/submission_master_BEST_rank_fusion_multidist.csv
    ;;

  status)
    "$PY" scripts/master_status.py
    ;;

  *)
    echo "Unknown STEP=$STEP (use: target | extract | score | status)"
    exit 1
    ;;
esac

log "Done STEP=$STEP"
