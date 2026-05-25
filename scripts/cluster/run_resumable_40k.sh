#!/bin/bash
# Resumable v002 on 40k probes — saves one JSON per suspect (safe to stop/restart).
# Usage (GPU machine / RunPod / cluster GPU node):
#   cd ~/tml26_task2   # or /workspace/tml-assignment2
#   bash scripts/cluster/run_resumable_40k.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
BATCH="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUTPUT:-submission_v002_logit_40k.csv}"
CACHE="${CACHE:-results/cache/target_logits_40k.pt}"

mkdir -p results/cluster_scores results/submissions runlogs

if [[ ! -f "$CACHE" ]]; then
  echo "Precomputing target logits (40k)..."
  "$PY" scripts/cluster/precompute_target_logits.py --batch-size "$BATCH" --device "$DEVICE" --cache "$CACHE"
else
  echo "Using existing cache: $CACHE"
fi

done=0
skip=0
for i in $(seq 0 359); do
  f="results/cluster_scores/suspect_$(printf '%03d' "$i").json"
  if [[ -f "$f" ]]; then
    ((skip++)) || true
    continue
  fi
  echo "[$(date +%H:%M:%S)] scoring suspect_$i ..."
  "$PY" scripts/cluster/score_one_suspect.py "$i" --batch-size "$BATCH" --device "$DEVICE" --cache "$CACHE"
  ((done++)) || true
done

echo "Scored $done new, skipped $skip existing."
"$PY" scripts/cluster/merge_scores.py --output "$OUT"
echo "Done -> results/submissions/$OUT"
