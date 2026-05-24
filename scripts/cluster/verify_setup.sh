#!/bin/bash
# Run on cluster: bash scripts/cluster/verify_setup.sh
# Checks that Assignment 2 pipeline files are present before condor_submit.
set -uo pipefail

BASE="${HOME}/tml26_task2"
cd "${BASE}" 2>/dev/null || { echo "MISSING directory: ${BASE}"; exit 1; }

ok=0
fail=0

check_file() {
  if [[ -f "$1" ]]; then
    echo "  OK   $1"
    ((ok++)) || true
  else
    echo "  MISS $1"
    ((fail++)) || true
  fi
}

check_dir_min() {
  local dir=$1 min=$2
  local n
  n=$(find "$dir" -maxdepth 1 -type f 2>/dev/null | wc -l)
  if [[ -d "$dir" ]] && [[ "$n" -ge "$min" ]]; then
    echo "  OK   $dir ($n files)"
    ((ok++)) || true
  else
    echo "  MISS $dir (need >= $min files, found ${n:-0})"
    ((fail++)) || true
  fi
}

echo "=== Assignment 2 cluster setup check ==="
echo "Base: ${BASE}"
echo

echo "[1] Source code"
check_file "src/paths.py"
check_file "src/model_utils.py"
check_file "src/probe_data.py"
check_file "src/logit_scoring.py"
echo

echo "[2] Scripts"
check_file "scripts/score_logit_similarity.py"
check_file "scripts/cluster/precompute_target_logits.py"
check_file "scripts/cluster/score_one_suspect.py"
check_file "scripts/cluster/merge_scores.py"
echo

echo "[3] Data (wget)"
check_file "target_model/weights.safetensors"
check_file "data/train_main_idx.json"
echo

echo "[4] Suspect models (wget loop — 360 total)"
n=$(ls suspect_models/suspect_*.safetensors 2>/dev/null | wc -l)
if [[ "$n" -eq 360 ]]; then
  echo "  OK   suspect_models ($n/360)"
  ((ok++)) || true
elif [[ "$n" -gt 0 ]]; then
  echo "  PARTIAL suspect_models ($n/360) — download still running?"
  ((fail++)) || true
else
  echo "  MISS suspect_models (0/360)"
  ((fail++)) || true
fi
echo

echo "[5] Condor submit files"
check_file "condor/precompute_target.sub"
check_file "condor/score_suspect.sub"
echo

echo "[6] Output directories"
for d in results/cache results/cluster_scores results/submissions runlogs; do
  if [[ -d "$d" ]]; then
    echo "  OK   $d/"
    ((ok++)) || true
  else
    echo "  MISS $d/"
    ((fail++)) || true
  fi
done
echo

echo "=== Summary: ${ok} OK, ${fail} missing/incomplete ==="
if [[ "$fail" -eq 0 ]]; then
  echo "Ready for: condor_submit condor/precompute_target.sub"
  exit 0
else
  echo "Fix MISSING items before submitting jobs."
  exit 1
fi
