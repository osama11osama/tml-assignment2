#!/bin/bash
# Quick inventory: target + how many of 360 suspects exist (and which IDs are missing).
set -uo pipefail

ROOT="${1:-$HOME/tml26_task2}"
cd "$ROOT" || { echo "Missing: $ROOT"; exit 1; }

echo "=== Model inventory: $ROOT ==="
echo

ok=0
miss=0

if [[ -f target_model/weights.safetensors ]]; then
  echo "  OK   target_model/weights.safetensors ($(du -h target_model/weights.safetensors | cut -f1))"
  ((ok++)) || true
else
  echo "  MISS target_model/weights.safetensors"
  ((miss++)) || true
fi

if [[ -f data/train_main_idx.json ]]; then
  echo "  OK   data/train_main_idx.json"
  ((ok++)) || true
else
  echo "  MISS data/train_main_idx.json"
  ((miss++)) || true
fi

n=$(ls suspect_models/suspect_*.safetensors 2>/dev/null | wc -l)
echo
echo "Suspect files on disk: $n / 360"

missing_list=$(mktemp)
for i in $(seq 0 359); do
  f=$(printf "suspect_models/suspect_%03d.safetensors" "$i")
  if [[ ! -f "$f" ]]; then
    printf "%03d\n" "$i" >> "$missing_list"
  fi
done

n_missing=$(wc -l < "$missing_list")
if [[ "$n_missing" -eq 0 ]]; then
  echo "  OK   all 360 suspect weights present"
  ((ok++)) || true
else
  echo "  MISS $n_missing suspect files"
  echo "  First 30 missing IDs:"
  head -30 "$missing_list" | tr '\n' ' '
  echo
  echo "  Full list: $ROOT/missing_suspect_ids.txt"
  cp "$missing_list" missing_suspect_ids.txt
  ((miss++)) || true
fi
rm -f "$missing_list"

echo
if [[ -f results/cache_master/config.json ]]; then
  echo "  OK   results/cache_master/config.json (target cache ready)"
else
  echo "  MISS results/cache_master/config.json — run master_target Condor job first"
fi

echo
echo "=== Summary: $n/360 suspects, target=$([[ -f target_model/weights.safetensors ]] && echo yes || echo no) ==="
[[ "$n" -eq 360 && -f target_model/weights.safetensors ]] && exit 0 || exit 1
