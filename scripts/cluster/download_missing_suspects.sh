#!/bin/bash
# Download only missing suspect_*.safetensors (resume-safe).
set -euo pipefail

ROOT="${1:-$HOME/tml26_task2}"
cd "$ROOT"

HF="https://huggingface.co/SprintML/tml26_task2/resolve/main"
mkdir -p suspect_models

for i in $(seq -f "%03g" 0 359); do
  f="suspect_models/suspect_${i}.safetensors"
  if [[ -f "$f" ]]; then
    continue
  fi
  echo "Downloading $f ..."
  wget -q "$HF/suspect_models/suspect_${i}.safetensors" -O "$f" || {
    echo "FAILED $f" >&2
    rm -f "$f"
    exit 1
  }
done

echo "Done. Count: $(ls suspect_models/suspect_*.safetensors | wc -l) / 360"
