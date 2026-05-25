#!/bin/bash
# Run on cluster login node: bash scripts/cluster/cluster_diagnose.sh
set -euo pipefail
BASE="${HOME}/tml26_task2"
cd "$BASE"

echo "=== 1) verify_setup ==="
bash scripts/cluster/verify_setup.sh || true

echo
echo "=== 2) python + torch ==="
/opt/conda/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), torch.__version__)" || python3 -c "import torch; print('cuda', torch.cuda.is_available())"

echo
echo "=== 3) smoke (2 suspects, 256 images) — needs GPU shell if cuda False here ==="
/opt/conda/bin/python scripts/score_logit_similarity.py --subset 256 --suspects 0,1 --device cuda --output submission_smoke.csv || true

echo
echo "=== 4) condor held jobs? ==="
condor_q -hold 2>/dev/null | head -20 || echo "(no condor_q or no held jobs)"

echo
echo "=== 5) last precompute error (if any) ==="
tail -30 runlogs/precompute.*.err 2>/dev/null || echo "no precompute err yet"
