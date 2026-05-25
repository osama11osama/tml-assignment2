# RunPod: resumable 40k (do not lose progress)

## Stop the old run (optional)

In Jupyter terminal: `Ctrl+C` on the current `score_logit_similarity.py` (it does **not** save per suspect).

## Start resumable pipeline (safe)

```bash
cd /workspace/tml-assignment2

# get latest script (if repo public)
wget -q -O scripts/cluster/run_resumable_40k.sh \
  https://raw.githubusercontent.com/osama11osama/tml-assignment2/main/scripts/cluster/run_resumable_40k.sh
chmod +x scripts/cluster/run_resumable_40k.sh

# run in background — close laptop / Jupyter OK
screen -S a2score
bash scripts/cluster/run_resumable_40k.sh 2>&1 | tee runlogs/resumable_40k.log
# Detach: Ctrl+A then D
# Reattach: screen -r a2score
```

Each suspect writes `results/cluster_scores/suspect_XXX.json` immediately.

If interrupted, run the **same command** again — it skips finished suspects.

## Download result

`results/submissions/submission_v002_logit_40k.csv` via Jupyter file browser.
