# HTCondor jobs (Saarland HPC)

Before submit, on the cluster login node:

```bash
export PROJECT=~/tml26_task2
cd "$PROJECT"
mkdir -p runlogs results/cache results/cluster_scores results/submissions
```

If `$(PROJECT)` is not expanded by your Condor version, edit both `.sub` files and replace `$(PROJECT)` with your full path, e.g. `/home/atml_team044/tml26_task2`.

## Master cache v2 (360 parallel jobs — fastest on busy cluster)

```bash
cd ~/tml26_task2
bash scripts/cluster/verify_models.sh          # must show 360/360 before extract
screen -S download
bash scripts/cluster/download_missing_suspects.sh
# Ctrl+A D

condor_submit condor/master_target.sub         # wait for config.json
condor_submit condor/master_extract_one.sub    # 360 jobs, 1 suspect each
condor_q
python scripts/master_status.py
python scripts/master_score_variants.py        # login node, CPU
```

| File | Jobs | What |
|------|------|------|
| `master_target.sub` | 1 | Target cache |
| `master_extract_one.sub` | 360 | `master_extract.py --suspects $(Process)` |

Do **not** submit `master_extract_one` until `verify_models.sh` exits 0 and `config.json` exists.
