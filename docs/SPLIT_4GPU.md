# Split master cache: 4 GPUs (1 local + 3 RunPod)

No shared cluster storage needed. Each machine processes **90 suspects** (360 ÷ 4).

| Worker | Machine | GPU | Suspect IDs |
|--------|---------|-----|-------------|
| **0** | Your PC (RTX 5060) | `cuda:0` | 0, 4, 8, …, 356 |
| **1** | RunPod | GPU 0 | 1, 5, 9, …, 357 |
| **2** | RunPod | GPU 1 | 2, 6, 10, …, 358 |
| **3** | RunPod | GPU 2 | 3, 7, 11, …, 359 |

**After extract:** copy RunPod `suspects/` → PC → one CPU scoring step.

---

## Prerequisites

- [ ] Models on **both** machines (`target_model/` + `suspect_models/` × 360)
- [ ] Same repo code on both (`git pull` or scp `src/`, `scripts/`)
- [ ] RunPod: **1 Pod** with **≥3 GPUs** (3090/4090/5090), **not** Instant Cluster
- [ ] Attach volume `bitter_turquoise_guppy` at `/workspace` (optional, keeps data)

---

## Step 1 — Target cache (both machines)

Target tensors must exist **on each machine** (suspects compare to local `target/` files).

### On your PC

```powershell
cd "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment2"
.\.venv\Scripts\Activate.ps1
python scripts/master_precompute_target.py --device cuda --import-legacy-40k --batch-size 128
```

### On RunPod (SSH)

```bash
cd /workspace/tml-assignment2   # or your path
pip install -r requirements.txt
chmod +x scripts/cluster/*.sh
STEP=target bash scripts/cluster/runpod_instant_cluster_master.sh
```

**Or** copy only target cache from PC (faster if PC already done):

```powershell
scp -r runpod-a2:/workspace/tml-assignment2/results/cache_master/target `
  # wrong direction - FROM pc TO runpod:
scp -r "results/cache_master/target" runpod-a2:/workspace/tml-assignment2/results/cache_master/
```

---

## Step 2 — Parallel extract (4 workers)

**Four standalone scripts** — run **any worker on PC or RunPod** (your choice).

| Worker | Suspects | Linux/Mac/Git Bash | Windows PowerShell |
|--------|----------|--------------------|--------------------|
| **0** | 0,4,8,… | `bash scripts/cluster/worker0_extract.sh` | `.\scripts\cluster\worker0_extract.ps1` |
| **1** | 1,5,9,… | `bash scripts/cluster/worker1_extract.sh` | `.\scripts\cluster\worker1_extract.ps1` |
| **2** | 2,6,10,… | `bash scripts/cluster/worker2_extract.sh` | `.\scripts\cluster\worker2_extract.ps1` |
| **3** | 3,7,11,… | `bash scripts/cluster/worker3_extract.sh` | `.\scripts\cluster\worker3_extract.ps1` |

Target cache once per machine (optional — each worker script auto-runs if missing):

```bash
bash scripts/cluster/precompute_target_only.sh
```

### Example A — PC worker 0 + RunPod workers 1–3

**PC:**

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\cluster\worker0_extract.ps1
```

**RunPod (3 GPUs at once):**

```bash
tmux new -s a2split
cd /workspace/tml-assignment2
bash scripts/cluster/runpod_workers_1_2_3.sh
tail -f runlogs/worker*.log
```

### Example B — all 4 on your PC (4 terminals, 1 GPU each — sequential)

Run **one at a time** on the same GPU (slower):

```powershell
.\scripts\cluster\worker0_extract.ps1
.\scripts\cluster\worker1_extract.ps1
# ... etc
```

### Example C — RunPod only (4 GPUs on one pod)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/cluster/worker0_extract.sh &
CUDA_VISIBLE_DEVICES=1 bash scripts/cluster/worker1_extract.sh &
CUDA_VISIBLE_DEVICES=2 bash scripts/cluster/worker2_extract.sh &
CUDA_VISIBLE_DEVICES=3 bash scripts/cluster/worker3_extract.sh &
wait
```

### Example D — single worker on RunPod (1 GPU)

```bash
CUDA_VISIBLE_DEVICES=0 WORKER_NAME=runpod-w1 bash scripts/cluster/worker1_extract.sh
```

---

## Step 3 — Check progress

**PC:**

```powershell
python scripts/master_status.py --worker-index 0 --num-workers 4
```

**RunPod:**

```bash
python scripts/master_status.py --worker-index 1 --num-workers 4
# ... or count folders:
ls results/cache_master/suspects | wc -l
```

Each worker should finish **~90** suspects. Total unique folders on PC after sync: **360**.

---

## Step 4 — Merge RunPod → PC

From **PowerShell** (only copy suspect dirs workers 1–3 processed):

```powershell
$RP = "runpod-a2"
$LOCAL = "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment2\results\cache_master\suspects"

# Sync all suspect folders from RunPod (overwrites none on PC for ids 1,2,3 mod 4)
scp -r "${RP}:/workspace/tml-assignment2/results/cache_master/suspects/*" $LOCAL
```

Or use rsync if available. **Do not delete** PC worker-0 folders before sync.

Verify:

```powershell
(Get-ChildItem results\cache_master\suspects).Count   # should be 360
python scripts/master_status.py
```

---

## Step 5 — Score on PC only (CPU, 2 min)

```powershell
python scripts/master_score_variants.py
python submission.py --validate-only results/submissions/submission_master_BEST_rank_fusion_multidist.csv
```

---

## Step 6 — Submit + stop RunPod

```powershell
python submission.py results/submissions/submission_master_BEST_rank_fusion_multidist.csv
```

Terminate RunPod Pod to stop billing.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| RunPod only 1 GPU | Run workers 1–3 **sequentially** on 1 GPU (slower), or use 2+2 split with PC |
| `config.json missing` on RunPod | Run `STEP=target` there first |
| Duplicate work | Same `NUM_WORKERS=4` and distinct `WORKER_INDEX` 0–3 |
| <360 after sync | List missing: `python scripts/master_status.py` |
| PC sleeps | Disable sleep; keep `--worker-name` logs |

---

## Time estimate

| Part | Time |
|------|------|
| Target (each machine) | ~5 min once |
| 90 suspects / GPU | ~25–40 min |
| **Wall clock** (parallel 4) | **~30–45 min** extract + sync + score |
| vs 1 GPU sequential | ~2–3 h |

---

## Quick reference

```text
PC:     worker 0, NUM_WORKERS=4
RunPod: workers 1,2,3  →  scripts/cluster/runpod_workers_1_2_3.sh
Merge:  scp suspects/* → PC
Score:  master_score_variants.py (PC only)
```
