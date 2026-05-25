# RunPod Instant Cluster — Master cache v2

Use this for the **next run** (`docs/CACHE_MASTER.md` v2): 40k train + 10k unused train + 10k test + aug.

**Important:** You do **not** need 15 nodes / 120 GPUs. That is for huge LLM training. For 360 suspects, **2 nodes × 1–8 GPUs** with **shared Network Volume** is enough.

---

## What you need from RunPod

| Item | Why |
|------|-----|
| **Network Volume** (~50–100 GB) | Shared `results/cache_master/` — survives pod stop, all workers see same files |
| **Instant Cluster** (2+ nodes) OR **2 GPU Pods** | Parallel extract |
| **PyTorch template** | CUDA + Python |
| **Models on volume** | `target_model/` + `suspect_models/` (~16 GB) |

Docs: [RunPod Instant Clusters](https://docs.runpod.io/instant-clusters) · [PyTorch deploy](https://docs.runpod.io/instant-clusters/pytorch)

---

## Architecture (your job)

```
Network Volume (/workspace/tml-assignment2)
├── target_model/ + suspect_models/ + data/
├── results/cache_master/          ← all GPUs write here (resumable)
│   ├── target/   (once)
│   └── suspects/suspect_XXX/
└── results/submissions/           ← CSVs after CPU score step

Node 0 (NODE_RANK=0):  STEP=target  (once, ~5 min)
All GPUs:              STEP=extract  (parallel, worker-index sharding)
Node 0:                STEP=score    (CPU, ~2 min)
```

**Sharding:** suspect `id` goes to worker `id % NUM_WORKERS`. No duplicate work if shared volume + claims work.

---

## Step-by-step setup

### 1. Create Network Volume (once)

1. RunPod console → **Storage** → **Network Volume**
2. Create **~80 GB** volume (models 16 GB + cache ~10 GB + headroom)
3. Pick a **region** you can get GPUs in (e.g. CA-MTL, EU, etc.)

### 2. Put project + weights on the volume

**Option A — Jupyter on a temp Pod (easiest)**

1. Start a **single GPU Pod**, mount the Network Volume at `/workspace`
2. Terminal:

```bash
cd /workspace
git clone https://github.com/osama11osama/tml-assignment2.git
cd tml-assignment2
pip install -r requirements.txt

# Download models ONCE to the volume
python scripts/download_models.py --target-only
python scripts/download_models.py
```

3. **Stop the Pod** — data stays on the volume.

**Option B — copy from your PC**

```powershell
# If you already have models locally, rsync/scp to volume via a mounted pod
scp -r "...\target_model" "...\suspect_models" "...\data" root@POD:/workspace/tml-assignment2/
```

### 3. Create Instant Cluster

1. **Instant Clusters** → **Create cluster**
2. Settings (recommended for this assignment):

| Setting | Value |
|---------|--------|
| Pod count | **2** (not 15) |
| GPU | **RTX 4090 / A100 / 3090** — whatever is cheap & available |
| GPUs per node | **1–8** (more = faster extract) |
| Template | **RunPod PyTorch** |
| **Network Volume** | Attach your volume at `/workspace` |
| Container disk | **20–50 GB** (code + CIFAR cache) |

3. **Deploy cluster** → wait until all nodes are **Running**

### 4. SSH from your PC (recommended)

**Configure SSH first:** [RUNPOD_SSH_SETUP.md](RUNPOD_SSH_SETUP.md) (Windows key, `~/.ssh/config`, SCP, tmux).

```powershell
# On your PC — test after deploy
ssh runpod-a2 "nvidia-smi -L"    # should list 8 GPUs
ssh runpod-a2                     # interactive shell
```

Inside SSH (use **tmux** so disconnect does not kill jobs):

```bash
tmux new -s a2master
cd /workspace/tml-assignment2
pip install -r requirements.txt   # once per fresh image
```

Check env (Instant Cluster sets these automatically):

```bash
echo "NODE_RANK=$NODE_RANK NUM_NODES=$NUM_NODES NUM_TRAINERS=$NUM_TRAINERS"
```

---

## Run pipeline

### Phase A — Target cache (node 0 only, ~5 min)

```bash
cd /workspace/tml-assignment2
chmod +x scripts/cluster/*.sh

STEP=target bash scripts/cluster/runpod_instant_cluster_master.sh
python scripts/master_status.py
```

Imports existing `target_logits_40k.pt` if present; adds `train_unused` + `test10k` logits.

---

### Phase B — Parallel extract (all GPUs)

#### Option 1 — Two nodes, one process per node (simple)

**Node 0:**
```bash
STEP=extract WORKER_INDEX=0 NUM_WORKERS=2 WORKER_NAME=node0 \
  bash scripts/cluster/runpod_instant_cluster_master.sh
```

**Node 1:**
```bash
STEP=extract WORKER_INDEX=1 NUM_WORKERS=2 WORKER_NAME=node1 \
  bash scripts/cluster/runpod_instant_cluster_master.sh
```

#### Option 2 — One node with 8 GPUs (faster)

```bash
NUM_NODES=1 NODE_RANK=0 NUM_LOCAL_WORKERS=8 \
  bash scripts/cluster/launch_workers_per_node.sh

# Monitor:
tail -f runlogs/extract_w*.log
python scripts/master_status.py
```

#### Option 3 — 2 nodes × 8 GPUs = 16 parallel workers

On **each node**:

```bash
export NUM_NODES=2
export NUM_WORKERS=16
# NODE_RANK is 0 on node0, 1 on node1
bash scripts/cluster/launch_workers_per_node.sh
```

(`launch_workers_per_node.sh` sets `WORKER_INDEX = NODE_RANK * 8 + local_gpu`)

**Resume:** stop cluster anytime → redeploy → rerun same `STEP=extract` — skips completed `suspect_XXX/` folders.

---

### Phase C — Score variants (node 0, CPU, ~2 min)

When `master_status.py` shows **360/360 complete**:

```bash
STEP=score bash scripts/cluster/runpod_instant_cluster_master.sh
```

Download from volume or Jupyter:

- `results/submissions/submission_master_BEST_rank_fusion_multidist.csv`

---

### Phase D — Submit from your PC

```powershell
scp root@...:/workspace/tml-assignment2/results/submissions/submission_master_BEST_rank_fusion_multidist.csv `
  "...\results\submissions\"

python submission.py results/submissions/submission_master_BEST_rank_fusion_multidist.csv
```

---

## Cost control

| Do | Don't |
|----|--------|
| **2 nodes**, 1–2 GPUs each | 15 nodes / 120 GPUs |
| **Terminate cluster** when 360/360 done | Leave cluster running overnight |
| Use **Network Volume** so you don't re-download 16 GB | Put weights only on container disk |
| **2 LB submits max** after score step | Grid-search on public leaderboard |

**Rough time (2 nodes, 1 GPU each):** ~1–1.5 h extract + 5 min target + 2 min score.

**Rough time (1 node, 8 GPUs):** ~20–30 min extract.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `config.json missing` | Run `STEP=target` on node 0 first |
| Two workers same suspect | Check `NUM_WORKERS` / `WORKER_INDEX`; claims in `cache_master/claims/` |
| CUDA OOM | `--batch-size 64` |
| CIFAR download slow | First job downloads to `data/cifar100/` on volume |
| Nodes don't share files | Network Volume not mounted on **all** nodes |
| NCCL / distributed errors | **Ignore** — you are not doing distributed training; only independent Python workers |

---

## vs single RunPod Pod

| | Single Pod (what you used for v004) | Instant Cluster + Volume |
|--|-------------------------------------|---------------------------|
| Parallelism | 1 GPU, sequential 360 | Many GPUs, parallel |
| Resume | per-suspect JSON (v002) or cache_master dirs | `cache_master/suspect_XXX/` |
| Cost | ~$10, ~2 h | Higher if huge cluster; **cheap if 2 nodes** |
| Best for | Quick 40k logit only | **Master cache v2** full pipeline |

---

## Quick checklist

- [ ] Network Volume created & mounted
- [ ] Models + repo on volume
- [ ] Cluster **2 nodes** (or 1 node × 8 GPU), not 15
- [ ] `STEP=target` on node 0
- [ ] `STEP=extract` on all workers with correct `WORKER_INDEX` / `NUM_WORKERS`
- [ ] `master_status.py` → 360/360
- [ ] `STEP=score` → download CSV → submit once
- [ ] **Terminate cluster**

See also: `docs/CACHE_MASTER.md` (what is cached & scoring formulas).
