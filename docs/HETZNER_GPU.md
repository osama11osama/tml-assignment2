# Hetzner GPU — run master cache (alternative to RunPod Cluster)

**Yes, Hetzner works** — but as **one Linux server with GPU(s) + SSH**, not as a managed “Instant Cluster.”

Same scripts as RunPod Pod: `master_precompute_target.py`, `master_extract.py`, `launch_workers_per_node.sh`.

---

## Hetzner vs RunPod Cluster

| | RunPod Instant Cluster | Hetzner GPU server |
|--|------------------------|-------------------|
| Setup | Click deploy | Rent server, install CUDA/PyTorch (or use image) |
| Multi-node | 2+ nodes, often H100 only | Usually **1 machine** |
| Cost | H100 cluster ≈ **$60+/hr** | Dedicated GPU ≈ **€1–3/hr** (model-dependent) |
| Volume | Network Volume | Local NVMe (+ optional Hetzner Volume) |
| Your scripts | ✅ | ✅ (identical SSH workflow) |

---

## What to rent on Hetzner

1. [Hetzner GPU servers](https://www.hetzner.com/dedicated-rootserver/matrix-gpu/) or **Cloud** GPU types (if in stock in your project)
2. Look for:
   - **1× RTX 4000 Ada / RTX 4090 / RTX 3090** — enough (slower than 8× parallel)
   - **Multi-GPU dedicated** — only if you need `NUM_LOCAL_WORKERS=8`

Minimum disk: **~100 GB** (models 16 GB + cache 10 GB + CIFAR).

OS: **Ubuntu 22.04** + NVIDIA drivers (or use a CUDA-ready image if offered).

---

## Setup (once on server)

```bash
# SSH from PC (your existing key in ~/.ssh/authorized_keys on server)
ssh root@YOUR_HETZNER_IP

apt update && apt install -y git tmux python3-pip python3-venv

# NVIDIA — if not preinstalled, follow Hetzner docs for your server type
nvidia-smi

cd /root  # or /mnt/volume
git clone https://github.com/osama11osama/tml-assignment2.git
cd tml-assignment2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
chmod +x scripts/cluster/*.sh

python scripts/download_models.py
```

---

## Run pipeline (same as RunPod)

```bash
tmux new -s a2master
cd /root/tml-assignment2   # adjust path
source .venv/bin/activate

# Optional: copy target 40k cache from PC
# scp results/cache/target_logits_40k.pt root@IP:.../results/cache/

STEP=target bash scripts/cluster/runpod_instant_cluster_master.sh

# 1 GPU:
STEP=extract WORKER_INDEX=0 NUM_WORKERS=1 bash scripts/cluster/runpod_instant_cluster_master.sh

# OR multi-GPU on one machine (if nvidia-smi shows 8 GPUs):
NUM_LOCAL_WORKERS=8 bash scripts/cluster/launch_workers_per_node.sh

python scripts/master_status.py
STEP=score bash scripts/cluster/runpod_instant_cluster_master.sh
```

Download CSV to PC:

```powershell
scp root@YOUR_HETZNER_IP:/root/tml-assignment2/results/submissions/submission_master_BEST_rank_fusion_multidist.csv `
  "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment2\results\submissions\"
```

---

## SSH config (Windows)

```sshconfig
Host hetzner-a2
    HostName YOUR_SERVER_IP
    User root
    IdentityFile C:/Users/Osama/.ssh/id_ed25519
    ServerAliveInterval 60
```

Add your **public key** in Hetzner robot/cloud → server → SSH keys (or paste into `~/.ssh/authorized_keys` on first login with password).

---

## Cost tip

- **Cancel/delete** server when `360/360` done — no “volume only” billing like RunPod
- **Snapshot** optional if you want to keep models between sessions

---

## Better options to try first (often free/cheaper)

| Option | When |
|--------|------|
| **RunPod Pod** (not Cluster) | 1× RTX 3090/4090 ~$0.5/hr — you already know this works |
| **Local RTX 5060** | Overnight `master_extract` |
| **Uni-Saarland HTCondor** | Free 360 parallel jobs — see `docs/CLUSTER.md` |

Use Hetzner if RunPod has no cheap GPU and course cluster SSH is down.
