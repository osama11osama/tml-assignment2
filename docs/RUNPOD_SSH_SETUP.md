# RunPod SSH setup (Windows) — before master cache run

Use this **once** on your PC, then connect with `ssh runpod-a2` for the whole assignment.

**Your plan:** 1 node × 8 GPUs + Network Volume → SSH + `tmux` + `launch_workers_per_node.sh`

---

## What to enable on RunPod (when creating cluster/pod)

| Setting | Why |
|---------|-----|
| **Network Volume** (~80 GB) | Persistent models + `cache_master/` |
| Mount path | `/workspace` |
| **Expose TCP port 22** (or use template with SSH) | Full SSH + **SCP** from Windows |
| **Public IP** | Required for direct SSH/SCP (not only `ssh.runpod.io` proxy) |
| Template | **RunPod PyTorch** (SSH daemon pre-installed) |

In the template / deploy UI, enable **SSH terminal access** / **Public IP** if there is a checkbox.

---

## Step 1 — Generate SSH key (PowerShell, on your PC)

```powershell
# Only if you don't already have a key:
ssh-keygen -t ed25519 -C "your-email@example.com" -f "$env:USERPROFILE\.ssh\id_ed25519_runpod"
```

Press Enter for empty passphrase (or set one — you'll type it each connect).

Show **public** key (copy entire line):

```powershell
Get-Content "$env:USERPROFILE\.ssh\id_ed25519_runpod.pub"
```

Starts with `ssh-ed25519 AAAA...`

---

## Step 2 — Add key to RunPod account

1. Open [RunPod Settings → SSH Public Keys](https://www.console.runpod.io/user/settings)
2. Paste the **full line** from `.pub` (not the SHA256 fingerprint)
3. **One key per line** if you add multiple keys
4. Save

---

## Step 3 — Deploy cluster, get SSH command

1. Create **Instant Cluster** (1 node, 8 GPUs, Network Volume at `/workspace`)
2. Wait until status = **Running**
3. Open the node → **Connect** tab
4. Copy **both** commands if shown:
   - **SSH** (proxied via `ssh.runpod.io`) — quick test
   - **SSH over exposed TCP** — use this for **SCP** and stable sessions

Example (yours will differ):

```text
ssh root@213.173.108.12 -p 17445 -i ~/.ssh/id_ed25519
```

---

## Step 4 — SSH config on Windows (recommended)

Create or edit `C:\Users\Osama\.ssh\config` (replace IP/port after deploy):

```sshconfig
Host runpod-a2
    HostName 213.173.108.12
    Port 17445
    User root
    IdentityFile C:/Users/Osama/.ssh/id_ed25519_runpod
    StrictHostKeyChecking accept-new
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

**After each new cluster:** update `HostName` and `Port` from the Connect tab.

Connect:

```powershell
ssh runpod-a2
```

---

## Step 5 — Test connection

```powershell
ssh runpod-a2 "echo OK && nvidia-smi -L"
```

You should see 8 GPU lines.

Check volume mount:

```powershell
ssh runpod-a2 "ls -la /workspace && df -h /workspace"
```

---

## Step 6 — SCP: copy code from PC (first time only)

From **PowerShell on your PC** (not inside SSH):

```powershell
$LOCAL = "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment2"
$RP = "runpod-a2"   # Host from ~/.ssh/config — not $Host (PowerShell reserved)

# Code only (no 16GB weights — download on pod)
scp -r "$LOCAL\src" "${RP}:/workspace/tml-assignment2/"
scp -r "$LOCAL\scripts" "${RP}:/workspace/tml-assignment2/"
scp -r "$LOCAL\data" "${RP}:/workspace/tml-assignment2/"
scp "$LOCAL\requirements.txt" "$LOCAL\submission.py" "$LOCAL\task_template.py" "${RP}:/workspace/tml-assignment2/"
scp -r "$LOCAL\docs" "${RP}:/workspace/tml-assignment2/"
```

If repo already on volume via `git clone`, skip SCP and only `git pull` over SSH.

Optional — copy **target 40k cache** from PC (saves ~5 min GPU):

```powershell
scp "$LOCAL\results\cache\target_logits_40k.pt" "${RP}:/workspace/tml-assignment2/results/cache/"
```

---

## Step 7 — `tmux` on the pod (survive disconnect)

```bash
ssh runpod-a2

apt-get update && apt-get install -y tmux   # if missing

tmux new -s a2master
cd /workspace/tml-assignment2
```

Detach (SSH can close, jobs keep running): `Ctrl+B` then `D`  
Reattach: `ssh runpod-a2` → `tmux attach -t a2master`

---

## Step 8 — One-time setup on pod (inside tmux)

```bash
cd /workspace/tml-assignment2
pip install -r requirements.txt
chmod +x scripts/cluster/*.sh

# Models on volume (once)
python scripts/download_models.py --target-only
python scripts/download_models.py

# Target cache (once, ~5 min)
STEP=target bash scripts/cluster/runpod_instant_cluster_master.sh
```

---

## Step 9 — Run 8 GPU workers (inside tmux)

```bash
cd /workspace/tml-assignment2
export NUM_NODES=1 NODE_RANK=0 NUM_LOCAL_WORKERS=8 NUM_WORKERS=8

bash scripts/cluster/launch_workers_per_node.sh

# Watch progress
tail -f runlogs/extract_w*.log
python scripts/master_status.py
```

When **360/360**:

```bash
STEP=score bash scripts/cluster/runpod_instant_cluster_master.sh
```

---

## Step 10 — Download results to PC

```powershell
scp runpod-a2:/workspace/tml-assignment2/results/submissions/submission_master_BEST_rank_fusion_multidist.csv `
  "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment2\results\submissions\"
```

Optional — backup full cache:

```powershell
scp -r runpod-a2:/workspace/tml-assignment2/results/cache_master/ `
  "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment2\results\"
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Permission denied (publickey)` | Wrong key path in config; key not in RunPod settings |
| Asks for **password** | Public key not pasted correctly; use `-i` path to **private** key |
| `bad permissions` on key | `icacls` fix below |
| `Connection refused` | Pod not running; wrong port; SSH not started |
| SCP fails, SSH works | Use **TCP** SSH command, not `ssh.runpod.io` proxy |
| Session dies overnight | Use **tmux**; set `ServerAliveInterval 60` in config |

**Windows key permissions:**

```powershell
icacls "$env:USERPROFILE\.ssh\id_ed25519_runpod" /inheritance:r
icacls "$env:USERPROFILE\.ssh\id_ed25519_runpod" /grant:r "$env:USERNAME:(R)"
```

---

## Quick checklist before the run

- [ ] SSH key generated + added to RunPod settings
- [ ] `~/.ssh/config` Host `runpod-a2` with correct IP/port
- [ ] `ssh runpod-a2 "nvidia-smi -L"` shows **8** GPUs
- [ ] `/workspace` has Network Volume (models persist)
- [ ] `tmux` session started
- [ ] `STEP=target` done
- [ ] `launch_workers_per_node.sh` running
- [ ] **Terminate cluster** when done (stop billing)

Next: [RUNPOD_INSTANT_CLUSTER.md](RUNPOD_INSTANT_CLUSTER.md) · Pipeline: [CACHE_MASTER.md](CACHE_MASTER.md)
