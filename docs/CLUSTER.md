# HPC cluster workflow (Assignment 2)

**Preferred:** download weights on the cluster (wget). Copy **only code** from your PC (~few MB).

Replace `atml_team044` with your CMS cluster username.

---

## Step 1 — On SSH only: create folders + download models

```bash
mkdir -p ~/tml26_task2/{target_model,suspect_models,data,results/cache,results/cluster_scores,results/submissions,runlogs,condor,src,scripts}
cd ~/tml26_task2

HF="https://huggingface.co/SprintML/tml26_task2/resolve/main"

# Target + indices (~45 MB + small json)
wget -nc "$HF/target_model/weights.safetensors" -O target_model/weights.safetensors
wget -nc "$HF/target_model/train_main_idx.json" -O data/train_main_idx.json
# (NOT data/train_main_idx.json on HuggingFace)

# All 360 suspects (~16 GB total — run in screen/tmux, takes hours)
for i in $(seq -f "%03g" 0 359); do
  f="suspect_models/suspect_${i}.safetensors"
  [[ -f "$f" ]] && continue
  echo "Downloading $f ..."
  wget -q "$HF/suspect_models/suspect_${i}.safetensors" -O "$f"
done

ls suspect_models/*.safetensors | wc -l   # should print 360
```

**Tip:** keep SSH alive during long download:

```bash
screen -S download
# paste loop above
# Ctrl+A then D to detach;  screen -r download  to reattach
```

Or copy script from PC once: `_private/setup/cluster_download_all.sh` → run `bash cluster_download_all.sh`

---

## Step 2 — Code only from your PC (small scp)

On **Windows PowerShell** (not inside SSH):

```powershell
$LOCAL = "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment2"
$REMOTE = "atml_team044@conduit2.hpc.uni-saarland.de"

scp -r "$LOCAL\src" "$LOCAL\scripts" "${REMOTE}:~/tml26_task2/"
scp "$LOCAL\scripts\cluster\condor\precompute_target.sub" "${REMOTE}:~/tml26_task2/condor/"
scp "$LOCAL\scripts\cluster\condor\score_suspect.sub" "${REMOTE}:~/tml26_task2/condor/"
# Important: Saarland uses universe=docker (NOT vanilla). Files in scripts/cluster/condor/
```

**Alternative (no scp):** clone your GitHub repo on cluster (code only, no weights in git):

```bash
cd ~
git clone https://github.com/osama11osama/tml-assignment2.git tml26_task2_code
cp -r tml26_task2_code/src tml26_task2_code/scripts ~/tml26_task2/
# still wget weights into ~/tml26_task2/ as in step 1
```

---

## Step 3 — Submit GPU jobs (SSH)

```bash
cd ~/tml26_task2
mkdir -p runlogs

condor_submit condor/precompute_target.sub
condor_q
# wait until:
ls -lh results/cache/target_logits_40k.pt

condor_submit condor/score_suspect.sub
condor_q

/opt/conda/bin/python scripts/cluster/merge_scores.py --output submission_v002_logit_40k.csv
```

CIFAR-100 auto-downloads to `data/cifar100/` on first job.

---

## Step 4 — Pull CSV to PC

```powershell
scp atml_team044@conduit2.hpc.uni-saarland.de:~/tml26_task2/results/submissions/submission_v002_logit_train.csv `
  "C:\Users\Osama\Master\SS2026\01_Trustworthy Machine Learning\Assignments\Assignment2\results\submissions\"
```

```powershell
python submission.py --validate-only results/submissions/submission_v002_logit_train.csv
```

---

## What to download vs copy

| Item | On cluster | From PC |
|------|------------|---------|
| Target + 360 suspects | wget HuggingFace | skip |
| train_main_idx.json | wget | skip |
| src/, scripts/cluster/ | — | scp or git clone |
| condor/*.sub | — | scp once |
| _private/, .env | never | never |
