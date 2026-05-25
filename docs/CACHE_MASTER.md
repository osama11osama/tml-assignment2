# Master cache v2 — multi-distribution forensic pipeline

One **resumable** GPU run caches logits on **three probe distributions**.
After that, **all CSV variants are CPU-only** (~2 min).

## Why v2? (your v004 result)

| Finding | Meaning |
|---------|---------|
| v002 1k vs v004 40k: Spearman **0.99**, top-18 **identical** | More **same** train probes = saturated |
| v003 vs v004: top-18 overlap **5/18** | **Different distributions** can change the cutoff |

**Do not replace** `train_main` 40k — **add** orthogonal probes:

| Probe set | Size | Source | Role |
|-----------|------|--------|------|
| **A — victim train** | 40,000 | `data/train_main_idx.json` | Memorization / ownership fingerprint |
| **B — victim non-train** | ~10,000 | CIFAR-100 train **minus** train_main | Same domain, **unseen by victim** |
| **C — official test** | 10,000 | CIFAR-100 `train=False` | Generalization fingerprint |
| **D — aug delta** | 256×4 views | Fixed-seed aug on train subset | Local decision geometry |

**No new download** beyond CIFAR-100 (already in `data/cifar100/`).
Unused train indices auto-cached to `data/train_unused_idx.json` on first run.

---

## Behavioral fingerprint profile (per suspect)

```
F(S) = [
  sim_train40k,           plain_cosine_40k
  sim_unused10k,          plain_cosine_unused10k   ← NEW
  sim_test10k,            plain_cosine_test10k     ← NEW
  gap_train − test,       gap_train40k_minus_test10k
  gap_train − unused,     gap_train40k_minus_unused10k
  aug_delta,              aug_delta_cosine
  top5_test,              top5_agreement_test10k
  … + weighted/trimmed/JS on 40k
]
```

**Final submit candidate:** `rank_fusion_multidist` (fixed weights, no LB tuning).

---

## Cache layout

```
results/cache_master/
  config.json                     # v2 probe sizes
  target/
    train40k_logits.fp16.pt       # [40000, 100] — import legacy OK
    train_unused_logits.fp16.pt   # [~10000, 100]  NEW
    test10k_logits.fp16.pt        # [10000, 100]    NEW
    train4k_logits.fp16.pt        # layer4 + legacy
    test2k_logits.fp16.pt
    aug_inputs.fp16.pt
    aug_*_logits.fp16.pt
    manifest.json                 # per-stage timestamps
  suspects/suspect_XXX/           # same files per suspect
  claims/                         # parallel worker locks
  variants/                       # analysis + comparison JSON
```

**Resume:** each `.fp16.pt` is one stage. Stop anytime → re-run → skips finished stages/suspects.

---

## Split 4 GPUs (1 local + 3 RunPod) — **use this if no cheap cluster**

No Instant Cluster needed. See **[SPLIT_4GPU.md](SPLIT_4GPU.md)**.

| Worker | Where | Command |
|--------|-------|---------|
| 0 | PC | `scripts/cluster/run_local_worker0.ps1` |
| 1–3 | RunPod Pod (3 GPUs) | `scripts/cluster/runpod_workers_1_2_3.sh` |
| Merge | PC | `scp` RunPod `suspects/*` → local |
| Score | PC | `master_score_variants.py` |

---

## RunPod Instant Cluster (optional — often H100-only / expensive)

Full guide: **[RUNPOD_INSTANT_CLUSTER.md](RUNPOD_INSTANT_CLUSTER.md)**

- Attach a **Network Volume** (~80 GB) — shared `cache_master/` across nodes
- Use **2 nodes** (not 15×8 GPUs) + `scripts/cluster/runpod_instant_cluster_master.sh`
- Phases: `STEP=target` → `STEP=extract` (parallel) → `STEP=score` (CPU)

---

## Procedure (next run)

### Step 0 — What you already have

| Asset | Status |
|-------|--------|
| `target/train40k_logits` | ✅ imported from `target_logits_40k.pt` |
| `target/train4k`, `test2k`, `aug_*` | ✅ from v1 precompute |
| `target/train_unused`, `test10k` | ❌ run Step 1 to add |
| `suspects/*` | ❌ not started (0/360) |

Existing v1 target files are **kept**; only **missing** stages are computed.

---

### Step 1 — Update target cache (~3 min GPU)

Adds unused-10k + full test-10k target logits (skips existing 40k/4k/aug):

```powershell
python scripts/master_precompute_target.py --device cuda --import-legacy-40k
```

This also writes `data/train_unused_idx.json` if missing.

---

### Step 2 — Suspect extract (resumable, ~2–3 h per GPU for half)

**One GPU per machine.** Split across local + RunPod:

```powershell
# Local RTX 5060 — even suspects 0,2,4,…
python scripts/master_extract.py --device cuda --worker-index 0 --num-workers 2 --worker-name local-5060

# RunPod RTX 3090 — odd suspects 1,3,5,…
python scripts/master_extract.py --device cuda --worker-index 1 --num-workers 2 --worker-name runpod-3090
```

**Controlling parallel jobs:**

| Setup | Command |
|-------|---------|
| 1 GPU only | `--num-workers 1 --worker-index 0` |
| Local + RunPod | `--num-workers 2`, index `0` vs `1` |
| 3 cloud pods | `--num-workers 3`, index `0`, `1`, `2` |
| **Never** | 2 extract scripts on **same** GPU |

Sync `results/cache_master/suspects/` between machines (copy **missing** folders only).

```powershell
python scripts/master_status.py
python scripts/master_status.py --suspect 42
```

---

### Step 3 — All CSV variants (CPU only)

```powershell
python scripts/master_score_variants.py
python submission.py --validate-only results/submissions/submission_master_BEST_rank_fusion_multidist.csv
```

Key outputs:

| CSV | Description |
|-----|-------------|
| `submission_master_plain_cosine_40k.csv` | v004 equivalent |
| `submission_master_plain_cosine_test10k.csv` | test-only similarity |
| `submission_master_plain_cosine_unused10k.csv` | held-out train similarity |
| `submission_master_gap_train40k_minus_test10k.csv` | lineage gap signal |
| `submission_master_BEST_rank_fusion_multidist.csv` | **recommended submit** |
| `results/cache_master/variants/variant_comparison.json` | top-18 overlap vs 40k baseline |

---

### Step 4 — Submit (max 2–3, respect 429 cooldown)

1. `submission_master_BEST_rank_fusion_multidist.csv`
2. Best single variant with **lowest top-18 overlap** vs plain 40k (from JSON)
3. Optional: keep v004 as fallback

---

## Scoring variants (all from cache, zero extra GPU)

| Variant | Formula |
|---------|---------|
| `plain_cosine_40k` | mean cos(z_T, z_S) on victim 40k |
| `plain_cosine_unused10k` | mean cos on **10k train not in train_main** |
| `plain_cosine_test10k` | mean cos on CIFAR-100 test |
| `gap_train40k_minus_test10k` | sim_train40k − sim_test10k |
| `gap_train40k_minus_unused10k` | sim_train40k − sim_unused10k |
| `conf_weighted_cosine_40k` | victim-confidence weighted |
| `trimmed_cosine_90_40k` | top 90% per-image cos |
| `neg_js_T4_40k` | −JS(softmax z/T) |
| `top5_agreement_test10k` | top-5 class overlap on test |
| `aug_delta_cosine` | cos(Δz_T, Δz_S) under fixed aug |
| `rank_fusion_multidist` | fixed multi-probe rank fusion |

---

## Where the data comes from (no exotic OOD)

```python
# A — already have
train_main_idx.json  → 40k victim train indices

# B — auto-generated
train_unused_idx.json  → 50_000 CIFAR train − train_main  ≈ 10k indices

# C — torchvision
CIFAR100(root="data/cifar100", train=False)  → 10k test images

# Same normalization as task_template.py (already in get_eval_transform())
```

**Skip for now:** CIFAR-10, TinyImageNet, noise/MNIST OOD — lower ROI before deadline.

---

## Disk / time estimate (v2)

| Item | Estimate |
|------|----------|
| Per suspect | ~20–30 MB (40k + 10k + 10k + aug + 4k) |
| Total 360 | ~8–12 GB |
| Target update | ~3 min GPU |
| Per suspect extract | ~25–35 s (RTX 5060) |
| 180 suspects / GPU | ~1.5–2 h |

---

## Relation to previous pipelines

| Old | v2 master cache |
|-----|-----------------|
| v004 (40k logit only) | `plain_cosine_40k` variant |
| v003 (4k+2k ensemble) | superseded by `rank_fusion_multidist` after cache |
| `forensic_extract.py` | optional; master cache is the main path |

---

## Report ablation table (fill after Step 3)

| Method | Probes | Public TPR |
|--------|--------|------------|
| v001 weight cosine | — | 0.000 |
| v002 logit 1k train | 1k | 0.537 |
| v004 logit 40k train | 40k | 0.537 |
| v003 forensic ensemble | 4k+2k | 0.537 |
| plain_cosine_test10k | 10k test | TBD |
| plain_cosine_unused10k | 10k held-out train | TBD |
| **rank_fusion_multidist** | 40k+10k+10k+aug | TBD |
