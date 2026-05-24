# Trustworthy ML 2026 — Assignment 2: Stolen Model Detection

Detect which of **360 suspect** ResNet-18 models were stolen or derived from a target model (CIFAR-100).

**Metric:** TPR @ 5% FPR  
**Task ID:** `19-stolen-model-detection`

---

## Repository layout

```
tml-assignment2/
├── README.md              ← this file
├── SUBMIT.md              ← checklist: what to hand in to the course
├── submission.py          ← validate + upload CSV
├── task_template.py       ← official model loading example
├── requirements.txt
├── src/                   ← model utilities
├── scripts/               ← download, score, validate pipeline
├── experiments/           ← experiment notes
├── results/submissions/   ← generated CSV files
├── data/                  ← metadata (weights downloaded locally)
├── docs/                  ← method documentation
└── report/                ← LaTeX report (final PDF)
```

**Local-only (not in git):** `_private/` — personal tools and notes. See `SUBMIT.md`.

---

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Download models (~16 GB) — run locally, not committed
python scripts/download_models.py --target-only
python scripts/download_models.py

# Score and submit
python scripts/score_models.py
python submission.py
```

---

## Methods

| Version | Method | Script |
|---------|--------|--------|
| v001 | Weight cosine similarity | `scripts/score_models.py` |
| v002 | Logit similarity (train_main_idx) | `scripts/score_logit_similarity.py` |
| v003 | **Forensic ensemble** (multi-stage) | `scripts/forensic_*.py` |

### Forensic pipeline (recommended for max score)

```powershell
pip install -r requirements.txt

# 1) Target reference cache (GPU)
python scripts/forensic_precompute_target.py --probe-train 4000 --probe-test 2000 --device cuda

# 2) Per-suspect features — resumable (GPU, ~hours for 360)
python scripts/forensic_extract.py --device cuda --probe-train 4000 --probe-test 2000

# 3) Ensemble → CSV
python scripts/forensic_ensemble.py --output submission_v003_forensic.csv
python submission.py --validate-only results/submissions/submission_v003_forensic.csv
```

Design: [docs/FORENSIC_STRATEGY.md](docs/FORENSIC_STRATEGY.md)

**Cluster (full 40k × 360):** see [docs/CLUSTER.md](docs/CLUSTER.md)

See `experiments/exp_notes.md` and `docs/METHOD.md`.

### Releases (leaderboard snapshots)

| Git tag | Method | Public TPR@5%FPR |
|---------|--------|------------------|
| `v0.2-logit-0.537` | v002 logit cosine, 1k `train_main` probes | **0.537037** |

Reproduce: `git checkout v0.2-logit-0.537` then follow `experiments/submissions/v002_team_XLVII.json`.

---

## Data

Models: [SprintML/tml26_task2](https://huggingface.co/SprintML/tml26_task2)  
Download via `scripts/download_models.py` — weights stay local (gitignored).

---

## API key

Set `TML_API_KEY` in `.env` (32-char hash from CMS). **Never commit `.env`.**
