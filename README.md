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
| v002 | Logit similarity (planned) | TBD |

See `experiments/exp_notes.md` and `docs/METHOD.md`.

---

## Data

Models: [SprintML/tml26_task2](https://huggingface.co/SprintML/tml26_task2)  
Download via `scripts/download_models.py` — weights stay local (gitignored).

---

## API key

Set `TML_API_KEY` in `.env` (32-char hash from CMS). **Never commit `.env`.**
