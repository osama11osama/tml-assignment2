# What to submit (course checklist)

Use this list at the end of the assignment. **Only items in this document go to the course** — not the whole GitHub repo.

---

## 1. Leaderboard submission (required)

| Item | Location in repo | How to submit |
|------|------------------|---------------|
| **CSV** | `results/submissions/submission_vXXX_*.csv` | Course server via `python submission.py` |

Format: 360 rows, columns `id` (0–359), `score` (numeric).

```powershell
python submission.py results/submissions/your_best.csv
```

---

## 2. Written report (required)

| Item | Location | Format |
|------|----------|--------|
| **Report** | `report/paper.tex` → compile to PDF | PDF upload per course instructions |

Include: methods, experiments, results (TPR@5%FPR), discussion.

---

## 3. Code archive (if required by course)

Submit **repo root contents only** — not `_private/`:

```
submission.py
task_template.py
requirements.txt
README.md
src/
scripts/
experiments/
results/submissions/
data/train_main_idx.json
report/          (sources + figures)
docs/
```

**Do NOT include:**
- `_private/` (local tools and notes)
- `.env`, API keys
- Model weights (`target_model/`, `suspect_models/`)
- `.venv/`, `.cache/`

---

## 4. GitHub vs course submission

| | GitHub (`osama11osama/tml-assignment2`) | Course platform |
|--|----------------------------------------|-----------------|
| Purpose | Version control, reproducible code | Grading |
| Includes `_private/` | **No** (gitignored) | **No** |
| Includes model weights | **No** | **No** |
| Includes `results/submissions/` | Yes | Upload best CSV only |

---

## Quick pre-flight

```powershell
python submission.py --validate-only results/submissions/your_best.csv
git status   # confirm _private/ and .env are NOT staged
```
