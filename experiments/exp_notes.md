# Experiment notes

## v001 — Weight cosine similarity

**Hypothesis:** Stolen models (copies, fine-tunes) have weight vectors close to the target.

**Method:** Flatten all safetensor weights → cosine similarity with target → use as `score`.

**Script:** `python scripts/score_models.py`

**Output:** `results/submissions/submission_v001_weight_cosine.csv`

**Leaderboard:** TPR @ 5% FPR = **0.0** (no usable ranking; scores ~1.0).

**Limitations:** May miss distilled/quantized models with altered weights but similar behavior.

---

## v002 — Logit similarity on train_main_idx ✅ SUBMITTED

**Hypothesis:** Stolen/derived models produce similar logits on the target's training images (`train_main_idx.json`).

**Method:** Precompute target logits on probe set → per suspect mean cosine similarity of per-image logits.

**Reproduce (exact run that produced the tagged CSV):**

```powershell
python scripts/score_logit_similarity.py --subset 1000 --device cuda --batch-size 64
python submission.py --validate-only results/submissions/submission_v002_logit_train.csv
python submission.py results/submissions/submission_v002_logit_train.csv
```

**Output:** `results/submissions/submission_v002_logit_train.csv`  
**Score stats:** min 0.120, max 1.0, mean 0.760, median 0.886

**Leaderboard (team_XLVII):** public TPR @ 5% FPR = **0.537037**  
**Git tag:** `v0.2-logit-0.537`  
**Full record:** `experiments/submissions/v002_team_XLVII.json`

**Cluster alternative (full 40k):** `docs/CLUSTER.md` — precompute + 360 jobs + `merge_scores.py`

---

## v003 — Forensic ensemble (multi-stage)

**Stages:** weights + behavioral (train_main + test) + CKA activations + augmentation fingerprint.  
**Ensemble:** rank-weighted fusion (no LB tuning; anti-overfit).  
**Scripts:** `forensic_precompute_target.py` → `forensic_extract.py` → `forensic_ensemble.py`  
**Docs:** `docs/FORENSIC_STRATEGY.md`

---
