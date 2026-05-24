# Experiment notes

## v001 — Weight cosine similarity

**Hypothesis:** Stolen models (copies, fine-tunes) have weight vectors close to the target.

**Method:** Flatten all safetensor weights → cosine similarity with target → use as `score`.

**Script:** `python scripts/score_models.py`

**Output:** `results/submissions/submission_v001_weight_cosine.csv`

**Limitations:** May miss distilled/quantized models with altered weights but similar behavior.

---

## v002 — Logit similarity (planned)

Compare model outputs on shared CIFAR-100 images.

---
