# Method overview

## Problem

Given one **target** model and **360 suspect** models (same architecture, CIFAR-100), assign a score to each suspect: higher = more likely stolen.

## Baseline v001 — weight cosine similarity

1. Flatten all weights from target and each suspect (safetensors).
2. Compute cosine similarity between weight vectors.
3. Use similarity as the submission score.

**Rationale:** Direct copies and fine-tunes retain similar weight vectors.

**Limitation:** Distilled or quantized models may differ in weights while behaving similarly → v002 will use logit comparison.

## Evaluation

Server metric: **TPR @ 5% FPR** — fraction of stolen models detected when false-positive rate is capped at 5%.

## Reproducing v001

```powershell
python scripts/score_models.py
python submission.py --validate-only
```

Output: `results/submissions/submission_v001_weight_cosine.csv`
