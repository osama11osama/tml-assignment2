# Method overview

## Problem

Given one **target** model and **360 suspect** models (same architecture, CIFAR-100), assign a score to each suspect: higher = more likely stolen.

## Baseline v001 — weight cosine similarity

1. Flatten all weights from target and each suspect (safetensors).
2. Compute cosine similarity between weight vectors.
3. Use similarity as the submission score.

**Rationale:** Direct copies and fine-tunes retain similar weight vectors.

**Limitation:** Distilled or quantized models may differ in weights while behaving similarly → use v002 logit comparison.

## v002 — logit cosine similarity (primary method)

1. Load probe images from `train_main_idx.json` (CIFAR-100 train, eval transform).
2. Precompute target logits on probe set; cache to `results/cache/target_logits_40k.pt`.
3. For each suspect: forward on same images → mean per-image cosine similarity vs target logits.
4. Higher score = more similar behavior = more likely stolen.

**Local smoke test:**
```powershell
python scripts/score_logit_similarity.py --subset 256 --suspects 0,1
```

**Full run (GPU recommended):**
```powershell
python scripts/score_logit_similarity.py --device cuda
```

**Cluster:** see `docs/CLUSTER.md`.

## Evaluation

Server metric: **TPR @ 5% FPR** — fraction of stolen models detected when false-positive rate is capped at 5%.

## Reproducing v001

```powershell
python scripts/score_models.py
python submission.py --validate-only
```

Output: `results/submissions/submission_v001_weight_cosine.csv`
