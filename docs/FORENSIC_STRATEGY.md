# Forensic ensemble strategy (Assignment 2)

## Critical evaluation (competition engineer view)

### What actually works on this task

| Signal family | Private-set expectation | Cost | v001 evidence |
|---------------|-------------------------|------|---------------|
| **Behavioral (logits)** | **High** — distillation, knockoffs | Medium GPU | v002: 0.12–1.0 spread |
| **Representation (activations)** | **High** — fine-tunes, partial copies | Medium GPU | Theory + literature |
| **Augmentation fingerprint** | **Medium–High** — recipe known | Medium | Assignment-specific |
| **Layer-wise weights** | **Low–Medium** — not alone | Low CPU | v001 global: **failed** |
| **Adversarial transfer** | Medium — noisy, expensive | High | Optional |
| **XGBoost on pseudo-labels** | **Risky** — overfits public 30% | Low | **Avoid as primary** |

### Stronger approach than “blind checklist”

1. **Multi-probe generalization** — `train_main_idx` (victim training set) + **CIFAR-100 test** (different distribution). Private 70% benefits from probes that are not all in-distribution.
2. **Rank fusion, not raw score sum** — TPR@5%FPR is rank-based; per-feature **rank normalization** across 360 suspects is robust without labels.
3. **No pseudo-label training on public LB** — weights are fixed from theory + ablation on **local** holdout (e.g. score stability across probes), not iterative LB tuning.
4. **CKA approximated** — linear CKA on pooled activations per layer (batched); full SVCCA only if needed.
5. **Skip adversarial by default** — enable with `--adversarial` after baseline ensemble works.

### Anti-overfitting rules (enforced in code)

- Default ensemble: **rank-weighted fusion** with fixed weights (behavioral > repr > aug > weights).
- Optional `--isolation-forest` as secondary signal (unsupervised), not LB-tuned.
- Document every submission variant in `experiments/exp_notes.md`.
- Do **not** grid-search weights using public leaderboard more than 1–2 times.

---

## Pipeline architecture

```
                    ┌─────────────────────┐
                    │  Target reference   │
                    │  (once, cached)     │
                    └──────────┬──────────┘
                               │
     ┌─────────────────────────┼─────────────────────────┐
     ▼                         ▼                         ▼
 Stage 1              Stage 2                 Stage 3–4
 weights              behavioral              repr + aug
 (CPU)                (GPU, logits)           (GPU, hooks)
     │                         │                         │
     └─────────────────────────┴─────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Per-suspect vector │
                    │  → parquet cache    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Rank ensemble      │
                    │  → submission.csv   │
                    └─────────────────────┘
```

---

## CLI workflow

```powershell
# 1) Precompute target caches (GPU)
python scripts/forensic_precompute_target.py --probe-train 4000 --probe-test 2000

# 2) Extract per-suspect features (resumable)
python scripts/forensic_extract.py --device cuda --probe-train 4000 --probe-test 2000

# 3) Build submission
python scripts/forensic_ensemble.py --output submission_v003_forensic.csv
```

See `README.md` for full options.
