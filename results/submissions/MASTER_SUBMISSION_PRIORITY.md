# Master Submission Priority Guide

This folder contains the full set of CPU-only submission CSVs produced from the completed master cache.

The files were renamed with priority prefixes so they sort naturally in the file explorer:

- `A_` = highest priority
- `B_` = next priority
- ...
- `T_` = lowest priority / mostly alias or specialist fallback

## How These CSVs Were Generated

- Source of truth: cluster project `~/tml26_task2`
- Cache status at scoring time: `360/360 suspects complete`
- Scoring entrypoint: `scripts/master_score_variants.py`
- Cluster wrapper used: `condor/run_master_score.sh`
- Condor submit file used: `condor/master_score.sub`
- Comparison report copied locally to: `results/cache_master/variants/variant_comparison.json`

All files in this folder are local copies of the generated cluster outputs.

## Priority Order

### A — `A_submission_master_BEST_rank_fusion_multidist.csv`
- Original name: `submission_master_BEST_rank_fusion_multidist.csv`
- Meaning: the main recommended final candidate.
- Formula: fixed multi-probe rank fusion over train-40k, test-10k, unused-train-10k, gap signals, augmentation delta, and top-5 test agreement.
- Why high priority: this is the explicitly recommended submit in the pipeline docs.

### B — `B_submission_master_BEST_rank_fusion.csv`
- Original name: `submission_master_BEST_rank_fusion.csv`
- Meaning: best train-heavy fusion fallback.
- Formula: default rank fusion over 40k-centered signals plus layer4 and augmentation.
- Why high priority: strongest simple fallback if the multidistribution fusion underperforms.

### C — `C_submission_master_neg_js_T2_40k.csv`
- Original name: `submission_master_neg_js_T2_40k.csv`
- Meaning: negative Jensen-Shannon divergence between softened target and suspect probabilities on train-40k with temperature 2.
- Why high priority: fairly strong correlation to baseline while still changing the top suspects more than most 40k-only variants.

### D — `D_submission_master_neg_js_T4_40k.csv`
- Original name: `submission_master_neg_js_T4_40k.csv`
- Meaning: negative Jensen-Shannon divergence on train-40k with temperature 4.
- Why high priority: another strong but more distribution-aware alternative to raw cosine.

### E — `E_submission_master_gap_train40k_minus_unused10k.csv`
- Original name: `submission_master_gap_train40k_minus_unused10k.csv`
- Meaning: train-40k similarity minus held-out-train similarity.
- Why high priority: highly orthogonal lineage-style signal; top-18 overlap with the plain 40k baseline was `0`.

### F — `F_submission_master_gap_train40k_minus_test10k.csv`
- Original name: `submission_master_gap_train40k_minus_test10k.csv`
- Meaning: train-40k similarity minus CIFAR-100 test similarity.
- Why high priority: another orthogonal lineage gap signal; top-18 overlap with the plain 40k baseline was `0`.

### G — `G_submission_master_train_test_gap_cosine.csv`
- Original name: `submission_master_train_test_gap_cosine.csv`
- Meaning: train-4k logits cosine minus test-2k logits cosine.
- Why high priority: broad train-vs-test gap fallback with `0` top-18 overlap against the plain 40k baseline.

### H — `H_submission_master_plain_cosine_40k.csv`
- Original name: `submission_master_plain_cosine_40k.csv`
- Meaning: plain mean cosine similarity on the victim train-40k probe set.
- Why here: strong reference baseline; essentially the v004-style score.

### I — `I_submission_master_conf_weighted_cosine_40k.csv`
- Original name: `submission_master_conf_weighted_cosine_40k.csv`
- Meaning: train-40k cosine weighted by victim confidence.
- Why here: small refinement of the plain 40k baseline.

### J — `J_submission_master_margin_weighted_cosine_40k.csv`
- Original name: `submission_master_margin_weighted_cosine_40k.csv`
- Meaning: train-40k cosine weighted by victim top1-top2 margin.
- Why here: another modest refinement of the plain 40k baseline.

### K — `K_submission_master_trimmed_cosine_90_40k.csv`
- Original name: `submission_master_trimmed_cosine_90_40k.csv`
- Meaning: mean of the best 90% per-image cosine scores on train-40k.
- Why here: robustified cosine that drops the lowest-matching tail.

### L — `L_submission_master_trimmed_cosine_80_40k.csv`
- Original name: `submission_master_trimmed_cosine_80_40k.csv`
- Meaning: mean of the best 80% per-image cosine scores on train-40k.
- Why here: more aggressive robust trimming than the 90% version.

### M — `M_submission_master_plain_cosine_test10k.csv`
- Original name: `submission_master_plain_cosine_test10k.csv`
- Meaning: plain mean cosine on CIFAR-100 official test images.
- Why here: clean out-of-train but same-domain probe.

### N — `N_submission_master_plain_cosine_unused10k.csv`
- Original name: `submission_master_plain_cosine_unused10k.csv`
- Meaning: plain mean cosine on CIFAR-100 train images not used by the victim.
- Why here: same-domain held-out training probe.

### O — `O_submission_master_top5_agreement_40k.csv`
- Original name: `submission_master_top5_agreement_40k.csv`
- Meaning: average top-5 class overlap on the victim train-40k probe set.
- Why here: class-ranking agreement signal rather than full-logit geometry.

### P — `P_submission_master_top5_agreement_test10k.csv`
- Original name: `submission_master_top5_agreement_test10k.csv`
- Meaning: average top-5 class overlap on the CIFAR-100 test-10k probe set.
- Why here: same ranking-style signal on the test distribution.

### Q — `Q_submission_master_aug_delta_cosine.csv`
- Original name: `submission_master_aug_delta_cosine.csv`
- Meaning: cosine similarity between target and suspect augmentation-response deltas.
- Why here: specialized local decision-geometry signal.

### R — `R_submission_master_layer4_cosine_4k.csv`
- Original name: `submission_master_layer4_cosine_4k.csv`
- Meaning: cosine similarity on layer4 features over the train-4k probe set.
- Why here: internal representation signal; useful as a specialist fallback.

### S — `S_submission_master_rank_fusion_multidist.csv`
- Original name: `submission_master_rank_fusion_multidist.csv`
- Meaning: same multidistribution rank-fusion formula as file `A`.
- Why lower: exact alias of `A`, kept only for traceability to the original generator output.

### T — `T_submission_master_rank_fusion_default.csv`
- Original name: `submission_master_rank_fusion_default.csv`
- Meaning: same default train-heavy rank-fusion formula as file `B`.
- Why lower: exact alias of `B`, kept only for traceability to the original generator output.

## Duplicate Pairs

- `A_submission_master_BEST_rank_fusion_multidist.csv` is byte-identical to `S_submission_master_rank_fusion_multidist.csv`
- `B_submission_master_BEST_rank_fusion.csv` is byte-identical to `T_submission_master_rank_fusion_default.csv`

## Practical Recommendation

If you only try a few files, start in this order:

1. `A_submission_master_BEST_rank_fusion_multidist.csv`
2. `B_submission_master_BEST_rank_fusion.csv`
3. `C_submission_master_neg_js_T2_40k.csv`
4. `E_submission_master_gap_train40k_minus_unused10k.csv`
5. `H_submission_master_plain_cosine_40k.csv`

## Notes

- These files were validated locally for format (`360` rows, columns `id,score`).
- No leaderboard/CMS submission was performed automatically.
