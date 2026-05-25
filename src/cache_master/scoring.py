"""CPU-only scoring variants from cached tensors."""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F

from src.cache_master.io import load_tensor
from src.cache_master.paths import (
    STAGE_AUG_BASE,
    STAGE_AUG_VIEWS,
    STAGE_TEST10K,
    STAGE_TEST2K_LAYER4,
    STAGE_TEST2K_LOGITS,
    STAGE_TRAIN40K,
    STAGE_TRAIN4K_LAYER4,
    STAGE_TRAIN4K_LOGITS,
    STAGE_TRAIN_UNUSED,
    TARGET_DIR,
    stage_path,
    suspect_dir,
)
from src.forensics.aggregates import rank_normalize_series

VARIANT_NAMES = (
    "plain_cosine_40k",
    "plain_cosine_unused10k",
    "plain_cosine_test10k",
    "conf_weighted_cosine_40k",
    "margin_weighted_cosine_40k",
    "trimmed_cosine_90_40k",
    "trimmed_cosine_80_40k",
    "neg_js_T2_40k",
    "neg_js_T4_40k",
    "top5_agreement_40k",
    "top5_agreement_test10k",
    "gap_train40k_minus_test10k",
    "gap_train40k_minus_unused10k",
    "train_test_gap_cosine",
    "layer4_cosine_4k",
    "aug_delta_cosine",
    "rank_fusion_default",
    "rank_fusion_multidist",
)


def _per_image_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=1)


def _softmax_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return F.softmax(logits.float() / temperature, dim=-1)


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1)
    kl_qm = (q * (q.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def _topk_agreement(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
    at = a.topk(k, dim=1).indices
    bt = b.topk(k, dim=1).indices
    out = []
    for i in range(a.shape[0]):
        out.append(len(set(at[i].tolist()) & set(bt[i].tolist())) / k)
    return torch.tensor(out, dtype=torch.float32)


def _trimmed_mean(values: torch.Tensor, keep_frac: float) -> float:
    n = values.numel()
    k = max(1, int(n * keep_frac))
    top, _ = torch.topk(values, k)
    return float(top.mean())


def _load_pair(stage: str, suspect_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    t = load_tensor(stage_path(TARGET_DIR, stage))
    s = load_tensor(stage_path(suspect_dir(suspect_id), stage))
    n = min(t.shape[0], s.shape[0])
    return t[:n], s[:n]


def _pair_exists(stage: str, suspect_id: int) -> bool:
    return (
        stage_path(TARGET_DIR, stage).exists()
        and stage_path(suspect_dir(suspect_id), stage).exists()
    )


def _mean_cosine_stage(stage: str, suspect_id: int) -> float:
    t, s = _load_pair(stage, suspect_id)
    return float(_per_image_cosine(t, s).mean())


def score_suspect_from_cache(suspect_id: int) -> dict[str, float]:
    """All scalar variants for one suspect (CPU)."""
    scores: dict[str, float] = {}

    if _pair_exists(STAGE_TRAIN40K, suspect_id):
        t40, s40 = _load_pair(STAGE_TRAIN40K, suspect_id)
        cos40 = _per_image_cosine(t40, s40)
        p_t = _softmax_probs(t40)
        conf = p_t.max(dim=1).values
        top2 = p_t.topk(2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]

        scores["plain_cosine_40k"] = float(cos40.mean())
        scores["conf_weighted_cosine_40k"] = float((conf * cos40).sum() / conf.sum().clamp_min(1e-8))
        scores["margin_weighted_cosine_40k"] = float((margin * cos40).sum() / margin.sum().clamp_min(1e-8))
        scores["trimmed_cosine_90_40k"] = _trimmed_mean(cos40, 0.90)
        scores["trimmed_cosine_80_40k"] = _trimmed_mean(cos40, 0.80)
        scores["neg_js_T2_40k"] = float(-_js_divergence(_softmax_probs(t40, 2), _softmax_probs(s40, 2)).mean())
        scores["neg_js_T4_40k"] = float(-_js_divergence(_softmax_probs(t40, 4), _softmax_probs(s40, 4)).mean())
        scores["top5_agreement_40k"] = float(_topk_agreement(t40, s40, 5).mean())

        sim_train40 = scores["plain_cosine_40k"]
        if _pair_exists(STAGE_TEST10K, suspect_id):
            sim_test = _mean_cosine_stage(STAGE_TEST10K, suspect_id)
            scores["plain_cosine_test10k"] = sim_test
            scores["gap_train40k_minus_test10k"] = sim_train40 - sim_test
            t10, s10 = _load_pair(STAGE_TEST10K, suspect_id)
            scores["top5_agreement_test10k"] = float(_topk_agreement(t10, s10, 5).mean())
        if _pair_exists(STAGE_TRAIN_UNUSED, suspect_id):
            sim_unused = _mean_cosine_stage(STAGE_TRAIN_UNUSED, suspect_id)
            scores["plain_cosine_unused10k"] = sim_unused
            scores["gap_train40k_minus_unused10k"] = sim_train40 - sim_unused

    if _pair_exists(STAGE_TRAIN4K_LOGITS, suspect_id) and _pair_exists(STAGE_TEST2K_LOGITS, suspect_id):
        scores["train_test_gap_cosine"] = (
            _mean_cosine_stage(STAGE_TRAIN4K_LOGITS, suspect_id)
            - _mean_cosine_stage(STAGE_TEST2K_LOGITS, suspect_id)
        )

    if _pair_exists(STAGE_TRAIN4K_LAYER4, suspect_id):
        tl, sl = _load_pair(STAGE_TRAIN4K_LAYER4, suspect_id)
        scores["layer4_cosine_4k"] = float(_per_image_cosine(tl, sl).mean())

    if _pair_exists(STAGE_AUG_BASE, suspect_id):
        tb, sb = _load_pair(STAGE_AUG_BASE, suspect_id)
        tv, sv = _load_pair(STAGE_AUG_VIEWS, suspect_id)
        dt = tv.mean(dim=1) - tb
        ds = sv.mean(dim=1) - sb
        scores["aug_delta_cosine"] = float(_per_image_cosine(dt, ds).mean())

    return scores


def build_variant_dataframe(suspect_ids: list[int] | None = None) -> pd.DataFrame:
    from src.paths import NUM_SUSPECTS

    ids = suspect_ids if suspect_ids is not None else list(range(NUM_SUSPECTS))
    rows = [{"id": sid, **score_suspect_from_cache(sid)} for sid in ids]
    return pd.DataFrame(rows)


def rank_fusion(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    weights = weights or {
        "plain_cosine_40k": 0.35,
        "conf_weighted_cosine_40k": 0.20,
        "trimmed_cosine_90_40k": 0.15,
        "neg_js_T4_40k": 0.10,
        "top5_agreement_40k": 0.10,
        "layer4_cosine_4k": 0.05,
        "aug_delta_cosine": 0.05,
    }
    return _weighted_rank_fusion(df, weights)


def rank_fusion_multidist(df: pd.DataFrame) -> pd.Series:
    """Multi-distribution fingerprint — orthogonal to saturated 40k train signal."""
    return _weighted_rank_fusion(
        df,
        {
            "plain_cosine_40k": 0.40,
            "plain_cosine_test10k": 0.20,
            "plain_cosine_unused10k": 0.10,
            "gap_train40k_minus_test10k": 0.10,
            "aug_delta_cosine": 0.10,
            "top5_agreement_test10k": 0.05,
            "gap_train40k_minus_unused10k": 0.05,
        },
    )


def _weighted_rank_fusion(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    fused = pd.Series(0.0, index=df.index)
    total_w = 0.0
    for col, w in weights.items():
        if col not in df.columns or df[col].isna().all():
            continue
        fused = fused + w * rank_normalize_series(df[col].fillna(df[col].min()))
        total_w += w
    if total_w > 0:
        fused = fused / total_w
    return fused


def variant_to_submission(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "rank_fusion_default":
        scores = rank_fusion(df)
    elif variant == "rank_fusion_multidist":
        scores = rank_fusion_multidist(df)
    elif variant not in df.columns:
        raise KeyError(f"Unknown variant: {variant}")
    else:
        scores = df[variant]
    out = pd.DataFrame({"id": df["id"].astype(int), "score": scores.astype(float)})
    return out.sort_values("id").reset_index(drop=True)


def compare_variants_to_baseline(
    df: pd.DataFrame,
    baseline_col: str = "plain_cosine_40k",
) -> pd.DataFrame:
    from scipy.stats import spearmanr

    k = max(1, int(len(df) * 0.05))
    base_top = set(df.nlargest(k, baseline_col)["id"].astype(int))
    rows = []
    for col in df.columns:
        if col == "id" or df[col].isna().all():
            continue
        sp = spearmanr(df[baseline_col], df[col], nan_policy="omit").statistic
        top = set(df.nlargest(k, col)["id"].astype(int))
        rows.append(
            {
                "variant": col,
                "spearman_vs_baseline": sp,
                f"top{k}_overlap": len(base_top & top),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_vs_baseline", ascending=False)
