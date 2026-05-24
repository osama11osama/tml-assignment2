"""Stage 2 — behavioral forensics from logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.forensics.aggregates import agg_named


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits.float(), dim=-1)


@torch.no_grad()
def compare_logits(target_logits: torch.Tensor, suspect_logits: torch.Tensor) -> dict[str, float]:
    """Per-image metrics aggregated; higher usually = more similar / stolen."""
    if target_logits.shape != suspect_logits.shape:
        raise ValueError(f"logit shape mismatch {target_logits.shape} vs {suspect_logits.shape}")

    t, s = target_logits.float(), suspect_logits.float()
    pt, ps = _softmax(t), _softmax(s)

    cos_sims = F.cosine_similarity(t, s, dim=1)
    kl = (pt * (pt.clamp_min(1e-8).log() - ps.clamp_min(1e-8).log())).sum(dim=1)
    m = 0.5 * (pt + ps)
    js = 0.5 * (
        (pt * (pt.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1)
        + (ps * (ps.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1)
    )

    top1_t, top1_s = t.argmax(dim=1), s.argmax(dim=1)
    top1_agree = (top1_t == top1_s).float()

    def topk_agree(k: int) -> torch.Tensor:
        kt = t.topk(k, dim=1).indices
        ks = s.topk(k, dim=1).indices
        return torch.tensor(
            [len(set(kt[i].tolist()) & set(ks[i].tolist())) / k for i in range(t.shape[0])],
            dtype=torch.float32,
        )

    conf_t = pt.max(dim=1).values
    conf_s = ps.max(dim=1).values
    margin_t = pt.topk(2, dim=1).values[:, 0] - pt.topk(2, dim=1).values[:, 1]
    margin_s = ps.topk(2, dim=1).values[:, 0] - ps.topk(2, dim=1).values[:, 1]

    feats: dict[str, float] = {}
    feats.update(agg_named(cos_sims.tolist(), "logit_cosine"))
    feats.update(agg_named((-kl).tolist(), "neg_kl"))  # higher = more similar
    feats.update(agg_named((-js).tolist(), "neg_js"))
    feats.update(agg_named(top1_agree.tolist(), "top1_agree"))
    feats.update(agg_named(topk_agree(3).tolist(), "top3_agree"))
    feats.update(agg_named(topk_agree(5).tolist(), "top5_agree"))
    feats.update(agg_named((conf_t - conf_s).abs().tolist(), "conf_absdiff"))
    feats.update(agg_named(1.0 - (conf_t - conf_s).abs(), "conf_sim"))
    feats.update(agg_named((margin_t - margin_s).abs().tolist(), "margin_absdiff"))
    feats.update(agg_named(1.0 - (margin_t - margin_s).abs(), "margin_sim"))

  # Calibration: mean predicted prob on true class (if labels unknown use max prob correlation)
    feats["calibration_maxprob_corr"] = float(
        torch.corrcoef(torch.stack([conf_t, conf_s]))[0, 1].item()
        if conf_t.numel() > 1
        else 1.0
    )
    return feats
