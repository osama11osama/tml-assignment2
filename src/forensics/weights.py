"""Stage 1 — weight forensics (CPU, no forward pass)."""

from __future__ import annotations

from typing import Any

import torch

from src.forensics.aggregates import tensor_stats
from src.model_utils import cosine_similarity, flatten_weights, load_state_dict


def _frobenius(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.norm(a.float() - b.float(), p="fro").item())


def _spectral_norm(w: torch.Tensor) -> float:
    w = w.float()
    if w.ndim >= 2:
        return float(torch.linalg.svdvals(w.reshape(w.shape[0], -1))[0].item())
    return float(w.abs().max().item())


def extract_weight_features(
    target_sd: dict[str, torch.Tensor],
    suspect_sd: dict[str, torch.Tensor],
) -> dict[str, float]:
    feats: dict[str, float] = {}
    t_flat = flatten_weights(target_sd)
    s_flat = flatten_weights(suspect_sd)
    feats["w_global_cosine"] = cosine_similarity(t_flat, s_flat)
    feats["w_global_frobenius"] = -_frobenius(t_flat, s_flat)  # higher = more similar

    layer_sims: list[float] = []
    for key in sorted(target_sd.keys()):
        if target_sd[key].dtype not in (torch.float32, torch.float16, torch.bfloat16):
            continue
        t, s = target_sd[key].float(), suspect_sd[key].float()
        if t.shape != s.shape:
            continue
        sim = cosine_similarity(t.flatten(), s.flatten())
        layer_sims.append(sim)
        feats[f"w_layer_cosine__{key}"] = sim
        feats[f"w_layer_frobenius__{key}"] = -_frobenius(t, s)
        feats[f"w_layer_specdiff__{key}"] = -abs(_spectral_norm(t) - _spectral_norm(s))

    feats.update(
        {
            "w_layer_cosine_mean": float(sum(layer_sims) / max(len(layer_sims), 1)),
            "w_layer_cosine_min": float(min(layer_sims)) if layer_sims else 0.0,
        }
    )

    # Histogram fingerprint on flattened weights (coarse bins)
    bins = 64
    t_hist = torch.histc(t_flat, bins=bins, min=-1.0, max=1.0)
    s_hist = torch.histc(s_flat, bins=bins, min=-1.0, max=1.0)
    feats["w_hist_cosine"] = cosine_similarity(t_hist, s_hist)

    for prefix, sd in [("t", target_sd), ("s", suspect_sd)]:
        flat = flatten_weights(sd)
        for k, v in tensor_stats(flat).items():
            feats[f"w_{prefix}_flat_{k}"] = v
    feats["w_flat_mean_diff"] = -abs(feats.get("t_flat_mean", 0) - feats.get("s_flat_mean", 0))

    return feats
