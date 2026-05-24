"""Stage 4 — augmentation fingerprint (target training recipe approximation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.forensics.aggregates import agg_named
from src.model_utils import CIFAR100_MEAN, CIFAR100_STD


def get_training_like_transform() -> transforms.Compose:
    """Approximate target training augmentations (Tutorial 3)."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


@torch.no_grad()
def augmentation_fingerprint(
    model: nn.Module,
    base_images: torch.Tensor,
    device: str,
    n_views: int = 4,
) -> dict[str, float]:
    """
    Measure output stability under random augmentations.
    Returns stats for this model only (compare target vs suspect in pipeline).
    """
    model.eval()
    aug = get_training_like_transform()
    kl_drifts: list[float] = []
    margin_drifts: list[float] = []

    for i in range(base_images.shape[0]):
        img_pil = transforms.ToPILImage()(base_images[i].cpu())
        probs = []
        for _ in range(n_views):
            x = aug(img_pil).unsqueeze(0).to(device)
            p = F.softmax(model(x), dim=-1).cpu().squeeze(0)
            probs.append(p)
        p_stack = torch.stack(probs)
        p_mean = p_stack.mean(0)
        kl = (p_stack * (p_stack.clamp_min(1e-8).log() - p_mean.clamp_min(1e-8).log())).sum(dim=1)
        margins = p_stack.topk(2, dim=1).values[:, 0] - p_stack.topk(2, dim=1).values[:, 1]
        kl_drifts.append(float(kl.mean()))
        margin_drifts.append(float(margins.std()))

    return {
        **agg_named(kl_drifts, "aug_kl_drift"),
        **agg_named(margin_drifts, "aug_margin_drift"),
    }


def compare_aug_fingerprints(target_fp: dict[str, float], suspect_fp: dict[str, float]) -> dict[str, float]:
    feats: dict[str, float] = {}
    for k in target_fp:
        if k in suspect_fp:
            feats[f"aug_match__{k}"] = -abs(target_fp[k] - suspect_fp[k])
    keys = [k for k in feats if k.startswith("aug_match__")]
    if keys:
        feats["aug_match_mean"] = float(sum(feats[k] for k in keys) / len(keys))
    return feats
