"""Stage 5 — adversarial lineage (optional, small probe)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.forensics.aggregates import agg_named


@torch.no_grad()
def fgsm_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, eps: float) -> torch.Tensor:
    images = images.clone().detach().requires_grad_(True)
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    adv = images + eps * images.grad.sign()
    return adv.detach()


@torch.no_grad()
def adversarial_transfer_features(
    target_model: nn.Module,
    suspect_model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    eps: float = 8 / 255,
) -> dict[str, float]:
    """FGSM on target; measure suspect prediction change vs clean."""
    images = images.to(device)
    labels = labels.to(device)
    adv = fgsm_attack(target_model, images, labels, eps)

    clean_t = target_model(images).argmax(1)
    adv_t = target_model(adv).argmax(1)
    clean_s = suspect_model(images).argmax(1)
    adv_s = suspect_model(adv).argmax(1)

    target_flip = (clean_t != adv_t).float()
    suspect_flip = (clean_s != adv_s).float()
    transfer = ((clean_t != adv_t) & (clean_s != adv_s)).float()

    return {
        **agg_named(target_flip.tolist(), "adv_target_flip"),
        **agg_named(suspect_flip.tolist(), "adv_suspect_flip"),
        **agg_named(transfer.tolist(), "adv_transfer"),
        "adv_transfer_rate": float(transfer.mean()),
    }
