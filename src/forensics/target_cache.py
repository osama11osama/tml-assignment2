"""Precompute and load target reference artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.forensics.augmentation import augmentation_fingerprint
from src.forensics.representations import forward_collect
from src.model_utils import load_model
from src.paths import FORENSIC_TARGET_CACHE, TARGET_WEIGHTS


@dataclass
class TargetCache:
    logits: dict[str, torch.Tensor] = field(default_factory=dict)
    activations: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    aug_fingerprint: dict[str, float] = field(default_factory=dict)
    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)


def precompute_target_cache(
    loaders: dict[str, DataLoader],
    device: str,
    aug_images: torch.Tensor | None = None,
    n_aug_views: int = 4,
) -> TargetCache:
    model = load_model(str(TARGET_WEIGHTS), device=device)
    cache = TargetCache(state_dict={k: v.cpu() for k, v in model.state_dict().items()})

    for name, loader in loaders.items():
        logits, acts = forward_collect(model, loader, device)
        cache.logits[name] = logits
        cache.activations[name] = acts

    if aug_images is not None:
        cache.aug_fingerprint = augmentation_fingerprint(
            model, aug_images, device, n_views=n_aug_views
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


def save_target_cache(cache: TargetCache, path: Path | None = None) -> Path:
    path = path or FORENSIC_TARGET_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "logits": cache.logits,
            "activations": cache.activations,
            "aug_fingerprint": cache.aug_fingerprint,
            "state_dict": cache.state_dict,
        },
        path,
    )
    return path


def load_target_cache(path: Path | None = None) -> TargetCache:
    path = path or FORENSIC_TARGET_CACHE
    if not path.exists():
        raise FileNotFoundError(f"Missing target cache: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    return TargetCache(
        logits=data["logits"],
        activations=data["activations"],
        aug_fingerprint=data["aug_fingerprint"],
        state_dict=data["state_dict"],
    )
