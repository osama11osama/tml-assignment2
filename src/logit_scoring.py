"""Logit-based stolen-model scoring (v002)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model_utils import load_model
from src.paths import TARGET_LOGITS_CACHE, TARGET_WEIGHTS


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: str) -> torch.Tensor:
    """Run model on loader; return logits [N, num_classes] on CPU."""
    model.eval()
    chunks: list[torch.Tensor] = []
    for images, _ in tqdm(loader, desc="forward", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        chunks.append(logits.cpu())
    return torch.cat(chunks, dim=0)


def mean_logit_cosine_similarity(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """Mean per-image cosine similarity between two logit matrices [N, C]."""
    if logits_a.shape != logits_b.shape:
        raise ValueError(f"Shape mismatch: {logits_a.shape} vs {logits_b.shape}")
    sims = torch.nn.functional.cosine_similarity(logits_a.float(), logits_b.float(), dim=1)
    return float(sims.mean().item())


def precompute_target_logits(
    loader: DataLoader,
    device: str,
    cache_path: Path | None = None,
) -> torch.Tensor:
    """Forward target model; optionally save cache."""
    if not TARGET_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing target weights: {TARGET_WEIGHTS}")
    target = load_model(str(TARGET_WEIGHTS), device=device)
    logits = collect_logits(target, loader, device)
    del target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cache_path = cache_path or TARGET_LOGITS_CACHE
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"logits": logits, "num_images": logits.shape[0]}, cache_path)
    return logits


def load_target_logits_cache(cache_path: Path | None = None) -> torch.Tensor:
    cache_path = cache_path or TARGET_LOGITS_CACHE
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Target logit cache missing: {cache_path}\n"
            "Run precompute first (local or cluster)."
        )
    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    return data["logits"]


def score_suspect_logit_similarity(
    suspect_path: Path,
    target_logits: torch.Tensor,
    loader: DataLoader,
    device: str,
) -> float:
    """Mean cosine similarity of logits vs precomputed target logits."""
    suspect = load_model(str(suspect_path), device=device)
    suspect_logits = collect_logits(suspect, loader, device)
    del suspect
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mean_logit_cosine_similarity(target_logits, suspect_logits)
