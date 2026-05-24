"""Orchestrate per-suspect forensic feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.forensics.adversarial import adversarial_transfer_features
from src.forensics.augmentation import augmentation_fingerprint, compare_aug_fingerprints
from src.forensics.behavioral import compare_logits
from src.forensics.representations import compare_representations, forward_collect
from src.forensics.target_cache import TargetCache
from src.forensics.weights import extract_weight_features
from src.model_utils import load_model, load_state_dict
from src.paths import FEATURES_DIR, SUSPECT_DIR, TARGET_WEIGHTS


@dataclass
class PipelineConfig:
    stages_weights: bool = True
    stages_behavioral: bool = True
    stages_representations: bool = True
    stages_augmentation: bool = True
    stages_adversarial: bool = False
    n_aug_views: int = 4
    adv_epsilon: float = 8 / 255


def suspect_feature_path(suspect_id: int) -> Path:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    return FEATURES_DIR / f"suspect_{suspect_id:03d}.parquet"


def extract_suspect_features(
    suspect_id: int,
    target_cache: TargetCache,
    loaders: dict[str, DataLoader],
    device: str,
    cfg: PipelineConfig,
    aug_images: torch.Tensor | None = None,
    adv_batch: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, float]:
    suspect_path = SUSPECT_DIR / f"suspect_{suspect_id:03d}.safetensors"
    suspect_sd = load_state_dict(str(suspect_path))
    feats: dict[str, float] = {}

    if cfg.stages_weights:
        feats.update(extract_weight_features(target_cache.state_dict, suspect_sd))

    need_model = (
        cfg.stages_behavioral
        or cfg.stages_representations
        or cfg.stages_augmentation
        or cfg.stages_adversarial
    )
    model = load_model(str(suspect_path), device=device) if need_model else None

    try:
        for probe_name, loader in loaders.items():
            if probe_name not in target_cache.logits:
                continue
            logits_s, acts_s = forward_collect(model, loader, device)

            if cfg.stages_behavioral:
                feats.update(
                    {
                        f"{probe_name}__{k}": v
                        for k, v in compare_logits(
                            target_cache.logits[probe_name], logits_s
                        ).items()
                    }
                )
            if cfg.stages_representations:
                feats.update(
                    {
                        f"{probe_name}__{k}": v
                        for k, v in compare_representations(
                            target_cache.activations[probe_name],
                            acts_s,
                            target_cache.logits[probe_name],
                            logits_s,
                        ).items()
                    }
                )

        if cfg.stages_augmentation and aug_images is not None and model is not None:
            s_fp = augmentation_fingerprint(
                model, aug_images, device, n_views=cfg.n_aug_views
            )
            feats.update(compare_aug_fingerprints(target_cache.aug_fingerprint, s_fp))

        if cfg.stages_adversarial and adv_batch is not None and model is not None:
            images, labels = adv_batch
            target_model = load_model(str(TARGET_WEIGHTS), device=device)
            feats.update(
                adversarial_transfer_features(
                    target_model, model, images, labels, device, eps=cfg.adv_epsilon
                )
            )
            del target_model
    finally:
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return feats


def save_suspect_features(suspect_id: int, feats: dict[str, float]) -> Path:
    path = suspect_feature_path(suspect_id)
    df = pd.DataFrame([{"id": suspect_id, **feats}])
    df.to_parquet(path, index=False)
    return path


def load_all_feature_tables() -> pd.DataFrame:
    rows = []
    for path in sorted(FEATURES_DIR.glob("suspect_*.parquet")):
        rows.append(pd.read_parquet(path))
    if not rows:
        raise FileNotFoundError(f"No feature files in {FEATURES_DIR}")
    return pd.concat(rows, ignore_index=True)
