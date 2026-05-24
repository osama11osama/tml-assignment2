"""CIFAR-100 probe set indexed by train_main_idx.json."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from src.model_utils import get_eval_transform
from src.paths import CIFAR100_DIR, TRAIN_MAIN_IDX


def load_train_indices(path: Path | None = None, subset: int | None = None) -> list[int]:
    """Load CIFAR-100 train indices from train_main_idx.json."""
    path = path or TRAIN_MAIN_IDX
    with open(path, encoding="utf-8") as f:
        indices = json.load(f)
    if subset is not None:
        indices = indices[:subset]
    return indices


class _IndexedCIFAR100(Dataset):
    """Wrap CIFAR-100 train set; Subset uses these indices."""

    def __init__(self, root: Path, indices: list[int]):
        self.base = datasets.CIFAR100(
            root=str(root),
            train=True,
            download=True,
            transform=get_eval_transform(),
        )
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[self.indices[i]]


def get_probe_dataloader(
    subset: int | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    indices_path: Path | None = None,
    cifar_root: Path | None = None,
) -> tuple[DataLoader, list[int]]:
    """DataLoader over train_main_idx probe images (eval transform only)."""
    indices = load_train_indices(indices_path, subset=subset)
    dataset = _IndexedCIFAR100(cifar_root or CIFAR100_DIR, indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, indices
