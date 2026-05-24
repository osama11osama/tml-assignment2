"""Probe dataset builders for forensic pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from src.model_utils import get_eval_transform
from src.paths import CIFAR100_DIR, TRAIN_MAIN_IDX

ProbeName = Literal["train_main", "test"]


def load_train_main_indices(path: Path | None = None, subset: int | None = None) -> list[int]:
    path = path or TRAIN_MAIN_IDX
    with open(path, encoding="utf-8") as f:
        indices = json.load(f)
    if subset is not None:
        indices = indices[:subset]
    return indices


class _IndexedCIFAR100Train(Dataset):
    def __init__(self, root: Path, indices: list[int]):
        self.base = datasets.CIFAR100(
            root=str(root), train=True, download=True, transform=get_eval_transform()
        )
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[self.indices[i]]


def get_probe_dataloader(
    probe: ProbeName,
    subset: int | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    cifar_root: Path | None = None,
) -> tuple[DataLoader, int]:
    root = cifar_root or CIFAR100_DIR
    if probe == "train_main":
        indices = load_train_main_indices(subset=subset)
        ds: Dataset = _IndexedCIFAR100Train(root, indices)
        n = len(indices)
    elif probe == "test":
        full = datasets.CIFAR100(
            root=str(root), train=False, download=True, transform=get_eval_transform()
        )
        n_full = len(full)
        n = n_full if subset is None else min(subset, n_full)
        ds = torch.utils.data.Subset(full, list(range(n)))
    else:
        raise ValueError(probe)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, n


def load_probe_images_for_aug(
    probe: ProbeName,
    n_images: int,
    cifar_root: Path | None = None,
) -> torch.Tensor:
    """Load first n_images as [0,1] tensors for augmentation fingerprint."""
    from torchvision import transforms as T

    to_tensor = T.Compose([T.ToTensor()])
    root = cifar_root or CIFAR100_DIR
    if probe == "train_main":
        indices = load_train_main_indices(subset=n_images)
        base = datasets.CIFAR100(root=str(root), train=True, download=True, transform=to_tensor)
        images = torch.stack([base[indices[i]][0] for i in range(len(indices))])
    else:
        base = datasets.CIFAR100(root=str(root), train=False, download=True, transform=to_tensor)
        n = min(n_images, len(base))
        images = torch.stack([base[i][0] for i in range(n)])
    return images
