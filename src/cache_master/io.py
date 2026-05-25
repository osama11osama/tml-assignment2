"""Atomic tensor I/O for master cache."""

from __future__ import annotations

from pathlib import Path

import torch


def save_tensor(path: Path, tensor: torch.Tensor, *, fp16: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = tensor.half() if fp16 else tensor.float()
    torch.save({"tensor": payload, "shape": list(tensor.shape)}, tmp)
    tmp.replace(path)


def load_tensor(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["tensor"].float()
