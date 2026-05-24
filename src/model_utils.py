"""Model architecture and checkpoint loading (matches official task_template.py)."""

from __future__ import annotations

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.models import resnet18

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def make_model(num_classes: int = 100) -> nn.Module:
    """ResNet-18 adapted for CIFAR-100 (same as task_template.py)."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def load_state_dict(checkpoint_path: str, device: str = "cpu") -> dict:
    return load_file(checkpoint_path, device=device)


def load_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    model = make_model()
    state_dict = load_state_dict(checkpoint_path, device=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def flatten_weights(state_dict: dict) -> torch.Tensor:
    """Concatenate all tensor values into one 1-D vector."""
    parts = []
    for key in sorted(state_dict.keys()):
        parts.append(state_dict[key].reshape(-1).float())
    return torch.cat(parts)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 0.0
    return float(torch.dot(a, b) / denom)
