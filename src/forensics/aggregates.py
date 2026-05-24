"""Statistical aggregates and rank utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.float().flatten()
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "skew": float(_skew(x)),
        "kurt": float(_kurt(x)),
    }


def _skew(x: torch.Tensor) -> float:
    m, s = x.mean(), x.std(unbiased=False)
    if s < 1e-12:
        return 0.0
    return float(((x - m) ** 3).mean() / (s**3))


def _kurt(x: torch.Tensor) -> float:
    m, s = x.mean(), x.std(unbiased=False)
    if s < 1e-12:
        return 0.0
    return float(((x - m) ** 4).mean() / (s**4) - 3.0)


def agg_named(values: Iterable[float], prefix: str) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
    }


def rank_normalize_series(s: pd.Series, higher_is_suspicious: bool = True) -> pd.Series:
    """Rank across suspects in [0, 1]; 1 = most suspicious."""
    r = s.rank(method="average", ascending=not higher_is_suspicious)
    return (r - 1) / max(len(s) - 1, 1)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between [N, D] activation matrices."""
    x = x.float() - x.float().mean(0)
    y = y.float() - y.float().mean(0)
    xy = (y.T @ x).norm() ** 2
    xx = (x.T @ x).norm()
    yy = (y.T @ y).norm()
    if xx < 1e-12 or yy < 1e-12:
        return 0.0
    return float(xy / (xx * yy))
