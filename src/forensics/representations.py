"""Stage 3 — representation forensics via forward hooks."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.forensics.aggregates import linear_cka
from src.model_utils import load_model


HOOK_LAYERS = ("conv1", "layer1", "layer2", "layer3", "layer4", "avgpool")


class ActivationCollector:
    """Accumulate per-layer activations [N, D] on CPU."""

    def __init__(self, model: nn.Module) -> None:
        self._bufs: dict[str, list[torch.Tensor]] = {n: [] for n in HOOK_LAYERS}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        def _hook(name: str) -> Callable:
            def fn(_mod, _inp, out):
                x = out.detach()
                if x.ndim == 4:
                    x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
                self._bufs[name].append(x.cpu())
            return fn

        for name in HOOK_LAYERS:
            mod = getattr(model, name)
            self._hooks.append(mod.register_forward_hook(_hook(name)))

    def close(self) -> None:
        for h in self._hooks:
            h.remove()

    def concat(self) -> dict[str, torch.Tensor]:
        return {k: torch.cat(v, dim=0) for k, v in self._bufs.items() if v}


@torch.no_grad()
def forward_collect(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    collector = ActivationCollector(model)
    logit_chunks: list[torch.Tensor] = []
    for images, _ in tqdm(loader, desc="repr-forward", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        logit_chunks.append(logits.cpu())
    acts = collector.concat()
    collector.close()
    return torch.cat(logit_chunks, dim=0), acts


def compare_representations(
    target_acts: dict[str, torch.Tensor],
    suspect_acts: dict[str, torch.Tensor],
    target_logits: torch.Tensor | None = None,
    suspect_logits: torch.Tensor | None = None,
) -> dict[str, float]:
    feats: dict[str, float] = {}
    for name in HOOK_LAYERS:
        if name not in target_acts or name not in suspect_acts:
            continue
        ta, sa = target_acts[name], suspect_acts[name]
        if ta.shape[0] != sa.shape[0]:
            n = min(ta.shape[0], sa.shape[0])
            ta, sa = ta[:n], sa[:n]
        feats[f"cka__{name}"] = linear_cka(ta, sa)
        # Linear reconstruction proxy: cosine(mean pooled)
        feats[f"act_cosine__{name}"] = float(
            torch.nn.functional.cosine_similarity(
                ta.mean(0), sa.mean(0), dim=0
            ).item()
        )
    cka_keys = [k for k in feats if k.startswith("cka__")]
    if cka_keys:
        feats["cka_mean"] = float(sum(feats[k] for k in cka_keys) / len(cka_keys))
    return feats
