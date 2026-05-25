"""GPU inference: collect and cache all tensors for one model."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.cache_master.config import MasterCacheConfig
from src.cache_master.io import load_tensor, save_tensor
from src.cache_master.paths import (
    ALL_STAGES,
    AUG_INPUTS_FILE,
    STAGE_AUG_BASE,
    STAGE_AUG_VIEWS,
    STAGE_TEST10K,
    STAGE_TEST2K_LAYER4,
    STAGE_TEST2K_LOGITS,
    STAGE_TRAIN40K,
    STAGE_TRAIN4K_LAYER4,
    STAGE_TRAIN4K_LOGITS,
    STAGE_TRAIN_UNUSED,
    stage_path,
)
from src.cache_master.status import mark_stage, stage_done
from src.forensics.augmentation import get_training_like_transform
from src.model_utils import get_eval_transform, load_model
from src.probe_sets import get_probe_dataloader


class Layer4Hook:
    """Capture layer4 pooled features [N, 512]."""

    def __init__(self, model: nn.Module) -> None:
        self._buf: list[torch.Tensor] = []
        self._handle = model.layer4.register_forward_hook(self._hook)

    def _hook(self, _mod, _inp, out) -> None:
        x = out.detach()
        if x.ndim == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        self._buf.append(x.cpu())

    def close(self) -> None:
        self._handle.remove()

    def concat(self) -> torch.Tensor:
        return torch.cat(self._buf, dim=0)


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: str) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    for images, _ in tqdm(loader, desc="logits", leave=False):
        images = images.to(device, non_blocking=True)
        chunks.append(model(images).cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def collect_logits_layer4(
    model: nn.Module, loader: DataLoader, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    hook = Layer4Hook(model)
    logit_chunks: list[torch.Tensor] = []
    for images, _ in tqdm(loader, desc="logits+layer4", leave=False):
        images = images.to(device, non_blocking=True)
        logit_chunks.append(model(images).cpu())
    logits = torch.cat(logit_chunks, dim=0)
    layer4 = hook.concat()
    hook.close()
    return logits, layer4


@torch.no_grad()
def forward_on_tensor(model: nn.Module, batch: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    out: list[torch.Tensor] = []
    bs = 64
    for i in range(0, batch.shape[0], bs):
        x = batch[i : i + bs].to(device, non_blocking=True)
        out.append(model(x).cpu())
    return torch.cat(out, dim=0)


def build_aug_inputs(cfg: MasterCacheConfig, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic aug inputs shared by target and all suspects.
    Returns:
      base_eval [N, 3, 32, 32] normalized
      aug_views [N, V, 3, 32, 32] normalized
    """
    from src.probe_sets import load_probe_images_for_aug

    raw = load_probe_images_for_aug("train_main", cfg.aug_images)  # [0,1]
    eval_tf = get_eval_transform()
    aug_tf = get_training_like_transform()
    to_pil = transforms.ToPILImage()

    base_list: list[torch.Tensor] = []
    view_list: list[torch.Tensor] = []

    for i in range(raw.shape[0]):
        pil = to_pil(raw[i].cpu())
        base_list.append(eval_tf(pil))
        views = []
        for v in range(cfg.aug_views):
            torch.manual_seed(cfg.aug_seed + i * cfg.aug_views + v)
            views.append(aug_tf(pil))
        view_list.append(torch.stack(views))

    base = torch.stack(base_list)
    views = torch.stack(view_list)
    return base, views


@torch.no_grad()
def collect_aug_logits(
    model: nn.Module,
    base: torch.Tensor,
    views: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """base [N,C,H,W], views [N,V,C,H,W] -> base_logits [N,100], view_logits [N,V,100]."""
    base_logits = forward_on_tensor(model, base, device)
    n, v, c, h, w = views.shape
    flat = views.reshape(n * v, c, h, w)
    flat_logits = forward_on_tensor(model, flat, device)
    return base_logits, flat_logits.reshape(n, v, -1)


def _run_stage(
    root: Path,
    stage: str,
    worker: str,
    fn,
) -> None:
    if stage_done(root, stage):
        return
    fn()
    mark_stage(root, stage, worker)


def extract_model_cache(
    weights_path: Path,
    root: Path,
    cfg: MasterCacheConfig,
    device: str,
    worker: str,
    batch_size: int = 128,
    skip_40k: bool = False,
) -> None:
    """Fill all master-cache stages for target or one suspect."""
    model = load_model(str(weights_path), device=device)

    if not skip_40k:

        def _40k() -> None:
            loader, _ = get_probe_dataloader("train_main", subset=cfg.probe_train_40k, batch_size=batch_size)
            logits = collect_logits(model, loader, device)
            save_tensor(stage_path(root, STAGE_TRAIN40K), logits)

        _run_stage(root, STAGE_TRAIN40K, worker, _40k)

    def _unused10k() -> None:
        loader, _ = get_probe_dataloader(
            "train_unused", subset=cfg.probe_train_unused, batch_size=batch_size
        )
        logits = collect_logits(model, loader, device)
        save_tensor(stage_path(root, STAGE_TRAIN_UNUSED), logits)

    if cfg.probe_train_unused > 0 and not stage_done(root, STAGE_TRAIN_UNUSED):
        _run_stage(root, STAGE_TRAIN_UNUSED, worker, _unused10k)

    def _test10k() -> None:
        loader, _ = get_probe_dataloader("test", subset=cfg.probe_test_10k, batch_size=batch_size)
        logits = collect_logits(model, loader, device)
        save_tensor(stage_path(root, STAGE_TEST10K), logits)

    if cfg.probe_test_10k > 0 and not stage_done(root, STAGE_TEST10K):
        _run_stage(root, STAGE_TEST10K, worker, _test10k)

    def _train4k() -> None:
        loader, _ = get_probe_dataloader("train_main", subset=cfg.probe_train_4k, batch_size=batch_size)
        logits, layer4 = collect_logits_layer4(model, loader, device)
        save_tensor(stage_path(root, STAGE_TRAIN4K_LOGITS), logits)
        save_tensor(stage_path(root, STAGE_TRAIN4K_LAYER4), layer4)
        mark_stage(root, STAGE_TRAIN4K_LOGITS, worker)
        mark_stage(root, STAGE_TRAIN4K_LAYER4, worker)

    if not stage_done(root, STAGE_TRAIN4K_LOGITS):
        _train4k()

    def _test2k() -> None:
        loader, _ = get_probe_dataloader("test", subset=cfg.probe_test_2k, batch_size=batch_size)
        logits, layer4 = collect_logits_layer4(model, loader, device)
        save_tensor(stage_path(root, STAGE_TEST2K_LOGITS), logits)
        save_tensor(stage_path(root, STAGE_TEST2K_LAYER4), layer4)
        mark_stage(root, STAGE_TEST2K_LOGITS, worker)
        mark_stage(root, STAGE_TEST2K_LAYER4, worker)

    if not stage_done(root, STAGE_TEST2K_LOGITS):
        _test2k()

    def _aug() -> None:
        from src.cache_master.paths import TARGET_DIR

        inputs_file = TARGET_DIR / AUG_INPUTS_FILE
        if not inputs_file.exists():
            raise FileNotFoundError(f"Missing aug inputs: {inputs_file} — run master_precompute_target first")
        data = torch.load(inputs_file, map_location="cpu", weights_only=True)
        base, views = data["base"].float(), data["views"].float()
        base_logits, view_logits = collect_aug_logits(model, base, views, device)
        save_tensor(stage_path(root, STAGE_AUG_BASE), base_logits)
        save_tensor(stage_path(root, STAGE_AUG_VIEWS), view_logits)
        mark_stage(root, STAGE_AUG_BASE, worker)
        mark_stage(root, STAGE_AUG_VIEWS, worker)

    if not stage_done(root, STAGE_AUG_BASE):
        _aug()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mf_stages = {s for s in ALL_STAGES if stage_done(root, s)}
    if skip_40k:
        mf_stages.discard(STAGE_TRAIN40K)
    from src.cache_master.status import load_manifest, save_manifest

    mf = load_manifest(root)
    mf["complete"] = all(stage_done(root, s) for s in ALL_STAGES if not (skip_40k and s == STAGE_TRAIN40K))
    if mf["complete"]:
        mf["completed_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    save_manifest(root, mf)
