#!/usr/bin/env python3
"""Extract per-suspect forensic features (resumable)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.forensics.aggregates import set_seed
from src.forensics.pipeline import PipelineConfig, extract_suspect_features, save_suspect_features, suspect_feature_path
from src.forensics.target_cache import load_target_cache
from src.paths import NUM_SUSPECTS
from src.probe_sets import get_probe_dataloader, load_probe_images_for_aug


def parse_ids(spec: str | None) -> list[int]:
    if not spec:
        return list(range(NUM_SUSPECTS))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> None:
    p = argparse.ArgumentParser(description="Extract forensic features per suspect.")
    p.add_argument("--suspects", type=str, default=None)
    p.add_argument("--probe-train", type=int, default=4000)
    p.add_argument("--probe-test", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--aug-images", type=int, default=256)
    p.add_argument("--aug-views", type=int, default=4)
    p.add_argument("--adversarial", action="store_true")
    p.add_argument("--no-weights", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    set_seed(args.seed)
    target_cache = load_target_cache()

    loaders = {}
    if args.probe_train > 0:
        loaders["train_main"] = get_probe_dataloader(
            "train_main", subset=args.probe_train, batch_size=args.batch_size
        )[0]
    if args.probe_test > 0:
        loaders["test"] = get_probe_dataloader(
            "test", subset=args.probe_test, batch_size=args.batch_size
        )[0]

    aug_images = load_probe_images_for_aug("train_main", args.aug_images) if args.aug_images else None

    adv_batch = None
    if args.adversarial:
        loader = get_probe_dataloader("train_main", subset=64, batch_size=64)[0]
        imgs, lbls = next(iter(loader))
        adv_batch = (imgs, lbls)

    cfg = PipelineConfig(
        stages_weights=not args.no_weights,
        stages_adversarial=args.adversarial,
        n_aug_views=args.aug_views,
    )

    for sid in tqdm(parse_ids(args.suspects), desc="extract"):
        out_path = suspect_feature_path(sid)
        if out_path.exists() and not args.force:
            continue
        feats = extract_suspect_features(
            sid, target_cache, loaders, args.device, cfg, aug_images, adv_batch
        )
        save_suspect_features(sid, feats)
        print(f"  suspect_{sid:03d}: {len(feats)} features -> {out_path.name}")


if __name__ == "__main__":
    main()
