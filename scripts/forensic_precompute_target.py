#!/usr/bin/env python3
"""Precompute target reference cache for forensic pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.forensics.aggregates import set_seed
from src.forensics.target_cache import load_target_cache, precompute_target_cache, save_target_cache
from src.probe_sets import get_probe_dataloader, load_probe_images_for_aug


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute target forensic cache.")
    p.add_argument("--probe-train", type=int, default=4000)
    p.add_argument("--probe-test", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--aug-images", type=int, default=256)
    p.add_argument("--aug-views", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("CUDA unavailable — using CPU.")

    set_seed(args.seed)
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
    cache = precompute_target_cache(
        loaders, args.device, aug_images=aug_images, n_aug_views=args.aug_views
    )
    path = save_target_cache(cache)
    print(f"Saved target cache -> {path}")


if __name__ == "__main__":
    main()
