#!/usr/bin/env python3
"""Cluster step 1: precompute target logits on full probe set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.logit_scoring import precompute_target_logits
from src.paths import TARGET_LOGITS_CACHE
from src.probe_data import get_probe_dataloader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", type=Path, default=TARGET_LOGITS_CACHE)
    args = parser.parse_args()

    loader, indices = get_probe_dataloader(subset=args.subset, batch_size=args.batch_size)
    print(f"Probe images: {len(indices)}")
    logits = precompute_target_logits(loader, device=args.device, cache_path=args.cache)
    print(f"Done. shape={tuple(logits.shape)} -> {args.cache}")


if __name__ == "__main__":
    main()
