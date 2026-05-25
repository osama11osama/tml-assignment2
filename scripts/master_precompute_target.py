#!/usr/bin/env python3
"""Step 1: Precompute target master cache + shared aug inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cache_master.config import MasterCacheConfig
from src.cache_master.extract import build_aug_inputs, extract_model_cache
from src.cache_master.io import save_tensor
from src.cache_master.paths import AUG_INPUTS_FILE, TARGET_DIR
from src.cache_master.status import print_status
from src.paths import TARGET_LOGITS_CACHE, TARGET_WEIGHTS


def _import_legacy_40k_target() -> bool:
    """Reuse results/cache/target_logits_40k.pt if present."""
    legacy = TARGET_LOGITS_CACHE
    if not legacy.exists():
        return False
    from src.cache_master.io import save_tensor as st
    from src.cache_master.paths import STAGE_TRAIN40K, stage_path
    import torch

    data = torch.load(legacy, map_location="cpu", weights_only=True)
    logits = data["logits"].float()
    out = stage_path(TARGET_DIR, STAGE_TRAIN40K)
    if out.exists():
        return True
    st(out, logits)
    print(f"Imported legacy 40k target logits -> {out}  shape={tuple(logits.shape)}")
    from src.cache_master.status import mark_stage

    mark_stage(TARGET_DIR, STAGE_TRAIN40K, "legacy-import")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Master cache: precompute target tensors.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-40k", action="store_true", help="Skip 40k forward (use --import-legacy-40k).")
    p.add_argument("--import-legacy-40k", action="store_true", help="Copy target_logits_40k.pt into master cache.")
    args = p.parse_args()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable — using CPU.")
        args.device = "cpu"

    cfg = MasterCacheConfig()
    cfg.save()
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if args.import_legacy_40k or args.skip_40k:
        _import_legacy_40k_target()

    # Shared deterministic augmentation inputs (all suspects reuse)
    aug_file = TARGET_DIR / AUG_INPUTS_FILE
    if not aug_file.exists():
        print("Building shared aug inputs...")
        base, views = build_aug_inputs(cfg, args.device)
        torch.save({"base": base.half(), "views": views.half()}, aug_file)
        print(f"Saved aug inputs -> {aug_file}")

    if not TARGET_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing {TARGET_WEIGHTS}")

    print("Extracting target master cache...")
    extract_model_cache(
        TARGET_WEIGHTS,
        TARGET_DIR,
        cfg,
        device=args.device,
        worker="target-precompute",
        batch_size=args.batch_size,
        skip_40k=args.skip_40k,
    )
    print("Target cache complete.")
    print_status()


if __name__ == "__main__":
    main()
