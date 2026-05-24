#!/usr/bin/env python3
"""Download target and suspect model weights from HuggingFace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from src.paths import HF_REPO, NUM_SUSPECTS, SUSPECT_DIR, TARGET_DIR, TARGET_WEIGHTS


def download_target() -> Path:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_WEIGHTS.exists():
        print(f"Target already present: {TARGET_WEIGHTS}")
        return TARGET_WEIGHTS

    print("Downloading target model (~45 MB)...")
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename="target_model/weights.safetensors",
        local_dir=ROOT,
    )
    print(f"Saved: {path}")
    return Path(path)


def download_suspects(start: int = 0, end: int | None = None) -> None:
    end = NUM_SUSPECTS if end is None else min(end, NUM_SUSPECTS)
    SUSPECT_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    for i in range(start, end):
        dest = SUSPECT_DIR / f"suspect_{i:03d}.safetensors"
        if not dest.exists():
            missing.append(i)

    if not missing:
        print(f"All suspect models {start:03d}..{end - 1:03d} already downloaded.")
        return

    print(f"Downloading {len(missing)} suspect models (~45 MB each)...")
    for i in tqdm(missing, desc="suspects"):
        hf_hub_download(
            repo_id=HF_REPO,
            filename=f"suspect_models/suspect_{i:03d}.safetensors",
            local_dir=ROOT,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Assignment 2 model weights.")
    parser.add_argument("--target-only", action="store_true", help="Download only the target model.")
    parser.add_argument("--start", type=int, default=0, help="First suspect id (default: 0).")
    parser.add_argument("--end", type=int, default=None, help="Exclusive end suspect id (default: 360).")
    args = parser.parse_args()

    download_target()
    if not args.target_only:
        download_suspects(start=args.start, end=args.end)


if __name__ == "__main__":
    main()
