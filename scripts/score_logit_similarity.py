#!/usr/bin/env python3
"""
Score suspects via mean logit cosine similarity vs target (v002).

Higher score = logits more similar to target on train_main_idx probe images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.logit_scoring import (
    load_target_logits_cache,
    precompute_target_logits,
    score_suspect_logit_similarity,
)
from src.paths import (
    CACHE_DIR,
    CLUSTER_SCORES_DIR,
    NUM_SUSPECTS,
    SUBMISSIONS_DIR,
    SUSPECT_DIR,
    TARGET_LOGITS_CACHE,
    TARGET_WEIGHTS,
)
from src.probe_data import get_probe_dataloader


def suspect_path(model_id: int) -> Path:
    return SUSPECT_DIR / f"suspect_{model_id:03d}.safetensors"


def parse_suspect_ids(spec: str | None) -> list[int]:
    if not spec:
        return list(range(NUM_SUSPECTS))
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def score_all(
    subset: int | None = None,
    suspect_ids: list[int] | None = None,
    batch_size: int = 64,
    device: str = "cuda",
    skip_precompute: bool = False,
    cache_path: Path | None = None,
    output_name: str = "submission_v002_logit_train.csv",
) -> Path:
    if not TARGET_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Target weights missing: {TARGET_WEIGHTS}\n"
            "Run: python scripts/download_models.py --target-only"
        )

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — using CPU (slow).")
        device = "cpu"

    cache_path = cache_path or TARGET_LOGITS_CACHE
    loader, _ = get_probe_dataloader(subset=subset, batch_size=batch_size)

    if skip_precompute and cache_path.exists():
        target_logits = load_target_logits_cache(cache_path)
    else:
        print(f"Precomputing target logits ({'subset ' + str(subset) if subset else 'full'})...")
        target_logits = precompute_target_logits(loader, device=device, cache_path=cache_path)
        print(f"Cached target logits -> {cache_path}  shape={tuple(target_logits.shape)}")

    suspect_ids = suspect_ids or list(range(NUM_SUSPECTS))
    rows = []
    for model_id in tqdm(suspect_ids, desc="suspects"):
        path = suspect_path(model_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        score = score_suspect_logit_similarity(path, target_logits, loader, device)
        rows.append({"id": model_id, "score": score})
        print(f"  suspect_{model_id:03d}: {score:.6f}")

    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTER_SCORES_DIR.mkdir(parents=True, exist_ok=True)

    out_path = SUBMISSIONS_DIR / output_name
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")
    print(f"Score range: [{df['score'].min():.6f}, {df['score'].max():.6f}]")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="v002 logit cosine similarity scoring.")
    parser.add_argument("--subset", type=int, default=None, help="Use first N train_main indices.")
    parser.add_argument("--suspects", type=str, default=None, help="e.g. 0,1,5-10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--skip-precompute", action="store_true", help="Reuse existing target logit cache.")
    parser.add_argument("--cache", type=Path, default=None, help="Target logits cache path.")
    parser.add_argument(
        "--output",
        default="submission_v002_logit_train.csv",
        help="Output filename in results/submissions/",
    )
    args = parser.parse_args()

    score_all(
        subset=args.subset,
        suspect_ids=parse_suspect_ids(args.suspects),
        batch_size=args.batch_size,
        device=args.device,
        skip_precompute=args.skip_precompute,
        cache_path=args.cache,
        output_name=args.output,
    )


if __name__ == "__main__":
    main()
