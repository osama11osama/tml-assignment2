#!/usr/bin/env python3
"""
Score suspects via mean logit cosine similarity vs target (v002).

Higher score = logits more similar to target on train_main_idx probe images.
"""

from __future__ import annotations

import argparse
import json
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


def _suspect_score_path(model_id: int, out_dir: Path) -> Path:
    return out_dir / f"suspect_{model_id:03d}.json"


def _load_cached_score(model_id: int, out_dir: Path) -> float | None:
    path = _suspect_score_path(model_id, out_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["score"])


def _save_cached_score(model_id: int, score: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _suspect_score_path(model_id, out_dir)
    path.write_text(
        json.dumps({"id": model_id, "score": score, "method": "logit_cosine"}),
        encoding="utf-8",
    )


def merge_cluster_scores(
    output_name: str = "submission_v002_logit_train.csv",
    in_dir: Path | None = None,
) -> Path:
    in_dir = in_dir or CLUSTER_SCORES_DIR
    rows = []
    for model_id in range(NUM_SUSPECTS):
        path = _suspect_score_path(model_id, in_dir)
        if not path.exists():
            raise FileNotFoundError(f"Missing score file: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"id": int(data["id"]), "score": float(data["score"])})

    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / output_name
    df.to_csv(out_path, index=False)
    print(f"Merged {len(df)} scores -> {out_path}")
    print(f"Score range: [{df['score'].min():.6f}, {df['score'].max():.6f}]")
    return out_path


def score_all(
    subset: int | None = None,
    suspect_ids: list[int] | None = None,
    batch_size: int = 64,
    device: str = "cuda",
    skip_precompute: bool = False,
    cache_path: Path | None = None,
    output_name: str = "submission_v002_logit_train.csv",
    resumable: bool = False,
    score_dir: Path | None = None,
    precompute_only: bool = False,
    merge_at_end: bool = True,
) -> Path:
    if not TARGET_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Target weights missing: {TARGET_WEIGHTS}\n"
            "Run: python scripts/download_models.py --target-only"
        )

    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available — using CPU (slow).")
        device = "cpu"
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cache_path = cache_path or TARGET_LOGITS_CACHE
    loader, _ = get_probe_dataloader(subset=subset, batch_size=batch_size)
    n_probe = len(loader.dataset)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    need_precompute = True
    if cache_path.exists():
        cached = load_target_logits_cache(cache_path)
        if cached.shape[0] == n_probe:
            target_logits = cached
            need_precompute = False
            print(f"Using cached target logits ({n_probe} images) -> {cache_path}")
        elif skip_precompute:
            raise ValueError(
                f"Cache has {cached.shape[0]} images but probe set has {n_probe}. "
                "Re-run without --skip-precompute."
            )
        else:
            print(f"Cache size mismatch ({cached.shape[0]} vs {n_probe}) — recomputing.")

    if need_precompute:
        print(f"Precomputing target logits ({'subset ' + str(subset) if subset else 'full'})...")
        target_logits = precompute_target_logits(loader, device=device, cache_path=cache_path)
        print(f"Cached target logits -> {cache_path}  shape={tuple(target_logits.shape)}")

    if precompute_only:
        return cache_path

    score_dir = score_dir or CLUSTER_SCORES_DIR
    score_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    suspect_ids = suspect_ids or list(range(NUM_SUSPECTS))
    for model_id in tqdm(suspect_ids, desc="suspects"):
        if resumable:
            cached_score = _load_cached_score(model_id, score_dir)
            if cached_score is not None:
                continue
        path = suspect_path(model_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        score = score_suspect_logit_similarity(path, target_logits, loader, device)
        _save_cached_score(model_id, score, score_dir)
        print(f"  suspect_{model_id:03d}: {score:.6f}")

    if merge_at_end:
        return merge_cluster_scores(output_name=output_name, in_dir=score_dir)
    return score_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="v002 logit cosine similarity scoring.")
    parser.add_argument("--subset", type=int, default=None, help="Use first N train_main indices.")
    parser.add_argument("--suspects", type=str, default=None, help="e.g. 0,1,5-10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu")
    parser.add_argument("--skip-precompute", action="store_true", help="Require matching target logit cache.")
    parser.add_argument("--precompute-only", action="store_true", help="Only cache target logits, then exit.")
    parser.add_argument("--merge-only", action="store_true", help="Merge results/cluster_scores/*.json -> CSV.")
    parser.add_argument(
        "--resumable",
        action="store_true",
        help="Skip suspects with existing JSON in results/cluster_scores/.",
    )
    parser.add_argument("--score-dir", type=Path, default=CLUSTER_SCORES_DIR)
    parser.add_argument("--cache", type=Path, default=None, help="Target logits cache path.")
    parser.add_argument(
        "--output",
        default="submission_v002_logit_train.csv",
        help="Output filename in results/submissions/",
    )
    args = parser.parse_args()

    if args.merge_only:
        merge_cluster_scores(output_name=args.output, in_dir=args.score_dir)
        return

    score_all(
        subset=args.subset,
        suspect_ids=parse_suspect_ids(args.suspects),
        batch_size=args.batch_size,
        device=args.device,
        skip_precompute=args.skip_precompute,
        cache_path=args.cache,
        output_name=args.output,
        resumable=args.resumable,
        score_dir=args.score_dir,
        precompute_only=args.precompute_only,
    )


if __name__ == "__main__":
    main()
