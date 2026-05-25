#!/usr/bin/env python3
"""
Parallel v002 scoring across multiple local GPUs (one suspect shard per GPU).

Single GPU: cannot parallelize CUDA inference — use cluster or resumable mode:
  python scripts/score_logit_similarity.py --resumable --batch-size 128 --output submission_v002_logit_40k.csv
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _shard(ids: list[int], n: int) -> list[list[int]]:
    shards: list[list[int]] = [[] for _ in range(n)]
    for i, sid in enumerate(ids):
        shards[i % n].append(sid)
    return [s for s in shards if s]


def _worker(
    gpu_id: int,
    suspect_ids: list[int],
    subset: int | None,
    batch_size: int,
    cache_path: str,
    score_dir: str,
) -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from scripts.score_logit_similarity import score_all

    score_all(
        subset=subset,
        suspect_ids=suspect_ids,
        batch_size=batch_size,
        device="cuda",
        skip_precompute=True,
        cache_path=Path(cache_path),
        resumable=True,
        score_dir=Path(score_dir),
        merge_at_end=False,
    )


def main() -> None:
    import torch

    from src.paths import CLUSTER_SCORES_DIR, NUM_SUSPECTS, TARGET_LOGITS_CACHE
    from scripts.score_logit_similarity import merge_cluster_scores, score_all

    p = argparse.ArgumentParser(description="Multi-GPU parallel v002 logit scoring.")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gpus", type=str, default=None, help="Comma GPU ids, e.g. 0,1. Default: all visible.")
    p.add_argument("--output", default="submission_v002_logit_40k.csv")
    p.add_argument("--cache", type=Path, default=TARGET_LOGITS_CACHE)
    p.add_argument("--score-dir", type=Path, default=CLUSTER_SCORES_DIR)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for parallel mode.")

    gpu_ids = (
        [int(x.strip()) for x in args.gpus.split(",")]
        if args.gpus
        else list(range(torch.cuda.device_count()))
    )
    if len(gpu_ids) < 2:
        print(
            f"Only {len(gpu_ids)} GPU(s) — parallel mode needs 2+.\n"
            "Single-GPU resumable run:\n"
            "  python scripts/score_logit_similarity.py --resumable --batch-size 128 "
            f"--output {args.output}"
        )
        raise SystemExit(1)

    print("Step 1 — precompute target logits (GPU 0)")
    score_all(
        subset=args.subset,
        suspect_ids=[],
        batch_size=args.batch_size,
        device="cuda:0",
        cache_path=args.cache,
        precompute_only=True,
        score_dir=args.score_dir,
        merge_at_end=False,
    )

    suspect_ids = list(range(NUM_SUSPECTS))
    shards = _shard(suspect_ids, len(gpu_ids))
    print(f"Step 2 — {len(gpu_ids)} GPU workers, shard sizes: {[len(s) for s in shards]}")

    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id, shard in zip(gpu_ids, shards):
        proc = ctx.Process(
            target=_worker,
            args=(
                gpu_id,
                shard,
                args.subset,
                args.batch_size,
                str(args.cache),
                str(args.score_dir),
            ),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise SystemExit(f"Worker failed with exit code {proc.exitcode}")

    print("Step 3 — merge JSON scores -> CSV")
    merge_cluster_scores(output_name=args.output, in_dir=args.score_dir)


if __name__ == "__main__":
    main()
