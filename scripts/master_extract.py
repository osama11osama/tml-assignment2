#!/usr/bin/env python3
"""
Step 2: Extract per-suspect master cache (resumable, multi-GPU / multi-machine).

Split work across local GPU + RunPod:
  Local:   --worker-index 0 --num-workers 2 --worker-name local-5060
  RunPod:  --worker-index 1 --num-workers 2 --worker-name runpod-3090

On ONE GPU never run two extract processes in parallel — use num-workers=2 only
across two machines (or one CPU + one GPU if desperate).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cache_master.config import MasterCacheConfig
from src.cache_master.extract import extract_model_cache
from src.cache_master.status import (
    is_suspect_complete,
    list_incomplete,
    print_status,
    release_claim,
    try_claim,
    write_progress,
)
from src.cache_master.paths import suspect_dir
from src.paths import NUM_SUSPECTS, SUSPECT_DIR


def parse_ids(spec: str | None) -> list[int] | None:
    if not spec:
        return None
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
    p = argparse.ArgumentParser(description="Master cache: extract suspect tensors.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--worker-index", type=int, default=0, help="This worker id (0 .. num-workers-1).")
    p.add_argument("--num-workers", type=int, default=1, help="Total parallel workers across all GPUs/machines.")
    p.add_argument("--worker-name", default="local", help="Label for logs/claims.")
    p.add_argument("--suspects", type=str, default=None, help="Optional subset e.g. 0,5,10-20")
    p.add_argument("--skip-40k", action="store_true", help="Skip 40k logits (faster; 40k variants unavailable).")
    p.add_argument("--no-claim", action="store_true", help="Disable file claims (single worker only).")
    args = p.parse_args()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable — using CPU.")
        args.device = "cpu"
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfg = MasterCacheConfig.load()
    explicit = parse_ids(args.suspects)
    if explicit is not None:
        todo = explicit
    else:
        todo = list_incomplete(args.worker_index, args.num_workers)

    print(f"Worker {args.worker_name} index={args.worker_index}/{args.num_workers} — {len(todo)} suspects queued")
    print_status(args.worker_index, args.num_workers)

    for sid in tqdm(todo, desc="master-extract"):
        if is_suspect_complete(sid):
            continue
        if not args.no_claim and not try_claim(sid, args.worker_name):
            continue

        path = SUSPECT_DIR / f"suspect_{sid:03d}.safetensors"
        if not path.exists():
            release_claim(sid, args.worker_name)
            raise FileNotFoundError(path)

        root = suspect_dir(sid)
        root.mkdir(parents=True, exist_ok=True)
        try:
            extract_model_cache(
                path,
                root,
                cfg,
                device=args.device,
                worker=args.worker_name,
                batch_size=args.batch_size,
                skip_40k=args.skip_40k,
            )
            write_progress(args.worker_name, sid)
            print(f"  suspect_{sid:03d}: complete -> {root}")
        finally:
            release_claim(sid, args.worker_name)

    print_status(args.worker_index, args.num_workers)


if __name__ == "__main__":
    main()
