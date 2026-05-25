#!/usr/bin/env python3
"""Show master cache progress and per-suspect stage completion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cache_master.status import print_status, stage_summary
from src.cache_master.paths import CONFIG_PATH, PROGRESS_PATH


def main() -> None:
    p = argparse.ArgumentParser(description="Master cache status.")
    p.add_argument("--suspect", type=int, default=None, help="Show stage detail for one suspect.")
    p.add_argument("--worker-index", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    args = p.parse_args()

    if CONFIG_PATH.exists():
        print(f"Config: {CONFIG_PATH.read_text(encoding='utf-8')[:200]}...")
    else:
        print("Config: not created yet — run master_precompute_target.py")

    print_status(args.worker_index, args.num_workers)

    if args.suspect is not None:
        print(f"\nSuspect {args.suspect} stages:")
        for stage, ok in stage_summary(args.suspect).items():
            print(f"  {'OK' if ok else 'MISSING':7}  {stage}")


if __name__ == "__main__":
    main()
