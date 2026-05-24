#!/usr/bin/env python3
"""Merge per-suspect JSON scores into submission CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.paths import CLUSTER_SCORES_DIR, NUM_SUSPECTS, SUBMISSIONS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=CLUSTER_SCORES_DIR)
    parser.add_argument(
        "--output",
        default="submission_v002_logit_train.csv",
        help="Filename under results/submissions/",
    )
    args = parser.parse_args()

    rows = []
    for i in range(NUM_SUSPECTS):
        path = args.in_dir / f"suspect_{i:03d}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"id": int(data["id"]), "score": float(data["score"])})

    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    if len(df) != NUM_SUSPECTS:
        raise ValueError(f"Expected {NUM_SUSPECTS} rows, got {len(df)}")

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSIONS_DIR / args.output
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(f"Score range: [{df['score'].min():.6f}, {df['score'].max():.6f}]")


if __name__ == "__main__":
    main()
