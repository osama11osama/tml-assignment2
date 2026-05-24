#!/usr/bin/env python3
"""Build submission CSV from cached forensic features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.forensics.ensemble import build_ensemble_scores
from src.forensics.pipeline import load_all_feature_tables
from src.paths import SUBMISSIONS_DIR


def main() -> None:
    p = argparse.ArgumentParser(description="Ensemble forensic features -> submission.csv")
    p.add_argument("--method", choices=["rank_weighted", "isolation_forest"], default="rank_weighted")
    p.add_argument("--output", default="submission_v003_forensic.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Dev only: fewer than 360 feature files (fills missing ids with 0)",
    )
    args = p.parse_args()

    df = load_all_feature_tables()
    if args.allow_partial and len(df) < 360:
        import pandas as pd
        from src.paths import NUM_SUSPECTS

        have = set(df["id"].astype(int))
        missing = [i for i in range(NUM_SUSPECTS) if i not in have]
        pad = pd.DataFrame({"id": missing})
        df = pd.concat([df, pad], ignore_index=True).fillna(0)
    scores = build_ensemble_scores(df, method=args.method, seed=args.seed)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBMISSIONS_DIR / args.output
    scores.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(f"Score range: [{scores['score'].min():.6f}, {scores['score'].max():.6f}]")


if __name__ == "__main__":
    main()
