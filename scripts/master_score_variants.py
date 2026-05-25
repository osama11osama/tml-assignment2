#!/usr/bin/env python3
"""Step 3: Generate all submission CSV variants from master cache (CPU only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cache_master.scoring import (
    VARIANT_NAMES,
    build_variant_dataframe,
    compare_variants_to_baseline,
    rank_fusion,
    variant_to_submission,
)
from src.cache_master.status import count_complete, is_suspect_complete
from src.cache_master.paths import VARIANTS_DIR
from src.paths import NUM_SUSPECTS, SUBMISSIONS_DIR


def main() -> None:
    p = argparse.ArgumentParser(description="Score all variants from master cache (no GPU).")
    p.add_argument("--output-dir", type=Path, default=SUBMISSIONS_DIR)
    p.add_argument("--analysis-dir", type=Path, default=VARIANTS_DIR)
    p.add_argument("--prefix", default="submission_master")
    p.add_argument("--variants", type=str, default=None, help="Comma-separated; default = all available")
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()

    done, total = count_complete()
    if done < total and not args.allow_partial:
        missing = [i for i in range(NUM_SUSPECTS) if not is_suspect_complete(i)]
        raise RuntimeError(
            f"Master cache incomplete: {done}/{total}. Missing suspects e.g. {missing[:5]}. "
            "Use --allow-partial for dev only."
        )

    print("Building variant score table (CPU)...")
    df = build_variant_dataframe()
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.analysis_dir / "all_variants.parquet", index=False)

    variants = args.variants.split(",") if args.variants else list(VARIANT_NAMES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for v in variants:
        v = v.strip()
        sub = variant_to_submission(df, v)
        out = args.output_dir / f"{args.prefix}_{v}.csv"
        sub.to_csv(out, index=False)
        print(f"  {v}: [{sub['score'].min():.4f}, {sub['score'].max():.4f}] -> {out.name}")

    if "plain_cosine_40k" in df.columns:
        report = compare_variants_to_baseline(df)
        report_path = args.analysis_dir / "variant_comparison.json"
        report_path.write_text(report.to_json(orient="records", indent=2), encoding="utf-8")
        print(f"Variant analysis -> {report_path}")
        print(report.to_string(index=False))

    # Best single + fusion recommendation
    if "plain_cosine_40k" in df.columns:
        k = max(1, int(len(df) * 0.05))
        base_top = set(df.nlargest(k, "plain_cosine_40k")["id"])
        best = None
        best_overlap = -1
        for v in variants:
            if v not in df.columns or v == "plain_cosine_40k":
                continue
            top = set(df.nlargest(k, v)["id"])
            ov = len(base_top & top)
            if ov > best_overlap:
                best_overlap, best = ov, v
        print(f"\nLargest top-{k} shake-up vs plain_cosine_40k: {best} (overlap {best_overlap}/{k})")

    fusion = variant_to_submission(df, "rank_fusion_multidist")
    best_out = args.output_dir / f"{args.prefix}_BEST_rank_fusion_multidist.csv"
    fusion.to_csv(best_out, index=False)
    print(f"\nRecommended multi-dist candidate -> {best_out}")

    fusion_v1 = variant_to_submission(df, "rank_fusion_default")
    v1_out = args.output_dir / f"{args.prefix}_BEST_rank_fusion.csv"
    fusion_v1.to_csv(v1_out, index=False)
    print(f"Train-only fusion candidate -> {v1_out}")


if __name__ == "__main__":
    main()
