#!/usr/bin/env python3
"""Validate submission CSV against official requirements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.paths import NUM_SUSPECTS


def validate(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    errors = []

    if list(df.columns) != ["id", "score"]:
        errors.append(f"Columns must be exactly ['id', 'score'], got {list(df.columns)}")

    if len(df) != NUM_SUSPECTS:
        errors.append(f"Expected {NUM_SUSPECTS} rows, got {len(df)}")

    ids = pd.to_numeric(df["id"], errors="coerce")
    scores = pd.to_numeric(df["score"], errors="coerce")

    if ids.isna().any():
        errors.append("Non-numeric id values found.")
    if scores.isna().any():
        errors.append("Non-numeric or NaN score values found.")
    if np.isinf(scores).any():
        errors.append("Infinite score values found.")

    if ids.min() != 0 or ids.max() != NUM_SUSPECTS - 1:
        errors.append(f"id range must be 0..{NUM_SUSPECTS - 1}, got [{ids.min()}, {ids.max()}]")

    if ids.nunique() != NUM_SUSPECTS:
        errors.append(f"Duplicate or missing ids (unique count: {ids.nunique()}).")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"OK: {path} ({len(df)} rows, score range [{scores.min():.6f}, {scores.max():.6f}])")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="Path to submission CSV")
    args = parser.parse_args()
    validate(args.csv)


if __name__ == "__main__":
    main()
