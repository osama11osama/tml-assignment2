"""
submission.py — upload a Stolen Model Detection CSV to the course server.

Requirements (summary):
  - Columns: id, score
  - Exactly 360 rows, ids 0–359, unique
  - Numeric finite scores (ranked for TPR@FPR=0.05)

Environment:
  TML_API_KEY   required — 32-char hash from CMS (also read from .env)

Examples:
  python submission.py
  python submission.py results/submissions/submission_v001_weight_cosine.csv
  python submission.py --validate-only results/submissions/submission_v001_weight_cosine.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "results" / "submissions" / "submission_v001_weight_cosine.csv"

BASE_URL = "http://34.63.153.158"
TASK_ID = "19-stolen-model-detection"
EXPECTED_ROWS = 360


def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def load_api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("TML_API_KEY", "").strip()
    if not key:
        die(
            "TML_API_KEY is not set.\n"
            "  Add it to .env:  TML_API_KEY=your_32_char_key\n"
            '  Or PowerShell:   $env:TML_API_KEY="your_32_char_key"',
            2,
        )
    return key


def validate_df(df: pd.DataFrame) -> None:
    import numpy as np

    if list(df.columns) != ["id", "score"]:
        raise ValueError(f"columns must be ['id', 'score'], got {list(df.columns)}")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(df)}")
    ids = pd.to_numeric(df["id"], errors="coerce")
    scores = pd.to_numeric(df["score"], errors="coerce")
    if ids.isna().any() or scores.isna().any():
        raise ValueError("NaN in id or score")
    if np.isinf(scores).any():
        raise ValueError("infinite scores")
    if ids.min() != 0 or ids.max() != EXPECTED_ROWS - 1:
        raise ValueError(f"id range must be 0..{EXPECTED_ROWS - 1}")
    if ids.nunique() != EXPECTED_ROWS:
        raise ValueError(f"duplicate or missing ids (unique: {ids.nunique()})")


def describe(df: pd.DataFrame) -> None:
    print(f"  rows            : {len(df):,}")
    print(f"  unique ids      : {df['id'].nunique():,}")
    print(f"  score min / max : {df['score'].min():.6f} / {df['score'].max():.6f}")
    print(f"  score mean      : {df['score'].mean():.6f}")


def submit_csv(csv_path: Path, api_key: str) -> int:
    url = f"{BASE_URL}/submit/{TASK_ID}"
    print(f"Submitting {csv_path.name}  ->  {url}")
    with open(csv_path, "rb") as f:
        resp = requests.post(
            url,
            headers={"X-API-Key": api_key},
            files={"file": (csv_path.name, f, "application/csv")},
            timeout=(10, 120),
        )

    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    print(f"HTTP status: {resp.status_code}")

    if resp.status_code == 401:
        die(f"Unauthorized — check TML_API_KEY.\nServer: {body.get('detail', body)}")
    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")
    if resp.status_code == 429:
        die(f"Rate limited — wait before retrying.\nServer: {body.get('detail', body)}")
    if not resp.ok:
        die(f"Submission failed.\nServer response: {body}")

    print("Successfully submitted.")
    print("Server response:", body)
    if submission_id := body.get("submission_id"):
        print(f"Submission ID: {submission_id}")
    return resp.status_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and submit Assignment 2 CSV.")
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Submission CSV (default: {DEFAULT_CSV.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate format only; do not upload.",
    )
    args = parser.parse_args(argv)

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        die(f"File not found: {csv_path}")

    api_key = load_api_key() if not args.validate_only else ""

    print("=" * 60)
    print(f"Step 1 — read CSV: {csv_path}")
    print("=" * 60)
    df = pd.read_csv(csv_path)
    describe(df)

    print()
    print("=" * 60)
    print("Step 2 — validate format")
    print("=" * 60)
    try:
        validate_df(df)
        print("  validation OK")
    except ValueError as e:
        die(f"  INVALID: {e}", 3)

    if args.validate_only:
        print("\nValidate-only mode — not uploading.")
        return 0

    print()
    print("=" * 60)
    print("Step 3 — upload to server")
    print("=" * 60)
    status = submit_csv(csv_path, api_key)
    return 0 if status == 200 else 4


if __name__ == "__main__":
    raise SystemExit(main())
