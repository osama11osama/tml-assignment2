#!/usr/bin/env python3
"""
Score all suspect models vs the target using weight cosine similarity (baseline v001).

Higher score = more likely stolen (weights closer to target).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model_utils import cosine_similarity, flatten_weights, load_state_dict
from src.paths import NUM_SUSPECTS, SCORES_DIR, SUBMISSIONS_DIR, SUSPECT_DIR, TARGET_WEIGHTS


def suspect_path(model_id: int) -> Path:
    return SUSPECT_DIR / f"suspect_{model_id:03d}.safetensors"


def score_all(output_name: str = "submission_v001_weight_cosine.csv") -> Path:
    if not TARGET_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Target weights missing: {TARGET_WEIGHTS}\n"
            "Run: python scripts/download_models.py --target-only"
        )

    target_vec = flatten_weights(load_state_dict(str(TARGET_WEIGHTS)))
    rows = []

    for model_id in tqdm(range(NUM_SUSPECTS), desc="scoring"):
        path = suspect_path(model_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download suspects first:\n"
                "  python scripts/download_models.py"
            )
        suspect_vec = flatten_weights(load_state_dict(str(path)))
        score = cosine_similarity(target_vec, suspect_vec)
        rows.append({"id": model_id, "score": score})

    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    out_path = SUBMISSIONS_DIR / output_name
    df.to_csv(out_path, index=False)

    scores_copy = SCORES_DIR / output_name
    df.to_csv(scores_copy, index=False)

    print(f"Wrote {len(df)} rows -> {out_path}")
    print(f"Score range: [{df['score'].min():.6f}, {df['score'].max():.6f}]")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate submission CSV from weight similarity.")
    parser.add_argument(
        "--output",
        default="submission_v001_weight_cosine.csv",
        help="Output filename inside results/submissions/",
    )
    args = parser.parse_args()
    score_all(output_name=args.output)


if __name__ == "__main__":
    main()
