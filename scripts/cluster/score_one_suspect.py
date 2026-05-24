#!/usr/bin/env python3
"""Cluster step 2: score one suspect vs cached target logits (array job)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.logit_scoring import load_target_logits_cache, score_suspect_logit_similarity
from src.paths import CLUSTER_SCORES_DIR, SUSPECT_DIR, TARGET_LOGITS_CACHE
from src.probe_data import get_probe_dataloader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suspect_id", type=int)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", type=Path, default=TARGET_LOGITS_CACHE)
    parser.add_argument("--out-dir", type=Path, default=CLUSTER_SCORES_DIR)
    args = parser.parse_args()

    suspect_path = SUSPECT_DIR / f"suspect_{args.suspect_id:03d}.safetensors"
    if not suspect_path.exists():
        raise FileNotFoundError(suspect_path)

    target_logits = load_target_logits_cache(args.cache)
    loader, _ = get_probe_dataloader(subset=args.subset, batch_size=args.batch_size)

    score = score_suspect_logit_similarity(
        suspect_path, target_logits, loader, device=args.device
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"suspect_{args.suspect_id:03d}.json"
    payload = {"id": args.suspect_id, "score": score, "method": "logit_cosine"}
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"suspect_{args.suspect_id:03d}: {score:.6f} -> {out_path}")


if __name__ == "__main__":
    main()
