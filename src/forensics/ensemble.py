"""Stage 6 — rank-based ensemble (robust, no label overfit)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.forensics.aggregates import rank_normalize_series
from src.paths import NUM_SUSPECTS


def _feature_groups(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "behavioral": [],
        "representations": [],
        "augmentation": [],
        "weights": [],
        "adversarial": [],
    }
    for c in columns:
        if c == "id":
            continue
        if c.startswith("w_") or "layer_cosine" in c or "w_hist" in c:
            groups["weights"].append(c)
        elif c.startswith("aug_"):
            groups["augmentation"].append(c)
        elif c.startswith("adv_"):
            groups["adversarial"].append(c)
        elif "cka" in c or "act_cosine" in c:
            groups["representations"].append(c)
        elif any(
            x in c
            for x in ("logit", "kl", "js", "top", "margin", "conf", "calibration", "neg_")
        ):
            groups["behavioral"].append(c)
        else:
            groups["behavioral"].append(c)
    return groups


def _group_score(df: pd.DataFrame, cols: list[str], higher_suspicious: bool = True) -> pd.Series:
    if not cols:
        return pd.Series(0.0, index=df.index)
    sub = df[cols].astype(float)
    # Per-feature rank then mean rank
    ranks = [rank_normalize_series(sub[c], higher_is_suspicious=higher_suspicious) for c in cols]
    return pd.concat(ranks, axis=1).mean(axis=1)


def build_ensemble_scores(
    feature_df: pd.DataFrame,
    method: str = "rank_weighted",
    weights: dict[str, float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns id, score (higher = more suspicious).
    """
    df = feature_df.sort_values("id").reset_index(drop=True)
    if len(df) != NUM_SUSPECTS:
        raise ValueError(f"Expected {NUM_SUSPECTS} rows, got {len(df)}")

    groups = _feature_groups([c for c in df.columns if c != "id"])
    default_w = {
        "behavioral": 3.0,
        "representations": 2.0,
        "augmentation": 1.5,
        "weights": 0.25,
        "adversarial": 0.5,
    }
    w = {**default_w, **(weights or {})}

    if method == "isolation_forest":
        feat_cols = [c for c in df.columns if c != "id"]
        x = df[feat_cols].astype(float).fillna(0).values
        clf = IsolationForest(random_state=seed, contamination=0.15)
        pred = clf.fit_predict(x)  # -1 = outlier = suspicious
        score = -clf.score_samples(x)
        out = pd.DataFrame({"id": df["id"].astype(int), "score": score})
        return out.sort_values("id").reset_index(drop=True)

    # rank_weighted (default)
    total = pd.Series(0.0, index=df.index)
    weight_sum = 0.0
    for gname, cols in groups.items():
        if not cols or w.get(gname, 0) <= 0:
            continue
        # Frobenius / diff features: higher (less negative) = more similar
        higher = True
        if gname == "weights" and all("frobenius" in c or "specdiff" in c for c in cols[:1]):
            higher = True
        gs = _group_score(df, cols, higher_suspicious=higher)
        total += w[gname] * gs
        weight_sum += w[gname]

    if weight_sum > 0:
        total /= weight_sum

    out = pd.DataFrame({"id": df["id"].astype(int), "score": total.astype(float)})
    return out.sort_values("id").reset_index(drop=True)
