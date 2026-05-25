"""Forensics helpers.

Keep imports lazy so callers that only need a small submodule (for example
`src.forensics.augmentation`) do not immediately pull in heavier optional
dependencies such as `sklearn`.
"""


def __getattr__(name: str):
    if name == "build_ensemble_scores":
        from src.forensics.ensemble import build_ensemble_scores

        return build_ensemble_scores
    if name == "extract_suspect_features":
        from src.forensics.pipeline import extract_suspect_features

        return extract_suspect_features
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["extract_suspect_features", "build_ensemble_scores"]
