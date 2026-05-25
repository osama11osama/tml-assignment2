"""Master cache helpers.

Keep scoring imports lazy so target/extract code does not require the full
forensics scoring stack just to import ``src.cache_master.config``.
"""

from src.cache_master.status import count_complete, is_suspect_complete, list_incomplete


def __getattr__(name: str):
    if name in {"VARIANT_NAMES", "score_suspect_from_cache"}:
        from src.cache_master.scoring import VARIANT_NAMES, score_suspect_from_cache

        exports = {
            "VARIANT_NAMES": VARIANT_NAMES,
            "score_suspect_from_cache": score_suspect_from_cache,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "VARIANT_NAMES",
    "score_suspect_from_cache",
    "count_complete",
    "is_suspect_complete",
    "list_incomplete",
]
