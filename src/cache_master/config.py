"""Master cache configuration (written once, read by extract + score)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.cache_master.paths import CONFIG_PATH


@dataclass
class MasterCacheConfig:
    # v2: multi-distribution forensic probes
    probe_train_40k: int = 40_000  # victim train_main (memorization fingerprint)
    probe_train_unused: int = 10_000  # CIFAR-100 train NOT in train_main (~10k)
    probe_test_10k: int = 10_000  # full CIFAR-100 test set
    probe_train_4k: int = 4_000  # subset for layer4 + legacy v003 alignment
    probe_test_2k: int = 2_000  # legacy subset
    aug_images: int = 256
    aug_views: int = 4
    aug_seed: int = 42
    version: int = 2

    def save(self, path: Path | None = None) -> Path:
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> MasterCacheConfig:
        path = path or CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Master cache config missing: {path}\n"
                "Run: python scripts/master_precompute_target.py"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        defaults = asdict(cls())
        defaults.update(data)
        return cls(**{k: defaults[k] for k in defaults})
