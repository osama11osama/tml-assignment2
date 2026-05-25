"""Paths and stage names for master cache layout."""

from __future__ import annotations

from pathlib import Path

from src.paths import CACHE_DIR, NUM_SUSPECTS, RESULTS_DIR

CACHE_MASTER_DIR = RESULTS_DIR / "cache_master"
TARGET_DIR = CACHE_MASTER_DIR / "target"
SUSPECTS_DIR = CACHE_MASTER_DIR / "suspects"
CLAIMS_DIR = CACHE_MASTER_DIR / "claims"
CONFIG_PATH = CACHE_MASTER_DIR / "config.json"
PROGRESS_PATH = CACHE_MASTER_DIR / "progress.json"
VARIANTS_DIR = CACHE_MASTER_DIR / "variants"

# Artifacts saved per model (target or suspect)
STAGE_TRAIN40K = "train40k_logits"
STAGE_TRAIN_UNUSED = "train_unused_logits"
STAGE_TEST10K = "test10k_logits"
STAGE_TRAIN4K_LOGITS = "train4k_logits"
STAGE_TRAIN4K_LAYER4 = "train4k_layer4"
STAGE_TEST2K_LOGITS = "test2k_logits"
STAGE_TEST2K_LAYER4 = "test2k_layer4"
STAGE_AUG_BASE = "aug_base_logits"
STAGE_AUG_VIEWS = "aug_view_logits"

ALL_STAGES = (
    STAGE_TRAIN40K,
    STAGE_TRAIN_UNUSED,
    STAGE_TEST10K,
    STAGE_TRAIN4K_LOGITS,
    STAGE_TRAIN4K_LAYER4,
    STAGE_TEST2K_LOGITS,
    STAGE_TEST2K_LAYER4,
    STAGE_AUG_BASE,
    STAGE_AUG_VIEWS,
)

STAGE_FILES = {
    STAGE_TRAIN40K: "train40k_logits.fp16.pt",
    STAGE_TRAIN_UNUSED: "train_unused_logits.fp16.pt",
    STAGE_TEST10K: "test10k_logits.fp16.pt",
    STAGE_TRAIN4K_LOGITS: "train4k_logits.fp16.pt",
    STAGE_TRAIN4K_LAYER4: "train4k_layer4.fp16.pt",
    STAGE_TEST2K_LOGITS: "test2k_logits.fp16.pt",
    STAGE_TEST2K_LAYER4: "test2k_layer4.fp16.pt",
    STAGE_AUG_BASE: "aug_base_logits.fp16.pt",
    STAGE_AUG_VIEWS: "aug_view_logits.fp16.pt",
}

AUG_INPUTS_FILE = "aug_inputs.fp16.pt"


def suspect_dir(suspect_id: int) -> Path:
    return SUSPECTS_DIR / f"suspect_{suspect_id:03d}"


def stage_path(root: Path, stage: str) -> Path:
    return root / STAGE_FILES[stage]


def manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def claim_path(suspect_id: int) -> Path:
    return CLAIMS_DIR / f"suspect_{suspect_id:03d}.claim"
