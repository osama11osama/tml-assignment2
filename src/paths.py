"""Central path definitions for the Assignment 2 workspace."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
# HuggingFace stores target weights at repo root: target_model/weights.safetensors
TARGET_DIR = ROOT / "target_model"
TARGET_WEIGHTS = TARGET_DIR / "weights.safetensors"
TRAIN_MAIN_IDX = DATA_DIR / "train_main_idx.json"
TRAIN_UNUSED_IDX = DATA_DIR / "train_unused_idx.json"  # ~10k CIFAR-100 train not in victim set

SUSPECT_DIR = ROOT / "suspect_models"
CIFAR100_DIR = DATA_DIR / "cifar100"

RESULTS_DIR = ROOT / "results"
SUBMISSIONS_DIR = RESULTS_DIR / "submissions"
SCORES_DIR = RESULTS_DIR / "scores"
LOGS_DIR = RESULTS_DIR / "logs"
CACHE_DIR = RESULTS_DIR / "cache"
CACHE_MASTER_DIR = RESULTS_DIR / "cache_master"
CLUSTER_SCORES_DIR = RESULTS_DIR / "cluster_scores"
FEATURES_DIR = RESULTS_DIR / "features"
FORENSIC_TARGET_CACHE = CACHE_DIR / "forensic_target.pt"

TARGET_LOGITS_CACHE = CACHE_DIR / "target_logits_40k.pt"

HF_REPO = "SprintML/tml26_task2"
NUM_SUSPECTS = 360
