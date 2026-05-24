"""Central path definitions for the Assignment 2 workspace."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
# HuggingFace stores target weights at repo root: target_model/weights.safetensors
TARGET_DIR = ROOT / "target_model"
TARGET_WEIGHTS = TARGET_DIR / "weights.safetensors"
TRAIN_MAIN_IDX = DATA_DIR / "train_main_idx.json"

SUSPECT_DIR = ROOT / "suspect_models"
CIFAR100_DIR = DATA_DIR / "cifar100"

RESULTS_DIR = ROOT / "results"
SUBMISSIONS_DIR = RESULTS_DIR / "submissions"
SCORES_DIR = RESULTS_DIR / "scores"
LOGS_DIR = RESULTS_DIR / "logs"

HF_REPO = "SprintML/tml26_task2"
NUM_SUSPECTS = 360
