"""
Project configuration.
"""

from pathlib import Path

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

DIV2K_TRAIN_DIR = DATA_DIR / "DIV2K" / "train"

DIV2K_VALID_DIR = DATA_DIR / "DIV2K" / "valid"

# ==========================================================
# DATASET
# ==========================================================

PATCH_SIZE = 96

SCALE_FACTOR = 2

CHANNELS = 3

# ==========================================================
# TRAINING
# ==========================================================

BATCH_SIZE = 16

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

NUM_WORKERS = 2

NUM_CHANNELS = 3

# ==========================================================
# RANDOMNESS
# ==========================================================

SEED = 42


# DataLoader
BATCH_SIZE = 16
NUM_WORKERS = 0
