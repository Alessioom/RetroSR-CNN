"""
Configuration file for the RetroSR-CNN project.
All project parameters are defined here.
"""

# ==========================================================
# DATASET
# ==========================================================

# High Resolution patch size
PATCH_SIZE = 96

# Super Resolution scale factor
SCALE_FACTOR = 2

# Number of image channels
CHANNELS = 3

# ==========================================================
# TRAINING
# ==========================================================

BATCH_SIZE = 16

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

NUM_WORKERS = 2

# ==========================================================
# PATHS
# ==========================================================

TRAIN_DIR = ""

VALID_DIR = ""

# ==========================================================
# RANDOM SEED
# ==========================================================

SEED = 42