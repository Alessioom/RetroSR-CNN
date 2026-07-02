"""
Utility functions for creating PyTorch DataLoaders.
"""

from torch.utils.data import DataLoader

from src.config import (
    DIV2K_TRAIN_DIR,
    BATCH_SIZE,
    NUM_WORKERS
)

from src.dataset import DIV2KDataset


def get_train_loader():
    """
    Create the DataLoader used during training.

    Returns
    -------
    DataLoader
        PyTorch DataLoader for the DIV2K training dataset.
    """

    dataset = DIV2KDataset(DIV2K_TRAIN_DIR)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    return train_loader