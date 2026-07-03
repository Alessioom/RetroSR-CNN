"""
Utilities for saving and loading model checkpoints.
"""

import torch


def save_model(model, filepath):
    """
    Save the model parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.

    filepath : str
        Destination path.
    """

    torch.save(model.state_dict(), filepath)


def load_model(model, filepath, device):
    """
    Load model parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Model instance.

    filepath : str
        Path of the saved checkpoint.

    device : torch.device
        Device where the model will be loaded.

    Returns
    -------
    torch.nn.Module
        Loaded model.
    """

    state_dict = torch.load(filepath, map_location=device)

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_psnr,
    filepath
):
    """
    Save a complete training checkpoint.
    """

    checkpoint = {
        "epoch": epoch,
        "best_psnr": best_psnr,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }

    torch.save(checkpoint, filepath)


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    filepath,
    device
):
    """
    Load a complete training checkpoint.
    """

    checkpoint = torch.load(
        filepath,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    model.to(device)

    start_epoch = checkpoint["epoch"] + 1
    best_psnr = checkpoint["best_psnr"]

    return (
        model,
        optimizer,
        scheduler,
        start_epoch,
        best_psnr
    )