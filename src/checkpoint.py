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