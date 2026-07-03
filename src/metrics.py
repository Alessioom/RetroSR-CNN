"""
Image quality evaluation metrics.
"""

import torch


def calculate_psnr(prediction, target):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Parameters
    ----------
    prediction : torch.Tensor
        Reconstructed image.

    target : torch.Tensor
        Ground truth image.

    Returns
    -------
    float
        PSNR value in dB.
    """

    mse = torch.mean((prediction - target) ** 2)

    if mse == 0:
        return float("inf")

    psnr = 10 * torch.log10(1.0 / mse)

    return psnr.item()