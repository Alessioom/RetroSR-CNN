"""
Image quality evaluation metrics.
"""

import torch
from skimage.metrics import structural_similarity as ssim


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

def calculate_ssim(prediction, target):
    """
    Compute Structural Similarity Index (SSIM).

    Parameters
    ----------
    prediction : torch.Tensor
        Reconstructed image.

    target : torch.Tensor
        Ground truth image.

    Returns
    -------
    float
        SSIM value.
    """

    prediction = prediction.detach().cpu()
    target = target.detach().cpu()

    batch_size = prediction.shape[0]

    total_ssim = 0.0

    for i in range(batch_size):

        pred = prediction[i].permute(1, 2, 0).numpy()
        gt = target[i].permute(1, 2, 0).numpy()

        total_ssim += ssim(
            gt,
            pred,
            channel_axis=2,
            data_range=1.0
        )

    return total_ssim / batch_size