"""
Evaluation utilities.

This module contains the validation loop used during training.
"""

import torch

from src.metrics import calculate_psnr


def evaluate(
    model,
    dataloader,
    criterion,
    device
):
    """
    Evaluate the model on the validation dataset.
    """

    model.eval()

    total_loss = 0
    total_psnr = 0

    with torch.no_grad():

        for lr_batch, hr_batch in dataloader:

            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            prediction = model(lr_batch)

            loss = criterion(
                prediction,
                hr_batch
            )

            psnr = calculate_psnr(
                prediction,
                hr_batch
            )

            total_loss += loss.item()
            total_psnr += psnr

    average_loss = total_loss / len(dataloader)
    average_psnr = total_psnr / len(dataloader)

    return average_loss, average_psnr