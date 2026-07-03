"""
Training utilities.

This module contains the functions used to train the SRCNN model.
"""

import torch


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    for lr_batch, hr_batch in dataloader:

        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        optimizer.zero_grad()

        output = model(lr_batch)

        loss = criterion(output, hr_batch)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss