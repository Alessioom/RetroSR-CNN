import torch
import torch.nn as nn


class ConvAutoencoderResidual(nn.Module):
    """
    Convolutional Autoencoder with Residual Learning
    for 2x Super Resolution.
    """

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):

        residual = self.encoder(x)

        residual = self.bottleneck(residual)

        residual = self.decoder(residual)

        output = x + residual

        output = torch.clamp(output, 0.0, 1.0)

        return output