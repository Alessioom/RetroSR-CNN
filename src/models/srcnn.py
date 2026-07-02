"""
SRCNN model implementation.

Reference:
Chao Dong et al.,
Image Super-Resolution Using Deep Convolutional Networks (ECCV 2014)
"""

import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extraction = nn.Sequential(

        nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=9,
            padding=4
        ),

        nn.ReLU(inplace=True)
    )
        self.mapping = nn.Sequential(

        nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=1
        ),

        nn.ReLU(inplace=True)
    )
        self.reconstruction = nn.Conv2d(

        in_channels=32,
        out_channels=3,
        kernel_size=5,
        padding=2
    )
    def forward(self, x):

        x = self.feature_extraction(x)

        x = self.mapping(x)

        x = self.reconstruction(x)

        return x