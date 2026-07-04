"""
FSRCNN model implementation.

Reference:
Chao Dong et al.,
Accelerating the Super-Resolution Convolutional Neural Network (ECCV 2016)
"""

import torch.nn as nn


class FSRCNN(nn.Module):

    def __init__(
        self,
        d=56,
        s=12,
        m=4
    ):

        super().__init__()

        self.feature_extraction = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=d,
                kernel_size=5,
                padding=2
            ),

            nn.PReLU(d)
        )

        self.shrinking = nn.Sequential(

            nn.Conv2d(
                d,
                s,
                kernel_size=1
            ),

            nn.PReLU(s)
        )

        mapping = []

        for _ in range(m):

            mapping.append(

                nn.Conv2d(
                    s,
                    s,
                    kernel_size=3,
                    padding=1
                )

            )

            mapping.append(

                nn.PReLU(s)

            )

        self.mapping = nn.Sequential(*mapping)

        self.expanding = nn.Sequential(

            nn.Conv2d(
                s,
                d,
                kernel_size=1
            ),

            nn.PReLU(d)
        )

        self.reconstruction = nn.Conv2d(

            d,
            3,
            kernel_size=5,
            padding=2
        )

    def forward(self, x):

        x = self.feature_extraction(x)

        x = self.shrinking(x)

        x = self.mapping(x)

        x = self.expanding(x)

        x = self.reconstruction(x)

        return x