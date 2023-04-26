# RyanXingQL @2022
import torch.nn as nn

from ..registry import BACKBONES
from .base import BaseNet


@BACKBONES.register_module()
class DCAD(BaseNet):
    """DCAD network structure.

    Paper: https://ieeexplore.ieee.org/document/7923714

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_blocks (int): Block number in the trunk network. Default: 8.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 mid_channels=64,
                 num_blocks=8):

        super().__init__()

        layers = []

        # input conv
        layers.append(nn.Conv2d(in_channels, mid_channels, 3, padding=1))

        # body
        for _ in range(num_blocks):
            layers += [
                nn.ReLU(inplace=False),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
            ]

        # output conv
        layers += [
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return self.layers(x) + x
