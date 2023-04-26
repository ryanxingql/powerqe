# RyanXingQL @2022
import torch.nn as nn

from ..registry import BACKBONES
from .base import BaseNet


@BACKBONES.register_module()
class DnCNN(BaseNet):
    """DnCNN network structure.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_blocks (int): Block number in the trunk network. Default: 15.
        if_bn (bool): If use BN layer. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 mid_channels=64,
                 num_blocks=15,
                 if_bn=False):

        super().__init__()

        layers = []

        # input conv
        layers.append(nn.Conv2d(in_channels, mid_channels, 3, padding=1))

        # body
        for _ in range(num_blocks):
            layers.append(nn.ReLU(inplace=True))
            if if_bn:
                layers += [
                    # if used before BN, unnecessary bias is turned off
                    nn.Conv2d(mid_channels,
                              mid_channels,
                              3,
                              padding=1,
                              bias=False),
                    # momentum=0.9 in
                    # https://github.com/cszn/KAIR/blob/
                    # 7e51c16c6f55ff94b59c218c2af8e6b49fe0668b/
                    # models/basicblock.py#L69
                    # but the default momentum=0.1 in PyTorch
                    nn.BatchNorm2d(num_features=mid_channels,
                                   momentum=0.9,
                                   eps=1e-04,
                                   affine=True)
                ]
            else:
                layers.append(
                    nn.Conv2d(mid_channels, mid_channels, 3, padding=1))

        # output conv
        layers += [
            nn.ReLU(inplace=True),
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
