"""Copyright 2023 RyanXingQL.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at
https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from ..registry import BACKBONES
from .base import BaseNet


@BACKBONES.register_module()
class DCAD(BaseNet):
    """DCAD network structure.

    Args:
        io_channels (int): Number of I/O channels.
        mid_channels (int): Channel number of intermediate features.
        num_blocks (int): Block number in the trunk network.
    """

    def __init__(self, io_channels=3, mid_channels=64, num_blocks=8):
        super().__init__()

        layers = []

        # input conv
        layers.append(nn.Conv2d(io_channels, mid_channels, 3, padding=1))

        # body
        for _ in range(num_blocks):
            layers += [
                nn.ReLU(inplace=False),
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
            ]

        # output conv
        layers += [
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, io_channels, 3, padding=1)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with the shape of (N, C, H, W).

        Returns:
            Tensor
        """
        return self.layers(x) + x
