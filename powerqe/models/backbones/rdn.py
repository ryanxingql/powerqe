"""Copyright (c) OpenMMLab. All rights reserved.

Copyright 2023 RyanXingQL

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

import math

import torch
from mmedit.models.backbones.sr_backbones.rdn import RDB
from torch import nn

from ..registry import BACKBONES
from .base import BaseNet


class Interpolate(nn.Module):
    # Ref: "https://discuss.pytorch.org/t
    # /using-nn-function-interpolate-inside-nn-sequential/23588/2"

    def __init__(self, scale_factor, mode):
        super().__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x,
                        scale_factor=self.scale_factor,
                        mode=self.mode,
                        align_corners=False)
        return x


@BACKBONES.register_module()
class RDNQE(BaseNet):
    """RDN for quality enhancement.

    Differences to the RDN in MMEditing:
        Support rescaling before/after enhancement.

    Args:
        rescale (int): Rescaling factor.
        io_channels (int): Number of I/O channels.
        mid_channels (int): Channel number of intermediate features.
        num_blocks (int): Block number in the trunk network.
        upscale_factor (int): Upsampling factor. Support 2^n and 3.
        num_layer (int): Layer number in the Residual Dense Block.
        channel_growth (int): Channels growth in each layer of RDB.
    """

    def __init__(
            self,
            rescale,
            io_channels,
            mid_channels=64,
            num_blocks=8,
            # upscale_factor=4,
            num_layers=8,
            channel_growth=64):
        super().__init__()

        self.rescale = rescale
        self.mid_channels = mid_channels
        self.channel_growth = channel_growth
        self.num_blocks = num_blocks
        self.num_layers = num_layers

        if not math.log2(rescale).is_integer():
            raise ValueError(
                f'Rescale factor ({rescale}) should be a power of 2.')

        if rescale == 1:
            self.downscale = nn.Identity()
        else:
            self.downscale = Interpolate(scale_factor=1. / rescale,
                                         mode='bicubic')

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(io_channels,
                              mid_channels,
                              kernel_size=3,
                              padding=3 // 2)
        self.sfe2 = nn.Conv2d(mid_channels,
                              mid_channels,
                              kernel_size=3,
                              padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList()
        for _ in range(self.num_blocks):
            self.rdbs.append(
                RDB(self.mid_channels, self.channel_growth, self.num_layers))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.mid_channels * self.num_blocks,
                      self.mid_channels,
                      kernel_size=1),
            nn.Conv2d(self.mid_channels,
                      self.mid_channels,
                      kernel_size=3,
                      padding=3 // 2))

        # upsampling
        if rescale == 1:
            self.upscale = nn.Identity()
        else:
            self.upscale = []
            for _ in range(rescale // 2):
                self.upscale.extend([
                    nn.Conv2d(self.mid_channels,
                              self.mid_channels * (2**2),
                              kernel_size=3,
                              padding=3 // 2),
                    nn.PixelShuffle(2)
                ])
            self.upscale = nn.Sequential(*self.upscale)

        self.output = nn.Conv2d(self.mid_channels,
                                io_channels,
                                kernel_size=3,
                                padding=3 // 2)

    def forward(self, x):
        """Forward.

        Args:
            x (Tensor): Input tensor with the shape of (N, C, H, W).

        Returns:
            Tensor
        """
        x = self.downscale(x)

        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features,
                               1)) + sfe1  # global residual learning

        x = self.upscale(x)
        x = self.output(x)
        return x
