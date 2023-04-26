# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL @2022
import math

import torch
from mmedit.models.backbones.sr_backbones.rdn import RDB
from torch import nn

from ..registry import BACKBONES
from .base import BaseNet


class Interpolate(nn.Module):
    """
    https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588/2
    """

    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
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

    Difference to the RDN in mmedit:
    1. Support rescaling before/after enhancement.

    New args: rescale (int): rescaling factor.
    """

    def __init__(
            self,
            rescale,
            in_channels,
            out_channels,
            mid_channels=64,
            num_blocks=8,
            #  upscale_factor=4,
            num_layers=8,
            channel_growth=64):

        super().__init__()
        self.rescale = rescale
        self.mid_channels = mid_channels
        self.channel_growth = channel_growth
        self.num_blocks = num_blocks
        self.num_layers = num_layers

        assert math.log2(rescale).is_integer()
        if rescale == 1:
            self.downscale = nn.Identity()
        else:
            self.downscale = Interpolate(scale_factor=1. / rescale,
                                         mode='bicubic')

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(in_channels,
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

        # up-sampling
        # assert 2 <= upscale_factor <= 4
        # if upscale_factor == 2 or upscale_factor == 4:
        #     self.upscale = []
        #     for _ in range(upscale_factor // 2):
        #         self.upscale.extend([
        #             nn.Conv2d(
        #                 self.mid_channels,
        #                 self.mid_channels * (2**2),
        #                 kernel_size=3,
        #                 padding=3 // 2),
        #             nn.PixelShuffle(2)
        #         ])
        #     self.upscale = nn.Sequential(*self.upscale)
        # else:
        #     self.upscale = nn.Sequential(
        #         nn.Conv2d(
        #             self.mid_channels,
        #             self.mid_channels * (upscale_factor**2),
        #             kernel_size=3,
        #             padding=3 // 2), nn.PixelShuffle(upscale_factor))
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
                                out_channels,
                                kernel_size=3,
                                padding=3 // 2)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
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
