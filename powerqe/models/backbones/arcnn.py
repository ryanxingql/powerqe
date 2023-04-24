# RyanXingQL @2022
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmedit.utils import get_root_logger

from ..registry import BACKBONES


@BACKBONES.register_module()
class ARCNN(nn.Module):
    """AR-CNN network structure.

    Paper: https://arxiv.org/pdf/1504.06993.pdf

    Args:
        in_channels (int): Channel number of inputs.
            Default: 3.
        mid_channels_1 (int): Channel number of the first intermediate
            features.
            Default: 64.
        mid_channels_2 (int): Channel number of the second intermediate
            features.
            Default: 32.
        mid_channels_3 (int): Channel number of the third intermediate
            features.
            Default: 16.
        out_channels (int): Channel number of outputs.
            Default: 3.
        in_kernel_size (int): Kernel size of the first convolution.
            Default: 9.
        mid_kernel_size (int): Kernel size of the first intermediate
            convolution.
            Default: 7.
        mid_kernel_size (int): Kernel size of the second intermediate
            convolution.
            Default: 1.
        out_kernel_size (int): Kernel size of the last convolution.
            Default: 5.
    """

    def __init__(self,
                 in_channels=3,
                 mid_channels_1=64,
                 mid_channels_2=32,
                 mid_channels_3=16,
                 out_channels=3,
                 in_kernel_size=9,
                 mid_kernel_size_1=7,
                 mid_kernel_size_2=1,
                 out_kernel_size=5):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels_1,
                      in_kernel_size,
                      padding=in_kernel_size // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels_1,
                      mid_channels_2,
                      mid_kernel_size_1,
                      padding=mid_kernel_size_1 // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels_2,
                      mid_channels_3,
                      mid_kernel_size_2,
                      padding=mid_kernel_size_2 // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels_3,
                      out_channels,
                      out_kernel_size,
                      padding=out_kernel_size // 2),
        )

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return self.layers(x) + x

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass  # use default initialization
        else:
            raise TypeError('"pretrained" must be a str or None.'
                            f' But received {type(pretrained)}.')
