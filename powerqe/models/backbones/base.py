"""
Author: RyanXingQL
"""
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmedit.utils import get_root_logger


class BaseNet(nn.Module):
    """Base network with the function init_weights."""

    def __init__(self) -> None:
        super().__init__()

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str): Path for pretrained weights.
                If given None, pretrained weights will not be loaded.
                Default: None.
            strict (bool): Whether strictly load the pretrained model.
                Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass  # use default initialization
        else:
            raise TypeError('"pretrained" must be a string or None;'
                            f' received "{type(pretrained)}".')
