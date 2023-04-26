# RyanXingQL @2023
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmedit.utils import get_root_logger


class BaseNet(nn.Module):
    """Base network with an init_weights function."""

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
