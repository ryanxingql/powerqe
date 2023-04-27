# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL @2022
from mmcv.runner import load_checkpoint
from mmedit.models import RRDBNet
from mmedit.models.common import default_init_weights
from mmedit.utils import get_root_logger

from ..registry import BACKBONES


@BACKBONES.register_module()
class RRDBNetQE(RRDBNet):
    """See RRDBNet."""

    def init_weights(self,
                     pretrained=None,
                     strict=True,
                     revise_keys=[(r'^module\.', '')]):
        """Init weights for models.

        Accept `revise_keys` for restorer `ESRGANQE`. Default value is equal to
        that of `load_checkpoint`.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self,
                            pretrained,
                            strict=strict,
                            logger=logger,
                            revise_keys=revise_keys)
        elif pretrained is None:
            # Use smaller std for better stability and performance. We
            # use 0.1. See more details in "ESRGAN: Enhanced Super-Resolution
            # Generative Adversarial Networks"
            for m in [
                    self.conv_first, self.conv_body, self.conv_up1,
                    self.conv_up2, self.conv_hr, self.conv_last
            ]:
                default_init_weights(m, 0.1)
        else:
            raise TypeError('`pretrained` must be a string or `None`;'
                            f' received `{type(pretrained)}`.')
