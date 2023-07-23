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
from mmcv.runner import load_checkpoint
from mmedit.models import RRDBNet
from mmedit.models.common import default_init_weights
from mmedit.utils import get_root_logger

from ..registry import BACKBONES


@BACKBONES.register_module()
class RRDBNetQE(RRDBNet):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN and Real-ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.
    Currently, it supports [x1/x2/x4] upsampling scale factor.

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
        num_blocks (int): Block number in the trunk network.
        growth_channels (int): Channels for each growth.
        upscale_factor (int): Upsampling factor. Support x1, x2 and x4.
    """

    def __init__(self,
                 io_channels,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32,
                 upscale_factor=4):
        super().__init__(in_channels=io_channels,
                         out_channels=io_channels,
                         mid_channels=mid_channels,
                         num_blocks=num_blocks,
                         growth_channels=growth_channels,
                         upscale_factor=upscale_factor)

    def init_weights(self,
                     pretrained=None,
                     strict=True,
                     revise_keys=[(r'^module\.', '')]):
        """Init weights for models.

        Accept revise_keys for restorer ESRGANRestorer.
        Default value is equal to that of load_checkpoint.

        Args:
            pretrained (str, optional): Path for pretrained weights.
                If given None, pretrained weights will not be loaded.
            strict (boo, optional): Whether strictly load the pretrained model.
            revise_keys (list): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations.
                Default: strip the prefix 'module.' by [(r'^module\\.', '')].
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
            raise TypeError('"pretrained" must be a string or None;'
                            f' received "{type(pretrained)}".')
