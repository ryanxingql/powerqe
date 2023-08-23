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
            raise TypeError(
                '"pretrained" must be a string or None;'
                f' received "{type(pretrained)}".'
            )
