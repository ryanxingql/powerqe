# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL @2022
from collections.abc import Sequence

from mmcv.utils import build_from_cfg
from mmedit.datasets.pipelines import Compose as MMEditCompose

from ..registry import PIPELINES


class Compose(MMEditCompose):
    """
    Difference to MMEditCompose:
        1. Use the PIPELINES in powerqe.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')
