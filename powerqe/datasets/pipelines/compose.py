# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL @2022
from collections.abc import Sequence

from mmcv.utils import build_from_cfg
from mmedit.datasets.pipelines import Compose as MMEditCompose

from ..registry import PIPELINES


class Compose(MMEditCompose):
    """Compose a data pipeline with a sequence of transforms.

    Differences to `MMEditCompose`:
    - Use the `PIPELINES` in powerqe.
    """

    def __init__(self, transforms):
        if not isinstance(transforms, Sequence):
            raise TypeError(
                '`transforms` should be an instance of `Sequence`.')
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('`transform` should be callable or a dict;'
                                f' received `{type(transform)}`.')
