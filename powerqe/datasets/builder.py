"""Copyright (c) OpenMMLab.

All rights reserved.
Author: RyanXingQL
"""
import platform

from mmcv.utils import build_from_cfg
from mmedit.datasets.builder import _concat_dataset
from mmedit.datasets.dataset_wrappers import RepeatDataset
from torch.utils.data import ConcatDataset

from .registry import DATASETS

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def build_dataset(cfg, default_args=None):
    """Build dataset.

    Difference to that in MMEditing: Use the DATASETS in PowerQE.
    """
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(build_dataset(cfg['dataset'], default_args),
                                cfg['times'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
