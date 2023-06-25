"""
Author: RyanXingQL
"""
from .builder import build_dataset
from .paired_same_size_video_dataset import (
    PairedSameSizeVideoDataset, PairedSameSizeVideoKeyAnnotationsDataset,
    PairedSameSizeVideoKeyFramesDataset)
from .registry import DATASETS

__all__ = [
    'DATASETS',
    'build_dataset',
    'PairedSameSizeVideoDataset',
    'PairedSameSizeVideoKeyFramesDataset',
    'PairedSameSizeVideoKeyAnnotationsDataset',
]
