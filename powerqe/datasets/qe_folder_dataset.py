# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL, 2022
import os.path as osp

from mmedit.datasets import SRFolderDataset

from .registry import DATASETS


@DATASETS.register_module()
class QEFolderDataset(SRFolderDataset):
    """General paired image folder dataset for image restoration.

    Adapted from the SRFolderDataset in MMEditing 0.15.
    Difference:
    1. Scale == 1
    2. Support different extension between GT and LQ

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{}.png'.
    """
    def __init__(self,
                 lq_folder,
                 gt_folder,
                 pipeline,
                 test_mode=False,
                 filename_tmpl='{}.png'):
        super().__init__(lq_folder=lq_folder,
                         gt_folder=gt_folder,
                         pipeline=pipeline,
                         scale=1,
                         test_mode=test_mode,
                         filename_tmpl=filename_tmpl)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.filename_tmpl = filename_tmpl
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotations for SR dataset.

        It loads the LQ and GT image path from folders.

        Returns:
            list[dict]: A list of dicts for paired paths of LQ and GT.
        """
        data_infos = []
        lq_paths = self.scan_folder(self.lq_folder)
        gt_paths = self.scan_folder(self.gt_folder)
        assert len(lq_paths) == len(gt_paths), (
            f'gt and lq datasets have different number of images: '
            f'{len(lq_paths)}, {len(gt_paths)}.')
        for gt_path in gt_paths:
            basename, _ = osp.splitext(osp.basename(gt_path))
            lq_path = osp.join(self.lq_folder,
                               (f'{self.filename_tmpl.format(basename)}'))
            assert lq_path in lq_paths, f'{lq_path} is not in lq_paths.'
            data_infos.append(dict(lq_path=lq_path, gt_path=gt_path))
        return data_infos
