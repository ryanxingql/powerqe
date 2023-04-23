# RyanXingQL @2023
import copy
import os
import os.path as osp

from mmedit.datasets import BaseVFIDataset

from .pipelines.compose import Compose
from .registry import DATASETS


@DATASETS.register_module()
class Vimeo90KTripletCenterGTDataset(BaseVFIDataset):
    """Vimeo90K triplet dataset.

    The dataset loads three input frames and a center GT (Ground-Truth) frame.
    Then it applies specified transforms and finally returns a dict containing
    paired data and other information.

    It reads Vimeo90K keys from the txt file.
    Each line contains:

    Examples:

    ::

        00001/0389
        00001/0402

    Args:
        pipeline (list[dict | callable]): A sequence of data transformations.
        folder (str | :obj:`Path`): Path to the folder.
        ann_file (str | :obj:`Path`): Path to the annotation file.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        gt_folder: for GT im2.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{}.png'.
    """

    def __init__(self,
                 pipeline,
                 folder,
                 gt_folder,
                 ann_file,
                 test_mode=False,
                 filename_tmpl='{}.png'):
        super().__init__(pipeline, folder, ann_file, test_mode)

        self.lq_folder = self.folder
        self.gt_folder = str(gt_folder)
        self.filename_tmpl = filename_tmpl
        self.data_infos = self.load_annotations()

        # BaseDataset cannot accept the new pipeline outside MMEdit
        # we have to create pipeline manually
        self.pipeline = Compose(pipeline)

    def __getitem__(self, idx):
        """
        Different to the __getitem__ of BaseVFIDataset:
            1. Add results['scale'] = 1 for PairedRandomCrop.
        """
        results = copy.deepcopy(self.data_infos[idx])
        results['folder'] = self.folder
        results['ann_file'] = self.ann_file
        results['scale'] = 1
        return self.pipeline(results)

    def load_annotations(self):
        """Load annotations for Vimeo90K dataset.

        Returns:
            list[dict]: A list of dicts for paired paths and other information.
        """
        # get keys
        with open(self.ann_file, 'r') as f:
            keys = f.read().split('\n')
            keys = [
                k.strip() for k in keys if (k.strip() is not None and k != '')
            ]

        data_infos = []
        for key in keys:
            key = key.replace('/', os.sep)
            key_folder = osp.join(self.lq_folder, key)

            # GT: im1
            lq_path = [
                osp.join(key_folder, f'{self.filename_tmpl.format("im1")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im1")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im2")}')
            ]
            gt_path = osp.join(self.gt_folder, key, 'im1.png')
            data_infos.append(
                dict(
                    lq_path=lq_path,
                    gt_path=gt_path,
                    key=key + '/im1',
                ))

            # GT: im2
            lq_path = [
                osp.join(key_folder, f'{self.filename_tmpl.format("im1")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im2")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im3")}')
            ]
            gt_path = osp.join(self.gt_folder, key, 'im2.png')
            data_infos.append(
                dict(
                    lq_path=lq_path,
                    gt_path=gt_path,
                    key=key + '/im2',
                ))

            # GT: im3
            lq_path = [
                osp.join(key_folder, f'{self.filename_tmpl.format("im2")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im3")}'),
                osp.join(key_folder, f'{self.filename_tmpl.format("im3")}')
            ]
            gt_path = osp.join(self.gt_folder, key, 'im3.png')
            data_infos.append(
                dict(
                    lq_path=lq_path,
                    gt_path=gt_path,
                    key=key + '/im3',
                ))

        return data_infos
