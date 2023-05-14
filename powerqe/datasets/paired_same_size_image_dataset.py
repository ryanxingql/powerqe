# RyanXingQL @2022
import os.path as osp

from mmedit.datasets import SRFolderDataset

from .pipelines.compose import Compose
from .registry import DATASETS


@DATASETS.register_module()
class PairedSameSizeImageDataset(SRFolderDataset):
    """Paired image dataset. GT and LQ are with the same size.

    Differences to `SRFolderDataset`:
    - `scale` is set to `1`.
    - Support different extensions between GT and LQ. See `load_annotations`.
    - Use the `Compose` in powerqe.

    Args:
    - `lq_folder` (str | :obj:`Path`): LQ folder.
    - `gt_folder` (str | :obj:`Path`): GT folder.
    - `pipeline` (List[dict | callable]): A sequence of data transformations.
    - `test_mode` (bool): Store `True` when building test dataset.
      Default: `False`.
    - `lq_ext` (str): Extension of LQ filenames.
      Default: `.png`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 pipeline,
                 test_mode=False,
                 lq_ext='.png'):
        self.lq_ext = lq_ext
        # `BaseDataset` cannot accept any pipelines outside mmedit;
        # Pass `[]` into `__init__`.
        super().__init__(lq_folder=lq_folder,
                         gt_folder=gt_folder,
                         pipeline=[],
                         scale=1,
                         test_mode=test_mode)
        self.pipeline = Compose(pipeline)

    def load_annotations(self):
        """Scan GT and LQ folders and record samples.

        The GT folder includes all images by default.
        LQ images are matches by `self.lq_ext`.
        LQ images can use a different image extension than GT images,
        which is indicated in `self.lq_ext`.

        Returns:
        - list[dict]: Sample information.
        """
        gt_paths = self.scan_folder(self.gt_folder)
        lq_paths = self.scan_folder(self.lq_folder)
        if len(gt_paths) != len(lq_paths):
            raise ValueError(
                'GT and LQ folders should have the same number of images;'
                f' found `{len(gt_paths)}` and `{len(lq_paths)}` images,'
                ' respectively.')

        data_infos = []
        for gt_path in gt_paths:
            basename, _ = osp.splitext(osp.basename(gt_path))
            lq_path = osp.join(self.lq_folder, basename + self.lq_ext)
            if lq_path not in lq_paths:
                raise FileNotFoundError(
                    f'Cannot find `{lq_path}` in `{self.lq_folder}`.')
            data_infos.append(dict(gt_path=gt_path, lq_path=lq_path))
        return data_infos
