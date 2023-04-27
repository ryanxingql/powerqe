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

    Args:
    - `lq_folder` (str | `Path` object): Path to a lq folder.
    - `gt_folder` (str | `Path` object): Path to a gt folder.
    - `pipeline` (List[dict | callable]): A sequence of data transformations.
    - `test_mode` (bool): Store `True` when building test dataset.
    Default: `False`.
    - `filename_tmpl` (str): Template for LQ filename. Default: `{}.png`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 pipeline,
                 test_mode=False,
                 filename_tmpl='{}.png'):
        # `BaseDataset` cannot accept any pipelines outside mmedit
        # pass `[]` into `__init__`
        super().__init__(lq_folder=lq_folder,
                         gt_folder=gt_folder,
                         pipeline=[],
                         scale=1,
                         test_mode=test_mode,
                         filename_tmpl=filename_tmpl)
        self.pipeline = Compose(pipeline)

    def load_annotations(self):
        """Scan GT and LQ folders and record samples.

        The GT folder includes all images by default.
        LQ images are matches by `self.filename_tmpl`.
        LQ images can use a different image extension than GT images,
        which is indicated in `self.filename_tmpl`.

        Returns:
        - list[dict]: Sample information.
        """
        gt_paths = self.scan_folder(self.gt_folder)
        lq_paths = self.scan_folder(self.lq_folder)
        if len(gt_paths) != len(lq_paths):
            raise ValueError(
                'GT and LQ folders should have the same number of images;'
                f' received `{len(gt_paths)}` vs. `{len(lq_paths)}`.')

        data_infos = []
        for gt_path in gt_paths:
            basename, _ = osp.splitext(osp.basename(gt_path))
            lq_path = osp.join(self.lq_folder,
                               f'{self.filename_tmpl.format(basename)}')
            if lq_path not in lq_paths:
                raise FileNotFoundError(
                    f'Cannot find `{lq_path}` in `{self.lq_folder}`.')
            data_infos.append(dict(gt_path=gt_path, lq_path=lq_path))
        return data_infos
