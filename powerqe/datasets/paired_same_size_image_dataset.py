# RyanXingQL @2022
import os.path as osp

from mmedit.datasets import SRFolderDataset

from .pipelines.compose import Compose
from .registry import DATASETS


@DATASETS.register_module()
class PairedSameSizeImageDataset(SRFolderDataset):
    """Paired image dataset. GT and LQ are with the same size.

    Difference to `SRFolderDataset`:
    - `scale` is set to `1`.
    - Support different extensions between GT and LQ.

    Args:
    - `lq_folder` (str | :obj:`Path`): Path to a lq folder.
    - `gt_folder` (str | :obj:`Path`): Path to a gt folder.
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
        super().__init__(
            lq_folder=lq_folder,
            gt_folder=gt_folder,
            pipeline=[],  # BaseDataset cannot accept any new
            # pipelines outside MMEdit
            scale=1,
            test_mode=test_mode,
            filename_tmpl=filename_tmpl)

        # BaseDataset cannot accept any new pipelines outside MMEdit
        # we have to create pipeline manually
        self.pipeline = Compose(pipeline)

    def load_annotations(self):
        """Load annotations and record samples.

        Difference to that of `SRFolderDataset`:
        - Support different extensions between GT and LQ
        by `self.filename_tmpl`.
        """
        data_infos = []
        lq_paths = self.scan_folder(self.lq_folder)
        gt_paths = self.scan_folder(self.gt_folder)
        if len(lq_paths) != len(gt_paths):
            raise ValueError(
                'GT and LQ folders should have the same number of images;'
                f' received `{len(lq_paths)}` vs. `{len(gt_paths)}`.')

        for gt_path in gt_paths:
            basename, _ = osp.splitext(osp.basename(gt_path))
            lq_path = osp.join(self.lq_folder,
                               f'{self.filename_tmpl.format(basename)}')
            if lq_path not in lq_paths:
                raise FileNotFoundError(
                    f'Cannot find `{lq_path}` in `{self.lq_folder}`.')
            data_infos.append(dict(lq_path=lq_path, gt_path=gt_path))
        return data_infos
