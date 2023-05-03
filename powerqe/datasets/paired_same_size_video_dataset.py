# RyanXingQL @2023
import os
import os.path as osp

from .paired_same_size_image_dataset import PairedSameSizeImageDataset
from .registry import DATASETS


@DATASETS.register_module()
class PairedSameSizeVideoDataset(PairedSameSizeImageDataset):
    """Paired video dataset. GT and LQ are with the same size.

    Differences to `PairedSameSizeImageDataset`:
    - Support video loading. See arguments.

    Suppose the video sequences are stored as:

    ```txt
    powerqe
    `-- gt_folder
    `   `-- 001
    `   `   `-- im1.png
    `   `   `-- im2.png
    `   `   `-- ...
    `   `   `-- im7.png
    `   `-- 002
    `   `-- ...
    `   `-- 100
    `-- lq_folder
    ```

    Then the annotation file should be:

    ```txt
    001
    002
    ...
    100
    ```

    Suppose the video sequences are stored as:

    ```txt
    powerqe
    `-- gt_folder
    `   `-- 001
    `   `   `-- 0001
    `   `   `   `-- im1.png
    `   `   `   `-- im2.png
    `   `   `   `-- ...
    `   `   `   `-- im7.png
    `   `   `-- 0002
    `   `   `-- ...
    `   `   `-- 1000
    `   `-- 002
    `   `-- ...
    `   `-- 100
    `-- lq_folder
    ```

    Then the annotation file should be:

    ```txt
    001/0001
    001/0002
    ...
    001/1000
    002/0001
    ...
    100/1000
    ```

    Args:
    - `gt_folder` (str | :obj:`Path`): GT folder.
    - `lq_folder` (str | :obj:`Path`): LQ folder.
    - `ann_file` (str | :obj:`Path`): Path to the annotation file.
      Each line records a sequence path relative to the GT/LQ folder.
    - `pipeline` (List[dict | callable]): A list of data transformations.
    - `test_mode` (bool): Store `True` when building test dataset.
      Default: `False`.
    - `filename_tmpl` (str): Template for LQ filename.
      Default: `{}.png`.
    - `samp_len` (int): Sample length.
      The default value `-1` corresponds to the sequence length.
      Default: `-1`.
    - `edge_padding` (bool): Set `True` to obtain more samples.
      Value `False` is recommended for training and `True` for testing.
      Default `False`.
    - `center_gt` (bool): If `True`, only the center frame is recorded in GT.
      The `samp_len` is required to be odd.
      Note that `gt_path` is always a list.
      Default: `False`.
    """

    def __init__(self,
                 gt_folder,
                 lq_folder,
                 ann_file,
                 pipeline,
                 test_mode=False,
                 filename_tmpl='{}.png',
                 samp_len=-1,
                 edge_padding=False,
                 center_gt=False):
        # Must be defined before `super().__init__(...)`
        # for `load_annotations` in `super().__init__(...)`.
        self.ann_file = ann_file
        self.samp_len = samp_len
        self.edge_padding = edge_padding
        self.center_gt = center_gt

        super().__init__(lq_folder=lq_folder,
                         gt_folder=gt_folder,
                         pipeline=pipeline,
                         test_mode=test_mode,
                         filename_tmpl=filename_tmpl)

    def load_annotations(self):
        """Load sequences according to the annotation file.

        The GT sequence includes all frames by default.
        LQ frames are matches by `self.filename_tmpl`.
        LQ frames can use a different image extension than GT frames,
        which is indicated in `self.filename_tmpl`.

        Returns:
        - list[dict]: Each dict records the information for a sub-sequence to
          serve as a sample in training or testing.
        """
        # get keys
        with open(self.ann_file, 'r') as f:
            keys = f.read().split('\n')
            keys = [
                k.strip() for k in keys if (k.strip() is not None and k != '')
            ]
        keys = [key.replace('/', os.sep) for key in keys]

        data_infos = []
        for key in keys:
            # get frames (sorted)
            gt_seq = osp.join(self.gt_folder, key)
            lq_seq = osp.join(self.lq_folder, key)
            gt_paths = self.scan_folder(gt_seq)
            if len(gt_paths) == 0:
                raise FileNotFoundError(f'No images were found in `{gt_seq}`.')
            lq_paths = self.scan_folder(lq_seq)
            if len(gt_paths) != len(lq_paths):
                raise ValueError(
                    f'The GT and LQ sequences for key `{key}` should have'
                    ' the same number of images;'
                    f' GT has `{len(gt_paths)}` images while'
                    f' LQ has `{len(lq_paths)}` images.')
            gt_names = []
            for gt_path in sorted(gt_paths):  # NOTE: sorted
                gt_name, _ = osp.splitext(osp.basename(gt_path))
                lq_path = osp.join(lq_seq,
                                   f'{self.filename_tmpl.format(gt_name)}')
                if lq_path not in lq_paths:
                    raise FileNotFoundError(
                        f'Cannot find `{lq_path}` in `{lq_seq}`.')
                gt_names.append(gt_name)

            # check
            if self.samp_len == -1:
                self.samp_len = len(gt_paths)
            if self.samp_len > len(gt_paths):
                raise ValueError(
                    f'The sample length (`{self.samp_len}`) should not be'
                    f' larger than the sequence length (`{len(gt_paths)}`).')
            if hasattr(self, 'seq_len'):
                if len(gt_paths) != self.seq_len:
                    raise ValueError(
                        'All sequences should have the same number of images;'
                        f' found two sequences with `{len(gt_paths)}` and'
                        f' `{self.seq_len}` images.')
            else:
                self.seq_len = len(gt_paths)
            if self.center_gt and (self.samp_len % 2 == 0):
                raise ValueError(
                    f'The sample length (`{self.samp_len}`) should be odd'
                    ' when requiring center GT.')

            # record samples
            idxs = list(range(self.seq_len))
            nfrms_left = self.samp_len // 2
            nfrms_right = self.samp_len - nfrms_left - 1
            if (self.samp_len == 1) or self.edge_padding:
                center_idxs = idxs
            else:
                center_idxs = idxs[nfrms_left:(-nfrms_right)]
            for center_idx in center_idxs:
                lq_idxs = list(
                    range(center_idx - nfrms_left,
                          center_idx + nfrms_right + 1))
                lq_idxs = [max(min(x, self.seq_len - 1), 0)
                           for x in lq_idxs]  # clip
                if self.center_gt:
                    gt_idxs = [center_idx]
                else:
                    gt_idxs = lq_idxs
                samp_gt_paths = [gt_paths[idx] for idx in gt_idxs]
                samp_lq_paths = [
                    osp.join(lq_seq,
                             f'{self.filename_tmpl.format(gt_names[idx])}')
                    for idx in lq_idxs
                ]
                data_infos.append(
                    dict(gt_path=samp_gt_paths, lq_path=samp_lq_paths,
                         key=key))
        return data_infos
