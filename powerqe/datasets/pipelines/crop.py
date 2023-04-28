# RyanXingQL @2022
from ..registry import PIPELINES


@PIPELINES.register_module()
class PairedCenterCrop:
    """Paired center crop.

    It crops a pair of LQ and GT images with corresponding locations.
    It also supports accepting LQ list and GT list.

    Required keys are `scale`, `lq`, and `gt`.
    Added or modified keys are `lq` and `gt`.

    Args:
    - `gt_patch_size` (int): Patch size for cropping GT.
    """

    def __init__(self, gt_patch_size):
        self.gt_patch_size = gt_patch_size

    def __call__(self, results):
        """Call.

        Args:
        - `results` (dict): A dict containing the necessary information and
        data for augmentation.

        Returns:
        - dict: Cropped LQ and GT.
        """
        scale = results['scale']
        lq_patch_size = self.gt_patch_size // scale

        lq_is_list = isinstance(results['lq'], list)
        if not lq_is_list:
            results['lq'] = [results['lq']]
        gt_is_list = isinstance(results['gt'], list)
        if not gt_is_list:
            results['gt'] = [results['gt']]

        h_lq, w_lq, _ = results['lq'][0].shape
        h_gt, w_gt, _ = results['gt'][0].shape

        if (h_gt != h_lq * scale) or (w_gt != w_lq * scale):
            raise ValueError(f'The GT size (`{h_gt}`, `{w_gt}`) should be'
                             f' `{scale}` times'
                             f' the LQ size (`{h_lq}`, `{w_lq}`).')
        if (h_lq < lq_patch_size) or (w_lq < lq_patch_size):
            raise ValueError(
                f'The LQ size (`{h_lq}`, `{w_lq}`) is smaller than the'
                f' patch size (`{lq_patch_size}`, `{lq_patch_size}`).')

        # center cropping offsets for lq
        top = (h_lq - lq_patch_size) // 2
        left = (w_lq - lq_patch_size) // 2

        # crop lq patch
        results['lq'] = [
            v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
            for v in results['lq']
        ]

        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        results['gt'] = [
            v[top_gt:top_gt + self.gt_patch_size,
              left_gt:left_gt + self.gt_patch_size, ...] for v in results['gt']
        ]

        if not lq_is_list:
            results['lq'] = results['lq'][0]
        if not gt_is_list:
            results['gt'] = results['gt'][0]
        return results
