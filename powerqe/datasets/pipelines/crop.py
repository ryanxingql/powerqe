# RyanXingQL @2022 - 2023
import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module()
class PairedRandomCropQE:
    """Paired random crop for quality enhancement.

    Differences to `PairedRandomCrop` in mmedit:
    - Support user-defined keys, e.g., `lq` and `gt`.
    - Scaling is not allowed.

    Args:
    - `patch_size` (int): Patch size.
    - `keys` (Sequence[str]): Images to be transformed.
    """

    def __init__(self, patch_size, keys):
        self.patch_size = patch_size
        self.keys = keys

    def choose_coordinates(self, h, w):
        top = np.random.randint(h - self.patch_size + 1)
        left = np.random.randint(w - self.patch_size + 1)
        return top, left

    def __call__(self, results):
        """Call function.

        Args:
        - `results` (dict[list | array]): Each value is a image (list) with
        the shape of (H, W, C).

        Returns:
        - dict: Cropped images. The shape is the same to the input.
        """
        is_list_flags = dict()
        check_flag = False
        for key in self.keys:
            # turn each results[key] into a list
            if isinstance(results[key], list):
                is_list_flags[key] = True
            else:
                is_list_flags[key] = False
                results[key] = [results[key]]

            # check shapes
            h_curr, w_curr = results[key][0].shape[:2]
            if not check_flag:
                h, w = h_curr, w_curr
                if h < self.patch_size or w < self.patch_size:
                    raise ValueError(
                        f'The image size (`{h}`, `{w}`) is smaller than the'
                        ' patch size'
                        f' (`{self.patch_size}`, `{self.patch_size}`).')
                check_flag = True
            else:
                if (h_curr != h) or (w_curr != w):
                    raise ValueError('Sizes of all keys should be the same.')

        # randomly choose top and left coordinates for patching
        top, left = self.choose_coordinates(h, w)

        for key in self.keys:
            # crop
            results[key] = [
                v[top:top + self.patch_size, left:left + self.patch_size, ...]
                for v in results[key]
            ]

            # revert if not a list
            if not is_list_flags[key]:
                results[key] = results[key][0]

        return results


@PIPELINES.register_module()
class PairedCenterCrop(PairedRandomCropQE):
    """Paired center crop for quality enhancement.

    Useful for testing.

    Differences to `PairedRandomCropQE`:
    - Center cropping instead of random cropping.
    """

    def choose_coordinates(self, h, w):
        top = (h - self.patch_size) // 2
        left = (w - self.patch_size) // 2
        return top, left
