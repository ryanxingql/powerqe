# RyanXingQL @2023
from mmedit.datasets.pipelines import LoadImageFromFile, LoadImageFromFileList

from ..registry import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFileMultiKeys:
    """Load image from file.

    Differences to the LoadImageFromFile in MMEditing:
        Accept multiple keys.

    Args:
        io_backend (str): io backend where images are stored.
            Default: 'disk'.
        keys (list[str]): Keys in results to find corresponding path.
            Default: ['gt', 'lq'].
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel. Candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        convert_to (str | None): The color space of the output image.
            If None, no conversion is conducted. Default: None.
        save_original_img (bool): If True, maintain a copy of the image in
            results dict with name of f'ori_{key}'. Default: False.
        use_cache (bool): If True, load all images at once. Default: False.
        backend (str): The image loading backend type.
            Options are 'cv2', 'pillow', and 'turbojpeg'.
            Default: None.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 io_backend='disk',
                 keys=['gt', 'lq'],
                 flag='color',
                 channel_order='bgr',
                 convert_to=None,
                 save_original_img=False,
                 use_cache=False,
                 backend=None,
                 **kwargs):
        for key in keys:
            setattr(
                self, f'loader_{key}',
                LoadImageFromFile(io_backend=io_backend,
                                  key=key,
                                  flag=flag,
                                  channel_order=channel_order,
                                  convert_to=convert_to,
                                  save_original_img=save_original_img,
                                  use_cache=use_cache,
                                  backend=backend,
                                  **kwargs))

        self.keys = keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            loader = getattr(self, f'loader_{key}')
            results = loader(results)
        return results


@PIPELINES.register_module()
class LoadImageFromFileListMultiKeys():
    """Load image from file list.

    Difference to the LoadImageFromFileList in MMEditing:
        Accept multiple keys.

    It accepts a list of path and read each frame from each path.
    A list of frames will be returned.

    Args:
        io_backend (str): io backend where images are stored. Default: 'disk'.
        keys (list[str]): Keys in results to find corresponding path.
            Default: ['gt', 'lq'].
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel. Candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        convert_to (str | None): The color space of the output image.
            If None, no conversion is conducted. Default: None.
        save_original_img (bool): If True, maintain a copy of the image in
            results dict with name of f'ori_{key}'. Default: False.
        use_cache (bool): If True, load all images at once. Default: False.
        backend (str): The image loading backend type.
            Options are 'cv2', 'pillow', and 'turbojpeg'. Default: None.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 io_backend='disk',
                 keys=['gt', 'lq'],
                 flag='color',
                 channel_order='bgr',
                 convert_to=None,
                 save_original_img=False,
                 use_cache=False,
                 backend=None,
                 **kwargs):
        for key in keys:
            setattr(
                self, f'loader_{key}',
                LoadImageFromFileList(io_backend=io_backend,
                                      key=key,
                                      flag=flag,
                                      channel_order=channel_order,
                                      convert_to=convert_to,
                                      save_original_img=save_original_img,
                                      use_cache=use_cache,
                                      backend=backend,
                                      **kwargs))

        self.keys = keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            loader = getattr(self, f'loader_{key}')
            results = loader(results)
        return results
