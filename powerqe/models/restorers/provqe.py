# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL @2022
import numbers
import os.path as osp

import mmcv
from mmcv.runner import auto_fp16
from mmedit.core import tensor2img

from ..registry import MODELS
from .basic_restorer import BasicVQERestorer


@MODELS.register_module()
class ProVQERestorer(BasicVQERestorer):
    """ProVQE restorer.

    Differences to BasicVQERestorer:
        Require key_frms in meta. See forward.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
        center_gt (bool): Only the center GT is provided and evaluated.
            Default: False.
    """

    @auto_fp16(apply_to=('lq'))
    def forward(self, lq, meta, gt=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        key_frms = [m['key_frms'] for m in meta]

        if test_mode:
            return self.forward_test(lq=lq, gt=gt, key_frms=key_frms, **kwargs)

        return self.forward_train(lq=lq, gt=gt, key_frms=key_frms)

    def forward_train(self, lq, gt, key_frms):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
            key_frms (list[list[int]]): Key-frame annotation of samples.

        Returns:
            Tensor: Output tensor.
        """
        losses = dict()
        output = self.generator(lq, key_frms)
        loss_pix = self.pixel_loss(output, gt)
        losses['loss_pix'] = loss_pix
        outputs = dict(losses=losses,
                       num_samples=len(gt.data),
                       results=dict(lq=lq.cpu(),
                                    gt=gt.cpu(),
                                    output=output.cpu()))
        return outputs

    def forward_test(self,
                     lq,
                     key_frms,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Test forward.

        For image saving, meta_keys of the transform Collect should contains
        key.

        Args:
            lq (Tensor): LQ images with the shape of (N=1, T, C, H, W)
            gt (Tensor): GT images with the shape of (N=1, T!=1, C, H, W)
                or (N=1, C, H, W). Default: None.
            meta (list): Meta information of samples. Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.
            key_frms (list[list[int]]): Key-frame annotation of samples.

        Returns:
            dict[dict]: A dict with a single key-value pair.
                The key is eval_result;
                the value is a dict of evaluation results.
        """
        if self.test_cfg is None:
            raise ValueError('"test_cfg" should be provided; received None.')

        if len(lq) != 1:
            raise ValueError(
                'Only one sample is allowed per batch to'
                ' (1) evaluate per-image metrics;'
                ' (2) extract the sequence name for saving (optional).')

        if 'unfolding' in self.test_cfg:
            raise NotImplementedError(
                'Unfolding is not supported yet for video tensor.')

        T = lq.shape[1]
        if self.center_gt and (T % 2 == 0):
            raise ValueError('Number of input frames should be odd'
                             ' when "center_gt" is True.')

        # inference
        output = self.generator(lq, key_frms)

        lq = lq.squeeze(0)  # (T, C, H, W)
        gt = gt.squeeze(0)  # (T, C, H, W) or (C, H, W)
        output = output.squeeze(0)  # (T, C, H, W) or (C, H, W)

        # save images
        if save_image:
            if len(meta) != 1:
                raise ValueError('Only one sample is allowed per batch to'
                                 ' extract the sequence name for saving.')
            key = meta[0]['key']  # sample id
            if self.center_gt:
                save_subpath = key + '.png'
            else:
                save_dir = '/'.join(key.split('/')[:-1])
                save_names = key.split('/')[-1].split(',')

            save_gt_lq = self.test_cfg.get('save_gt_lq', True)
            for it in range(T):  # note: T is the input lq idx
                if self.center_gt:  # save only the center frame
                    if it != (T // 2):
                        continue
                else:  # save every output frame
                    save_subpath = osp.join(save_dir, save_names[it] + '.png')

                if isinstance(iteration,
                              numbers.Number):  # val during training
                    if not save_gt_lq:
                        save_path_output = osp.join(save_path,
                                                    f'{iteration + 1}',
                                                    save_subpath)
                    else:
                        save_path_output = osp.join(save_path,
                                                    f'{iteration + 1}',
                                                    'output', save_subpath)
                        save_path_lq = osp.join(save_path, f'{iteration + 1}',
                                                'lq', save_subpath)
                        save_path_gt = osp.join(save_path, f'{iteration + 1}',
                                                'gt', save_subpath)
                elif iteration is None:  # testing
                    if not save_gt_lq:
                        save_path_output = osp.join(save_path, save_subpath)
                    else:
                        save_path_output = osp.join(save_path, 'output',
                                                    save_subpath)
                        save_path_lq = osp.join(save_path, 'lq', save_subpath)
                        save_path_gt = osp.join(save_path, 'gt', save_subpath)
                else:
                    raise TypeError('"iteration" should be a number or None;'
                                    f' received "{type(iteration)}".')

                if self.center_gt:
                    mmcv.imwrite(tensor2img(output), save_path_output)
                else:
                    mmcv.imwrite(tensor2img(output[it]), save_path_output)
                if save_gt_lq:
                    mmcv.imwrite(tensor2img(lq[it]), save_path_lq)
                    if self.center_gt:
                        mmcv.imwrite(tensor2img(gt), save_path_gt)
                    else:
                        mmcv.imwrite(tensor2img(gt[it]), save_path_gt)

        # evaluation
        if 'metrics' not in self.test_cfg:
            raise ValueError(
                'metrics should be provided in "test_cfg" for evaluation.')
        results = dict(eval_result=self.evaluate(
            metrics=self.test_cfg['metrics'], output=output, gt=gt, lq=lq))
        return results
