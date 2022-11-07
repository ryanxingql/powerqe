# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL, 2022
import numbers
import os.path as osp

import mmcv
from mmedit.core import tensor2img
from mmedit.models import BasicRestorer

from ..registry import MODELS


@MODELS.register_module()
class BasicRestorerQE(BasicRestorer):
    """Support LQ vs. GT testing.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """
    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(generator=generator,
                         pixel_loss=pixel_loss,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         pretrained=pretrained)

    # def evaluate(self, output, gt):
    def evaluate(self, output, gt, lq):
        """Evaluation function.

        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).
            lq (Tensor): LQ Tensor with shape (n, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border

        output = tensor2img(output)
        gt = tensor2img(gt)
        lq = tensor2img(lq)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            # eval_result[metric] = self.allowed_metrics[metric](output, gt,
            #                                                    crop_border)
            eval_result[metric + '-output'] = self.allowed_metrics[metric](
                output, gt, crop_border)
            eval_result[metric + '-LQ'] = self.allowed_metrics[metric](
                lq, gt, crop_border)
        return eval_result

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        output = self.generator(lq)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            # results = dict(eval_result=self.evaluate(output, gt))
            results = dict(eval_result=self.evaluate(
                output=output,
                gt=gt,
                lq=lq,
            ))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                """
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
                """
                save_path_output = osp.join(
                    save_path, folder_name, 'output',
                    f'{folder_name}-{iteration + 1:06d}.png')
                save_path_lq = osp.join(
                    save_path, folder_name, 'lq',
                    f'{folder_name}-{iteration + 1:06d}.png')
                save_path_gt = osp.join(
                    save_path, folder_name, 'gt',
                    f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                # save_path = osp.join(save_path, f'{folder_name}.png')
                save_path_output = osp.join(
                    save_path,
                    'output',
                    f'{folder_name}.png',
                )
                save_path_lq = osp.join(
                    save_path,
                    'lq',
                    f'{folder_name}.png',
                )
                save_path_gt = osp.join(
                    save_path,
                    'gt',
                    f'{folder_name}.png',
                )
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')

            mmcv.imwrite(tensor2img(output), save_path_output)
            mmcv.imwrite(tensor2img(lq), save_path_lq)
            if gt is not None:
                mmcv.imwrite(tensor2img(gt), save_path_gt)

        return results
