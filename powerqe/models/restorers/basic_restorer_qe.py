# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL @2022
import numbers
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmedit.core import tensor2img
from mmedit.models import BasicRestorer

from ..registry import MODELS


def cal_diff(sz, sz_mul):
    diff_sz = int(np.ceil(sz / sz_mul)) * sz_mul - sz
    return diff_sz


def pad_img(img, sz_mul):
    """
    img (tensor): (b c h w), float
    pad_info (tuple): (pad_left, pad_right, pad_top, pad_bottom)
    """
    h, w = img.shape[2:]
    assert (h >= sz_mul) and (w >= sz_mul)
    diff_h, diff_w = cal_diff(h, sz_mul), cal_diff(w, sz_mul)
    pad_info = ((diff_w // 2), (diff_w - diff_w // 2), (diff_h // 2),
                (diff_h - diff_h // 2))
    img_pad = F.pad(img, pad_info, mode='reflect')
    return img_pad, pad_info


def unfold_img(img, patch_sz):
    """
    img (tensor): (b c h w)

    https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html
    """
    patches = img.unfold(2, patch_sz, patch_sz).unfold(3, patch_sz, patch_sz)
    # unfold h: b c num_patch_h w patch_sz
    # unfold w: b c num_patch_h num_patch_w patch_sz patch_sz

    patches = patches.permute(0, 2, 3, 1, 4, 5)
    unfold_shape = patches.size()  # for reverting
    # b num_patch_h num_patch_w c patch_sz patch_sz
    patches = patches.contiguous().view(-1, img.size(1), patch_sz, patch_sz)
    # b*num_patch_h*num_patch_w c patch_sz patch_sz
    return patches, unfold_shape


def combine_patches(patches, unfold_shape):
    """
    patches (tensor): (b*num_patch_h*num_patch_w c patch_sz patch_sz)
    unfold_shape (tuple): (b num_patch_h num_patch_w c patch_sz patch_sz)
    """
    b, c = unfold_shape[0], unfold_shape[3]
    h_pad = unfold_shape[1] * unfold_shape[4]
    w_pad = unfold_shape[2] * unfold_shape[5]

    img = patches.view(
        unfold_shape)  # b num_patch_h num_patch_w c patch_sz patch_sz
    img = img.permute(
        0, 3, 1, 4, 2,
        5).contiguous()  # b c num_patch_h patch_sz num_patch_w patch_sz
    img = img.view(b, c, h_pad, w_pad)
    return img


def crop_img(img, pad_info):
    """
    img (tensor): (b c h w)
    pad_info (tuple): (pad_left, pad_right, pad_top, pad_bottom)
    """
    if pad_info[3] == 0 and pad_info[1] == 0:
        img = img[..., pad_info[2]:, pad_info[0]:]
    elif pad_info[3] == 0:
        img = img[..., pad_info[2]:, pad_info[0]:-pad_info[1]]
    elif pad_info[1] == 0:
        img = img[..., pad_info[2]:-pad_info[3], pad_info[0]:]
    else:
        img = img[..., pad_info[2]:-pad_info[3], pad_info[0]:-pad_info[1]]
    return img


@MODELS.register_module()
class BasicRestorerQE(BasicRestorer):
    """Support LQ vs. GT testing for BasicRestorer."""

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
        if self.test_cfg is not None and 'unfolding' in self.test_cfg:
            unfold_patch_sz = self.test_cfg.unfolding.patch_sz
            lq_pad, pad_info = pad_img(lq, unfold_patch_sz)
            lq_patches, unfold_shape = unfold_img(lq_pad, unfold_patch_sz)
            output_patches = []
            splits = self.test_cfg.unfolding.splits
            b_split = lq_patches.shape[0] // splits
            for split in range(splits):
                output_patches.append(
                    self.generator(lq_patches[split * b_split:(split + 1) *
                                              b_split]))
            if splits * b_split < lq_patches.shape[0]:
                output_patches.append(
                    self.generator(lq_patches[splits * b_split:]))
            output_patches = torch.cat(output_patches, dim=0)
            output = combine_patches(output_patches, unfold_shape)
            output = crop_img(output, pad_info)
        else:
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
