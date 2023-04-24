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
    """
    Difference to BasicRestorer:
         1. Support LQ vs. GT testing.
    """

    def evaluate(self, output, gt, lq):
        """
        New args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
        """
        crop_border = self.test_cfg.crop_border

        output = tensor2img(output)
        gt = tensor2img(gt)
        lq = tensor2img(lq)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
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
        """
        Difference to the forward_test of BasicRestorer:
            1. Support unfolding.
            2. Save LQ, output, and GT.
        """
        if self.test_cfg is not None and 'unfolding' in self.test_cfg:
            unfold_patch_sz = self.test_cfg.unfolding.patch_sz
            lq_pad, pad_info = pad_img(lq, unfold_patch_sz)
            lq_patches, unfold_shape = unfold_img(lq_pad, unfold_patch_sz)

            output_patches = []
            splits = self.test_cfg.unfolding.splits
            npatches = lq_patches.shape[0]
            if splits > npatches:
                splits = npatches
            b_split = npatches // splits
            for split in range(splits):
                output_patches.append(
                    self.generator(lq_patches[split * b_split:(split + 1) *
                                              b_split]))
            if splits * b_split < npatches:
                output_patches.append(
                    self.generator(lq_patches[splits * b_split:]))
            output_patches = torch.cat(output_patches, dim=0)
            output = combine_patches(output_patches, unfold_shape)
            output = crop_img(output, pad_info)
        else:
            output = self.generator(lq)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, 'Evaluation with metrics must have GT.'
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
            lq_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path_output = osp.join(save_path, 'output', lq_name,
                                            f'{iteration + 1}.png')
                save_path_lq = osp.join(save_path, 'lq', lq_name,
                                        f'{iteration + 1}.png')
                save_path_gt = osp.join(save_path, 'gt', lq_name,
                                        f'{iteration + 1}.png')
            elif iteration is None:
                save_path_output = osp.join(
                    save_path,
                    'output',
                    f'{lq_name}.png',
                )
                save_path_lq = osp.join(
                    save_path,
                    'lq',
                    f'{lq_name}.png',
                )
                save_path_gt = osp.join(
                    save_path,
                    'gt',
                    f'{lq_name}.png',
                )
            else:
                raise ValueError('"iteration" should be number or None,'
                                 f' but got {type(iteration)}.')

            mmcv.imwrite(tensor2img(output), save_path_output)
            mmcv.imwrite(tensor2img(lq), save_path_lq)
            if gt is not None:
                mmcv.imwrite(tensor2img(gt), save_path_gt)

        return results


@MODELS.register_module()
class BasicRestorerVQE(BasicRestorer):
    """
    Difference to BasicRestorer:
        1. Support LQ vs. GT testing.
        2. Support sequence LQ.
            GT corresponds to the center LQ.
    """

    def evaluate(self, output, gt, lq):
        """
        New args:
            lq (Tensor): LQ Tensor with shape (n, t, c, h, w).
        """
        crop_border = self.test_cfg.crop_border

        output = tensor2img(output)
        gt = tensor2img(gt)

        t = lq.shape[1]
        assert t % 2 == 1
        lq = tensor2img(lq[:, t // 2, ...])

        eval_result = dict()
        for metric in self.test_cfg.metrics:
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
        """
        Difference to the forward_test of BasicRestorer:
            1. Save LQ, output, and GT.
        """
        t = lq.shape[1]
        assert t % 2 == 1

        if self.test_cfg is not None and 'unfolding' in self.test_cfg:
            raise NotImplementedError(
                'Unfolding is currently not supported for video tensor.')
        else:
            output = self.generator(lq)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, 'Evaluation with metrics must have GT.'
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
            assert lq.shape[0] == 1, 'Batch size > 1 is not supported.'
            key_name = meta[0]['key']
            if isinstance(iteration, numbers.Number):
                save_path_output = osp.join(save_path, 'output', key_name,
                                            f'{iteration + 1}.png')
                save_path_lq = osp.join(save_path, 'lq', key_name,
                                        f'{iteration + 1}.png')
                save_path_gt = osp.join(save_path, 'gt', key_name,
                                        f'{iteration + 1}.png')
            elif iteration is None:
                save_path_output = osp.join(
                    save_path,
                    'output',
                    f'{key_name}.png',
                )
                save_path_lq = osp.join(
                    save_path,
                    'lq',
                    f'{key_name}.png',
                )
                save_path_gt = osp.join(
                    save_path,
                    'gt',
                    f'{key_name}.png',
                )
            else:
                raise ValueError('"iteration" should be number or None,'
                                 f' but got {type(iteration)}.')

            mmcv.imwrite(tensor2img(output), save_path_output)
            mmcv.imwrite(tensor2img(lq[:, t // 2, ...]),
                         save_path_lq)  # save the center frame
            if gt is not None:
                mmcv.imwrite(tensor2img(gt), save_path_gt)

        return results


@MODELS.register_module()
class BasicRestorerVQESequence(BasicRestorer):
    """
    Difference to BasicRestorer:
        1. Support LQ vs. GT testing.
        2. Support sequence LQ and sequence GT.
        3. Support parameter fix for some iters.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # fix pre-trained networks
        train_cfg = kwargs['train_cfg']
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
        self.fix_module = train_cfg.get('fix_module', []) if train_cfg else []
        self.is_weight_fixed = False

        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))

    def train_step(self, data_batch, optimizer):
        """
        Difference to the train_step of BasicRestorer:
            1. Support parameter fix for some iters.
        """
        # parameter fix
        if self.step_counter < self.fix_iter:
            if not self.is_weight_fixed:
                self.is_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    for fix_module in self.fix_module:
                        if fix_module in k:
                            v.requires_grad_(False)
                            break
        elif self.step_counter == self.fix_iter:
            # train all the parameters
            self.generator.requires_grad_(True)

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs

    def evaluate(self, output, gt, lq):
        """
        New args:
            lq (Tensor): LQ Tensor with shape (n, t, c, h, w).
        """
        crop_border = self.test_cfg.crop_border

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            output_results = []
            lq_results = []
            for it in range(output.shape[1]):
                output_it = tensor2img(output[:, it, ...])
                gt_it = tensor2img(gt[:, it, ...])
                lq_it = tensor2img(lq[:, it, ...])
                output_results.append(self.allowed_metrics[metric](
                    output_it,
                    gt_it,
                    crop_border,
                ))
                lq_results.append(self.allowed_metrics[metric](
                    lq_it,
                    gt_it,
                    crop_border,
                ))
            eval_result[metric + '-output'] = np.mean(output_results)
            eval_result[metric + '-LQ'] = np.mean(lq_results)
        return eval_result

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """
        Difference to the forward_test of BasicRestorer:
            1. Save LQ, output, and GT.
            2. Save sequences.
                Key: sequence name.
        """
        if self.test_cfg is not None and 'unfolding' in self.test_cfg:
            raise NotImplementedError(
                'Unfolding is currently not supported for video tensor.')
        else:
            output = self.generator(lq)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, 'Evaluation with metrics must have GT.'
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
            key_name = meta[0]['key']
            for it in range(output.shape[1]):
                if isinstance(iteration, numbers.Number):
                    save_path_output = osp.join(save_path, 'output', key_name,
                                                f'{iteration + 1}.png')
                    save_path_lq = osp.join(save_path, 'lq', key_name,
                                            f'{iteration + 1}.png')
                    save_path_gt = osp.join(save_path, 'gt', key_name,
                                            f'{iteration + 1}.png')
                elif iteration is None:
                    save_path_output = osp.join(
                        save_path,
                        'output',
                        key_name,
                        f'{it+1}.png',
                    )
                    save_path_lq = osp.join(
                        save_path,
                        'lq',
                        key_name,
                        f'{it+1}.png',
                    )
                    save_path_gt = osp.join(
                        save_path,
                        'gt',
                        key_name,
                        f'{it+1}.png',
                    )
                else:
                    raise ValueError('"iteration" should be number or None,'
                                     f' but got {type(iteration)}.')

                mmcv.imwrite(tensor2img(output[:, it, ...]), save_path_output)
                mmcv.imwrite(tensor2img(lq[:, it, ...]),
                             save_path_lq)  # save the center frame
                if gt is not None:
                    mmcv.imwrite(tensor2img(gt[:, it, ...]), save_path_gt)

        return results
