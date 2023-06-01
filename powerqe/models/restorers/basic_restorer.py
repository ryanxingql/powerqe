# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL @2022
import numbers
import os.path as osp

import mmcv
import numpy as np
import torch
from mmedit.core import psnr, ssim, tensor2img
from mmedit.models import BasicRestorer

from ..registry import MODELS
from .script import combine_patches, crop_img, pad_img, unfold_img


@MODELS.register_module()
class BasicQERestorer(BasicRestorer):
    """Basic restorer for quality enhancement.

    Differences to BasicRestorer:
        Support LQ vs. GT evaluation. See evaluate.
        Support saving GT and LQ. See forward_test.
        Support unfolding testing. See forward_test.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """
    supported_metrics = {'PSNR': psnr, 'SSIM': ssim}

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

    def evaluate(self, metrics, output, gt, lq):
        """Evaluation.

        Args:
            metrics (list): List of evaluation metrics.
            output (Tensor): Model output with the shape of (N=1, C, H, W).
            gt (Tensor): GT image with the shape of (N=1, C, H, W).
            lq (Tensor): LQ image with the shape of (N=1, C, H, W).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.get('crop_border', 0)

        output = tensor2img(output)
        gt = tensor2img(gt)
        lq = tensor2img(lq)

        eval_result = dict()
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(
                    f'Supported metrics include "{self.supported_metrics}";'
                    f' received "{metric}".')

            eval_result[metric] = self.supported_metrics[metric](output, gt,
                                                                 crop_border)
            eval_result[metric + '_baseline'] = self.supported_metrics[metric](
                lq, gt, crop_border)

        return eval_result

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Test forward.

        To save memory, image can be cut (or unfolded) into patches.
        Those patches can be tested separately.
        test_cfg must contains unfolding, which is a dict contains
        patchsize (patch size) and splits (number of testing splits).

        For image saving, meta_keys of Collect transform should contains
        lq_path.

        Args:
            lq (Tensor): LQ image with the shape of (N=1, C, H, W).
            gt (Tensor): GT image with the shape of (N=1, C, H, W).
                Default: None.
            meta (list): Meta information of samples.
                Default: None.
            save_image (bool): Whether to save image.
                Default: False.
            save_path (str): Path to save image.
                Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict[dict]: A dict with a single key-value pair.
                The key is eval_result;
                the value is a dict of evaluation results.
        """
        if self.test_cfg is None:
            raise ValueError(
                '"self.test_cfg" should be provided; received None.')

        if len(lq) != 1:
            raise ValueError(
                'Only one sample is allowed per batch to'
                ' (1) manage image unfolding (optional);'
                ' (2) evaluate image metrics;'
                ' (3) extract the image name for image saving (optional).')

        # inference
        if 'unfolding' in self.test_cfg:
            _cfg = self.test_cfg['unfolding']
            lq_pad, pad_info = pad_img(lq, _cfg['patchsize'])
            lq_patches, unfold_shape = unfold_img(lq_pad, _cfg['patchsize'])

            splits = _cfg['splits']
            npatches = lq_patches.shape[0]
            if splits > npatches:
                splits = npatches
            b_split = npatches // splits

            output_patches = []
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

        # denormalize before image saving and evaluation
        if 'denormalize' in self.test_cfg:
            device = output.device
            mean = torch.tensor(self.test_cfg['denormalize']['mean']).view(
                1, -1, 1, 1).to(device)
            std = torch.tensor(self.test_cfg['denormalize']['std']).view(
                1, -1, 1, 1).to(device)
            output = output * std + mean
            lq = lq * std + mean
            gt = gt * std + mean

        # save image
        if save_image:
            if len(meta) != 1:
                raise ValueError('Only one sample is allowed per batch to'
                                 ' extract the image name for image saving.')
            lq_path = meta[0]['lq_path']
            lq_name = osp.splitext(osp.basename(lq_path))[0]
            save_subpath = lq_name + '.png'

            save_gt_lq = self.test_cfg.get('save_gt_lq', True)
            if isinstance(iteration, numbers.Number):  # val during training
                if not save_gt_lq:
                    save_path_output = osp.join(save_path, f'{iteration + 1}',
                                                save_subpath)
                else:
                    save_path_output = osp.join(save_path, f'{iteration + 1}',
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

            mmcv.imwrite(tensor2img(output), save_path_output)
            if save_gt_lq:
                mmcv.imwrite(tensor2img(lq), save_path_lq)
                mmcv.imwrite(tensor2img(gt), save_path_gt)

        # evaluation
        if 'metrics' not in self.test_cfg:
            raise ValueError(
                'metrics should be provided in test_cfg for evaluation.')
        results = dict(eval_result=self.evaluate(
            metrics=self.test_cfg['metrics'], output=output, gt=gt, lq=lq))
        return results


@MODELS.register_module()
class BasicVQERestorer(BasicRestorer):
    """Basic restorer for video quality enhancement.

    Differences to BasicRestorer:
        Support LQ vs. GT testing.
        Support sequence LQ and sequence/center GT. See forward_test.
        Support parameter fix for some iters. See train_step.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
        center_gt (bool): Only the center GT is provided and evaluated.
            Default: False.
    """
    supported_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 center_gt=False):
        super().__init__(generator=generator,
                         pixel_loss=pixel_loss,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         pretrained=pretrained)

        # for evaluation
        self.center_gt = center_gt

        # fix pre-trained networks
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
        self.fix_module = train_cfg.get('fix_module', []) if train_cfg else []
        self.is_weight_fixed = False

        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))

    def train_step(self, data_batch, optimizer):
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

        # inference
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs.update({'log_vars': log_vars})

        self.step_counter += 1

        return outputs

    def evaluate(self, metrics, output, gt, lq):
        """Evaluation.

        Args:
            metrics (list): List of evaluation metrics.
            output (Tensor): Output images with the shape of (T!=1, C, H, W)
                or (C, H, W).
            gt (Tensor): GT images with the shape of (T!=1, C, H, W)
                or (C, H, W).
            lq (Tensor): LQ images with the shape of (T, C, H, W).

        Returns:
            dict: Evaluation results.
        """
        T = lq.shape[0]
        if self.center_gt and (T % 2 == 0):
            raise ValueError('Number of output frames should be odd.')

        crop_border = self.test_cfg.get('crop_border', 0)
        eval_result = dict()

        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(
                    f'Supported metrics include "{self.supported_metrics}";'
                    f' received "{metric}".')
            eval_func = self.supported_metrics[metric]

            results = dict(lq=[], output=[])
            for it in range(T):
                if self.center_gt and (it != (T // 2)):
                    continue

                if self.center_gt:
                    gt_it = tensor2img(gt)
                    output_it = tensor2img(output)
                else:
                    gt_it = tensor2img(gt[it])
                    output_it = tensor2img(output[it])
                lq_it = tensor2img(lq[it])

                result = eval_func(output_it, gt_it, crop_border)
                results['output'].append(result)

                result = eval_func(lq_it, gt_it, crop_border)
                results['lq'].append(result)

            eval_result[metric] = np.mean(results['output'])
            eval_result[metric + '_baseline'] = np.mean(results['lq'])

        return eval_result

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Test forward.

        For image saving, meta_keys of Collect transform should contains
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

        Returns:
            dict[dict]: A dict with a single key-value pair.
                The key is eval_result; the value is a dict of evaluation
                results.
        """
        if self.test_cfg is None:
            raise ValueError(
                '"self.test_cfg" should be provided; received None.')

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
        output = self.generator(lq)

        assert lq.shape[0] == 1, ('Only one sample is allowed per batch to'
                                  ' squeeze the first dim.')
        lq = lq.squeeze(0)  # (T, C, H, W)
        gt = gt.squeeze(0)  # (T, C, H, W) or (C, H, W)
        output = output.squeeze(0)  # (T, C, H, W) or (C, H, W)

        # denormalize before image saving and evaluation
        if 'denormalize' in self.test_cfg:
            device = output.device
            mean = torch.tensor(self.test_cfg['denormalize']['mean']).view(
                1, -1, 1, 1).to(device)
            std = torch.tensor(self.test_cfg['denormalize']['std']).view(
                1, -1, 1, 1).to(device)
            lq = lq * std + mean
            if gt.dim() == 3:
                mean = torch.tensor(self.test_cfg['denormalize']['mean']).view(
                    -1, 1, 1).to(device)
                std = torch.tensor(self.test_cfg['denormalize']['std']).view(
                    -1, 1, 1).to(device)
            gt = gt * std + mean
            output = output * std + mean

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
