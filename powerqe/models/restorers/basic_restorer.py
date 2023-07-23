"""Copyright 2023 RyanXingQL.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at
https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numbers
import os.path as osp

import mmcv
import numpy as np
import torch
from mmedit.core import tensor2img
from mmedit.models import BasicRestorer

from ...utils.unfolding import (combine_patches, crop_img, pad_img_min_sz,
                                pad_img_sz_mul, unfold_img)
from ..registry import MODELS


@MODELS.register_module()
class BasicQERestorer(BasicRestorer):
    """Basic restorer for quality enhancement.

    Differences to BasicRestorer:
        Support padding testing. See forward_test.
        Support unfolding testing. See forward_test.
        Support de-normalization for image saving.

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
        # Check
        assert self.test_cfg is not None, (
            '"self.test_cfg" should be provided; received None.')

        assert len(lq) == 1, ('Only one sample is allowed per batch to'
                              ' (1) manage image unfolding;'
                              ' (2) extract image names for image saving;'
                              ' (3) evaluate image metrics.')

        assert 'metrics' in self.test_cfg, (
            'metrics should be provided in test_cfg for evaluation.')

        # Inference
        if 'padding' in self.test_cfg:
            _cfg = self.test_cfg['padding']
            lq, pad_info = pad_img_min_sz(lq, _cfg['minSize'])

        if 'unfolding' in self.test_cfg:
            _cfg = self.test_cfg['unfolding']
            lq_pad, pad_info_unfold = pad_img_sz_mul(lq, _cfg['patchsize'])
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
            output = crop_img(output, pad_info_unfold)
        else:
            output = self.generator(lq)

        if 'padding' in self.test_cfg:
            output = crop_img(output, pad_info)

        # De-normalize before image saving and evaluation
        if 'denormalize' in self.test_cfg:
            device = output.device
            mean = torch.tensor(self.test_cfg['denormalize']['mean']).view(
                1, -1, 1, 1).to(device)
            std = torch.tensor(self.test_cfg['denormalize']['std']).view(
                1, -1, 1, 1).to(device)
            output = output * std + mean
            gt = gt * std + mean

        # Save image
        if save_image:
            lq_path = meta[0]['lq_path']
            lq_name = osp.splitext(osp.basename(lq_path))[0]
            save_subpath = lq_name + '.png'

            if isinstance(iteration, numbers.Number):  # val during training
                _save_path = osp.join(save_path, f'{iteration + 1}',
                                      save_subpath)
            elif iteration is None:  # testing
                _save_path = osp.join(save_path, save_subpath)
            else:
                raise TypeError('"iteration" should be a number or None;'
                                f' received "{type(iteration)}".')

            mmcv.imwrite(tensor2img(output), _save_path)

        # Evaluation
        results = dict(eval_result=self.evaluate(output=output, gt=gt))
        return results


@MODELS.register_module()
class BasicVQERestorer(BasicRestorer):
    """Basic restorer for video quality enhancement.

    Differences to BasicRestorer:
        Support padding testing. See forward_test.
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

        # For evaluation
        self.center_gt = center_gt

        # Fix pre-trained networks
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
        self.fix_module = train_cfg.get('fix_module', []) if train_cfg else []
        self.is_weight_fixed = False

        # Count training steps
        self.register_buffer('step_counter', torch.zeros(1))

    def train_step(self, data_batch, optimizer):
        # Fix parameter
        if self.step_counter < self.fix_iter:
            if not self.is_weight_fixed:
                self.is_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    for fix_module in self.fix_module:
                        if fix_module in k:
                            v.requires_grad_(False)
                            break
        elif self.step_counter == self.fix_iter:
            # Train all the parameters
            self.generator.requires_grad_(True)

        # Inference
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # Optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs.update({'log_vars': log_vars})

        self.step_counter += 1

        return outputs

    def evaluate(self, metrics, output, gt):
        """Evaluation.

        Args:
            metrics (list): List of evaluation metrics.
            output (Tensor): Output images with the shape of (T!=1, C, H, W)
                or (C, H, W).
            gt (Tensor): GT images with the shape of (T!=1, C, H, W)
                or (C, H, W).

        Returns:
            dict: Evaluation results.
        """
        T = gt.shape[0]
        if self.center_gt and (T % 2 == 0):
            raise ValueError('Number of output frames should be odd.')

        crop_border = self.test_cfg.get('crop_border', 0)
        eval_result = dict()

        for metric in metrics:
            if metric not in self.allowed_metrics:
                raise ValueError(
                    f'Supported metrics include "{self.allowed_metrics}";'
                    f' received "{metric}".')
            eval_func = self.allowed_metrics[metric]

            results = []
            for it in range(T):
                if self.center_gt and (it != (T // 2)):
                    continue

                if self.center_gt:
                    gt_it = tensor2img(gt)
                    output_it = tensor2img(output)
                else:
                    gt_it = tensor2img(gt[it])
                    output_it = tensor2img(output[it])

                result = eval_func(output_it, gt_it, crop_border)
                results.append(result)
            eval_result[metric] = np.mean(results)

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
        # Check
        assert self.test_cfg is not None, ValueError(
            '"self.test_cfg" should be provided; received None.')

        assert len(lq) == 1, (
            'Only one sample is allowed per batch to'
            ' (1) extract image names from "meta" for image saving;'
            ' (2) evaluate image metrics.')

        if 'unfolding' in self.test_cfg:
            raise NotImplementedError(
                'Unfolding is not supported yet for video tensor.')

        nfrms = lq.shape[1]
        if self.center_gt and (nfrms % 2 == 0):
            raise ValueError('Number of input frames should be odd'
                             ' when "center_gt" is True.')

        assert 'metrics' in self.test_cfg, (
            'metrics should be provided in "test_cfg" for evaluation.')

        # Inference
        if 'padding' in self.test_cfg:
            _cfg = self.test_cfg['padding']
            _tensors = []
            _pad_info = ()
            for it in range(nfrms):
                _lq_it, pad_info = pad_img_min_sz(lq[:, it, ...],
                                                  _cfg['minSize'])
                _tensors.append(_lq_it)
                if _pad_info:
                    assert pad_info == _pad_info
                else:
                    _pad_info = pad_info
            _lq = torch.stack(_tensors, dim=1)
            output = self.generator(_lq)
            _tensors = []
            for it in range(nfrms):
                _tensors.append(crop_img(output[:, it, ...], pad_info))
            output = torch.stack(_tensors, dim=1)
        else:
            output = self.generator(lq)

        # Squeeze dim B
        gt = gt.squeeze(0)  # (T, C, H, W) or (C, H, W)
        output = output.squeeze(0)  # (T, C, H, W) or (C, H, W)

        # Denormalize before image saving and evaluation
        if 'denormalize' in self.test_cfg:
            device = output.device
            mean = torch.tensor(self.test_cfg['denormalize']['mean']).view(
                1, -1, 1, 1).to(device)
            std = torch.tensor(self.test_cfg['denormalize']['std']).view(
                1, -1, 1, 1).to(device)
            if gt.dim() == 3:
                mean = torch.tensor(self.test_cfg['denormalize']['mean']).view(
                    -1, 1, 1).to(device)
                std = torch.tensor(self.test_cfg['denormalize']['std']).view(
                    -1, 1, 1).to(device)
            gt = gt * std + mean
            output = output * std + mean

        # Save images
        if save_image:
            key = meta[0]['key']
            if self.center_gt:
                key_dir = osp.dirname(key)
                key_stem = osp.splitext(osp.basename(key))[0]
                save_subpath = osp.join(key_dir, key_stem + '.png')
            else:
                key_dir = osp.dirname(key)
                key_names = osp.basename(key).split(',')
                key_stems = [
                    osp.splitext(key_name)[0] for key_name in key_names
                ]
                save_subpaths = [
                    osp.join(key_dir, key_stem + '.png')
                    for key_stem in key_stems
                ]

            for it in range(nfrms):  # note: T is the input lq idx
                if self.center_gt:  # save only the center frame
                    if it != (nfrms // 2):
                        continue
                else:  # save every output frame
                    save_subpath = save_subpaths[it]

                if isinstance(iteration,
                              numbers.Number):  # val during training
                    _save_path = osp.join(save_path, f'{iteration + 1}',
                                          save_subpath)
                elif iteration is None:  # testing
                    _save_path = osp.join(save_path, save_subpath)
                else:
                    raise TypeError('"iteration" should be a number or None;'
                                    f' received "{type(iteration)}".')

                if self.center_gt:
                    mmcv.imwrite(tensor2img(output), _save_path)
                else:
                    mmcv.imwrite(tensor2img(output[it]), _save_path)

        # Evaluation
        results = dict(eval_result=self.evaluate(
            metrics=self.test_cfg['metrics'], output=output, gt=gt))
        return results
