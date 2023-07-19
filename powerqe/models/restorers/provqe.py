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
import torch
from mmcv.runner import auto_fp16
from mmedit.core import tensor2img

from ...utils.unfolding import crop_img, pad_img_min_sz
from ..registry import MODELS
from .basic_restorer import BasicVQERestorer


@MODELS.register_module()
class ProVQERestorer(BasicVQERestorer):
    """ProVQE restorer.

    Differences to BasicVQERestorer:
        Require "key_frms" in meta. See forward.

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
    def forward(self, lq, gt=None, test_mode=False, meta=None, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        key_frms = [m['key_frms'] for m in meta]

        if test_mode:
            return self.forward_test(lq=lq,
                                     gt=gt,
                                     key_frms=key_frms,
                                     meta=meta,
                                     **kwargs)

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
        # Check
        assert self.test_cfg is not None, (
            '"test_cfg" should be provided; received None.')

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
            output = self.generator(_lq, key_frms)
            _tensors = []
            for it in range(nfrms):
                _tensors.append(crop_img(output[:, it, ...], pad_info))
            output = torch.stack(_tensors, dim=1)
        else:
            output = self.generator(lq, key_frms)

        # Squeeze dim B
        gt = gt.squeeze(0)  # (T, C, H, W) or (C, H, W)
        output = output.squeeze(0)  # (T, C, H, W) or (C, H, W)

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
