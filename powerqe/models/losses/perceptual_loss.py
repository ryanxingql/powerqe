"""Copyright (c) OpenMMLab. All rights reserved.

Copyright 2023 RyanXingQL

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
from mmedit.models.losses import PerceptualLoss

from ..registry import LOSSES


@LOSSES.register_module()
class PerceptualLossGray(PerceptualLoss):
    """Perceptual loss for gray-scale images.

    Differences to PerceptualLoss: Input x is a gray-scale image.
    """

    def forward(self, x, gt):
        # gray -> color
        # diff 1/1
        x = x.repeat(1, 3, 1, 1)

        if self.norm_img:
            x = (x + 1.0) * 0.5
            gt = (gt + 1.0) * 0.5
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += (
                    self.criterion(x_features[k], gt_features[k])
                    * self.layer_weights[k]
                )
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            if self.vgg_style is not None:
                x_features = self.vgg_style(x)
                gt_features = self.vgg_style(gt.detach())

            style_loss = 0
            for k in x_features.keys():
                style_loss += (
                    self.criterion(
                        self._gram_mat(x_features[k]), self._gram_mat(gt_features[k])
                    )
                    * self.layer_weights_style[k]
                )
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (Tensor): Tensor with the shape of (N, C, H, W).

        Returns:
            Tensor: Gram matrix.
        """
        (n, c, h, w) = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
