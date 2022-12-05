# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL @2022
from mmedit.models import PerceptualLoss

from ..registry import LOSSES


@LOSSES.register_module()
class PerceptualLossGray(PerceptualLoss):
    """Support PerceptualLoss for gray-scale images."""

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # gray -> color
        # diff 1/1
        x = x.repeat(1, 3, 1, 1)

        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(
                    x_features[k], gt_features[k]) * self.layer_weights[k]
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
                style_loss += self.criterion(self._gram_mat(
                    x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights_style[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        (n, c, h, w) = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
