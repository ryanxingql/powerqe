# Copyright (c) OpenMMLab. All rights reserved.
# Modified by RyanXingQL @2022
from mmedit.models import PerceptualLoss

from ..registry import LOSSES


@LOSSES.register_module()
class PerceptualLossGray(PerceptualLoss):
    """Perceptual loss for gray-scale images.

    Differences to PerceptualLoss:
        Input x is a gray-scale image.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature for
            perceptual loss. Here is an example: {'4': 1., '9': 1., '18': 1.},
            which means the 5th, 10th and 18th feature layer will be
            extracted with weight 1.0 in calculating losses.
        layers_weights_style (dict): The weight for each layer of vgg feature
            for style loss. If set to None, the weights are set equal to
            the weights for perceptual loss. Default: None.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If perceptual_weight > 0, the perceptual
            loss will be calculated and the loss will be multiplied by the
            weight. Default: 1.0.
        style_weight (float): If style_weight > 0, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 1.0.
        norm_img (bool): If True, the image will be normed to [0, 1].
            Note that this is different from the use_input_norm which norm the
            input in forward function of vgg according to the statistics of
            dataset. Importantly, the input image must be in range [-1, 1].
            Default: True.
        pretrained (str): Path for pretrained weights.
            Default: 'torchvision://vgg19'.
        criterion (str): Criterion type. Options are 'l1' and 'mse'.
            Default: 'l1'.
    """

    def forward(self, x, gt):
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
            x (Tensor): Tensor with the shape of (N, C, H, W).

        Returns:
            Tensor: Gram matrix.
        """
        (n, c, h, w) = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
