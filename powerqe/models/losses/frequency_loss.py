# RyanXingQL, 2022
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
from mmedit.models.losses.utils import masked_loss

from ..registry import LOSSES

_reduction_modes = ['none', 'mean', 'sum']


@masked_loss
def mse_loss(pred, target):
    """MSE loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated MSE loss.
    """
    return F.mse_loss(pred, target, reduction='none')


@LOSSES.register_module()
class DCTLowFreqLoss(nn.Module):
    """MSE of low-frequency DCT map.

    Args:
        ratio (float): Low frequency ratio.
        pre_cal_sides (list): Pre-calculate the weight maps for these sides.
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """
    def __init__(self,
                 ratio=0.4,
                 pre_cal_sides=[64, 128],
                 loss_weight=1.0,
                 reduction='mean',
                 sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.ratio = ratio
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

        self.pre_cal_weight(pre_cal_sides=pre_cal_sides)

    def cal_dct(self, tensor):
        """
        tensor: (N, 1, H, W) or (N, [RGB], H, W)
        """
        if tensor.size(1) == 3:
            tensor = 0.299 * tensor[:, 0:1, ...] + \
                    0.587 * tensor[:, 1:2, ...] + \
                    0.114 * tensor[:, 2:, ...]  # RGB -> Y; (N, 1, H, W)
        return dct.dct_2d(tensor)  # DCT-II of the signal over the last 2 dims

    def cal_weight(self, h, w):
        a = h * self.ratio
        b = w * self.ratio

        weight_map = torch.zeros(size=(1, 1, h, w), dtype=torch.int)
        for ih in range(h):
            for iw in range(w):
                r = math.pow(ih, 2) / math.pow(a, 2) + math.pow(
                    iw, 2) / math.pow(b, 2)
                if r < 1:  # inside the ellipse
                    weight_map[..., ih, iw] = 1
                else:
                    break
        return weight_map

    def pre_cal_weight(self, pre_cal_sides):
        self.weight_pre_cal = dict()
        for pre_cal_side in pre_cal_sides:
            self.weight_pre_cal[pre_cal_side] = self.cal_weight(h=pre_cal_side,
                                                                w=pre_cal_side)

    def return_weight(self, pred_dct):
        h, w = pred_dct.shape[2:]
        if h == w and (h in self.weight_pre_cal):
            return self.weight_pre_cal[h]
        else:
            return self.cal_weight(h, w)

    def forward(self, pred, target, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """

        pred_dct = self.cal_dct(pred)
        target_dct = self.cal_dct(target)
        weight = self.return_weight(pred_dct)  # (1, 1, H, W)
        device = pred_dct.device
        weight = weight.to(device)

        return self.loss_weight * mse_loss(pred_dct,
                                           target_dct,
                                           weight=weight,
                                           reduction=self.reduction,
                                           sample_wise=self.sample_wise)
