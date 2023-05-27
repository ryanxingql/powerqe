# RyanXingQL @2022
import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import BACKBONES
from .base import BaseNet


class ECA(nn.Module):
    """Efficient Channel Attention.

    Ref: "https://github.com/BangguWu/ECANet/blob
    /3adf7a99f829ffa2e94a0de1de8a362614d66958/models/eca_module.py#L5"

    Args:
        k_size: Kernel size.
    """

    def __init__(self, k_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channels=1,
                              out_channels=1,
                              kernel_size=k_size,
                              padding=(k_size - 1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # (N, C, H, W) -> (N, C, 1, 1)
        # -> (N, C, 1) -> (N, 1, C) -> Conv (just like FC, but ks=3)
        # -> (N, 1, C) -> (N, C, 1) -> (N, C, 1, 1)
        logic = self.avg_pool(x)
        logic = self.conv(logic.squeeze(-1).transpose(-1, -2)).transpose(
            -1, -2).unsqueeze(-1)
        logic = self.sigmoid(logic)
        return x * logic.expand_as(x)


class SeparableConv2d(nn.Module):

    def __init__(self, nf_in, nf_out):
        super().__init__()

        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=nf_in,
                out_channels=nf_in,
                kernel_size=3,
                padding=3 // 2,
                groups=nf_in  # each channel is convolved with its own filter
            ),
            nn.Conv2d(in_channels=nf_in,
                      out_channels=nf_out,
                      kernel_size=1,
                      groups=1)  # then point-wise
        )

    def forward(self, x):
        return self.separable_conv(x)


class GaussianSmoothing(nn.Module):
    """Apply gaussian smoothing on a 1d, 2d or 3d tensor.

    Filtering is performed separately for each channel
    in the input using a depthwise convolution.

    Args:
        channels (int, sequence): Number of channels of the input tensors.
            Output will have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
    """

    def __init__(self, channels, kernel_size, sigma, padding, dim=2):
        super().__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(
                -((mgrid - mean) / std)**2 /
                2)  # ignore the warning: it is a tensor

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(
            kernel)  # ignore the warning: it is a tensor

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.padding = padding

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise ValueError('Data with 1/2/3 dimensions is supported;'
                             f' received {dim} dimensions.')

    def forward(self, x):
        """Apply gaussian filter to input.

        Args:
            x (Tensor): Input to apply gaussian filter on.

        Returns:
            Tensor: Filtered output.
        """
        return self.conv(x,
                         weight=self.weight,
                         groups=self.groups,
                         padding=self.padding)


class IQAM:

    def __init__(self, comp_type='jpeg'):
        if comp_type == 'jpeg':
            self.patch_sz = 8

            self.tche_poly = torch.tensor(
                [[
                    0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536,
                    0.3536
                ],
                 [
                     -0.5401, -0.3858, -0.2315, -0.0772, 0.0772, 0.2315,
                     0.3858, 0.5401
                 ],
                 [
                     0.5401, 0.0772, -0.2315, -0.3858, -0.3858, -0.2315,
                     0.0772, 0.5401
                 ],
                 [
                     -0.4308, 0.3077, 0.4308, 0.1846, -0.1846, -0.4308,
                     -0.3077, 0.4308
                 ],
                 [
                     0.2820, -0.5238, -0.1209, 0.3626, 0.3626, -0.1209,
                     -0.5238, 0.2820
                 ],
                 [
                     -0.1498, 0.4922, -0.3638, -0.3210, 0.3210, 0.3638,
                     -0.4922, 0.1498
                 ],
                 [
                     0.0615, -0.3077, 0.5539, -0.3077, -0.3077, 0.5539,
                     -0.3077, 0.0615
                 ],
                 [
                     -0.0171, 0.1195, -0.3585, 0.5974, -0.5974, 0.3585,
                     -0.1195, 0.0171
                 ]],
                dtype=torch.float32).cuda()

            self.thr_out = 0.855

        elif comp_type == 'hevc':
            self.patch_sz = 4

            self.tche_poly = torch.tensor([[0.5000, 0.5000, 0.5000, 0.5000],
                                           [-0.6708, -0.2236, 0.2236, 0.6708],
                                           [0.5000, -0.5000, -0.5000, 0.5000],
                                           [-0.2236, 0.6708, -0.6708, 0.2236]],
                                          dtype=torch.float32).cuda()

            self.thr_out = 0.900

        self.tche_poly_transposed = self.tche_poly.permute(1, 0)  # h <-> w

        self.thr_smooth = torch.tensor(0.004)
        self.thr_jnd = torch.tensor(0.05)
        self.bigc = torch.tensor(1e-5)  # numerical stability
        self.alpha_block = 0.9  # [0, 1]

        self.gaussian_filter = GaussianSmoothing(channels=1,
                                                 kernel_size=3,
                                                 sigma=5,
                                                 padding=3 // 2).cuda()

    def cal_tchebichef_moments(self, x):
        x = x.clone()
        x /= torch.sqrt(self.patch_sz * self.patch_sz * (x.reshape(
            (-1, )).pow(2).mean()))
        x -= x.reshape((-1, )).mean()
        moments = torch.mm(torch.mm(self.tche_poly, x),
                           self.tche_poly_transposed)
        return moments

    def forward(self, x):
        """Forward.

        Only test one channel, e.g., red.

        Args:
            x (Tensor): Image with the shape of (B=1, C, H, W).
        """
        h, w = x.shape[2:]
        h_cut = h // self.patch_sz * self.patch_sz
        w_cut = w // self.patch_sz * self.patch_sz
        x = x[0, 0, :h_cut, :w_cut]  # (h_cut, w_cut)

        num_smooth = 0.
        num_textured = 0.
        score_blocky_smooth = 0.
        score_blurred_textured = 0.

        start_h = self.patch_sz // 2 - 1
        while start_h + self.patch_sz <= h_cut:
            start_w = self.patch_sz // 2 - 1

            while start_w + self.patch_sz <= w_cut:
                patch = x[start_h:(start_h + self.patch_sz),
                          start_w:(start_w + self.patch_sz)]

                sum_patch = torch.sum(torch.abs(patch))
                # will lead to NAN score of blocky smooth patch
                if sum_patch == 0:
                    num_smooth += 1
                    score_blocky_smooth = score_blocky_smooth + 1.

                else:
                    moments_patch = self.cal_tchebichef_moments(patch)

                    # smooth/textured patch
                    ssm = torch.sum(
                        moments_patch.pow(2)) - moments_patch[0, 0].pow(2)
                    if ssm > self.thr_smooth:
                        num_textured += 1

                        patch_blurred = torch.squeeze(
                            self.gaussian_filter(patch.clone().view(
                                1, 1, self.patch_sz, self.patch_sz)))
                        moments_patch_blurred = self.cal_tchebichef_moments(
                            patch_blurred)
                        similarity_matrix = torch.div(
                            (torch.mul(moments_patch, moments_patch_blurred) *
                             2. + self.bigc), (moments_patch.pow(2)) +
                            moments_patch_blurred.pow(2) + self.bigc)
                        score_blurred_textured += 1 - torch.mean(
                            similarity_matrix.reshape((-1)))

                    else:
                        num_smooth += 1

                        sum_moments = torch.sum(torch.abs(moments_patch))
                        strength_vertical = torch.sum(
                            torch.abs(moments_patch[self.patch_sz - 1, :])
                        ) / sum_moments - torch.abs(
                            moments_patch[0, 0]) + self.bigc
                        strength_horizontal = torch.sum(
                            torch.abs(moments_patch[:, self.patch_sz - 1])
                        ) / sum_moments - torch.abs(
                            moments_patch[0, 0]) + self.bigc

                        if strength_vertical > self.thr_jnd:
                            strength_vertical = self.thr_jnd
                        if strength_horizontal > self.thr_jnd:
                            strength_horizontal = self.thr_jnd
                        score_ = torch.log(1 - (
                            (strength_vertical + strength_horizontal) /
                            2)) / torch.log(1 - self.thr_jnd)

                        score_blocky_smooth = score_blocky_smooth + score_

                start_w += self.patch_sz
            start_h += self.patch_sz

        if num_textured != 0:
            score_blurred_textured /= num_textured
        else:
            score_blurred_textured = torch.tensor(1., dtype=torch.float32)
        if num_smooth != 0:
            score_blocky_smooth /= num_smooth
        else:
            score_blocky_smooth = torch.tensor(1., dtype=torch.float32)

        score_quality = (score_blocky_smooth.pow(self.alpha_block)) * (
            score_blurred_textured.pow(1 - self.alpha_block))
        if score_quality >= self.thr_out:
            return True
        else:
            return False


class Down(nn.Module):
    # downsample for one time, e.g., from C2,1 to C3,2

    def __init__(self, nf_in, nf_out, method, if_separable, if_eca):
        super().__init__()

        supported_methods = ['avepool2d', 'strideconv']
        if method not in supported_methods:
            raise NotImplementedError(
                f'Downsampling method should be in "{supported_methods}";'
                f' received "{method}".')

        if if_separable and if_eca:
            layers = nn.ModuleList(
                [ECA(k_size=3),
                 SeparableConv2d(nf_in=nf_in, nf_out=nf_in)])
        elif if_separable and (not if_eca):
            layers = nn.ModuleList(SeparableConv2d(nf_in=nf_in, nf_out=nf_in))
        elif (not if_separable) and if_eca:
            layers = nn.ModuleList([
                ECA(k_size=3),
                nn.Conv2d(in_channels=nf_in,
                          out_channels=nf_in,
                          kernel_size=3,
                          padding=3 // 2)
            ])
        else:
            layers = nn.ModuleList([
                nn.Conv2d(in_channels=nf_in,
                          out_channels=nf_in,
                          kernel_size=3,
                          padding=3 // 2)
            ])

        if method == 'avepool2d':
            layers.append(nn.AvgPool2d(kernel_size=2))
        elif method == 'strideconv':
            layers.append(
                nn.Conv2d(in_channels=nf_in,
                          out_channels=nf_out,
                          kernel_size=3,
                          padding=3 // 2,
                          stride=2))

        if if_separable and if_eca:
            layers += [
                ECA(k_size=3),
                SeparableConv2d(nf_in=nf_out, nf_out=nf_in),
            ]
        elif if_separable and (not if_eca):
            layers.append(SeparableConv2d(nf_in=nf_out, nf_out=nf_in))
        elif (not if_separable) and if_eca:
            layers += [
                ECA(k_size=3),
                nn.Conv2d(in_channels=nf_out,
                          out_channels=nf_out,
                          kernel_size=3,
                          padding=3 // 2),
            ]
        else:
            layers.append(
                nn.Conv2d(in_channels=nf_out,
                          out_channels=nf_out,
                          kernel_size=3,
                          padding=3 // 2))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):
    # upsample for one time, e.g., from C3,1 and C2,1 to C2,2

    def __init__(self, nf_in_s, nf_in, nf_out, method, if_separable, if_eca):
        super().__init__()

        supported_methods = ['upsample', 'transpose2d']
        if method not in supported_methods:
            raise NotImplementedError(
                f'Upsampling method should be in "{supported_methods}";'
                f' received "{method}".')

        if method == 'upsample':
            self.up = nn.Upsample(scale_factor=2)
        elif method == 'transpose2d':
            self.up = nn.ConvTranspose2d(in_channels=nf_in_s,
                                         out_channels=nf_out,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1)

        if if_separable and if_eca:
            layers = nn.ModuleList([
                ECA(k_size=3),
                SeparableConv2d(nf_in=nf_in, nf_out=nf_out),
                nn.ReLU(inplace=True),
                ECA(k_size=3),
                SeparableConv2d(nf_in=nf_out, nf_out=nf_out),
            ])
        elif if_separable and (not if_eca):
            layers = nn.ModuleList([
                SeparableConv2d(nf_in=nf_in, nf_out=nf_out),
                nn.ReLU(inplace=True),
                SeparableConv2d(nf_in=nf_out, nf_out=nf_out),
            ])
        elif (not if_separable) and if_eca:
            layers = nn.ModuleList([
                ECA(k_size=3),
                nn.Conv2d(in_channels=nf_in,
                          out_channels=nf_out,
                          kernel_size=3,
                          padding=3 // 2),
                nn.ReLU(inplace=True),
                ECA(k_size=3),
                nn.Conv2d(in_channels=nf_out,
                          out_channels=nf_out,
                          kernel_size=3,
                          padding=3 // 2)
            ])
        else:
            layers = nn.ModuleList([
                nn.Conv2d(in_channels=nf_in,
                          out_channels=nf_out,
                          kernel_size=3,
                          padding=3 // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=nf_out,
                          out_channels=nf_out,
                          kernel_size=3,
                          padding=3 // 2)
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, small_t, *normal_t_list):
        feat = self.up(small_t)

        # pad feat according to a normal_t
        if len(normal_t_list) > 0:
            h_s, w_s = feat.size()[2:]  # (N, C, H, W)
            h, w = normal_t_list[0].size()[2:]
            dh = h - h_s
            dw = w - w_s

            if dh < 0:
                feat = feat[:, :, :h, :]
                dh = 0
            if dw < 0:
                feat = feat[:, :, :, :w]
                dw = 0
            feat = F.pad(
                input=feat,
                pad=[dw // 2, (dw - dw // 2), dh // 2, (dh - dh // 2)],
                mode='constant',
                value=0)

            feat = torch.cat((feat, *normal_t_list), dim=1)

        return self.layers(feat)


@BACKBONES.register_module()
class RBQE(BaseNet):

    def __init__(self,
                 nf_io=3,
                 nf_base=32,
                 nlevel=5,
                 down_method='strideconv',
                 up_method='transpose2d',
                 if_separable=False,
                 if_eca=True,
                 if_only_last_output=True,
                 comp_type='hevc'):
        super().__init__()

        self.nlevel = nlevel
        self.if_only_last_output = if_only_last_output

        # input conv
        if if_separable:
            self.in_conv_seq = nn.Sequential(
                SeparableConv2d(nf_in=nf_io, nf_out=nf_base),
                nn.ReLU(inplace=True),
                SeparableConv2d(nf_in=nf_base, nf_out=nf_base),
            )
        else:
            self.in_conv_seq = nn.Sequential(
                nn.Conv2d(in_channels=nf_io,
                          out_channels=nf_base,
                          kernel_size=3,
                          padding=3 // 2), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=nf_base,
                          out_channels=nf_base,
                          kernel_size=3,
                          padding=3 // 2))

        # down then up at each nested u-net
        for idx_unet in range(nlevel):
            setattr(
                self, f'down_{idx_unet}',
                Down(nf_in=nf_base,
                     nf_out=nf_base,
                     method=down_method,
                     if_separable=if_separable,
                     if_eca=if_eca))
            for idx_up in range(idx_unet + 1):
                setattr(
                    self,
                    f'up_{idx_unet}_{idx_up}',
                    Up(
                        nf_in_s=nf_base,
                        nf_in=nf_base * (2 + idx_up),  # dense connection
                        nf_out=nf_base,
                        method=up_method,
                        if_separable=if_separable,
                        if_eca=if_eca))

        # output side
        self.out_layers = nn.ModuleList()
        if if_only_last_output:  # single exit
            repeat_times = 1
        else:  # multi exits
            repeat_times = nlevel
        for _ in range(repeat_times):
            if if_separable and if_eca:
                self.out_layers.append(
                    nn.Sequential(ECA(k_size=3),
                                  SeparableConv2d(nf_in=nf_base,
                                                  nf_out=nf_io)))
            elif if_separable and (not if_eca):
                self.out_layers.append(
                    SeparableConv2d(nf_in=nf_base, nf_out=nf_io))
            elif (not if_separable) and if_eca:
                self.out_layers.append(
                    nn.Sequential(
                        ECA(k_size=3),
                        nn.Conv2d(in_channels=nf_base,
                                  out_channels=nf_io,
                                  kernel_size=3,
                                  padding=3 // 2)))
            else:
                self.out_layers.append(
                    nn.Conv2d(in_channels=nf_base,
                              out_channels=nf_io,
                              kernel_size=3,
                              padding=3 // 2))

        # IQA module
        # no trainable parameters
        if not if_only_last_output:  # multi-exit network
            self.iqam = IQAM(comp_type=comp_type)

    def forward(self, x, idx_out=None):
        """Forward.

        Args:
            x (Tensor): Image with the shape of (B=1, C, H, W).
            idx_out (int):
                -2: Determined by IQAM.
                -1: Output all images from all outputs for training.
                0 | 1 | ... | self.nlevel-1: Output from the assigned exit.
                None: Output from the last exit.
        """
        if self.if_only_last_output:
            if idx_out is not None:
                raise ValueError('Exit cannot be indicated'
                                 ' since there is only one exit.')
            idx_out = self.nlevel - 1

        feat = self.in_conv_seq(x)
        feat_level_unet = [[feat]
                           ]  # the first level feature of the first U-Net

        if idx_out == -1:  # to record output images from all exits
            out_img_list = []

        for idx_unet in range(self.nlevel):  # per U-Net
            down = getattr(self, f'down_{idx_unet}')
            feat = down(
                feat_level_unet[-1][0])  # the previous U-Net, the first level
            feat_up_list = [feat]

            # for the first u-net (idx=0), up one time
            for idx_up in range(idx_unet + 1):
                dense_inp_list = []
                # To obtain C2,4
                # It is the second upsampling, idx_up == 2.
                # It needs C2,1 to C2,3 at feat_level_unet[1][0],
                # feat_level_unet[2][1] and feat_level_unet[3][2].
                # feat_level_unet now contains 4 lists.
                for idx_, feat_level in enumerate(
                        feat_level_unet[-(idx_up + 1):]):
                    dense_inp_list.append(
                        feat_level[idx_]
                    )  # append features from previous U-Nets at the same level

                up = getattr(self, f'up_{idx_unet}_{idx_up}')
                feat_up = up(
                    feat_up_list[-1],
                    *dense_inp_list,
                )
                feat_up_list.append(feat_up)

            if idx_out in [-1, -2, idx_unet]:  # if go to the output side
                if self.if_only_last_output:
                    out_conv_seq = self.out_layers[0]
                else:
                    out_conv_seq = self.out_layers[idx_unet]
                out_img = out_conv_seq(feat_up_list[-1]) + x

                if idx_out == -1:
                    out_img_list.append(out_img)

                # if at the last level, no need to IQA
                if (idx_out == -2) and (idx_unet < (self.nlevel - 1)):
                    if_out = self.iqam.forward(out_img)
                    if if_out:
                        break

            feat_level_unet.append(feat_up_list)

        if idx_out == -1:
            return torch.stack(out_img_list,
                               dim=0)  # (self.nlevel, N, C, H, W)
        else:
            return out_img  # (B=1, C, H, W)
