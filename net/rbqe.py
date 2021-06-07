import math
import torch
import numbers
import torch.nn as nn
import torch.nn.functional as F

from .ecanet import ECA

from utils import BaseNet, Timer


class Up(nn.Module):
    def __init__(self, nf_in_s, nf_in, nf_out, method, if_separable, if_eca):
        assert method in ['upsample','transpose2d'], '> not supported yet.'

        super().__init__()

        if method == 'upsample':
            self.up = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),
            )
        elif method == 'transpose2d':
            self.up = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=nf_in_s,
                    out_channels=nf_out,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            )

        if if_separable:
            if if_eca:
                self.conv_lst = nn.ModuleList([
                        nn.ReLU(inplace=False),
                        ECA(k_size=3),
                        SeparableConv2d(nf_in=nf_in, nf_out=nf_out),
                        nn.ReLU(inplace=False),
                        ECA(k_size=3),
                        SeparableConv2d(nf_in=nf_out, nf_out=nf_out),
                    ]
                )
            else:
                self.conv_lst = nn.ModuleList([
                        nn.ReLU(inplace=False),
                        SeparableConv2d(nf_in=nf_in, nf_out=nf_out),
                        nn.ReLU(inplace=False),
                        SeparableConv2d(nf_in=nf_out, nf_out=nf_out),
                    ]
                )                
        else:
            if if_eca:
                self.conv_lst = nn.ModuleList([
                        nn.ReLU(inplace=False),
                        ECA(k_size=3),
                        nn.Conv2d(
                            in_channels=nf_in,
                            out_channels=nf_out,
                            kernel_size=3,
                            padding=3//2,
                        ),
                        nn.ReLU(inplace=False),
                        ECA(k_size=3),
                        nn.Conv2d(
                            in_channels=nf_out,
                            out_channels=nf_out,
                            kernel_size=3,
                            padding=3//2,
                        ),
                    ]
                )                
            else:
                self.conv_lst = nn.ModuleList([
                        nn.ReLU(inplace=False),
                        nn.Conv2d(
                            in_channels=nf_in,
                            out_channels=nf_out,
                            kernel_size=3,
                            padding=3//2,
                        ),
                        nn.ReLU(inplace=False),
                        nn.Conv2d(
                            in_channels=nf_out,
                            out_channels=nf_out,
                            kernel_size=3,
                            padding=3//2,
                        ),
                    ]
                )

    def forward(self, small_t, *normal_t_lst):
        feat = self.up(small_t)

        if len(normal_t_lst) > 0:
            h_s, w_s = feat.size()[2:]  # B C H W
            h, w = normal_t_lst[0].size()[2:]
            diffH = h - h_s
            diffW = w - w_s
            if diffH < 0:
                feat = feat[:,:,:h,:]
                diffH = 0
            if diffW < 0:
                feat = feat[:,:,:,:w]
                diffW = 0
            feat = F.pad(
                input=feat,
                pad=(
                    diffW//2, (diffW-diffW//2),  # only pad H and W; left (diffW//2); right remaining (diffW - diffW//2)
                    diffH//2, (diffH-diffH//2),
                ),
                mode='constant',
                value=0, # pad with constant 0
            )

            feat = torch.cat((feat, *normal_t_lst), dim=1)

        for module_ in self.conv_lst:
            feat = module_(feat)
        out_t = feat

        return out_t

class Down(nn.Module):
    def __init__(self, nf_in, nf_out, method, if_separable, if_eca):
        super().__init__()

        self.down_lst = nn.ModuleList([nn.ReLU(inplace=False)])
        if if_separable:
            if if_eca:
                self.down_lst += [
                    ECA(k_size=3),
                    SeparableConv2d(nf_in=nf_in, nf_out=nf_in),
                ]
            else:
                self.down_lst.append(
                    SeparableConv2d(nf_in=nf_in, nf_out=nf_in)
                )
        else:
            if if_eca:
                self.down_lst += [
                    ECA(k_size=3),
                    nn.Conv2d(
                        in_channels=nf_in,
                        out_channels=nf_in,
                        kernel_size=3,
                        padding=3//2,
                    ),
                ]
            else:
                self.down_lst.append(
                    nn.Conv2d(
                        in_channels=nf_in,
                        out_channels=nf_in,
                        kernel_size=3,
                        padding=3//2,
                    )
                )                
        self.down_lst.append(nn.ReLU(inplace=False))
        if method == 'avepool2d':
            self.down_lst.append(nn.AvgPool2d(kernel_size=2))
        elif method == 'strideconv':
            self.down_lst += [
                nn.Conv2d(
                    in_channels=nf_in,
                    out_channels=nf_out,
                    kernel_size=3,
                    padding=3//2,
                    stride=2,
                ),
                nn.ReLU(inplace=False),
            ]
        if if_separable:
            if if_eca:
                self.down_lst += [
                    ECA(k_size=3),
                    SeparableConv2d(nf_in=nf_out, nf_out=nf_in),
                ]
            else:
                self.down_lst.append(
                    SeparableConv2d(nf_in=nf_out, nf_out=nf_in)
                )
        else:
            if if_eca:
                self.down_lst += [
                    ECA(k_size=3),
                    nn.Conv2d(
                        in_channels=nf_out,
                        out_channels=nf_out,
                        kernel_size=3,
                        padding=3//2,
                    ),
                ]
            else:
                self.down_lst.append(
                    nn.Conv2d(
                        in_channels=nf_out,
                        out_channels=nf_out,
                        kernel_size=3,
                        padding=3//2,
                    )
                )

    def forward(self, inp_t):
        feat = inp_t
        for module_ in self.down_lst:
            feat = module_(feat)
        out_t = feat
        return out_t

class SeparableConv2d(nn.Module):
    def __init__(self, nf_in, nf_out):
        super().__init__()

        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=nf_in,
                out_channels=nf_in,
                kernel_size=3,
                padding=3//2,
                groups=nf_in,
            ), # groups=inch: each channel is convolved with its own filter
            nn.Conv2d(
                in_channels=nf_in,
                out_channels=nf_out,
                kernel_size=1,
                groups=1,
            ) # then point-wise
        )
    
    def forward(self, inp_t):
        out_t = self.separable_conv(inp_t)
        return out_t

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, padding, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

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
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)

class IQAM():
    def __init__(self, comp_type='jpeg'):
        if comp_type == 'jpeg':
            self.patch_sz = 8

            self.tche_poly = torch.tensor(
                [
                    [0.3536,0.3536,0.3536,0.3536,0.3536,0.3536,0.3536,0.3536],
                    [-0.5401,-0.3858,-0.2315,-0.0772,0.0772,0.2315,0.3858,0.5401],
                    [0.5401,0.0772,-0.2315,-0.3858,-0.3858,-0.2315,0.0772,0.5401],
                    [-0.4308,0.3077,0.4308,0.1846,-0.1846,-0.4308,-0.3077,0.4308],
                    [0.2820,-0.5238,-0.1209,0.3626,0.3626,-0.1209,-0.5238,0.2820],
                    [-0.1498,0.4922,-0.3638,-0.3210,0.3210,0.3638,-0.4922,0.1498],
                    [0.0615,-0.3077,0.5539,-0.3077,-0.3077,0.5539,-0.3077,0.0615],
                    [-0.0171,0.1195,-0.3585,0.5974,-0.5974,0.3585,-0.1195,0.0171],
                ], dtype=torch.float32
            ).cuda()

            self.thr_out = 0.855

        elif comp_type == 'hevc':
            self.patch_sz = 4

            self.tche_poly = torch.tensor(
                [
                    [0.5000,0.5000,0.5000,0.5000],
                    [-0.6708,-0.2236,0.2236,0.6708],
                    [0.5000,-0.5000,-0.5000,0.5000],
                    [-0.2236,0.6708,-0.6708,0.2236],
                ], dtype=torch.float32
            ).cuda()
            
            self.thr_out = 0.900

        self.tche_poly_transposed = self.tche_poly.permute(1, 0)  # h <-> w

        self.thr_smooth = torch.tensor(0.004)
        self.thr_jnd = torch.tensor(0.05)
        self.bigc = torch.tensor(1e-5)  # numerical stability
        self.alpha_block = 0.9  # [0, 1]

        self.gaussian_filter = GaussianSmoothing(channels=1, kernel_size=3, sigma=5, padding=3//2).cuda()

    def cal_tchebichef_moments(self, inp_t):
        inp_t = inp_t.clone()
        inp_t /= torch.sqrt(self.patch_sz * self.patch_sz * (inp_t.reshape((-1,)).pow(2).mean()))
        inp_t -= inp_t.reshape((-1,)).mean()
        moments = torch.mm(torch.mm(self.tche_poly, inp_t), self.tche_poly_transposed)
        return moments

    def forward(self, inp_t):
        """
        (B=1 C H W)
        only test one channel, e.g., Red
        """
        h, w = inp_t.shape[2:]
        h_cut = h // self.patch_sz * self.patch_sz
        w_cut = w // self.patch_sz * self.patch_sz
        inp_t = inp_t[0, 0, :h_cut, :w_cut]  # (h_cut, w_cut)

        num_smooth = 0.
        num_textured = 0.
        score_blocky_smooth = 0.
        score_blurred_textured = 0.
        
        start_h = self.patch_sz // 2 - 1
        while (start_h + self.patch_sz <= h_cut):
            start_w = self.patch_sz // 2 - 1
            while (start_w + self.patch_sz <= w_cut):
                patch = inp_t[start_h:(start_h + self.patch_sz), start_w:(start_w + self.patch_sz)]

                sum_patch = torch.sum(torch.abs(patch))
                if sum_patch == 0:  # will lead to NAN score of blocky smooth patch
                    num_smooth += 1
                    score_blocky_smooth = score_blocky_smooth + 1.
                
                else:
                    moments_patch = self.cal_tchebichef_moments(patch)

                    # smooth/textured patch
                    ssm = torch.sum(moments_patch.pow(2)) - moments_patch[0,0].pow(2)
                    if ssm > self.thr_smooth:
                        num_textured += 1

                        patch_blurred = torch.squeeze(self.gaussian_filter(patch.clone().view(1, 1, self.patch_sz, self.patch_sz)))
                        moments_patch_blurred = self.cal_tchebichef_moments(patch_blurred)
                        similarity_matrix = torch.div((torch.mul(moments_patch, moments_patch_blurred) * 2. + self.bigc), (moments_patch.pow(2)) + moments_patch_blurred.pow(2) + self.bigc)
                        score_blurred_textured = score_blurred_textured + 1 - torch.mean(similarity_matrix.reshape((-1,)))

                    else:
                        num_smooth += 1

                        sum_moments = torch.sum(torch.abs(moments_patch))
                        strength_vertical = torch.sum(torch.abs(moments_patch[self.patch_sz-1, :])) / sum_moments - torch.abs(moments_patch[0,0]) + self.bigc
                        strength_horizontal = torch.sum(torch.abs(moments_patch[:, self.patch_sz-1])) / sum_moments - torch.abs(moments_patch[0,0]) + self.bigc
                        
                        strength_vertical = strength_vertical if strength_vertical <= self.thr_jnd else self.thr_jnd
                        strength_horizontal = strength_horizontal if strength_horizontal <= self.thr_jnd else self.thr_jnd
                        score_ = torch.log(1 - ((strength_vertical + strength_horizontal) / 2)) / torch.log(1 - self.thr_jnd)

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

        score_quality = (score_blocky_smooth.pow(self.alpha_block)) * (score_blurred_textured.pow(1 - self.alpha_block))
        if score_quality >= self.thr_out:
            return True
        else:
            return False

class Network(BaseNet):
    def __init__(
            self,
            if_train,
            nf_in=3,
            nf_base=32,
            nlevel=5,  # 5 outputs
            down_method='strideconv',
            up_method='transpose2d',
            if_separable=False,
            if_eca=False,
            nf_out=3,
            if_residual=True,
            comp_type='hevc',
        ):
        assert down_method in ['avepool2d','strideconv'], '> not supported!'
        
        super().__init__()

        self.if_train = if_train
        self.nlevel = nlevel
        self.if_residual = if_residual

        # input conv
        if if_separable:
            self.inconv_lst = nn.ModuleList([
                    SeparableConv2d(nf_in=nf_in, nf_out=nf_base),
                    nn.ReLU(inplace=False),
                    SeparableConv2d(nf_in=nf_base, nf_out=nf_base),
                ]
            )
        else:
            self.inconv_lst = nn.ModuleList([
                    nn.Conv2d(
                        in_channels=nf_in,
                        out_channels=nf_base,
                        kernel_size=3,
                        padding=3//2,
                    ),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        in_channels=nf_base,
                        out_channels=nf_base,
                        kernel_size=3,
                        padding=3//2,
                    ),
                ]
            )

        # down then up at each nested u-net
        for idx_level in range(nlevel):
            setattr(self, f'down_{idx_level}', Down(
                nf_in=nf_base,
                nf_out=nf_base,
                method=down_method,
                if_separable=if_separable,
                if_eca=if_eca,
            ))
            for idx_up in range(idx_level+1):
                setattr(self, f'up_{idx_level}_{idx_up}', Up(
                    nf_in_s=nf_base,
                    nf_in=nf_base*(2+idx_up),
                    nf_out=nf_base,
                    method=up_method,
                    if_separable=if_separable,
                    if_eca=if_eca,
                ))

        # out
        self.outconv_lst = nn.ModuleList([])
        for _ in range(nlevel):
            outconv_lst= [nn.ReLU(inplace=False)]
            if if_separable:
                if if_eca:
                    outconv_lst += [
                        ECA(k_size=3),
                        SeparableConv2d(nf_in=nf_base, nf_out=nf_out),
                    ]
                else:
                    outconv_lst.append(
                        SeparableConv2d(nf_in=nf_base, nf_out=nf_out)
                    )                
            else:
                if if_eca:
                    outconv_lst += [
                        ECA(k_size=3),
                        nn.Conv2d(
                            in_channels=nf_base,
                            out_channels=nf_out,
                            kernel_size=3,
                            padding=3//2,
                        ),
                    ]
                else:
                    outconv_lst.append(
                        nn.Conv2d(
                            in_channels=nf_base,
                            out_channels=nf_out,
                            kernel_size=3,
                            padding=3//2,
                        )
                    )
            self.outconv_lst.append(nn.Sequential(*outconv_lst))

        # IQA module
        self.iqam = IQAM(
            comp_type=comp_type,
        )

    def forward(self, inp_t, idx_out=-1):
        """
        idx_out:
            -1, output all images from all outputs for training.
            -2, judge by IQAM.
            0-4, output from the assigned exit.
        """
        if idx_out == -2:
            timer_wo_iqam = Timer()
            timer_wo_iqam.record()

        feat_lst_lst = []

        feat = inp_t
        for module_ in self.inconv_lst:
            feat = module_(feat)
        feat_lst = [feat]
        
        out_t_lst = []
        for idx_level in range(self.nlevel):
            feat_lst_lst.append(feat_lst)

            down = getattr(self, f'down_{idx_level}')
            feat = down(feat_lst_lst[-1][0])
            feat_lst = [feat]

            for idx_up in range(idx_level+1):
                inp_lst = []
                for pre_feat_lst in feat_lst_lst:
                    ndepth = idx_level + 1 - idx_up
                    if len(pre_feat_lst) >= ndepth:
                        inp_lst.append(pre_feat_lst[-ndepth])

                up = getattr(self, f'up_{idx_level}_{idx_up}')
                feat = up(
                    feat_lst[-1],
                    *inp_lst,
                )
                feat_lst.append(feat)

            if (idx_out == -1):
                out_conv = self.outconv_lst[idx_level]
                out_t = out_conv(feat_lst[-1])
                if self.if_residual:
                    out_t += inp_t
                out_t_lst.append(out_t)

            elif (idx_out == -2):
                out_conv = self.outconv_lst[idx_level]
                out_t = out_conv(feat_lst[-1])
                if self.if_residual:
                    out_t += inp_t
                timer_wo_iqam.record_inter()
                if_out = self.iqam.forward(out_t)
                if (idx_level == (self.nlevel - 1)) or if_out:
                    break
                else:
                    timer_wo_iqam.record()

            elif (idx_level == idx_out):
                out_conv = self.outconv_lst[idx_level]
                out_t = out_conv(feat_lst[-1])
                if self.if_residual:
                    out_t += inp_t
                break

        if (idx_out == -1):
            return torch.stack(out_t_lst, dim=0)  # nlevel B C H W
        elif (idx_out == -2):
            return sum(timer_wo_iqam.inter_lst), out_t  # B=1 C H W
        else:
            return out_t  # B=1 C H W

class RBQEModel(nn.Module):
    def __init__(self, opts_dict, if_train=True):
        super().__init__()
        
        self.opts_dict = opts_dict

        net = Network(if_train, **self.opts_dict)
        self.module_lst = dict(net=net)
        self.msg_lst = dict(
            net=f'> RBQE model is created with {net.cal_num_params():d} params (rank 0 solely).'
        )
