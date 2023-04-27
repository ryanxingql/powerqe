# RyanXingQL @2023
import torch
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2d

from ..registry import BACKBONES
from .base import BaseNet


class STDF(nn.Module):
    """STDF.

    Args:
    - `in_nc` (int): Num of input channels.
    - `out_nc` (int): Num of output channels.
    - `nf` (int): Num of channels (filters) of each conv layer.
    - `nb` (int): Num of conv layers.
    - `deform_ks` (int): Size of the deformable kernel.
    """

    def __init__(self,
                 in_nc=3,
                 out_nc=64,
                 nf=32,
                 nb=3,
                 base_ks=3,
                 deform_ks=3):
        super().__init__()

        self.in_nc = in_nc
        self.nb = nb
        self.deform_ks = deform_ks
        self.size_dk = deform_ks**2

        # U-shape backbone for learning offsets

        self.in_conv = nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2)

        for i in range(1, nb):
            setattr(
                self, 'dw_conv{}'.format(i),
                nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2)))
            setattr(
                self, 'up_conv{}'.format(i),
                nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1)))

        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1))

        self.out_conv = nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2)

        # deformable fusion using learned offsets

        # regression head
        # why in_nc*3*size_dk?
        # in_nc: each map use individual offset and mask
        # 2*size_dk: 2 coordinates for each point
        # 1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(nf,
                                     in_nc * 3 * self.size_dk,
                                     base_ks,
                                     padding=base_ks // 2)

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv2d(in_channels=in_nc,
                                                 out_channels=out_nc,
                                                 kernel_size=deform_ks,
                                                 padding=deform_ks // 2,
                                                 deform_groups=in_nc)

    def forward(self, x):
        b, _, _, h, w = x.shape
        x = x.view(b, -1, h, w)

        # U-shape backbone for learning offsets

        # feature extraction (with down-sampling)
        maps = [self.in_conv(x)]  # store feature maps for skip connections
        for i in range(1, self.nb):
            dw_conv = getattr(self, 'dw_conv{}'.format(i))
            maps.append(dw_conv(maps[-1]))

        # down x1 then up x1 at the bottom
        out = self.tr_conv(maps[-1])

        # feature reconstruction (with up-sampling)
        for i in range(self.nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(torch.cat([out, maps[i]], 1))  # skip connection
        out = self.out_conv(out)

        # deformable fusion using learned offsets

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        off_msk = self.offset_mask(out)
        nc_off = self.in_nc * 2 * self.size_dk
        off = off_msk[:, :nc_off, ...]
        msk = torch.sigmoid(off_msk[:, nc_off:, ...])
        fused_feat = self.deform_conv(x, off, msk)
        return fused_feat


class QENet(nn.Module):
    """QE subnet.

    Args:
    - `in_nc` (int): Num of input channels from STDF.
    - `nf` (int): Num of channels (filters) of each conv layer.
    - `nb` (int): Num of conv layers.
    - `out_nc` (int): Num of output channel. 3 for RGB, 1 for Y.
    """

    def __init__(self, in_nc=64, nf=48, nb=6, out_nc=3, base_ks=3):
        super().__init__()

        padding = base_ks // 2

        body = [nn.Conv2d(in_nc, nf, base_ks, padding=padding)]
        for _ in range(nb):
            body += [
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, base_ks, padding=padding)
            ]
        body += [
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, out_nc, base_ks, padding=padding)
        ]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


@BACKBONES.register_module()
class STDFNet(BaseNet):
    """STDF network structure.

    Ref: https://github.com/ryanxingql/stdf-pytorch

    Args:
    - `in_channels` (int): Channel number of inputs.
    - `out_channels` (int): Channel number of outputs.
    - `radius`: Frames number before the center frame.
    - `nf_stdf` (int): Channel number of intermediate features of STDF module.
    - `nb_stdf` (int): Block number of STDF module.
    - `nf_qe` (int): Channel number of intermediate features of QE module.
    - `nb_qe` (int): Block number of QE module.
    - `deform_ks` (int): Kernel size of deformable convolutions.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 radius=3,
                 nf_stdf=32,
                 nb_stdf=3,
                 nf_stdf_out=64,
                 deform_ks=3,
                 nf_qe=48,
                 nb_qe=6):
        super().__init__()

        self.radius = radius
        self.nf_qe = nf_qe
        self.nb_qe = nb_qe

        self.stdf = STDF(in_nc=in_channels * (radius * 2 + 1),
                         out_nc=nf_stdf_out,
                         nf=nf_stdf,
                         nb=nb_stdf,
                         deform_ks=deform_ks)
        self.qe_net = QENet(in_nc=nf_stdf_out,
                            nf=nf_qe,
                            nb=nb_qe,
                            out_nc=out_channels)

    def forward(self, x):
        """Forward.

        Args:
        - `x` (Tensor): Input tensor with the shape of (N, T, C, H, W).

        Returns:
        - `out` (Tensor): Out center frame with the shape of (N, C, H, W).
        """
        out = self.stdf(x)
        out = self.qe_net(out)
        out = out + x[:, self.radius, ...]  # residual learning
        return out
