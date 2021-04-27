import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcnv2 import DCNv2
from utils import BaseNet

class Fusion(nn.Module):
    """Spatio-temporal deformable fusion."""
    def __init__(
            self,
            in_nc=1,
            out_nc=64,
            nf=32,
            nb=3,
            base_ks=3,
            deform_ks=3
        ):
        super().__init__()

        self.nb = nb
        self.in_nc = in_nc

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
        )

        self.deform_conv = DCNv2(
            nf_in_msk=nf,
            nf_in=in_nc,
            nf_out=out_nc,
            ks=deform_ks,
            stride=1,
            padding=deform_ks//2,
            ngroups=in_nc,
            bias=False,
        )

    def forward(self, inp_t):
        nb = self.nb
        # feature extraction (with downsampling)
        enc_lst = [self.in_conv(inp_t)]  # record feature maps for skip connections of U-Net
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            enc_lst.append(dn_conv(enc_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(enc_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, enc_lst[i]], 1)
            )
        inp_t_msk = self.out_conv(out)

        feat = self.deform_conv(inp_t_msk=inp_t_msk, inp_t=inp_t)
        fused_feat = F.relu(
            feat,
            inplace=True,
        )

        return fused_feat

class QE(nn.Module):
    """Quality enhancement for fused features."""
    def __init__(self, in_nc=64, nf=48, nb=8, out_nc=3, base_ks=3):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_nc,
                out_channels=nf,
                kernel_size=base_ks,
                padding=1,
            ),
            nn.ReLU(inplace=True)
        )

        hid_conv_lst = []
        for _ in range(nb - 2):
            hid_conv_lst += [
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf,
                    kernel_size=base_ks,
                    padding=1,
                ),
                nn.ReLU(inplace=True)
            ]
        self.hid_conv = nn.Sequential(*hid_conv_lst)

        self.out_conv = nn.Conv2d(
            in_channels=nf,
            out_channels=out_nc,
            kernel_size=base_ks,
            padding=1,
        )

    def forward(self, inputs):
        out = self.in_conv(inputs)
        out = self.hid_conv(out)
        out = self.out_conv(out)
        return out

class Network(BaseNet):
    """STDF -> QE -> residual."""
    def __init__(
            self,
            radius,
            in_nc=1,
            nf_fusion=32,
            nb_fusion=3,
            out_nc_fusion=64,
            deform_ks=3,
            nf_qe=48,
            nb_qe=8,
            out_nc=1,
        ):
        super().__init__()

        self.radius = radius
        self.in_nc = in_nc
        self.out_nc_fusion = out_nc_fusion
        self.nf_fusion = nf_fusion
        self.nb_fusion = nb_fusion
        self.deform_ks = deform_ks

        self.nf_qe = nf_qe
        self.nb_qe = nb_qe
        self.out_nc = out_nc

        self.input_len = 2 * self.radius + 1

        self.fusion = Fusion(
            in_nc=self.in_nc*self.input_len, 
            out_nc=self.out_nc_fusion, 
            nf=self.nf_fusion, 
            nb=self.nb_fusion, 
            deform_ks=self.deform_ks
        )
        self.qe = QE(
            in_nc=self.out_nc_fusion,  
            nf=self.nf_qe, 
            nb=self.nb_qe, 
            out_nc=self.out_nc
        )

    def forward(self, inp_t):
        """
        inp_t:
            BGR: B C=[B1 B2 B3 R1 R2 R3 G1 G2 G3] H W
            Y: B C=[Y1 Y2 Y3] H W
        """
        feat = self.fusion(inp_t)
        out_t = self.qe(feat)

        frm_lst = [
            self.radius + idx_c * self.input_len for idx_c in range(self.in_nc)
        ]
        out_t += inp_t[:, frm_lst, ...]  # res: add middle frame

        return out_t

class STDFModel(nn.Module):
    def __init__(self, opts_dict, if_train=True):
        super().__init__()
        
        self.opts_dict = opts_dict

        net = Network(**self.opts_dict)
        self.module_lst = dict(net=net)
        self.msg_lst = dict(
            net=f'> STDF model is created with {net.cal_num_params():d} params (rank 0 solely).'
        )

