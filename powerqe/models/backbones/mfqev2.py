# RyanXingQL @2023
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmedit.models.backbones.sr_backbones.basicvsr_net import SPyNet
from mmedit.models.common import flow_warp
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class MFQEv2(nn.Module):
    """MFQEv2 network structure.

    Note: We use a pre-trained SpyNet instead of the MC subnet in the paper.
    This way, we can train our model without the MC loss.

    Ref: https://github.com/ryanxingql/mfqev2.0/blob/master/net_MFCNN.py

    Args:
        in_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        nf (int): Channel number of intermediate features.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 nf=32,
                 spynet_pretrained=None):

        super().__init__()

        # for frame alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        self.ks3_conv_list = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=nf,
                kernel_size=3,
                padding=3 // 2,
            ) for _ in range(3)
        ])
        self.ks5_conv_list = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=nf,
                kernel_size=5,
                padding=5 // 2,
            ) for _ in range(3)
        ])
        self.ks7_conv_list = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=nf,
                kernel_size=7,
                padding=7 // 2,
            ) for _ in range(3)
        ])

        self.rec_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=9 * nf,
                    out_channels=nf,
                    kernel_size=3,
                    padding=3 // 2,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(num_features=nf),
            ),  # c10
            nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf,
                    kernel_size=3,
                    padding=3 // 2,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(num_features=nf),
            ),  # c11
            nn.Sequential(
                nn.Conv2d(
                    in_channels=2 * nf,
                    out_channels=nf,
                    kernel_size=3,
                    padding=3 // 2,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(num_features=nf),
            ),  # c12
            nn.Sequential(
                nn.Conv2d(
                    in_channels=3 * nf,
                    out_channels=nf,
                    kernel_size=3,
                    padding=3 // 2,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(num_features=nf),
            ),  # c13
            nn.Sequential(
                nn.Conv2d(
                    in_channels=4 * nf,
                    out_channels=nf,
                    kernel_size=3,
                    padding=3 // 2,
                ),
                nn.PReLU(),
                nn.BatchNorm2d(num_features=nf),
            ),  # c14
            nn.Conv2d(
                in_channels=nf,
                out_channels=out_channels,
                kernel_size=3,
                padding=3 // 2,
            ),  # c15
        ])

    def align_frm(self, inp_frm, ref_frm):
        flow = self.spynet(ref_frm, inp_frm)  # n 2 h w
        aligned_frm = flow_warp(inp_frm, flow.permute(0, 2, 3, 1))  # n h w 2
        return aligned_frm

    def forward(self, x):
        """Forward function for MFQEv2.

        Args:
            x (Tensor): Input tensor with shape (n, 3, c, h, w).

        Returns:
            Tensor: Out center frame with shape (n, c, h, w).
        """
        # alignment
        center_frm = x[:, 1, ...]  # n c=3 h w
        aligned_left_pqf = self.align_frm(inp_frm=x[:, 0, ...],
                                          ref_frm=center_frm)
        aligned_right_pqf = self.align_frm(inp_frm=x[:, 2, ...],
                                           ref_frm=center_frm)

        # feature extraction
        ks3_feat_left_pqf = self.ks3_conv_list[0](aligned_left_pqf)
        ks3_feat_center_frm = self.ks3_conv_list[1](center_frm)
        ks3_feat_right_pqf = self.ks3_conv_list[2](aligned_right_pqf)

        ks5_feat_left_pqf = self.ks5_conv_list[0](aligned_left_pqf)
        ks5_feat_center_frm = self.ks5_conv_list[1](center_frm)
        ks5_feat_right_pqf = self.ks5_conv_list[2](aligned_right_pqf)

        ks7_feat_left_pqf = self.ks7_conv_list[0](aligned_left_pqf)
        ks7_feat_center_frm = self.ks7_conv_list[1](center_frm)
        ks7_feat_right_pqf = self.ks7_conv_list[2](aligned_right_pqf)

        feat_ = torch.cat(
            (ks3_feat_left_pqf, ks3_feat_center_frm, ks3_feat_right_pqf,
             ks5_feat_left_pqf, ks5_feat_center_frm, ks5_feat_right_pqf,
             ks7_feat_left_pqf, ks7_feat_center_frm, ks7_feat_right_pqf),
            dim=1)  # n c=9* h w

        # image reconstruction
        out_list = list()
        out_list.append(self.rec_conv[0](feat_))  # c10 in the paper
        for idx_dense in range(3):
            out_list.append(self.rec_conv[idx_dense + 1](torch.cat(
                out_list, dim=1)))  # c11, c12, c13 in the paper
        out = self.rec_conv[4](torch.cat(out_list, dim=1))  # c14 in the paper
        out = self.rec_conv[5](out)  # c15 in the paper
        out += center_frm  # res: add middle frame
        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError('"pretrained" must be a str or None.'
                            f' But received {type(pretrained)}.')
