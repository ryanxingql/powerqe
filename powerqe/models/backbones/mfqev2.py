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
import torch
import torch.nn as nn
from mmedit.models.backbones.sr_backbones.basicvsr_net import SPyNet
from mmedit.models.common import flow_warp

from ..registry import BACKBONES
from .base import BaseNet


@BACKBONES.register_module()
class MFQEv2(BaseNet):
    """MFQEv2 network structure.

    Note: We use a pre-trained SpyNet instead of the MC subnet in the paper.
    This way, we can train our model without the MC loss.

    Ref: "https://github.com/ryanxingql/mfqev2.0/blob/master/net_MFCNN.py"

    Args:
        io_channels (int): Number of I/O channels.
        nf (int): Channel number of intermediate features.
    """

    def __init__(self, io_channels=3, nf=32, spynet_pretrained=None):
        super().__init__()

        # for frame alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        self.ks3_conv_list = nn.ModuleList([
            nn.Conv2d(in_channels=io_channels,
                      out_channels=nf,
                      kernel_size=3,
                      padding=3 // 2) for _ in range(3)
        ])
        self.ks5_conv_list = nn.ModuleList([
            nn.Conv2d(in_channels=io_channels,
                      out_channels=nf,
                      kernel_size=5,
                      padding=5 // 2) for _ in range(3)
        ])
        self.ks7_conv_list = nn.ModuleList([
            nn.Conv2d(in_channels=io_channels,
                      out_channels=nf,
                      kernel_size=7,
                      padding=7 // 2) for _ in range(3)
        ])

        self.rec_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=9 * nf,
                          out_channels=nf,
                          kernel_size=3,
                          padding=3 // 2), nn.PReLU(),
                nn.BatchNorm2d(num_features=nf)),  # c10
            nn.Sequential(
                nn.Conv2d(in_channels=nf,
                          out_channels=nf,
                          kernel_size=3,
                          padding=3 // 2), nn.PReLU(),
                nn.BatchNorm2d(num_features=nf)),  # c11
            nn.Sequential(
                nn.Conv2d(in_channels=2 * nf,
                          out_channels=nf,
                          kernel_size=3,
                          padding=3 // 2), nn.PReLU(),
                nn.BatchNorm2d(num_features=nf)),  # c12
            nn.Sequential(
                nn.Conv2d(in_channels=3 * nf,
                          out_channels=nf,
                          kernel_size=3,
                          padding=3 // 2), nn.PReLU(),
                nn.BatchNorm2d(num_features=nf)),  # c13
            nn.Sequential(
                nn.Conv2d(in_channels=4 * nf,
                          out_channels=nf,
                          kernel_size=3,
                          padding=3 // 2), nn.PReLU(),
                nn.BatchNorm2d(num_features=nf)),  # c14
            nn.Conv2d(in_channels=nf,
                      out_channels=io_channels,
                      kernel_size=3,
                      padding=3 // 2)  # c15
        ])

    def align_frm(self, inp_frm, ref_frm):
        flow = self.spynet(ref_frm, inp_frm)  # n 2 h w
        aligned_frm = flow_warp(inp_frm, flow.permute(0, 2, 3, 1))  # n h w 2
        return aligned_frm

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with the shape of (N, T=3, C, H, W).

        Returns:
            Tensor: Output center frame with the shape of (N, C, H, W).
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
