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
import torch
from mmedit.models import EDVRNet

from ..registry import BACKBONES


@BACKBONES.register_module()
class EDVRNetQE(EDVRNet):
    """EDVR network structure for quality enhancement.

    Differences to EDVRNet:
        Comment all upsamplings since the input is with high resolution.
            See forward.

    Args:
        io_channels (int): Number of I/O channels.
        mid_channels (int): Channel number of intermediate features.
        num_frames (int): Number of input frames.
        deform_groups (int): Deformable groups.
        num_blocks_extraction (int): Number of blocks for feature extraction.
        num_blocks_reconstruction (int): Number of blocks for reconstruction.
        center_frame_idx (int): The index of center frame.
            Frame counting from 0.
        with_tsa (bool): Whether to use TSA module.
    """

    def __init__(self,
                 io_channels,
                 mid_channels=64,
                 num_frames=5,
                 deform_groups=8,
                 num_blocks_extraction=5,
                 num_blocks_reconstruction=10,
                 center_frame_idx=2,
                 with_tsa=True):
        super().__init__(in_channels=io_channels,
                         out_channels=io_channels,
                         mid_channels=mid_channels,
                         num_frames=num_frames,
                         deform_groups=deform_groups,
                         num_blocks_extraction=num_blocks_extraction,
                         num_blocks_reconstruction=num_blocks_reconstruction,
                         center_frame_idx=center_frame_idx,
                         with_tsa=with_tsa)

        # remove unused parameters
        delattr(self, 'upsample1')
        delattr(self, 'upsample2')
        delattr(self, 'img_upsample')

    def forward(self, x):
        n, t, c, h, w = x.size()
        if h % 4 != 0 or w % 4 != 0:
            raise ValueError(
                f'Height ({h}) and width ({w}) should be divisible by 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract LR features
        # L1
        l1_feat = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        l1_feat = self.feature_extraction(l1_feat)
        # L2
        l2_feat = self.feat_l2_conv2(self.feat_l2_conv1(l1_feat))
        # L3
        l3_feat = self.feat_l3_conv2(self.feat_l3_conv1(l2_feat))

        l1_feat = l1_feat.view(n, t, -1, h, w)
        l2_feat = l2_feat.view(n, t, -1, h // 2, w // 2)
        l3_feat = l3_feat.view(n, t, -1, h // 4, w // 4)

        # pcd alignment
        ref_feats = [  # reference feature list
            l1_feat[:, self.center_frame_idx, :, :, :].clone(),
            l2_feat[:, self.center_frame_idx, :, :, :].clone(),
            l3_feat[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            neighbor_feats = [
                l1_feat[:, i, :, :, :].clone(), l2_feat[:, i, :, :, :].clone(),
                l3_feat[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_alignment(neighbor_feats, ref_feats))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (n, t, c, h, w)

        if self.with_tsa:
            feat = self.fusion(aligned_feat)
        else:
            aligned_feat = aligned_feat.view(n, -1, h, w)
            feat = self.fusion(aligned_feat)

        # reconstruction
        out = self.reconstruction(feat)
        # out = self.lrelu(self.upsample1(out))
        # out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        # base = self.img_upsample(x_center)
        # out += base
        out += x_center
        return out
