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
import torch.nn.functional as F
from mmedit.models import BasicVSRPlusPlus
from mmedit.models.common import flow_warp
from mmedit.models.registry import BACKBONES


@BACKBONES.register_module()
class ProVQE(BasicVSRPlusPlus):
    """ProVQE network structure.

    Support either x4 upsampling or same size output.

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def propagate(self, feats, flows, module_name, key_frms):
        """Propagate the latent features throughout the sequence.

        Args:
            feats (dict(list[tensor])): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propagation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
            key_frms (list[list[int]]): Key-frame annotation of samples.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()  # (N, T-1, 2, H, W)

        T = t + 1
        if 'forward' in module_name:
            frame_idx = range(0, T)  # 0, 1, ..., T-1
            flow_idx = range(-1, T - 1)  # -1, 0, ..., T-2
        elif 'backward' in module_name:
            frame_idx = range(T - 1, -1, -1)  # T-1, T-2, ..., 0
            flow_idx = range(T - 1, -1, -1)  # T-1, T-2, ..., 0
            key_frms = [kfs[::-1] for kfs in key_frms]

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][idx]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()

            # deformable alignment
            if i > 0:  # has at least one previous frame
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(
                    0, 2, 3, 1))  # warp propagated feature from i-1 to i
                # NOTE: feat_prop is also the feats[module_name][-1]

                # initialize second-order features
                # being zeros if not replaced
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                # key frame
                if i > 1:  # has at least two previous frames
                    for ib in range(
                            n
                    ):  # each sample may have different key-frame annotation
                        if sum(key_frms[ib]
                               [:i - 1]) > 0:  # has at least one key frame
                            i_key = max(
                                [j for j in range(i - 1) if key_frms[ib][j]])
                        else:
                            i_key = i - 2

                        ngaps = i - i_key
                        feat_n2_ib = feats[module_name][-ngaps][ib:ib + 1]
                        if self.cpu_cache:
                            feat_n2_ib = feat_n2_ib.cuda()

                        flow_left = flows[ib:ib + 1,
                                          flow_idx[i_key + 1], :, :, :]
                        if self.cpu_cache:
                            flow_left = flow_left.cuda()
                        for j in range(1, ngaps):
                            flow_right = flows[ib:ib + 1, flow_idx[i_key + 1 +
                                                                   j], :, :, :]
                            if self.cpu_cache:
                                flow_right = flow_right.cuda()
                            flow_left = flow_right + flow_warp(
                                flow_left, flow_right.permute(0, 2, 3, 1))
                        flow_n2_ib = flow_left

                        cond_n2[ib:ib + 1] = flow_warp(
                            feat_n2_ib, flow_n2_ib.permute(0, 2, 3, 1)
                        )  # warp key propagated feature from i_key to i

                        feat_n2[ib:ib + 1] = feat_n2_ib
                        flow_n2[ib:ib + 1] = flow_n2_ib

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](
                    feat_prop, cond, flow_n1,
                    flow_n2)  # align propagated feature with cond and flow

            # concatenate
            # 1. spatial
            # 2. other modules' output
            # 3. propagated (aligned) for current module
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]
            feat = torch.cat(feat, dim=1)

            # residual blocks
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)
            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def forward(self, lqs, key_frms):
        """Forward function for ProVQE.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            key_frms (list[list[int]]): Key-frame annotation of samples.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU (no effect if using CPU)
        if t > self.cpu_cache_length and lqs.is_cuda:
            self.cpu_cache = True
        else:
            self.cpu_cache = False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(lqs.view(-1, c, h, w),
                                           scale_factor=0.25,
                                           mode='bicubic').view(
                                               n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :]
                                for i in range(0, t)]  # [t * (n, c, h, w)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of LR inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propagation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module_name = f'{direction}_{iter_}'

                feats[module_name] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module_name, key_frms)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)
