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
import torch.nn.functional as nn_func

from ..registry import BACKBONES
from .base import BaseNet


class Up(nn.Module):
    def __init__(self, method, nf_in=None):
        super().__init__()

        supported_methods = ["upsample", "transpose2d"]
        if method not in supported_methods:
            raise NotImplementedError(
                f'Upsampling method should be in "{supported_methods}";'
                f' received "{method}".'
            )

        if method == "upsample":
            self.up = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=False)
        elif method == "transpose2d":
            self.up = nn.ConvTranspose2d(
                in_channels=nf_in,
                out_channels=nf_in // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            )

    def forward(self, inp_t, ref_big):
        feat = self.up(inp_t)

        diff_h = ref_big.size()[2] - feat.size()[2]  # (N, C, H, W); H
        diff_w = ref_big.size()[3] - feat.size()[3]  # W

        if diff_h < 0:
            feat = feat[:, :, : ref_big.size()[2], :]
            diff_h = 0
        if diff_w < 0:
            feat = feat[:, :, :, : ref_big.size()[3]]
            diff_w = 0

        # only pad H and W; left (diff_w//2)
        # right remaining (diff_w - diff_w//2)
        # pad with constant 0
        out_t = nn_func.pad(
            input=feat,
            pad=[
                diff_w // 2,
                (diff_w - diff_w // 2),
                diff_h // 2,
                (diff_h - diff_h // 2),
            ],
            mode="constant",
            value=0,
        )

        return out_t


@BACKBONES.register_module()
class UNet(BaseNet):
    def __init__(
        self,
        nf_in,
        nf_out,
        nlevel,
        nf_base,
        nf_max=1024,
        nf_gr=2,
        nl_base=1,
        nl_max=8,
        nl_gr=2,
        down="avepool2d",
        up="transpose2d",
        reduce="concat",
        residual=True,
    ):
        super().__init__()

        supported_up_methods = ["upsample", "transpose2d"]
        if up not in supported_up_methods:
            raise NotImplementedError(
                f'Upsampling method should be in "{supported_up_methods}";'
                f' received "{up}".'
            )

        supported_down_methods = ["avepool2d", "strideconv"]
        if down not in supported_down_methods:
            raise NotImplementedError(
                f'Downsampling method should be in "{supported_down_methods}";'
                f' received "{down}".'
            )

        supported_reduce_methods = ["add", "concat"]
        if reduce not in supported_reduce_methods:
            raise NotImplementedError(
                f'Reduce method should be in "{supported_reduce_methods}";'
                f' received "{reduce}".'
            )

        if residual and (nf_in != nf_out):
            raise ValueError(
                "The input channel number should be equal to the"
                " output channel number."
            )

        self.nlevel = nlevel
        self.reduce = reduce
        self.residual = residual

        self.inc = nn.Sequential(
            nn.Conv2d(
                in_channels=nf_in, out_channels=nf_base, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
        )

        nf_lst = [nf_base]
        nl_lst = [nl_base]
        for idx_level in range(1, nlevel):
            nf_new = nf_lst[-1] * nf_gr if (nf_lst[-1] * nf_gr) <= nf_max else nf_max
            nf_lst.append(nf_new)
            nl_new = nl_lst[-1] * nl_gr if ((nl_lst[-1] * nl_gr) <= nl_max) else nl_max
            nl_lst.append(nl_new)

            # define downsampling operator

            if down == "avepool2d":
                setattr(self, f"down_{idx_level}", nn.AvgPool2d(kernel_size=2))
            elif down == "strideconv":
                setattr(
                    self,
                    f"down_{idx_level}",
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=nf_lst[-2],
                            out_channels=nf_lst[-2],
                            kernel_size=3,
                            stride=2,
                            padding=3 // 2,
                        ),
                        nn.ReLU(inplace=True),
                    ),
                )

            # define encoding operator

            module_lst = [
                nn.Conv2d(
                    in_channels=nf_lst[-2],
                    out_channels=nf_lst[-1],
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
            ]
            for _ in range(nl_lst[-1]):
                module_lst += [
                    nn.Conv2d(
                        in_channels=nf_lst[-1],
                        out_channels=nf_lst[-1],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                ]
            setattr(self, f"enc_{idx_level}", nn.Sequential(*module_lst))

        for idx_level in range((nlevel - 2), -1, -1):
            # define upsampling operator
            setattr(self, f"up_{idx_level}", Up(nf_in=nf_lst[idx_level + 1], method=up))

            # define decoding operator

            if reduce == "add":
                module_lst = [
                    nn.Conv2d(
                        in_channels=nf_lst[idx_level],
                        out_channels=nf_lst[idx_level],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                ]
            else:
                module_lst = [
                    nn.Conv2d(
                        in_channels=nf_lst[idx_level + 1],
                        out_channels=nf_lst[idx_level],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                ]
            for _ in range(nl_lst[idx_level]):
                module_lst += [
                    nn.Conv2d(
                        in_channels=nf_lst[idx_level],
                        out_channels=nf_lst[idx_level],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(inplace=True),
                ]
            setattr(self, f"dec_{idx_level}", nn.Sequential(*module_lst))

        self.outc = nn.Conv2d(
            in_channels=nf_base, out_channels=nf_out, kernel_size=3, padding=1
        )

    def forward(self, inp_t):
        feat = self.inc(inp_t)

        # down

        map_lst = []  # guidance maps
        for idx_level in range(1, self.nlevel):
            map_lst.append(feat)  # from level 0, 1, ..., (nlevel-1)
            down = getattr(self, f"down_{idx_level}")
            enc = getattr(self, f"enc_{idx_level}")
            feat = enc(down(feat))

        # up

        for idx_level in range((self.nlevel - 2), -1, -1):
            up = getattr(self, f"up_{idx_level}")
            dec = getattr(self, f"dec_{idx_level}")
            g_map = map_lst[idx_level]
            up_feat = up(inp_t=feat, ref_big=g_map)

            if self.reduce == "add":
                feat = up_feat + g_map
            elif self.reduce == "concat":
                feat = torch.cat((up_feat, g_map), dim=1)
            feat = dec(feat)

        out_t = self.outc(feat)

        if self.residual:
            out_t += inp_t

        return out_t
