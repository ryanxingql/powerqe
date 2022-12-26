# RyanXingQL @2022
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmedit.utils import get_root_logger

from ..registry import BACKBONES

up_method_lst = ['upsample', 'transpose2d']
down_method_lst = ['avepool2d', 'strideconv']
reduce_method_lst = ['add', 'concat']


class Up(nn.Module):

    def __init__(self, method, nf_in=None):
        assert method in up_method_lst, 'NOT SUPPORTED YET!'

        super().__init__()

        if method == 'upsample':
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bicubic',
                                  align_corners=False)
        elif method == 'transpose2d':
            self.up = nn.ConvTranspose2d(
                in_channels=nf_in,
                out_channels=nf_in // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            )

    def forward(self, inp_t, ref_big):
        feat = self.up(inp_t)

        diff_h = ref_big.size()[2] - feat.size()[2]  # B C H W, H
        diff_w = ref_big.size()[3] - feat.size()[3]  # W

        if diff_h < 0:
            feat = feat[:, :, :ref_big.size()[2], :]
            diff_h = 0
        if diff_w < 0:
            feat = feat[:, :, :, :ref_big.size()[3]]
            diff_w = 0

        out_t = F.pad(
            input=feat,
            pad=[
                diff_w // 2,
                (diff_w - diff_w // 2),
                # only pad H and W; left (diff_w//2);
                # right remaining (diff_w - diff_w//2)
                diff_h // 2,
                (diff_h - diff_h // 2),
            ],
            mode='constant',
            value=0,  # pad with constant 0
        )

        return out_t


@BACKBONES.register_module()
class UNet(nn.Module):
    """U-Net for enhancement."""

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
        down='avepool2d',
        up='transpose2d',
        reduce='concat',
        residual=True,
    ):
        assert down in down_method_lst, 'NOT SUPPORTED YET!'
        assert up in up_method_lst, 'NOT SUPPORTED YET!'
        assert reduce in reduce_method_lst, 'NOT SUPPORTED YET!'

        super().__init__()

        self.nlevel = nlevel
        self.reduce = reduce
        self.residual = residual

        if residual:
            assert nf_in == nf_out, (f'Expect nf_in [{nf_in}] '
                                     f'== nf_out [{nf_out}]')

        self.inc = nn.Sequential(
            nn.Conv2d(in_channels=nf_in,
                      out_channels=nf_base,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
        )

        nf_lst = [nf_base]
        nl_lst = [nl_base]
        for idx_level in range(1, nlevel):
            nf_new = nf_lst[-1] * nf_gr if (nf_lst[-1] *
                                            nf_gr) <= nf_max else nf_max
            nf_lst.append(nf_new)
            nl_new = nl_lst[-1] * nl_gr if (
                (nl_lst[-1] * nl_gr) <= nl_max) else nl_max
            nl_lst.append(nl_new)

            # define downsampling operator

            if down == 'avepool2d':
                setattr(
                    self,
                    f'down_{idx_level}',
                    nn.AvgPool2d(kernel_size=2),
                )
            elif down == 'strideconv':
                setattr(
                    self,
                    f'down_{idx_level}',
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
                nn.Conv2d(in_channels=nf_lst[-2],
                          out_channels=nf_lst[-1],
                          kernel_size=3,
                          padding=1),
                nn.ReLU(inplace=True),
            ]
            for _ in range(nl_lst[-1]):
                module_lst += [
                    nn.Conv2d(in_channels=nf_lst[-1],
                              out_channels=nf_lst[-1],
                              kernel_size=3,
                              padding=1),
                    nn.ReLU(inplace=True),
                ]
            setattr(self, f'enc_{idx_level}', nn.Sequential(*module_lst))

        for idx_level in range((nlevel - 2), -1, -1):
            # define upsampling operator
            setattr(self, f'up_{idx_level}',
                    Up(nf_in=nf_lst[idx_level + 1], method=up))

            # define decoding operator

            if reduce == 'add':
                module_lst = [
                    nn.Conv2d(in_channels=nf_lst[idx_level],
                              out_channels=nf_lst[idx_level],
                              kernel_size=3,
                              padding=1),
                    nn.ReLU(inplace=True),
                ]
            else:
                module_lst = [
                    nn.Conv2d(in_channels=nf_lst[idx_level + 1],
                              out_channels=nf_lst[idx_level],
                              kernel_size=3,
                              padding=1),
                    nn.ReLU(inplace=True),
                ]
            for _ in range(nl_lst[idx_level]):
                module_lst += [
                    nn.Conv2d(in_channels=nf_lst[idx_level],
                              out_channels=nf_lst[idx_level],
                              kernel_size=3,
                              padding=1),
                    nn.ReLU(inplace=True),
                ]
            setattr(self, f'dec_{idx_level}', nn.Sequential(*module_lst))

        self.outc = nn.Conv2d(
            in_channels=nf_base,
            out_channels=nf_out,
            kernel_size=3,
            padding=1,
        )

    def forward(self, inp_t):
        feat = self.inc(inp_t)

        # down

        map_lst = []  # guidance maps
        for idx_level in range(1, self.nlevel):
            map_lst.append(feat)  # from level 0, 1, ..., (nlevel-1)
            down = getattr(self, f'down_{idx_level}')
            enc = getattr(self, f'enc_{idx_level}')
            feat = enc(down(feat))

        # up

        for idx_level in range((self.nlevel - 2), -1, -1):
            up = getattr(self, f'up_{idx_level}')
            dec = getattr(self, f'dec_{idx_level}')
            g_map = map_lst[idx_level]
            up_feat = up(inp_t=feat, ref_big=g_map)

            if self.reduce == 'add':
                feat = up_feat + g_map
            elif self.reduce == 'concat':
                feat = torch.cat((up_feat, g_map), dim=1)
            feat = dec(feat)

        out_t = self.outc(feat)

        if self.residual:
            out_t += inp_t

        return out_t

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass  # use default initialization
        else:
            raise TypeError('"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
