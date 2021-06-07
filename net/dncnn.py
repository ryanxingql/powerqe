import torch.nn as nn

from utils import BaseNet


class Network(nn.Module):
    def __init__(self, nf_in=3, nf=64, ndepth=17, if_bn=False, nf_out=3):
        super().__init__()

        blk_lst = [
            nn.Conv2d(
                in_channels=nf_in,
                out_channels=nf,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        ]

        if if_bn:
            blk_lst += [
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=nf, affine=True),
                nn.ReLU(inplace=True),
            ] * (ndepth - 2)
        else:
            blk_lst += [
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
            ] * (ndepth - 2)            

        blk_lst.append(
            nn.Conv2d(
                in_channels=nf,
                out_channels=nf_out,
                kernel_size=3,
                padding=1,
            )
        )
        
        self.blk_lst = nn.Sequential(*blk_lst)

    def forward(self, inp_t, **_):
        out_t = self.blk_lst(inp_t)
        out_t += inp_t
        return out_t


class DnCNNModel(BaseNet):
    def __init__(self, opts_dict, if_train=False):
        self.net = dict(net=Network(**opts_dict['net']))
        super().__init__(opts_dict=opts_dict, if_train=if_train, infer_subnet='net')
