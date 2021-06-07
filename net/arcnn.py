import torch.nn as nn

from utils import BaseNet


class Network(nn.Module):
    def __init__(self, nf_in=3, nf_out=3, nf1=64, nf2=32, ks1=9, ks2=1, ks3=5):
        super().__init__()

        blk_lst = [
            nn.Conv2d(
                in_channels=nf_in,
                out_channels=nf1,
                kernel_size=ks1,
                padding=ks1//2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=nf1,
                out_channels=nf2,
                kernel_size=ks2,
                padding=ks2//2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=nf2,
                out_channels=nf_out,
                kernel_size=ks3,
                padding=ks3//2,
            )
        ]
        
        self.blk_lst = nn.Sequential(*blk_lst)

    def forward(self, inp_t, **_):
        out_t = self.blk_lst(inp_t)
        return out_t


class ARCNNModel(BaseNet):
    def __init__(self, opts_dict, if_train=False):
        self.net = dict(net=Network(**opts_dict['net']))
        super().__init__(opts_dict=opts_dict, if_train=if_train, infer_subnet='net')
