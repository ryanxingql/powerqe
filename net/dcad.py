import torch.nn as nn
from utils import BaseNet

class Network(BaseNet):
    def __init__(self, nf_in=3, nf_out=3, nf=64, nblk=8):
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

        blk_lst += [
            nn.Conv2d(
                in_channels=nf,
                out_channels=nf,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        ] * nblk

        blk_lst.append(
            nn.Conv2d(
                in_channels=nf,
                out_channels=nf_out,
                kernel_size=3,
                padding=1,
            )
        )
        
        self.blk_lst = nn.Sequential(*blk_lst)

    def forward(self, inp_t):
        out_t = self.blk_lst(inp_t)
        out_t += inp_t
        return out_t

class DCADModel(nn.Module):
    def __init__(self, opts_dict, if_train=True):
        """
        if_train is useless, but left for utils.
        """
        super().__init__()
        
        self.opts_dict = opts_dict

        net = Network(**self.opts_dict)
        self.module_lst = dict(net=net)
        self.msg_lst = dict(
            net=f'> DCAD model is created with {net.cal_num_params():d} params (rank 0 solely).'
        )
