import torch
import torch.nn as nn
from utils import BaseNet
from .unet import UNet

class Network(BaseNet):
    def __init__(
            self,
            nf_in=3,
            nf_estimate=32,
            nf_out=3,
            nlevel_denoise=3,
            nf_base_denoise=64,
            nf_gr_denoise=2,
            nl_base_denoise=1,
            nl_gr_denoise=2,
            down_denoise='avepool2d',
            up_denoise='transpose2d',
            reduce_denoise='add',
            ):
        super().__init__()

        estimate_lst = [
            nn.Conv2d(
                in_channels=nf_in,
                out_channels=nf_estimate,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
        ]
        estimate_lst += [
            nn.Conv2d(
                in_channels=nf_estimate,
                out_channels=nf_estimate,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
        ] * 3
        estimate_lst += [
            nn.Conv2d(nf_estimate, nf_out, 3, padding=1),
            nn.ReLU(inplace=True)
        ]
        self.estimate = nn.Sequential(*estimate_lst)

        self.denoise = UNet(
            nf_in=nf_in*2,
            nf_out=nf_out,
            nlevel=nlevel_denoise,
            nf_base=nf_base_denoise,
            nf_gr=nf_gr_denoise,
            nl_base=nl_base_denoise,
            nl_gr=nl_gr_denoise,
            down=down_denoise,
            up=up_denoise,
            reduce=reduce_denoise,
            residual=False,
        )

    def forward(self, inp_t):
        estimated_noise_map = self.estimate(inp_t)
        concat_inp_t = torch.cat([inp_t, estimated_noise_map], dim=1)
        feat = self.denoise(concat_inp_t)
        out_t = feat + inp_t
        return out_t#, estimated_noise_map

class CBDNetModel(nn.Module):
    def __init__(self, opts_dict, if_train=True):
        super().__init__()
        
        self.opts_dict = opts_dict

        net = Network(**self.opts_dict)
        self.module_lst = dict(net=net)
        self.msg_lst = dict(
            net=f'> CBDNet model is created with {net.cal_num_params():d} params (rank 0 solely).'
        )
