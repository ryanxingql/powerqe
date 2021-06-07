import torch
import torch.nn as nn

from .vgg import VGGStyleDiscriminator

from utils import BaseNet


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32, rescale=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat+1*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat+2*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat+3*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat+4*num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.rescale = rescale

    def forward(self, x0):
        x1 = self.lrelu(self.conv1(x0))
        x2 = self.lrelu(self.conv2(torch.cat((x0, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x0, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x0, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x0, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        out = x5 * self.rescale + x0
        return out


class RRDB(nn.Module):
    """Residual in residual."""
    def __init__(self, num_feat=64, num_grow_ch=32, rescale=0.2):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch, rescale)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch, rescale)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch, rescale)
        self.rescale = rescale

    def forward(self, x0):
        x1 = self.rdb1(x0)
        x2 = self.rdb2(x1)
        out = self.rdb3(x2)
        # Empirically, we use 0.2 to scale the residual for better performance
        out = out * self.rescale + x0
        return out


class ESRGANGenerator(nn.Module):
    """Residual in residual."""
    def __init__(self, num_in_ch, num_feat, num_block, num_grow_ch, num_out_ch):
        super().__init__()

        self.conv_in = nn.Conv2d(num_in_ch, num_feat, 3, stride=1, padding=1)
        
        layers = []
        for _ in range(num_block):
            layers.append(RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch))
        self.rrdb_lst = nn.Sequential(*layers)
        
        self.conv_mid = nn.Conv2d(num_feat, num_feat, 3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(num_feat, num_out_ch, 3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, inp_t, **_):
        x1 = self.conv_in(inp_t)  # no lrelu
        x2 = self.rrdb_lst(x1)
        x3 = self.conv_mid(x2)
        x4 = x1 + x3
        out = self.conv_out(self.lrelu(x4))
        return out


class ESRGANModel(BaseNet):
    def __init__(self, opts_dict, if_train=False):
        self.net = dict(gen=ESRGANGenerator(**opts_dict['generator']))
        if if_train:
            self.net.update(dict(dis=VGGStyleDiscriminator(**opts_dict['discriminator'])))

        super().__init__(opts_dict=opts_dict, if_train=if_train, infer_subnet='gen')
