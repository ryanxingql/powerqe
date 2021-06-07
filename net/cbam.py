import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # B 1 H W
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)  # B 2 H W
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    ref: https://github.com/luuuyi/CBAM.PyTorch/blob/83d3312c8c542d71dfbb60ee3a15454ba253a2b0/model/resnet_cbam.py#L25
    """

    def __init__(self, nf_in=64, ratio_ca=16, ks_sa=7):
        super().__init__()

        self.nf_in = nf_in
        self.ratio_ca = ratio_ca
        self.ks_sa = ks_sa

        self.ca = ChannelAttention(in_planes=self.nf_in, ratio=self.ratio_ca)
        self.sa = SpatialAttention(kernel_size=self.ks_sa)

    def forward(self, inp_t):
        feat = self.ca(inp_t) * inp_t
        feat = self.sa(feat) * feat
        out_t = feat + inp_t
        return out_t
