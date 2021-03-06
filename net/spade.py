import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.utils.spectral_norm as spectral_norm


class SPADE(nn.Module):
    """SPADE: normalize input -> generate learnable affine params by map -> affine transform to input.
    
    noutc_map must be in accordance with noutc.
        Since BN will not change the num of feature maps (ninc == noutc), the noutc_map should be equal to ninc.
    https://arxiv.org/pdf/1903.07291.pdf, Fig.2.
    """
    def __init__(self, ninc, ninc_map, nhidc_max, if_bn):
        super().__init__()

        self.ninc = ninc
        self.ninc_map = ninc_map
        self.nhidc_max = nhidc_max
        self.if_bn = if_bn

        self.norm_layer = nn.BatchNorm2d(num_features=self.ninc, affine=True) if self.if_bn else None
        # RyanXingQL: replace the original SynchronizedBatchNorm2d
        self.nhidc = self.nhidc_max if (self.ninc > self.nhidc_max) else self.ninc
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(self.ninc_map, self.nhidc, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mlp_gamma = nn.Conv2d(self.nhidc, self.ninc, kernel_size=3, padding=1, bias=False)
        self.mlp_beta = nn.Conv2d(self.nhidc, self.ninc, kernel_size=3, padding=1, bias=False)

    def forward(self, inp_t, guide_map):
        inp_t = nnf.interpolate(inp_t, size=guide_map.size()[2:], mode='nearest')  # so that the map is as big as inp_t

        normed_feat = self.norm_layer(inp_t) if self.if_bn else inp_t

        feat = self.mlp_shared(guide_map)
        gamma = self.mlp_gamma(feat)
        beta = self.mlp_beta(feat)

        out_t = normed_feat * gamma + beta
        return out_t


class SPADEResNetBlock(nn.Module):
    """SPADE ResNet Block.
    
    https://arxiv.org/pdf/1903.07291.pdf, Fig.4 left.
    input ─> ((SPADE ─> ReLU ─> 3x3 Conv) * 2) ─> + ─>
    └─────────────────────────────────────────────┘
    Leaky ReLU is used in the open-sourced code.
    """
    def __init__(self, ninc, noutc, nhidc_max, if_bn, if_spectral):
        super().__init__()
        
        self.ninc = ninc
        self.noutc = noutc
        self.nhidc_max = nhidc_max
        self.if_bn = if_bn
        self.if_spectral = if_spectral

        self.nmidc = min(self.ninc, self.noutc)
        self.learned_shortcut = (self.ninc != self.noutc)
        # if ninc != noutc, the shortcut should be learnable; else, the shortcut is straightforward
        
        # SPADE norm

        self.norm_s = SPADE(ninc=self.ninc, ninc_map=self.ninc, nhidc_max=self.nhidc_max, if_bn=self.if_bn) \
            if self.learned_shortcut else None
        self.norm_0 = SPADE(ninc=self.ninc, ninc_map=self.ninc, nhidc_max=self.nhidc_max, if_bn=self.if_bn)
        self.norm_1 = SPADE(ninc=self.nmidc, ninc_map=self.ninc, nhidc_max=self.nhidc_max, if_bn=self.if_bn)

        self.actv = nn.LeakyReLU(0.2)

        # Conv

        self.conv_s = nn.Conv2d(self.ninc, self.noutc, kernel_size=1, bias=False) if self.learned_shortcut else None
        self.conv_0 = nn.Conv2d(self.ninc, self.nmidc, kernel_size=3, stride=1, padding=3//2)
        self.conv_1 = nn.Conv2d(self.nmidc, self.noutc, kernel_size=3, stride=1, padding=3//2)

        # Spectral norm: applied to the module params. can stabilize GAN training

        if self.if_spectral:
            self.conv_s = spectral_norm(self.conv_s) if self.learned_shortcut else self.conv_s
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)

    def shortcut(self, inp_t, guide_map):
        out_t = self.conv_s(self.norm_s(inp_t, guide_map)) if self.learned_shortcut else inp_t
        return out_t

    def forward(self, inp_t, guide_map):
        inp_t_shortcut = self.shortcut(inp_t, guide_map)
        feat = self.conv_0(self.actv(self.norm_0(inp_t, guide_map)))
        feat = self.conv_1(self.actv(self.norm_1(feat, guide_map)))  # guide twice
        out_t = feat + inp_t_shortcut
        return out_t
