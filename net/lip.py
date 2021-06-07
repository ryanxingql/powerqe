import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SoftGate(nn.Module):
    def __init__(self, coeff):
        super().__init__()
        
        self.coeff = coeff
    
    def forward(self, inp_t):
        gated_inp_t = torch.sigmoid(inp_t).mul(self.coeff)
        # sigmoid for each [C H W] and then multiplied by coeff (e.g., 12)
        return gated_inp_t


class SimplifiedLIP(nn.Module):
    """Local importance pooling.
    
    A learnable pooling for downsampling at U-Net encoder.
    https://arxiv.org/pdf/1908.04156.pdf, Fig.3.
    """
    def __init__(self, nc, coeff, if_instancenorm):
        super().__init__()
        
        self.nc = nc
        self.coeff = coeff

        if if_instancenorm:
            self.logit = nn.Sequential(
                nn.Conv2d(self.nc, self.nc, 3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(self.nc, affine=True),
                SoftGate(self.coeff)
                )
        else:
            self.logit = nn.Sequential(
                nn.Conv2d(self.nc, self.nc, 3, stride=1, padding=1, bias=False),
                SoftGate(self.coeff)
                )            
    
    @staticmethod
    def lip2d(x, logit, kernel=3, stride=2, padding=1):
        weight = logit.exp()
        return nnf.avg_pool2d(x*weight, kernel, stride, padding) / nnf.avg_pool2d(weight, kernel, stride, padding)

    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)
    
    def forward(self, inp_t):
        logit = self.logit(inp_t)
        out_t = self.lip2d(inp_t, logit)  # pooling, just like downsampling
        return out_t
