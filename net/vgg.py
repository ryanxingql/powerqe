import torch.nn as nn
from utils import BaseNet

class VGGStyleDiscriminator(BaseNet):
    def __init__(self, ninc, nf, nhidc, gt_h, gt_w):
        super().__init__()

        self.ninc = ninc
        self.nf = nf
        self.nhidc = nhidc

        self.conv0_0 = nn.Conv2d(self.ninc, self.nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv0_1 = nn.Conv2d(self.nf, self.nf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(self.nf, affine=True)

        self.conv1_0 = nn.Conv2d(self.nf, self.nf * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(self.nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(self.nf * 2, self.nf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(self.nf * 2, affine=True)

        self.conv2_0 = nn.Conv2d(self.nf * 2, self.nf * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(self.nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(self.nf * 4, self.nf * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(self.nf * 4, affine=True)

        self.conv3_0 = nn.Conv2d(self.nf * 4, self.nf * 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(self.nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(self.nf * 8, self.nf * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(self.nf * 8, affine=True)

        self.conv4_0 = nn.Conv2d(self.nf * 8, self.nf * 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(self.nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(self.nf * 8, self.nf * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.nf * 8, affine=True)

        inp_h = gt_h // 2 // 2 // 2 // 2 // 2
        inp_w = gt_w // 2 // 2 // 2 // 2 // 2
        self.linear1 = nn.Linear(self.nf * 8 * inp_h * inp_w, self.nhidc)
        self.linear2 = nn.Linear(self.nhidc, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input_t):
        feat = self.lrelu(self.conv0_0(input_t))  # (B ? 128 128)
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # (B ? 64 64)
        
        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # (B ? 32 32)
        
        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # (B ? 16 16)
        
        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # (B ? 8 8)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # (B ? 4 4)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out_t = self.linear2(feat)
        return out_t
    