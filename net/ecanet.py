import torch.nn as nn

class ECA(nn.Module):
    """Efficient Channel Attention.
    ref: https://github.com/BangguWu/ECANet/blob/3adf7a99f829ffa2e94a0de1de8a362614d66958/models/eca_module.py#L5
    """
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size-1)//2,
            bias=False
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_t):
        logic = self.avg_pool(inp_t)  # B C H W -> B C 1 1 
        logic = self.conv(
            logic.squeeze(-1).transpose(-1, -2)
        ).transpose(-1, -2).unsqueeze(-1)  # B C 1 1 -> B C 1 -> B 1 C -> conv (just like FC, but ks=3) -> B 1 C -> B C 1 -> B C 1 1
        logic = self.sigmoid(logic)

        out_t = inp_t * logic.expand_as(inp_t)
        return out_t
        