# ultralytics/nn/modules/attn_extra.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- CBAM ----------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # [B,C,1,1]
        return x * w, w

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # [B,1,H,W]
        return x * a, a

class CBAM(nn.Module):
    """CBAM with optional attention-map return."""
    def __init__(self, channels, reduction=16, sa_kernel=7, return_maps=False):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(sa_kernel)
        self.return_maps = return_maps
        self.last_maps = {}  # {"ca": [B,C,1,1], "sa": [B,1,H,W]}

    def forward(self, x):
        x1, ca = self.ca(x)
        y, sa = self.sa(x1)
        if self.return_maps:
            self.last_maps = {"ca": ca.detach(), "sa": sa.detach()}
        return y

# ---------- ERF-style receptive-field block ----------
class ERFBlock(nn.Module):
    """
    Lightweight receptive-field expansion via dilated 3x3 branches + 1x1 fuse (+ residual).
    """
    def __init__(self, c, c_hidden=None, dilations=(1, 3), return_feats=False):
        super().__init__()
        c_hidden = c_hidden or max(8, c // 2)
        self.reduce = nn.Conv2d(c, c_hidden, 1, bias=False)
        self.branches = nn.ModuleList([
            nn.Conv2d(c_hidden, c_hidden, 3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        self.fuse = nn.Conv2d(c_hidden * len(dilations), c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)
        self.return_feats = return_feats
        self.last_feats = None  # concatenated branch feature map

    def forward(self, x):
        y0 = self.reduce(x)
        ys = [b(y0) for b in self.branches]
        y = torch.cat(ys, dim=1)
        out = self.bn(self.fuse(y))
        out = self.act(x + out)
        if self.return_feats:
            self.last_feats = y.detach()  # [B, c_hidden*#branches, H, W]
        return out
