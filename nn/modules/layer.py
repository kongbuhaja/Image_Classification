import torch.nn as nn
from .common import autopad
from DCNv4 import DCNv4

class DConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, e=1):
        super().__init__()
        assert k==3
        c = int(c1 * e)
        # self.cv1 = nn.Linear(c1, c)
        dg = c//16
        self.cv1 = nn.Conv2d(c1, c, 1, 1, groups=1)
        self.conv = DCNv4(c, c, k, s, autopad(k, p, d), group=dg, dw_kernel_size=None, without_pointwise=True, output_bias=False)
        # self.cv2 = nn.Linear(c, c2, bias=False)
        self.cv2 = nn.Conv2d(c, c2, 1, 1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else None

    def forward(self, x):
        b, c, h, w = x.shape
        # x = x.permute(0, 2, 3, 1).view(b, h*w, -1).contiguous()
        x = self.cv1(x)
        x = self.conv(x)
        x = self.cv2(x)
        # x = x.view(b, h, w, -1).permute(0, 3, 1, 2)
        x = self.bn(x)
        return self.act(x) if self.act else x

class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.act(x) if self.act else x
    
    def forward_fuse(self, x):
        x = self.conv(x)
        return self.act(x) if self.act else x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = self.head_dim * num_heads + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) * k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x