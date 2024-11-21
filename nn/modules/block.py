import torch
import torch.nn as nn
from .layer import *
from .common import autopad

class ResBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.conv1 = Conv(c1, c2, k, s, p)
        self.conv2 = Conv(c2, c2, k=1, act=False)
        self.downsample = Conv(c1, c2, 1, s, act=False) if c1 != c2 else None
        # self.act = nn.SiLU()
    
    def forward(self, x):
        branch = self.downsample(x) if self.downsample else x
        x = self.conv2(self.conv1(x))
        return x + branch
        # return self.act(x + branch)

class ResBlock2(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.conv1 = Conv(c1, c2, k, s, p)
        self.conv2 = Conv(c2, c2, k=1, act=False)
        self.downsample = Conv(c1, c2, 1, s, act=False) if c1 != c2 else None
        self.act = nn.SiLU()
    
    def forward(self, x):
        branch = self.downsample(x) if self.downsample else x
        x = self.conv2(self.conv1(x))
        return self.act(x + branch)
    
class DResBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.conv1 = DConv(c1, c2, k, s, p, e=1.19)
        self.conv2 = Conv(c2, c2, k=1, act=False)
        self.downsample = Conv(c1, c2, 1, s, act=False) if c1 != c2 else None
        # self.act = nn.SiLU()

    def forward(self, x):
        branch = self.downsample(x) if self.downsample else x
        x = self.conv2(self.conv1(x))
        return x + branch
        # return self.act(x + branch)

class DResBlock2(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.conv1 = DConv(c1, c2, k, s, p, e=1.2)
        self.conv2 = Conv(c2, c2, k=1, act=False)
        self.downsample = Conv(c1, c2, 1, s, act=False) if c1 != c2 else None
        self.act = nn.SiLU()

    def forward(self, x):
        branch = self.downsample(x) if self.downsample else x
        x = self.conv2(self.conv1(x))
        # return x + branch
        return self.act(x + branch)
    
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, c3, k=3, s=1, p=1):
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, 1)
        self.conv2 = Conv(c2, c2, k, s)
        self.conv3 = Conv(c2, c3, 1, 1)
        self.downsample = Conv(c1, c3, 1, s, act=False) if c1 != c3 else None
        self.act = nn.SiLU()
    
    def forward(self, x):
        branch = self.downsample(x) if self.downsample else x
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.act(x + branch)
    
class PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c//64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1),
                                 Conv(self.c * 2, self.c, 1, act=False))
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))
    
class PSD(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = DConv(self.c, self.c, k=3, e=0.7)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1),
                                 Conv(self.c * 2, self.c, 1, act=False))
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))
    
class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1),
                                 Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
    
class PSDBlock(nn.Module):
    def __init__(self, c, k=3, shortcut=True):
        super().__init__()
        self.attn = DConv(c, c, k=k)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1),
                                 Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
    
class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1*e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class C2PSD(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1*e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSDBlock(self.c, k=3) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
