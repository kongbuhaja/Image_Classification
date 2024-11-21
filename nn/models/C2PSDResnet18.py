import torch
import torch.nn as nn
from nn.modules import *

class C2PSDResNet18(nn.Module):
    def __init__(self, nc, c=64):
        super().__init__()
        # self.conv = Conv(3, c, 7, 2)
        self.conv = Conv(3, c, 3, 1)
        # self.maxpool = nn.MaxPool2d(3, 2, autopad(3))
        self.layer1 = nn.Sequential(ResBlock(c*2**0, c*2**0, 3, 1), 
                                    ResBlock(c*2**0, c*2**0, 3, 1))
        self.layer2 = nn.Sequential(ResBlock(c*2**0, c*2**1, 3, 2),
                                    ResBlock(c*2**1, c*2**1, 3, 1))
        self.layer3 = nn.Sequential(ResBlock(c*2**1, c*2**2, 3, 2),
                                    ResBlock(c*2**2, c*2**2, 3, 1))
        self.layer4 = nn.Sequential(ResBlock(c*2**2, c*2**3, 3, 2),
                                    ResBlock(c*2**3, c*2**3, 3, 1))
        self.c2psd = C2PSD(c*2**3, c*2**3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c*2**3, nc)

    def forward(self, x):
        x = self.conv(x)
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.c2psd(x)
        x = self.avgpool(x)
        x = self.fc(torch.flatten(x, start_dim=1))
        return x