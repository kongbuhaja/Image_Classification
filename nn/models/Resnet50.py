import torch
import torch.nn as nn
from nn.modules import *

class ResNet50(nn.Module):
    def __init__(self, nc, c=64):
        super().__init__()
        # self.conv = Conv(3, c, 7, 2)
        self.conv = Conv(3, c, 3, 1)
        # self.maxpool = nn.MaxPool2d(3, 2, autopad(3))
        self.layer1 = nn.Sequential(Bottleneck(c*2**0, c*2**0, c*2**2, 3, 1), 
                                    *[Bottleneck(c*2**2, c*2**0, c*2**2, 3, 1) for n in range(2)])
        self.layer2 = nn.Sequential(Bottleneck(c*2**2, c*2**1, c*2**3, 3, 2),
                                    *[Bottleneck(c*2**3, c*2**1, c*2**3, 3, 1) for n in range(3)])
        self.layer3 = nn.Sequential(Bottleneck(c*2**3, c*2**2, c*2**4, 3, 2),
                                    *[Bottleneck(c*2**4, c*2**2, c*2**4, 3, 1) for n in range(5)])
        self.layer4 = nn.Sequential(Bottleneck(c*2**4, c*2**3, c*2**5, 3, 2),
                                    *[Bottleneck(c*2**5, c*2**3, c*2**5, 3, 1) for n in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c*2**5, nc)
        self._reset_parameters()
    
    def _reset_parameters(self):
        xavier_uniform_(self.fc.weight.data)
        constant_(self.fc.bias.data, 0.)

    def forward(self, x):
        x = self.conv(x)
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(torch.flatten(x, start_dim=1))
        return x
    
    def gradcam(self, x):
        return self.forward(x)