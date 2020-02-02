#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(LCJ_Hust@126.com)

import torch
import torch.nn as nn

class DetBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, extra=False):
        super(DetBottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes, planes, stride, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, stride, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.extra = extra

        if self.extra:
            self.extra_conv = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x) :
        if self.extra:
            identity = self.extra_conv(x)
        else:
            identity = x
        out = self.bottleneck(x)
        out += identity
        out = self.relu(out)
        return out

if __name__ == "__main__":
    detnet = DetBottleneck(64, 64, 1, False)
    detnet = detnet.cuda()

    inputs = torch.randn(1, 64, 224, 224).float().cuda()
    output = detnet(inputs)

    print(output.shape)

