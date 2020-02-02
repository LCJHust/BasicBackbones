#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(caojialiang@deepmotion.ai)


import torch
import torch.nn as nn

class BottleNeck(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super(BottleNeck, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_dim)
        )

        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        out = self.bottleneck(x)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

if __name__ == '__main__':
    module = BottleNeck(32, 64)
    module = module.cuda()

    x = torch.randn((3, 64, 160, 320)).float().cuda()
    output = module(x)

    print("output shape:", output.shape)
