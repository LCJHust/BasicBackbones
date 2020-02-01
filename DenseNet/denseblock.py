#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(caojialiang@deepmotion.ai)

import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleNeck(nn.Module):
    def __init__(self, inChannels, growthRate):
        super(BottleNeck, self).__init__()

        interChannels = growthRate * 4
        self.bn1 = nn.BatchNorm2d(inChannels)
        self.conv1 = nn.Conv2d(inChannels, interChannels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], dim=1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, inChannels, growthRate, nDenseBlocks):
        super(DenseBlock, self).__init__()
        self.layers = []
        for i in range(int(nDenseBlocks)):
            self.layers.append(BottleNeck(inChannels, growthRate))
            inChannels += growthRate

        self.denseblock = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.denseblock(x)

if __name__ == "__main__":
    inter_feas = torch.randn(4, 64, 128, 64).float().cuda()

    module = DenseBlock(64, 32, 4)
    module = module.cuda()

    out_features = module(inter_feas)

    print("out shape:", out_features.shape)



