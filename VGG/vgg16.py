#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(LCJ_Hust@126.com)

import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        layers = []
        in_dim = 3
        out_dim = 64

        for i in range(13):
            layers += [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.ReLU(inplace=True)]
            in_dim = out_dim
            if i == 1 or i == 3 or i == 6 or i ==9 or i == 12:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if i != 9:
                    out_dim *= 2

        self.features = nn.Sequential(*layers)
        self.classifer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifer(x)
        return x

if __name__ == "__main__":
    inputs = torch.randn(1, 3, 224, 224).float().cuda()
    net = VGG16(21)
    net = net.cuda()

    output = net(inputs)
    print("inputs: ", inputs.shape)
    print("output: ", output.shape)