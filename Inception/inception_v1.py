#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(LCJ_Hust@126.com)
# Reference:https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py#L240
# InceptionV1 -- InceptionB

import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)

class InceptionV1(nn.Module):
    def __init__(self, in_dim, hid_1_1, hid_2_1, hid_2_3, hid_3_1, out_3_5, out_4_1):
        super(InceptionV1, self).__init__()
        self.branch_1x1 = BasicConv2d(in_dim, hid_1_1, 1)
        self.branch_3x3 = nn.Sequential(
            BasicConv2d(in_dim, hid_2_1, 1),
            BasicConv2d(hid_2_1, hid_2_3, 3, padding=1)
        )
        self.branch_5x5 = nn.Sequential(
            BasicConv2d(in_dim, hid_3_1, 1),
            BasicConv2d(hid_3_1, out_3_5, 5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            BasicConv2d(in_dim, out_4_1, 1)
        )

    def forward(self, x):
        x1 = self.branch_1x1(x)
        x2 = self.branch_3x3(x)
        x3 = self.branch_5x5(x)
        x4 = self.branch_pool(x)
        print("branch shape: ", x1.shape, x2.shape, x3.shape, x4.shape)

        out = torch.cat([x1, x2, x3, x4], dim=1)

        return out

if __name__ == "__main__":
    inputs = torch.randn((3, 3, 64, 128)).cuda()
    net = InceptionV1(3, 64, 32, 64, 64, 96, 32)
    net = net.cuda()

    print("input:", inputs.shape)
    output = net(inputs)
    print("..")