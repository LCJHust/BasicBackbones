#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(LCJ_Hust@126.com)
# Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py#L202
# InceptionV2 -- InceptionA

import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class InceptionV2(nn.Module):
    def __init__(self, in_dims, out_1_dims, hid_2_1, out_2_dims, hid_3_1, hid_3_3, out_3_dims, out_4_dims):
        super(InceptionV2, self).__init__()

        self.branch_1 = BasicConv2d(in_dims, out_1_dims)
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_dims, hid_2_1, kernel_size=1),
            BasicConv2d(hid_2_1, out_2_dims, kernel_size=3, padding=1)
        )
        self.branch_3 = nn.Sequential(
            BasicConv2d(in_dims, hid_3_1, kernel_size=1),
            BasicConv2d(hid_3_1, hid_3_3, kernel_size=3, padding=1),
            BasicConv2d(hid_3_3, out_3_dims, kernel_size=3, padding=1)
        )
        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_dims, out_4_dims)
        )

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)

        return out


if __name__ == "__main__":
    inputs = torch.randn(3, 3, 64, 128).float().cuda()
    net = InceptionV2(3, 96, 48, 64, 64, 96, 96, 64)
    net = net.cuda()

    output = net(inputs)

    print("inputs: ", inputs.shape)
    print("outputs: ", output.shape)


