#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(LCJ_Hust@126.com)

import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, hid_channels=None):
        super(BasicConv2d, self).__init__()
        if kernel_size < 2:
            self.basic_module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.basic_module = nn.Sequential(
                nn.Conv2d(in_channels, hid_channels, kernel_size=(1, kernel_size), padding=(0, padding)),
                nn.Conv2d(hid_channels, out_channels, kernel_size=(kernel_size, 1), padding=(padding, 0)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.basic_module(x)

class InceptionV2_improved(nn.Module):
    def __init__(self, in_dims, out_1_dims, hid_2_1, out_2_dims, hid_3_1, hid_3_3, out_3_dims, out_4_dims):
        super(InceptionV2_improved, self).__init__()

        self.branch_1 = BasicConv2d(in_dims, out_1_dims, kernel_size=1, padding=0)
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_dims, hid_2_1, kernel_size=1, padding=0),
            BasicConv2d(hid_2_1, out_2_dims, kernel_size=3, padding=1, hid_channels=hid_2_1),
        )
        self.branch_3 = nn.Sequential(
            BasicConv2d(in_dims, hid_3_1, kernel_size=1, padding=0),
            BasicConv2d(hid_3_1, hid_3_3, kernel_size=5, padding=2, hid_channels=hid_3_1),
            BasicConv2d(hid_3_3, out_3_dims, kernel_size=5, padding=2, hid_channels=hid_3_3),
        )
        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_dims, out_4_dims, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)

        return out


if __name__ == "__main__":
    inputs = torch.randn(3, 3, 64, 64).float().cuda()
    net = InceptionV2_improved(3, 96, 48, 64, 64, 96, 96, 64)
    net = net.cuda()

    output = net(inputs)
    print("inputs: ", inputs.shape)
    print("output: ", output.shape)
