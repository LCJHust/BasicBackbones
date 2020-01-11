#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(caojialiang@deepmotion.ai)

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    """
    DenseLayer means BottleNeck structure.
    """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        num_inter_features = growth_rate * bn_size
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, num_inter_features,
                                                kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(growth_rate * 4))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(num_inter_features, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout2d(new_features, p=self.drop_rate)
        return torch.cat([new_features, x], dim=1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module("denselayer{}".format(str(i+1)), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, block_config=(6, 12, 24, 16),num_input_features=64,
                 growth_rate=32, bn_size=4, drop_rate=0,):
        super(DenseNet, self).__init__()

        # First norm
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_input_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_input_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        ## each dense block
        num_features = num_input_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(block_config[i], num_features, growth_rate, bn_size, drop_rate)
            self.features.add_module('denseblock{}'.format(str(i+1)), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) -1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module('transition{}'.format(str(i+1)), trans)
                num_features = num_features // 2

        ## final norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def densenet121(**kwargs):
    model = DenseNet(num_input_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model

def densenet169(**kwargs):
    model = DenseNet(num_input_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    return model

def densenet201(**kwargs):
    model = DenseNet(num_input_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    return model

def densenet161(**kwargs):
    model = DenseNet(num_input_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs)
    return model

class DenseNet_Bb(nn.Module):
    def __init__(self, num_layers, pretrained_path):
        super(DenseNet_Bb, self).__init__()
        densenets = {
            121: densenet121,
            161: densenet161,
            169: densenet169,
            201: densenet201
        }

        if num_layers not in densenets:
            raise ValueError("{} is not a valid number of densenet layers".format(num_layers))

        self.densenet_bb = densenets[num_layers]()

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            self.densenet_bb.load_state_dict(checkpoint)

    def forward(self, x):
        out = self.densenet_bb.features.conv0(x)
        out = self.densenet_bb.features.norm0(out)
        out = self.densenet_bb.features.relu0(out)

        return out


if __name__ == "__main__":
    # inputs = torch.randn((3, 3, 352, 1216))
    # net = densenet161()
    # print("..")


    pretrained_path = '/home/caojia/pretrained_models/densenet121.pth'
    densenet_backbone = DenseNet_Bb(121, pretrained_path)
    densenet_backbone = densenet_backbone.cuda()

    inputs = torch.randn((2, 3, 512, 768)).float().cuda()
    output = densenet_backbone.forward(inputs)

    print("output shape: ", output.shape)



















