#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(LCJ_Hust@126.com)

import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def _make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

class VGG_backbone(nn.Module):
    def __init__(self, num_layers, batch_norm, pretrained_path=None, **kwargs):
        super(VGG_backbone, self).__init__()

        vgg_nets = {
            11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }

        if num_layers not in vgg_nets:
            raise ValueError("{} is not a valid number of vgg layers".format(num_layers))

        if pretrained_path is not None:
            kwargs['init_weights'] = False

        self.vgg_bb = VGG(_make_layers(vgg_nets[num_layers], batch_norm=batch_norm), **kwargs)

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            self.vgg_bb.load_state_dict(checkpoint, strict=False)

    def forward(self, x):
        x = self.vgg_bb.features(x)

        return x

if __name__ == "__main__":
    pretrained_path = '/home/lcj/.cache/torch/checkpoints/vgg19_bn.pth'
    vgg = VGG_backbone(19, True, pretrained_path)
    inputs = torch.randn(1, 3, 224, 224).float()
    outputs = vgg(inputs)
    print("outputs: ", outputs.shape)
    print("*")


