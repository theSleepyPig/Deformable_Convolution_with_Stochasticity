
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import numpy as np


BatchNorm2d = nn.BatchNorm2d
Conv2d = nn.Conv2d

class BasicBlock(nn.Module):
    '''BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = BatchNorm2d(planes)
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes,
                       kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    '''Bottleneck.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = BatchNorm2d(planes)
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = BatchNorm2d(self.expansion*planes)
        self.conv3 = Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = F.relu(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, device = torch.device(0), normalize = None, pos = 0, eot = False, lb = 2048):
        super(ResNet, self).__init__()
        self.eot = eot
        self.pos = pos
        self.lb = lb
        self.device = device
        self.in_planes = 64
        shapes = [(1, 3, 32, 32), (1, 64, 32, 32), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8), (1, 512, 4, 4)]
        self.w_shape = shapes[self.pos]
        Mu_ = torch.ones(self.w_shape).to(self.device)
        SD_ = torch.ones(self.w_shape).to(self.device)
        self.Mu_ = nn.Parameter(Mu_)
        self.SD_ = nn.Parameter(SD_)
        self.rand_weight = torch.normal(0, 1, self.w_shape).to(self.device)
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.normalize = normalize

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)
        
        if self.eot:
            self.rand_weight = torch.normal(0, 1, self.w_shape).to(self.device)

        if self.pos == 0:
            out = x * (self.rand_weight * self.SD_ + self.Mu_)
        else:
            out = x
        out = self.conv1(out)
        out = F.relu(self.bn1(out))

        if self.pos == 1:
            out = out * (self.rand_weight * self.SD_ + self.Mu_)
        
        for i, op in enumerate(self.layer1):
            out = op(out)

        if self.pos == 2:
            out = out * (self.rand_weight * self.SD_ + self.Mu_)
        for i, op in enumerate(self.layer2):
            out = op(out)

        if self.pos == 3:
            out = out * (self.rand_weight * self.SD_ + self.Mu_)
        for i, op in enumerate(self.layer3):
            out = op(out)

        if self.pos == 4:
            out = out * (self.rand_weight * self.SD_ + self.Mu_)
        for i, op in enumerate(self.layer4):
            out = op(out)

        if self.pos == 5:
            out = out * (self.rand_weight * self.SD_ + self.Mu_)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def set_rands(self):
        min_SD = float(self.SD_.min())
        mean_lb = (self.lb / (self.w_shape[1] * self.w_shape[2] * self.w_shape[3]))
        if min_SD < mean_lb:
            with torch.no_grad():
                self.SD_[self.SD_ < mean_lb] += mean_lb - min_SD
        if not self.eot:
            self.rand_weight = torch.normal(0, 1, self.w_shape).to(self.device)

def ResNet18(num_classes=10, normalize=None, device = torch.device(0), pos = 0, eot = False, lb = 2048):
    return ResNet(BasicBlock, [2,2,2,2], normalize = normalize, num_classes=num_classes, pos = pos, device = device, eot = eot, lb = lb)
