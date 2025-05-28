import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import numpy as np

from typing import Optional, List, Tuple, Union
from torch. nn. modules. utils import _single, _pair, _triple, _reverse_repeat_tuple

BatchNorm2d = nn.BatchNorm2d
Conv2d = nn.Conv2d

from randpos_multi_resnet import MaskedConv2d

class RandonBasicBlock(nn.Module):
    '''BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mask=None, device=None):
        super(RandonBasicBlock, self).__init__()
        self.bn1 = BatchNorm2d(planes)
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=5, stride=stride, padding=2, bias=False, mask=mask, device=device)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes,
                       kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion*planes)
            )
            
        self.print_flag = True
        
        # self.last_selected = None  # 用来存储上次选择的点

    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out
    
    
    def set_rand_mask11(self):
        mask = np.zeros((5, 5))
        mask[1, 1] = 1
        mask[1, 2] = 1
        mask[1, 3] = 1
        mask[2, 1] = 1
        mask[2, 2] = 1
        # mask[2, 3] = 1
        mask[3, 1] = 1
        mask[3, 2] = 1
        # mask[3, 3] = 1
        
        # right_bottom = [(3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)] #2 in 6
        right_bottom = [(3, 4), (4, 3), (4, 4), (2, 4)] #2 in 4
        # right_bottom = [(3, 4), (4, 3), (4, 4), (3, 3), (2, 4)] #2 in 5
        # right_bottom = [(3, 3), (3, 4), (4, 3), (4, 4), (2, 4)] #1 in 5
        
        selected_right_bottom = np.random.choice(len(right_bottom), 2, replace=False)
        for pos in selected_right_bottom:
            mask[right_bottom[pos]] = 1
        
        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))  # 根据 conv1 权重的形状调整
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
        if self.print_flag:
            print(self.conv1.mask)
            self.print_flag = False
            
            
    def set_rand_mask12(self, global_corner_count):
        mask = np.zeros((5, 5))
        
        # 将中心九个点设为1
        mask[1, 1] = 1
        mask[1, 2] = 1
        mask[1, 3] = 1
        mask[2, 1] = 1
        mask[2, 2] = 1
        mask[2, 3] = 1
        mask[3, 1] = 1
        mask[3, 2] = 1
        mask[3, 3] = 1
        
        # patch location
        left_top = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        right_top = [(0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2)]
        left_bottom = [(3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2)]
        right_bottom = [(3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]
        
        corners = [left_top, left_bottom, right_top, right_bottom]
        
        # current patch
        current_corner = corners[global_corner_count % 4]
        
        # # selected patch
        # if global_corner_count % 4 == 0:
        #     print("Current corner: left_top")
        # elif global_corner_count % 4 == 1:
        #     print("Current corner: left_bottom")
        # elif global_corner_count % 4 == 2:
        #     print("Current corner: right_top")
        # else:
        #     print("Current corner: right_bottom")

        for pos in current_corner:
            mask[pos] = 0
        selected_positions = np.random.choice(len(current_corner), 1, replace=False)
        for pos in selected_positions:
            mask[current_corner[pos]] = 1
        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
        if self.print_flag:
            print(self.conv1.mask)
            self.print_flag = False


    def set_rand_mask1(self):
        mask = np.zeros((5, 5))
        
        top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]

        selected_top = np.random.choice(len(top), 4, replace=False)
        
        for pos in selected_top:
            mask[top[pos]] = 1

        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
        if self.print_flag:
            print(self.conv1.mask)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印
