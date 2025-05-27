import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

import numpy as np

BatchNorm2d = nn.BatchNorm2d
Conv2d = nn.Conv2d

from typing import Optional, List, Tuple, Union
from torch. nn. modules. utils import _single, _pair, _triple, _reverse_repeat_tuple

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class BasicBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock2, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MaskedConv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn2 = BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        
        self.print_flag = True
        
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    
    
    def set_rand_mask111(self):
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
            print(self.conv1.mask.shape)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印
            
    def set_rand_mask11(self):
        mask = np.zeros((5, 5))
        # mask[2, 2] = 1  # 中间一个固定为1
        mask[0, 3] = 1
        mask[1, 0] = 1
        mask[4, 1] = 1
        mask[3, 4] = 1

        
        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
        if self.print_flag:
            print(self.conv1.mask)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印

# non repeated 5*5
class RandomBasicBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(RandomBasicBlock2, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MaskedConv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn2 = BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        
        self.print_flag = True
        
        self.last_selected = None  # 用来存储上次选择的点
        
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    
    
    # def set_rand_mask111(self):
    #     mask = np.zeros((5, 5))
        
    #     top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]    

    #     selected_top = np.random.choice(len(top), 4, replace=False)

    #     for pos in selected_top:
    #         mask[top[pos]] = 1
            
    #     # 根据 conv1 权重的形状调整 mask
    #     mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
    #     self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
    #     if self.print_flag:
    #         print(self.conv1.mask)
    #         print(self.conv1.mask.shape)
    #         self.print_flag = False  # 设置标志为 False，以便以后不再打印
            
    def set_rand_mask11(self):
        mask = np.zeros((5, 5))
        
        top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]

        # selected_top = np.random.choice(len(top), 4, replace=False)
        
        if self.last_selected is not None:
            available_choices = [idx for idx in range(len(top)) if idx not in self.last_selected]
        else:
            available_choices = list(range(len(top)))
            
        selected_top = np.random.choice(available_choices, 4, replace=False)
        self.last_selected = selected_top  
        
        for pos in selected_top:
            mask[top[pos]] = 1

        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
        if self.print_flag:
            print(self.conv1.mask)
            print(self.conv1.mask.shape)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印
            
# non repeated 5*5 3 point
class RandomBasicBlock23(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(RandomBasicBlock23, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MaskedConv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn2 = BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        
        self.print_flag = True
        
        self.last_selected = None  # 用来存储上次选择的点
        
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    
    
    # def set_rand_mask111(self):
    #     mask = np.zeros((5, 5))
        
    #     top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]    

    #     selected_top = np.random.choice(len(top), 4, replace=False)

    #     for pos in selected_top:
    #         mask[top[pos]] = 1
            
    #     # 根据 conv1 权重的形状调整 mask
    #     mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
    #     self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
    #     if self.print_flag:
    #         print(self.conv1.mask)
    #         print(self.conv1.mask.shape)
    #         self.print_flag = False  # 设置标志为 False，以便以后不再打印
            
    def set_rand_mask11(self):
        mask = np.zeros((5, 5))
        
        top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]

        # selected_top = np.random.choice(len(top), 4, replace=False)
        
        if self.last_selected is not None:
            available_choices = [idx for idx in range(len(top)) if idx not in self.last_selected]
        else:
            available_choices = list(range(len(top)))
            
        selected_top = np.random.choice(available_choices, 3, replace=False)
        self.last_selected = selected_top  
        
        for pos in selected_top:
            mask[top[pos]] = 1

        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
        if self.print_flag:
            print(self.conv1.mask)
            print(self.conv1.mask.shape)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印

# non repeated 5*5 2 point
class RandomBasicBlock22(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(RandomBasicBlock22, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MaskedConv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn2 = BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        
        self.print_flag = True
        
        self.last_selected = None  # 用来存储上次选择的点
        
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    
    
    # def set_rand_mask111(self):
    #     mask = np.zeros((5, 5))
        
    #     top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]    

    #     selected_top = np.random.choice(len(top), 4, replace=False)

    #     for pos in selected_top:
    #         mask[top[pos]] = 1
            
    #     # 根据 conv1 权重的形状调整 mask
    #     mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
    #     self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
    #     if self.print_flag:
    #         print(self.conv1.mask)
    #         print(self.conv1.mask.shape)
    #         self.print_flag = False  # 设置标志为 False，以便以后不再打印
            
    def set_rand_mask11(self):
        mask = np.zeros((5, 5))
        
        top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]

        # selected_top = np.random.choice(len(top), 4, replace=False)
        
        if self.last_selected is not None:
            available_choices = [idx for idx in range(len(top)) if idx not in self.last_selected]
        else:
            available_choices = list(range(len(top)))
            
        selected_top = np.random.choice(available_choices, 2, replace=False)
        self.last_selected = selected_top  
        
        for pos in selected_top:
            mask[top[pos]] = 1

        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
        if self.print_flag:
            print(self.conv1.mask)
            print(self.conv1.mask.shape)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印

# non repeated 5*5 2 point
class RandomBasicBlock21(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(RandomBasicBlock21, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MaskedConv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn2 = BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        
        self.print_flag = True
        
        self.last_selected = None  # 用来存储上次选择的点
        
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    
    
    # def set_rand_mask111(self):
    #     mask = np.zeros((5, 5))
        
    #     top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]    

    #     selected_top = np.random.choice(len(top), 4, replace=False)

    #     for pos in selected_top:
    #         mask[top[pos]] = 1
            
    #     # 根据 conv1 权重的形状调整 mask
    #     mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
    #     self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
    #     if self.print_flag:
    #         print(self.conv1.mask)
    #         print(self.conv1.mask.shape)
    #         self.print_flag = False  # 设置标志为 False，以便以后不再打印
            
    def set_rand_mask11(self):
        mask = np.zeros((5, 5))
        
        top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]

        # selected_top = np.random.choice(len(top), 4, replace=False)
        
        if self.last_selected is not None:
            available_choices = [idx for idx in range(len(top)) if idx not in self.last_selected]
        else:
            available_choices = list(range(len(top)))
            
        selected_top = np.random.choice(available_choices, 1, replace=False)
        self.last_selected = selected_top  
        
        for pos in selected_top:
            mask[top[pos]] = 1

        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
        if self.print_flag:
            print(self.conv1.mask)
            print(self.conv1.mask.shape)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印
            
# repeated 5*5 2 point
class RandomBasicBlock2222(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(RandomBasicBlock2222, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MaskedConv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn2 = BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        
        self.print_flag = True
        
        # self.last_selected = None  # 用来存储上次选择的点
        
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    
    
    def set_rand_mask11(self):
        mask = np.zeros((5, 5))
        
        top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]    

        selected_top = np.random.choice(len(top), 2, replace=False)

        for pos in selected_top:
            mask[top[pos]] = 1
            
        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
        if self.print_flag:
            print(self.conv1.mask)
            print(self.conv1.mask.shape)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印
            
    # def set_rand_mask11(self):
    #     mask = np.zeros((5, 5))
        
    #     top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]

    #     # selected_top = np.random.choice(len(top), 4, replace=False)
        
    #     if self.last_selected is not None:
    #         available_choices = [idx for idx in range(len(top)) if idx not in self.last_selected]
    #     else:
    #         available_choices = list(range(len(top)))
            
    #     selected_top = np.random.choice(available_choices, 5, replace=False)
    #     self.last_selected = selected_top  
        
    #     for pos in selected_top:
    #         mask[top[pos]] = 1

    #     # 根据 conv1 权重的形状调整 mask
    #     mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
    #     self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
    #     if self.print_flag:
    #         print(self.conv1.mask)
    #         print(self.conv1.mask.shape)
    #         self.print_flag = False  # 设置标志为 False，以便以后不再打印

# repeated 5*5 5 point
class RandomBasicBlock2225(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(RandomBasicBlock2225, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MaskedConv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn2 = BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        
        self.print_flag = True
        
        # self.last_selected = None  # 用来存储上次选择的点
        
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    
    
    def set_rand_mask11(self):
        mask = np.zeros((5, 5))
        
        top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]    

        selected_top = np.random.choice(len(top), 5, replace=False)

        for pos in selected_top:
            mask[top[pos]] = 1
            
        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
        if self.print_flag:
            print(self.conv1.mask)
            print(self.conv1.mask.shape)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印
            
# non repeated 7*7
class RandomBasicBlock20(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(RandomBasicBlock20, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MaskedConv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn2 = BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        
        self.print_flag = True
        
        self.last_selected = None  # 用来存储上次选择的点
        
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    
    
    def set_rand_mask11(self):
        mask = np.zeros((7, 7))
        
        # 生成所有的 7x7 网格坐标
        all_coords = [(i, j) for i in range(7) for j in range(7)]
    
        # 过滤出保留的坐标（满足 i+j >= 3 的条件）
        top = [coord for coord in all_coords if sum(coord) >= 3]
        # selected_top = np.random.choice(len(top), 4, replace=False)
        
        if self.last_selected is not None:
            available_choices = [idx for idx in range(len(top)) if idx not in self.last_selected]
        else:
            available_choices = list(range(len(top)))
            
        selected_top = np.random.choice(available_choices, 4, replace=False)
        self.last_selected = selected_top  
        
        for pos in selected_top:
            mask[top[pos]] = 1

        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
        if self.print_flag:
            print(self.conv1.mask)
            print(self.conv1.mask.shape)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印

# 7*7 不相干 2592
class RandomBasicBlock35(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(RandomBasicBlock35, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MaskedConv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn2 = BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        
        # 2592
        self.groups = [
            [16, 18, 20, 30, 32, 34],  # blue
            [10, 12, 24, 26, 38, 40],   # red
            [15, 17, 19, 21, 29, 31, 33, 35],             # grey
            [9, 11, 13, 23, 25, 27, 37, 39, 41]   # green...
        ]
        
        self.mask_size = (7, 7)  # Mask size is 7*7
        
        self.print_counter = 0
        self.max_prints = 6
        
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
    
    
    def set_rand_mask11(self):
        """
        Randomly select one point from each group to form a mask.
        """
        # Initialize a zero mask
        mask = np.zeros(self.mask_size)

        # For each group, randomly select one point
        for group in self.groups:
            selected_position = np.random.choice(group) - 1  # Convert to 0-based index
            row, col = divmod(selected_position, 7)
            mask[row, col] = 1.0

        # Convert mask to torch tensor and expand it to match the convolutional weights
        selected_mask = torch.tensor(mask, dtype=torch.float32)
        expanded_mask = selected_mask.repeat(self.conv1.weight.shape[0], 1, 1, 1)  # expand mask to match the convolutional weights
        self.conv1.mask = expanded_mask

        if self.print_counter < self.max_prints:
            print(self.conv1.mask)
            self.print_counter += 1
            
# 5*5 不相干 2592 3 point
class RandomBasicBlock353(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(RandomBasicBlock353, self).__init__()
        self.bn1 = BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MaskedConv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn2 = BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        
        # Grouping the points according to your request
        self.groups = [
            [1, 3, 5, 11, 13, 15, 21, 23, 25],  # Group 1
            [2, 4, 12, 14, 22, 24],             # Group 2
            [6, 8, 10, 16, 18, 20]             # Group 3
            # [7, 9, 17, 19]                      # Group 4
        ]

        self.mask_size = (5, 5)  # Mask size is 5x5
        
        self.print_counter = 0
        self.max_prints = 3

    def set_rand_mask11(self):
        """
        Randomly select one point from each group to form a mask.
        """
        # Initialize a 5x5 zero mask
        mask = np.zeros(self.mask_size)

        # For each group, randomly select one point
        for group in self.groups:
            selected_position = np.random.choice(group) - 1  # Convert to 0-based index
            row, col = divmod(selected_position, 5)
            mask[row, col] = 1.0

        # Convert mask to torch tensor and expand it to match the convolutional weights
        selected_mask = torch.tensor(mask, dtype=torch.float32)
        expanded_mask = selected_mask.repeat(self.conv1.weight.shape[0], 1, 1, 1)  # expand mask to match the convolutional weights
        self.conv1.mask = expanded_mask

        if self.print_counter < self.max_prints:
            print(self.conv1.mask)
            self.print_counter += 1
        
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

            

class NetworkBlock2(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block1, block2, stride, dropRate=0.0):
        super(NetworkBlock2, self).__init__()
        self.layer = self._make_layer(block1, block2, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block1, block2, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        nb_layers = int(nb_layers) - 1
        for i in range(int(nb_layers)):
            layers.append(block1(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        layers.append(block2(out_planes, out_planes, 1, dropRate))
        
        # for i in range(int(nb_layers)):
        #     if i == nb_layers - 1:
        #         layers.append(block2(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, dropRate))
        #     else:
        #         layers.append(block1(in_planes if i == 0 else out_planes, out_planes, i == 0 and stride or 1, dropRate))    
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

    # def set_rand_mask1(self):
    #     for layer in self.layer:
    #         layer.set_rand_mask11()
    
class NetworkBlock3(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block1, block2, stride, dropRate=0.0):
        super(NetworkBlock3, self).__init__()
        self.layer = self._make_layer(block1, block2, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block1, block2, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        nb_layers = int(nb_layers) - 2
        for i in range(int(nb_layers)):
            layers.append(block1(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        layers.append(block2(out_planes, out_planes, 1, dropRate))
        layers.append(block2(out_planes, out_planes, 1, dropRate))
        # layers.append(block2(out_planes, out_planes, 1, dropRate))
        
        # for i in range(int(nb_layers)):
        #     if i == nb_layers - 1:
        #         layers.append(block2(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, dropRate))
        #     else:
        #         layers.append(block1(in_planes if i == 0 else out_planes, out_planes, i == 0 and stride or 1, dropRate))    
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
    
    
class NetworkBlock4(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block1, block2, stride, dropRate=0.0):
        super(NetworkBlock4, self).__init__()
        self.layer = self._make_layer(block1, block2, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block1, block2, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        nb_layers = int(nb_layers) - 4
        for i in range(int(nb_layers)):
            layers.append(block1(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        layers.append(block2(out_planes, out_planes, 1, dropRate))
        layers.append(block2(out_planes, out_planes, 1, dropRate))
        layers.append(block2(out_planes, out_planes, 1, dropRate))
        layers.append(block2(out_planes, out_planes, 1, dropRate))
        # layers.append(block2(out_planes, out_planes, 1, dropRate))
        
        # for i in range(int(nb_layers)):
        #     if i == nb_layers - 1:
        #         layers.append(block2(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, dropRate))
        #     else:
        #         layers.append(block1(in_planes if i == 0 else out_planes, out_planes, i == 0 and stride or 1, dropRate))    
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)        
     
class NetworkBlock4layers(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block1, block2, stride, dropRate=0.0):
        super(NetworkBlock4layers, self).__init__()
        self.layer = self._make_layer(block1, block2, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block1, block2, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        nb_layers = int(nb_layers) - 1
        for i in range(int(nb_layers)):
            layers.append(block1(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        layers.append(block2(out_planes, out_planes, 1, dropRate))
        
        # for i in range(int(nb_layers)):
        #     if i == nb_layers - 1:
        #         layers.append(block2(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, dropRate))
        #     else:
        #         layers.append(block1(in_planes if i == 0 else out_planes, out_planes, i == 0 and stride or 1, dropRate))    
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)   
    
class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple],
                 stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0,
                 dilation: Union[int, tuple] = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', device=None, dtype=None, mask=None):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias, padding_mode, device, dtype)


        self.device = device
        # Ensure mask is a tensor with the same shape as the weight
        if mask is not None:
            if isinstance(mask, list):
                mask = np.array(mask)  # Convert list to NumPy array
            if isinstance(mask, np.ndarray):
                mask = torch.tensor(mask, dtype=torch.float32)  # Convert NumPy array to PyTorch tensor
            print(self.weight.shape)
            print(mask)
            assert mask.shape == self.weight.shape, "Mask shape must match weight shape"
            self.mask = mask.to(device)
        else:
            self.mask = torch.ones_like(self.weight).to(device)

        self.to(self.device)

    def _conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):

        weight = weight.to(input.device)
        self.mask = self.mask.to(input.device)
        masked_weight = weight * self.mask
        masked_weight = masked_weight.to(input.device)
        
        # print("Original weight:", weight)
        # print("Mask:", self.mask)
        # print("Masked weight:", masked_weight)

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            masked_weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, masked_weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, self.weight, self.bias)
    


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, normalize=None):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.normalize = normalize

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class WideResNet2(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, normalize=None, is_n_repeat = True):
        super(WideResNet2, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        block2 = BasicBlock2
        # 1st conv before any network block
        self.conv1 = Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block2, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock3(n, nChannels[1], nChannels[2], block2, block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.normalize = normalize

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
    def set_rand_mask(self):
        for layer in self.block1.layer:
            if hasattr(layer, 'set_rand_mask11'):
                layer.set_rand_mask11()
        for layer in self.block2.layer:
            if hasattr(layer, 'set_rand_mask11'):
                layer.set_rand_mask11()

class WideResNet21(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, normalize=None, is_n_repeat = True):
        super(WideResNet21, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        if  is_n_repeat == True:
            block2 = RandomBasicBlock20
        elif  is_n_repeat == False:
            block2 = BasicBlock2
        # 1st conv before any network block
        self.conv1 = Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block2, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock3(n, nChannels[1], nChannels[2], block2, block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.normalize = normalize

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
    def set_rand_mask(self):
        for layer in self.block1.layer:
            if hasattr(layer, 'set_rand_mask11'):
                layer.set_rand_mask11()
        for layer in self.block2.layer:
            if hasattr(layer, 'set_rand_mask11'):
                layer.set_rand_mask11() 

class WideResNet3(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, normalize=None , is_n_repeat = False):
        super(WideResNet3, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        if  is_n_repeat == True:
            block2 = RandomBasicBlock20
        elif  is_n_repeat == False:
            block2 = RandomBasicBlock35
        # 1st conv before any network block
        self.conv1 = Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock4(n, nChannels[0], nChannels[1], block2, block, 1, dropRate)
        # self.block1 = NetworkBlock4(n, nChannels[0], nChannels[1], block2, block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.normalize = normalize

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
    def set_rand_mask(self):
        for layer in self.block1.layer:
            if hasattr(layer, 'set_rand_mask11'):
                layer.set_rand_mask11()
        for layer in self.block2.layer:
            if hasattr(layer, 'set_rand_mask11'):
                layer.set_rand_mask11()

class WideResNet4(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, normalize=None , is_n_repeat = False):
        super(WideResNet4, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        if  is_n_repeat == True:
            block2 = RandomBasicBlock20
        elif  is_n_repeat == False:
            block2 = RandomBasicBlock35
        # 1st conv before any network block
        self.conv1 = Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock4(n, nChannels[0], nChannels[1], block2, block, 1, dropRate)
        # self.block1 = NetworkBlock4(n, nChannels[0], nChannels[1], block2, block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.normalize = normalize

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
    def set_rand_mask(self):
        for layer in self.block1.layer:
            if hasattr(layer, 'set_rand_mask11'):
                layer.set_rand_mask11()
        for layer in self.block2.layer:
            if hasattr(layer, 'set_rand_mask11'):
                layer.set_rand_mask11()
# 3 point
class WideResNet5(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, normalize=None , is_n_repeat = False):
        super(WideResNet5, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # non repeated
        if  is_n_repeat == True:
            block2 = RandomBasicBlock23
        # non related
        elif  is_n_repeat == False:
            block2 = RandomBasicBlock353
        # 1st conv before any network block
        self.conv1 = Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock4(n, nChannels[0], nChannels[1], block2, block, 1, dropRate)
        # self.block1 = NetworkBlock4(n, nChannels[0], nChannels[1], block2, block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.normalize = normalize

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
    def set_rand_mask(self):
        for layer in self.block1.layer:
            if hasattr(layer, 'set_rand_mask11'):
                layer.set_rand_mask11()
        for layer in self.block2.layer:
            if hasattr(layer, 'set_rand_mask11'):
                layer.set_rand_mask11()
                
# 2 point
class WideResNet6(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, normalize=None , is_n_repeat = False):
        super(WideResNet6, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # non repeated
        if  is_n_repeat == True:
            block2 = RandomBasicBlock22
        # non related
        elif  is_n_repeat == False:
            block2 = RandomBasicBlock2222
        # 1st conv before any network block
        self.conv1 = Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock4(n, nChannels[0], nChannels[1], block2, block, 1, dropRate)
        # self.block1 = NetworkBlock4(n, nChannels[0], nChannels[1], block2, block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.normalize = normalize

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
    def set_rand_mask(self):
        for layer in self.block1.layer:
            if hasattr(layer, 'set_rand_mask11'):
                layer.set_rand_mask11()
        for layer in self.block2.layer:
            if hasattr(layer, 'set_rand_mask11'):
                layer.set_rand_mask11()

def WideResNet34(num_classes=10, normalize=None, device = torch.device(0), is_n_repeat = True, pos = 0, eot = False, lb = 2048, reNum=5):

    return WideResNet21(num_classes=num_classes, normalize = normalize, is_n_repeat = is_n_repeat)

def WideResNet34_1layer(num_classes=10, normalize=None, device = torch.device(0), is_n_repeat = True, pos = 0, eot = False, lb = 2048, reNum=5):

    return WideResNet6(num_classes=num_classes, normalize = normalize, is_n_repeat = is_n_repeat)

