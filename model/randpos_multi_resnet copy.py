import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import numpy as np

from typing import Optional, List, Tuple, Union
from torch. nn. modules. utils import _single, _pair, _triple, _reverse_repeat_tuple

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


# non repeated
class RandonBasicBlock2(nn.Module):
    '''BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mask=None, device=None):
        super(RandonBasicBlock2, self).__init__()
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
        
        self.last_selected = None  # 用来存储上次选择的点

    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out

    def set_rand_mask1(self):
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
            self.print_flag = False  # 设置标志为 False，以便以后不再打印
            
# non repeated 7*7
class RandonBasicBlock20(nn.Module):
    '''BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mask=None, device=None, reNum=5):
        super(RandonBasicBlock20, self).__init__()
        self.bn1 = BatchNorm2d(planes)
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=7, stride=stride, padding=3, bias=False, mask=mask, device=device)
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
        
        self.last_selected = None  # 用来存储上次选择的点

    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out

    def set_rand_mask1(self):
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

# non repeated, ablation, 指定mask测0,1,2,3,4个点重复的准确率
class RandonBasicBlock21(nn.Module):
    '''BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mask=None, device=None, reNum=5):
        super(RandonBasicBlock21, self).__init__()
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
            
        # self.print_flag = True
        self.print_counter = 0
        self.max_prints = 2
        
        self.current_mask_idx = 0  # 用来跟踪当前使用的mask
        repeatedNum = reNum
        # 预定义的mask列表
        if repeatedNum == 0:
        # 0个点重复
            print("0个点重复")
            self.predefined_masks = [
                np.array([[[0., 1., 0., 0., 0.],
                        [0., 1., 1., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0.]]]),
                np.array([[[0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0.],
                        [0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 1.]]])
            ]
        
        
        elif repeatedNum == 1:                
            # 1个点重复
            print("1个点重复")
            self.predefined_masks = [
                np.array([[[0., 1., 0., 0., 0.],
                           [0., 1., 1., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 0.]]]),
                np.array([[[0., 1., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0.],
                           [0., 0., 0., 1., 0.],
                           [0., 0., 0., 0., 1.]]])
            ]
        
        elif repeatedNum == 2: 
            # 2个点重复
            print("2个点重复")
            self.predefined_masks = [
                np.array([[[0., 1., 0., 0., 0.],
                           [0., 1., 1., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 0.]]]),
                np.array([[[0., 1., 0., 0., 0.],
                           [0., 0., 1., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 0.],
                           [0., 0., 0., 0., 1.]]])
            ]
        
        elif repeatedNum == 3: 
            # 3个点重复
            print("3个点重复")
            self.predefined_masks = [
                np.array([[[0., 1., 0., 0., 0.],
                           [0., 1., 1., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 0.]]]),
                np.array([[[0., 1., 0., 0., 0.],
                           [0., 1., 1., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 1.]]])
            ]
        
        elif repeatedNum == 4:
            # 4个点重复
            print("4个点重复")
            self.predefined_masks = [
                np.array([[[0., 1., 0., 0., 0.],
                           [0., 1., 1., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 0.]]]),
                np.array([[[0., 1., 0., 0., 0.],
                           [0., 1., 1., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 1., 0.]]])
            ]
        
        else:
            print("无效的 repeatedNum 值，必须在 0 到 4 之间。")
        


    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out

    def set_rand_mask1(self):
        mask = self.predefined_masks[self.current_mask_idx]

        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
        self.current_mask_idx = (self.current_mask_idx + 1) % len(self.predefined_masks)
        
        # if self.print_flag:
        #     print(self.conv1.mask)
        #     self.print_flag = False
        if self.print_counter < self.max_prints:
            print(self.conv1.mask)
            self.print_counter += 1

# 六选一
class RandonBasicBlock3(nn.Module):
    '''BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mask=None, device=None):
        super(RandonBasicBlock3, self).__init__()
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
            
        self.top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), 
                    (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), 
                    (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]
        
        self.masks = []
        self.generate_random_masks()

        # self.print_flag = True
        self.print_counter = 0
        self.max_prints = 1

    def generate_random_masks(self):
        all_selected_points = set()
        
        for _ in range(6):
            available_choices = [idx for idx in range(len(self.top)) if idx not in all_selected_points]
            # print(11)
            if len(available_choices) < 4:
                print("没有足够的点来生成新的 mask")
                break

            selected_top = np.random.choice(available_choices, 4, replace=False)
            self.masks.append(selected_top)
            all_selected_points.update(selected_top)

    def set_rand_mask1(self):
        mask = np.zeros((5, 5))

        selected_mask_idx = np.random.choice(len(self.masks))
        selected_top = self.masks[selected_mask_idx]

        for pos in selected_top:
            mask[self.top[pos]] = 1

        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)

        # if self.print_flag:
        #     print(self.conv1.mask)
        #     self.print_flag = False
        if self.print_counter < self.max_prints:
            print(self.conv1.mask)
            self.print_counter += 1
            

    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out
    
# 六选一 选定mask
class RandonBasicBlock32(nn.Module):
    '''BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mask=None, device=None):
        super(RandonBasicBlock32, self).__init__()
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
            
        # Predefined masks
        self.masks = [
            np.array([[[0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 1.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [1., 0., 1., 1., 0.]]]),
            np.array([[[0., 0., 0., 0., 1.],
                       [1., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 1.],
                       [1., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.]]]),
            np.array([[[0., 0., 0., 1., 0.],
                       [0., 0., 0., 1., 0.],
                       [1., 1., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.]]]),
            np.array([[[0., 0., 1., 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 1., 1., 0.],
                       [0., 0., 0., 0., 0.]]]),
            np.array([[[0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 1., 0.],
                       [0., 1., 0., 0., 1.],
                       [0., 1., 0., 0., 0.]]]),
            np.array([[[1., 1., 0., 0., 0.],
                       [0., 1., 0., 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.]]])          
        ]

        # Convert masks to PyTorch tensors
        self.masks = [torch.tensor(mask, dtype=torch.float32) for mask in self.masks]
        # if device:
        #     self.masks = [mask.to(device) for mask in self.masks]
        self.print_counter = 0
        self.max_prints = 6


    def set_rand_mask1(self):
        
        mask_idx = np.random.randint(len(self.masks))
        selected_mask = self.masks[mask_idx]
        expanded_mask = selected_mask.repeat(self.conv1.weight.shape[0], 1, 1, 1)  # expand mask to match the convolutional weights
        self.conv1.mask = expanded_mask
        
        if self.print_counter < self.max_prints:
            print(self.conv1.mask)
            self.print_counter += 1
    

    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out
    
#不相干 四组每个选一个
class RandonBasicBlock33(nn.Module):
    '''BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mask=None, device=None):
        super(RandonBasicBlock33, self).__init__()
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
            
        # Grouping the points according to your request
        self.groups = [
            [1, 3, 5, 11, 13, 15, 21, 23, 25],  # Group 1
            [2, 4, 12, 14, 22, 24],             # Group 2
            [6, 8, 10, 16, 18, 20],             # Group 3
            [7, 9, 17, 19]                      # Group 4
        ]

        self.mask_size = (5, 5)  # Mask size is 5x5
        
        self.print_counter = 0
        self.max_prints = 6

    def set_rand_mask1(self):
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

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out
    
#不相干 7*7 四组每个选一个 2304
class RandonBasicBlock34(nn.Module):
    '''BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mask=None, device=None):
        super(RandonBasicBlock34, self).__init__()
        self.bn1 = BatchNorm2d(planes)
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=7, stride=stride, padding=3, bias=False, mask=mask, device=device)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes,
                       kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion*planes)
            )
            
        # 2304
        self.groups = [
            [4, 16, 18, 20, 30, 32, 34, 46],  # blue
            [10, 12, 22, 24, 26, 28, 38, 40],   # red
            [17, 19, 31, 33],             # grey
            [9, 11, 13, 23, 25, 27, 37, 39, 41]   # green...
        ]
        # 2592
        # self.groups = [
        #     [16, 18, 20, 30, 32, 34],  # blue
        #     [10, 12, 24, 26, 38, 40],   # red
        #     [15, 17, 19, 21, 29, 31, 33, 35],             # grey
        #     [9, 11, 13, 23, 25, 27, 37, 39, 41]   # green...
        # ]
        self.mask_size = (7, 7)  # Mask size is 7*7
        
        self.print_counter = 0
        self.max_prints = 6

    def set_rand_mask1(self):
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
    

    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out

#不相干 7*7 四组每个选一个 2592
class RandonBasicBlock35(nn.Module):
    '''BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mask=None, device=None):
        super(RandonBasicBlock35, self).__init__()
        self.bn1 = BatchNorm2d(planes)
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=7, stride=stride, padding=3, bias=False, mask=mask, device=device)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes,
                       kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion*planes)
            )
            
        # # 2304
        # self.groups = [
        #     [4, 16, 18, 20, 30, 32, 34, 46],  # blue
        #     [10, 12, 22, 24, 26, 28, 38, 40],   # red
        #     [17, 19, 31, 33],             # grey
        #     [9, 11, 13, 23, 25, 27, 37, 39, 41]   # green...
        # ]
        
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

    def set_rand_mask1(self):
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
    

    def forward(self, x):

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out

class DeformConv2d(nn.Module):
    def __init__(self,
                inc,
                outc,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=None,
                modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable
            Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc,  # 该卷积用于最终的卷积
                            outc,
                            kernel_size=kernel_size,
                            stride=kernel_size,
                            bias=bias)

        self.p_conv = nn.Conv2d(inc,  # 该卷积用于从input中学习offset
                                2 * kernel_size * kernel_size,
                                kernel_size=3,
                                padding=1,
                                stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation  # 该部分是DeformableConv V2版本的，可以暂时不看
        if modulation:
            self.m_conv = nn.Conv2d(inc,
                                    kernel_size * kernel_size,
                                    kernel_size=3,
                                    padding=1,
                                    stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # print("111")
        # print("111")
        offset = self.p_conv(x)  # 此处得到offset
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):  # 求
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class Rand_Weight_DConv(DeformConv2d):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False, lb = 2048):
        super().__init__(inc, outc, kernel_size, padding, stride, bias, modulation)

        print("called")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w_shape = self.p_conv.weight.data.shape

        self.rand_weight = torch.normal(0, 1, self.p_conv.weight.data.shape).to(self.device)
        self.lb = lb
        Mu_ = torch.ones_like(self.p_conv.weight.data).to(self.device)
        SD_ = torch.ones_like(self.p_conv.weight.data).to(self.device)
        self.Mu_ = nn.Parameter(Mu_)
        self.SD_ = nn.Parameter(SD_)

    def forward(self, x):
        # print("forward called")

        self.set_rands()
        offset = self.p_conv(x)  # 此处得到offset
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                        dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                        dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)


        #out = super(Rand_Weight_DConv, self).forward(x)
        return out

    def set_rands(self):
        # print("set_rands called")
        min_SD = float(self.SD_.min())
        mean_lb = (self.lb / (self.w_shape[1] * self.w_shape[2] * self.w_shape[3]))
        # print(mean_lb)

        if min_SD < mean_lb:
            with torch.no_grad():
                self.SD_[self.SD_ < mean_lb] += mean_lb - min_SD
        #if not self.eot:
        self.rand_weight = torch.normal(0, 1, self.w_shape).to(self.device)

        self.p_conv.weight.data = self.rand_weight * self.SD_ + self.Mu_
        if self.modulation:
            self.p_conv.weight.data = self.rand_weight * self.SD_ + self.Mu_


class Rand_Pos_DConv(DeformConv2d):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False, lb = 2048):
        super().__init__(inc, outc, kernel_size, padding, stride, bias, modulation)
        #TODO: rewrite deformable convolution with random position, and enumerate the convolutions

    def set_rands(self):
        #TODO: Add random choice abt positions
        pass


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
    

class VarBasicBlock(nn.Module):
    '''BasicBlock.'''
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(VarBasicBlock, self).__init__()
        self.bn1 = BatchNorm2d(planes)
        self.conv1 = Rand_Weight_DConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Rand_Weight_DConv(in_planes, self.expansion*planes,
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


class VarRandonBasicBlock(nn.Module):
    '''BasicBlock.'''
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(VarBasicBlock, self).__init__()
        self.bn1 = BatchNorm2d(planes)
        self.conv1 = Rand_Pos_DConv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Rand_Weight_DConv(in_planes, self.expansion*planes,
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
    def __init__(self, block, num_blocks, num_classes=10, normalize=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
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

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        for i, op in enumerate(self.layer1):
            out = op(out)

        for i, op in enumerate(self.layer2):
            out = op(out)

        for i, op in enumerate(self.layer3):
            out = op(out)

        for i, op in enumerate(self.layer4):
            out = op(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #print("sum of sigma:", float(out.sum().cpu().data))
        out = self.linear(out)
        return out


class ResNet3(nn.Module):
    def __init__(self, block1, block2, num_blocks, num_classes=10, normalize=None):
        super(ResNet3, self).__init__()
        self.in_planes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.layer0 = self._make_layer(block1, 64, num_blocks[0], stride=1)
        self.layer1 = self._make_layer(block2, 64, num_blocks[1], stride=1)
        self.layer2 = self._make_layer(block2, 128, num_blocks[2], stride=2)
        self.layer3 = self._make_layer(block2, 256, num_blocks[3], stride=2)
        self.layer4 = self._make_layer(block2, 512, num_blocks[4], stride=2)
        self.linear = nn.Linear(512 * block2.expansion, num_classes)

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

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        #out1 = out.clone()
        #out2 = out.clone()

        for i, op in enumerate(self.layer0):
            out = op(out)

        for i, op in enumerate(self.layer1):
            out = op(out)

        for i, op in enumerate(self.layer2):
            out = op(out)

        for i, op in enumerate(self.layer3):
            out = op(out)

        for i, op in enumerate(self.layer4):
            out = op(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #print("sum of sigma:", float(out.sum().cpu().data))
        out = self.linear(out)
        return out


class ResNetmask(nn.Module):

    def __init__(self, block, num_blocks, mask, device, num_classes=10, normalize=None):
        super(ResNetmask, self).__init__()
        self.in_planes = 64
        self.conv1 = MaskedConv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False, mask=mask, device=device)
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

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        for i, op in enumerate(self.layer1):
            out = op(out)

        for i, op in enumerate(self.layer2):
            out = op(out)

        for i, op in enumerate(self.layer3):
            out = op(out)

        for i, op in enumerate(self.layer4):
            out = op(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #print("sum of sigma:", float(out.sum().cpu().data))
        out = self.linear(out)
        return out
    
    #随机选择
    def set_rand_mask1(self):
        mask = np.zeros((5, 5))
        mask[2, 2] = 1

        positions = [(i, j) for i in range(5) for j in range(5) if (i, j) != (2, 2)]
        selected_positions = np.random.choice(len(positions), 1, replace=False)

        for pos in selected_positions:
            mask[positions[pos]] = 1

        # print(mask)
        mask = np.tile(mask, (64, 3, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)

    # 随机移动除了中心之外的n个点，分四个块
    def set_rand_mask2(self):
        mask = np.zeros((5, 5))
        mask[2, 2] = 1  # 中间一个固定为1

        left_top = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        right_top = [(0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2)]
        left_bottom = [(3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2)]
        right_bottom = [(3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]

        selected_left_top = np.random.choice(len(left_top), 2, replace=False)
        selected_right_top = np.random.choice(len(right_top), 2, replace=False)
        selected_left_bottom = np.random.choice(len(left_bottom), 2, replace=False)
        selected_right_bottom = np.random.choice(len(right_bottom), 2, replace=False)
        # selected_left_top = np.random.choice(len(left_top), 1, replace=False)
        # selected_right_top = np.random.choice(len(right_top), 1, replace=False)
        # selected_left_bottom = np.random.choice(len(left_bottom), 1, replace=False)
        # selected_right_bottom = np.random.choice(len(right_bottom), 1, replace=False)

        for pos in selected_left_top:
            mask[left_top[pos]] = 1
        for pos in selected_right_top:
            mask[right_top[pos]] = 1
        for pos in selected_left_bottom:
            mask[left_bottom[pos]] = 1
        for pos in selected_right_bottom:
            mask[right_bottom[pos]] = 1

        mask = np.tile(mask, (64, 3, 1, 1))
        # print(mask)
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)

    # 随机移动四个点，中间十字位置五个点固定
    def set_rand_mask3(self):
        mask = np.zeros((5, 5))
        mask[2, 2] = 1
        mask[1, 2] = 1
        mask[2, 1] = 1
        mask[2, 3] = 1
        mask[3, 2] = 1

        left_top = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
        right_top = [(0, 3), (0, 4), (1, 3), (1, 4), (0, 2)]
        left_bottom = [(3, 0), (3, 1), (4, 0), (4, 1), (4, 2)]
        right_bottom = [(3, 3), (3, 4), (4, 3), (4, 4), (2, 4)]

        selected_left_top = np.random.choice(len(left_top), 1, replace=False)
        selected_right_top = np.random.choice(len(right_top), 1, replace=False)
        selected_left_bottom = np.random.choice(len(left_bottom), 1, replace=False)
        selected_right_bottom = np.random.choice(len(right_bottom), 1, replace=False)

        for pos in selected_left_top:
            mask[left_top[pos]] = 1
        for pos in selected_right_top:
            mask[right_top[pos]] = 1
        for pos in selected_left_bottom:
            mask[left_bottom[pos]] = 1
        for pos in selected_right_bottom:
            mask[right_bottom[pos]] = 1

        mask = np.tile(mask, (64, 3, 1, 1))
        # print(mask)
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)

    #两个区，一个区选一个随机点
    def set_rand_mask4(self):
        mask = np.zeros((5, 5))
        mask[2, 2] = 1  # 中间一个固定为1

        left = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2)]
        right = [(0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]

        selected_left = np.random.choice(len(left), 1, replace=False)
        selected_right = np.random.choice(len(right), 1, replace=False)

        for pos in selected_left:
            mask[left[pos]] = 1
        for pos in selected_right:
            mask[right[pos]] = 1
            
        # print(mask)
        mask = np.tile(mask, (64, 3, 1, 1))

        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
    #随机选择
    def set_rand_mask5(self):
        mask = np.zeros((5, 5))

        positions = [(i, j) for i in range(5) for j in range(5)]
        selected_positions = np.random.choice(len(positions), 5, replace=False)

        for pos in selected_positions:
            mask[positions[pos]] = 1

        # print(mask)
        mask = np.tile(mask, (64, 3, 1, 1))
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)

    # 随机选择五个点,五个区域每个区域一个点
    def set_rand_mask(self):
        mask = np.zeros((5, 5))

        left_top = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
        right_top = [(0, 3), (0, 4), (1, 3), (1, 4), (0, 2)]
        left_bottom = [(3, 0), (3, 1), (4, 0), (4, 1), (4, 2)]
        right_bottom = [(3, 3), (3, 4), (4, 3), (4, 4), (2, 4)]
        center = [(2, 2), (1, 2), (2, 1), (2, 3), (3, 2)]

        selected_left_top = np.random.choice(len(left_top), 1, replace=False)
        selected_right_top = np.random.choice(len(right_top), 1, replace=False)
        selected_left_bottom = np.random.choice(len(left_bottom), 1, replace=False)
        selected_right_bottom = np.random.choice(len(right_bottom), 1, replace=False)
        selected_center = np.random.choice(len(center), 1, replace=False)

        for pos in selected_left_top:
            mask[left_top[pos]] = 1
        for pos in selected_right_top:
            mask[right_top[pos]] = 1
        for pos in selected_left_bottom:
            mask[left_bottom[pos]] = 1
        for pos in selected_right_bottom:
            mask[right_bottom[pos]] = 1
        for pos in selected_center:
            mask[right_bottom[pos]] = 1        

        mask = np.tile(mask, (64, 3, 1, 1))
        # print(mask)
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
    def set_1_mask(self):
        mask = np.ones((5, 5))

        mask = np.tile(mask, (64, 3, 1, 1))
        # print(mask)
        self.conv1.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)


class ResNetpath(nn.Module):
    def __init__(self, block, num_blocks, device, num_classes=10, normalize=None):
        super(ResNetpath, self).__init__()
        self.in_planes = 64
        self.conv1_layers = nn.ModuleList([MaskedConv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False, device=device , mask=generate_mask_by_patch()) for _ in range(2)])
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

        self.conv1 = np.random.choice(self.conv1_layers)
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        for i, op in enumerate(self.layer1):
            out = op(out)

        for i, op in enumerate(self.layer2):
            out = op(out)

        for i, op in enumerate(self.layer3):
            out = op(out)

        for i, op in enumerate(self.layer4):
            out = op(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #print("sum of sigma:", float(out.sum().cpu().data))
        out = self.linear(out)
        return out


class ResNetPartmask(nn.Module):
    def __init__(self, block1, block2, num_blocks, mask, device, num_classes=10, normalize=None, reNum=0):
        super(ResNetPartmask, self).__init__()
        self.in_planes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(block1, 64, num_blocks[0], stride=1, mask=mask, device=device, reNum=reNum)
        self.layer2 = self._make_layer(block1, 128, num_blocks[1], stride=2, mask=mask, device=device, reNum=reNum)
        # self.layer3 = self._make_layer(block1, 256, num_blocks[2], stride=2, mask=mask, device=device)
        # self.layer4 = self._make_layer(block2, 512, num_blocks[3], stride=2, mask=mask, device=device)
        # self.layer2 = self._make_layer2(block2, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer2(block2, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer2(block2, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block2.expansion, num_classes)

        self.normalize = normalize
        
        self.corner_count = 0  # 初始化角计数器

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

    def _make_layer(self, block, planes, num_blocks, stride, mask, device, reNum):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, mask=mask, device=device, reNum=reNum))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)
    
    def _make_layer2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)
    
    def set_rand_mask(self):
        for layer in self.layer1:
            layer.set_rand_mask1()
        for layer in self.layer2:
            layer.set_rand_mask1()
        # for layer in self.layer3:
        #     layer.set_rand_mask1()
        # for layer in self.layer1:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        # for layer in self.layer2:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        
        # for layer in self.layer3:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        # for layer in self.layer4:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        for i, op in enumerate(self.layer1):
            out = op(out)

        for i, op in enumerate(self.layer2):
            out = op(out)

        for i, op in enumerate(self.layer3):
            out = op(out)

        for i, op in enumerate(self.layer4):
            out = op(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #print("sum of sigma:", float(out.sum().cpu().data))
        out = self.linear(out)
        return out


class ResNetPartmask3(nn.Module):
    def __init__(self, block1, block2, num_blocks, mask, device, num_classes=10, normalize=None):
        super(ResNetPartmask3, self).__init__()
        self.in_planes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(block1, 64, num_blocks[0], stride=1, mask=mask, device=device)
        self.layer2 = self._make_layer(block1, 128, num_blocks[1], stride=2, mask=mask, device=device)
        # self.layer3 = self._make_layer(block1, 256, num_blocks[2], stride=2, mask=mask, device=device)
        # self.layer4 = self._make_layer(block2, 512, num_blocks[3], stride=2, mask=mask, device=device)
        # self.layer2 = self._make_layer2(block2, 128, num_blocks[1], stride=2)
            
        self.layer22 = self._make_layer(block1, 256, 1, stride=2, mask=mask, device=device)
        self.layer3 = self._make_layer2(block2, 256, 1, stride=1)
        
        # self.layer3 = self._make_layer2(block2, 256, num_blocks[2], stride=2)
        
        self.layer4 = self._make_layer2(block2, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block2.expansion, num_classes)

        self.normalize = normalize
        
        self.corner_count = 0  # 初始化角计数器

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

    def _make_layer(self, block, planes, num_blocks, stride, mask, device):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, mask=mask, device=device))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)
    
    def _make_layer2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)
    
    def set_rand_mask(self):
        for layer in self.layer1:
            layer.set_rand_mask1()
        for layer in self.layer2:
            layer.set_rand_mask1()
        for layer in self.layer22:
            layer.set_rand_mask1()        
        # for layer in self.layer3:
        #     layer.set_rand_mask1()
        # for layer in self.layer1:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        # for layer in self.layer2:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        
        # for layer in self.layer3:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        # for layer in self.layer4:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        # print('Conv1 output size:', out.size())

        for i, op in enumerate(self.layer1):
            out = op(out)
            # print(f'Layer1[{i}] output size:', out.size())
            
        for i, op in enumerate(self.layer2):
            out = op(out)
            # print(f'Layer2[{i}] output size:', out.size())
                        
        for i, op in enumerate(self.layer22):
            out = op(out)
            # print(f'Layer22[{i}] output size:', out.size())
            
        for i, op in enumerate(self.layer3):
            out = op(out)
            # print(f'Layer3[{i}] output size:', out.size())

        for i, op in enumerate(self.layer4):
            out = op(out)
            # print(f'Layer4[{i}] output size:', out.size())

        out = F.avg_pool2d(out, 4)
        # print('AvgPool output size:', out.size())
        out = out.view(out.size(0), -1)
        #print("sum of sigma:", float(out.sum().cpu().data))
        out = self.linear(out)
        return out
    
    
class ResNetPartmask4(nn.Module):
    def __init__(self, block1, block2, num_blocks, mask, device, num_classes=10, normalize=None):
        super(ResNetPartmask4, self).__init__()
        self.in_planes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(block1, 64, 1, stride=1, mask=mask, device=device)
        self.layer11 = self._make_layer2(block2, 64, 1, stride=1)
        self.layer2 = self._make_layer(block1, 128, 1, stride=2, mask=mask, device=device)
        self.layer21 = self._make_layer2(block2, 128, 1, stride=1)
        self.layer3 = self._make_layer(block1, 256, 1, stride=2, mask=mask, device=device)
        self.layer31 = self._make_layer2(block2, 256, 1, stride=1)
        self.layer4 = self._make_layer(block1, 512, 1, stride=2, mask=mask, device=device)
        self.layer41 = self._make_layer2(block2, 512, 1, stride=1)
        # self.layer2 = self._make_layer2(block2, 128, 1, stride=1)
            
        # self.layer22 = self._make_layer(block1, 256, 1, stride=2, mask=mask, device=device)
        # self.layer3 = self._make_layer2(block2, 256, 1, stride=1)
        
        # self.layer3 = self._make_layer2(block2, 256, num_blocks[2], stride=2)
        
        # self.layer4 = self._make_layer2(block2, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block2.expansion, num_classes)

        self.normalize = normalize
        
        self.corner_count = 0  # 初始化角计数器

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

    def _make_layer(self, block, planes, num_blocks, stride, mask, device):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, mask=mask, device=device))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)
    
    def _make_layer2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)
    
    def set_rand_mask(self):
        for layer in self.layer1:
            layer.set_rand_mask1()
        for layer in self.layer2:
            layer.set_rand_mask1() 
        for layer in self.layer3:
            layer.set_rand_mask1()
        for layer in self.layer4:
            layer.set_rand_mask1() 
        # for layer in self.layer2:
        #     layer.set_rand_mask1()
        # for layer in self.layer22:
        #     layer.set_rand_mask1()        
        # for layer in self.layer3:
        #     layer.set_rand_mask1()
        # for layer in self.layer1:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        # for layer in self.layer2:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        
        # for layer in self.layer3:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        # for layer in self.layer4:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        # print('Conv1 output size:', out.size())

        for i, op in enumerate(self.layer1):
            out = op(out)
            # print(f'Layer1[{i}] output size:', out.size())       
        for i, op in enumerate(self.layer11):
            out = op(out)
            # print(f'Layer22[{i}] output size:', out.size())
            
        for i, op in enumerate(self.layer2):
            out = op(out)
            # print(f'Layer2[{i}] output size:', out.size())
        for i, op in enumerate(self.layer21):
            out = op(out)
            
        for i, op in enumerate(self.layer3):
            out = op(out)
            # print(f'Layer3[{i}] output size:', out.size())
        for i, op in enumerate(self.layer31):
            out = op(out)
            
        for i, op in enumerate(self.layer4):
            out = op(out)
            # print(f'Layer4[{i}] output size:', out.size())
        for i, op in enumerate(self.layer41):
            out = op(out)
            
        out = F.avg_pool2d(out, 4)
        # print('AvgPool output size:', out.size())
        out = out.view(out.size(0), -1)
        #print("sum of sigma:", float(out.sum().cpu().data))
        out = self.linear(out)
        return out
    
class ResNetPartmask5(nn.Module):
    def __init__(self, block1, block2, num_blocks, mask, device, num_classes=10, normalize=None):
        super(ResNetPartmask5, self).__init__()
        self.in_planes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(block1, 64, 1, stride=1, mask=mask, device=device)
        self.layer11 = self._make_layer2(block2, 64, 1, stride=1)
        # self.layer2 = self._make_layer(block1, 128, 1, stride=2, mask=mask, device=device)
        # self.layer21 = self._make_layer2(block2, 128, 1, stride=1)
        # self.layer3 = self._make_layer(block1, 256, 1, stride=2, mask=mask, device=device)
        # self.layer31 = self._make_layer2(block2, 256, 1, stride=1)
        # self.layer4 = self._make_layer(block1, 512, 1, stride=2, mask=mask, device=device)
        # self.layer41 = self._make_layer2(block2, 512, 1, stride=1)
        
        self.layer2 = self._make_layer2(block2, 128, num_blocks[1], stride=2)
            
        # self.layer22 = self._make_layer(block1, 256, 1, stride=2, mask=mask, device=device)
        # self.layer3 = self._make_layer2(block2, 256, 1, stride=1)
        
        self.layer3 = self._make_layer2(block2, 256, num_blocks[2], stride=2)
        
        self.layer4 = self._make_layer2(block2, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block2.expansion, num_classes)

        self.normalize = normalize
        
        self.corner_count = 0  # 初始化角计数器

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

    def _make_layer(self, block, planes, num_blocks, stride, mask, device):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, mask=mask, device=device))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)
    
    def _make_layer2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)
    
    def set_rand_mask(self):
        for layer in self.layer1:
            layer.set_rand_mask1()
        # for layer in self.layer2:
        #     layer.set_rand_mask1() 
        # for layer in self.layer3:
        #     layer.set_rand_mask1()
        # for layer in self.layer4:
            layer.set_rand_mask1() 
        # for layer in self.layer2:
        #     layer.set_rand_mask1()
        # for layer in self.layer22:
        #     layer.set_rand_mask1()        
        # for layer in self.layer3:
        #     layer.set_rand_mask1()
        # for layer in self.layer1:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        # for layer in self.layer2:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        
        # for layer in self.layer3:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        # for layer in self.layer4:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        # print('Conv1 output size:', out.size())

        for i, op in enumerate(self.layer1):
            out = op(out)
            # print(f'Layer1[{i}] output size:', out.size())       
        for i, op in enumerate(self.layer11):
            out = op(out)
            # print(f'Layer22[{i}] output size:', out.size())
            
        for i, op in enumerate(self.layer2):
            out = op(out)
            # print(f'Layer2[{i}] output size:', out.size())

            
        for i, op in enumerate(self.layer3):
            out = op(out)
            # print(f'Layer3[{i}] output size:', out.size())

            
        for i, op in enumerate(self.layer4):
            out = op(out)
            # print(f'Layer4[{i}] output size:', out.size())

            
        out = F.avg_pool2d(out, 4)
        # print('AvgPool output size:', out.size())
        out = out.view(out.size(0), -1)
        #print("sum of sigma:", float(out.sum().cpu().data))
        out = self.linear(out)
        return out
    
class ResNetPartmask5(nn.Module):
    def __init__(self, block1, block2, num_blocks, mask, device, num_classes=10, normalize=None):
        super(ResNetPartmask5, self).__init__()
        self.in_planes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(block1, 64, 1, stride=1, mask=mask, device=device)
        self.layer11 = self._make_layer2(block2, 64, 1, stride=1)
        # self.layer2 = self._make_layer(block1, 128, 1, stride=2, mask=mask, device=device)
        # self.layer21 = self._make_layer2(block2, 128, 1, stride=1)
        # self.layer3 = self._make_layer(block1, 256, 1, stride=2, mask=mask, device=device)
        # self.layer31 = self._make_layer2(block2, 256, 1, stride=1)
        # self.layer4 = self._make_layer(block1, 512, 1, stride=2, mask=mask, device=device)
        # self.layer41 = self._make_layer2(block2, 512, 1, stride=1)
        
        self.layer2 = self._make_layer2(block2, 128, num_blocks[1], stride=2)
            
        # self.layer22 = self._make_layer(block1, 256, 1, stride=2, mask=mask, device=device)
        # self.layer3 = self._make_layer2(block2, 256, 1, stride=1)
        
        self.layer3 = self._make_layer2(block2, 256, num_blocks[2], stride=2)
        
        self.layer4 = self._make_layer2(block2, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block2.expansion, num_classes)

        self.normalize = normalize
        
        self.corner_count = 0  # 初始化角计数器

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

    def _make_layer(self, block, planes, num_blocks, stride, mask, device):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, mask=mask, device=device))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)
    
    def _make_layer2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)
    
    def set_rand_mask(self):
        for layer in self.layer1:
            layer.set_rand_mask1()
        # for layer in self.layer2:
        #     layer.set_rand_mask1() 
        # for layer in self.layer3:
        #     layer.set_rand_mask1()
        # for layer in self.layer4:
            # layer.set_rand_mask1() 
        # for layer in self.layer2:
        #     layer.set_rand_mask1()
        # for layer in self.layer22:
        #     layer.set_rand_mask1()        
        # for layer in self.layer3:
        #     layer.set_rand_mask1()
        # for layer in self.layer1:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        # for layer in self.layer2:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        
        # for layer in self.layer3:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1
        # for layer in self.layer4:
        #     layer.set_rand_mask1(self.corner_count)
        #     self.corner_count += 1

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)

        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        # print('Conv1 output size:', out.size())

        for i, op in enumerate(self.layer1):
            out = op(out)
            # print(f'Layer1[{i}] output size:', out.size())       
        for i, op in enumerate(self.layer11):
            out = op(out)
            # print(f'Layer22[{i}] output size:', out.size())
            
        for i, op in enumerate(self.layer2):
            out = op(out)
            # print(f'Layer2[{i}] output size:', out.size())

            
        for i, op in enumerate(self.layer3):
            out = op(out)
            # print(f'Layer3[{i}] output size:', out.size())

            
        for i, op in enumerate(self.layer4):
            out = op(out)
            # print(f'Layer4[{i}] output size:', out.size())

            
        out = F.avg_pool2d(out, 4)
        # print('AvgPool output size:', out.size())
        out = out.view(out.size(0), -1)
        #print("sum of sigma:", float(out.sum().cpu().data))
        out = self.linear(out)
        return out
    
    
def generate_mask():
    mask = np.zeros((5, 5))
    mask[2, 2] = 1

    positions = [(i, j) for i in range(5) for j in range(5) if (i, j) != (2, 2)]
    selected_positions = np.random.choice(len(positions), 1, replace=False)

    for pos in selected_positions:
        mask[positions[pos]] = 1

    # print(mask)
    mask = np.tile(mask, (64, 3, 1, 1))

def generate_mask_by_patch():
    mask = np.zeros((5, 5))
    mask[2, 2] = 1

    left_top = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    right_top = [(0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2)]
    left_bottom = [(3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2)]
    right_bottom = [(3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]

    selected_left_top = np.random.choice(len(left_top), 1, replace=False)
    selected_right_top = np.random.choice(len(right_top), 1, replace=False)
    selected_left_bottom = np.random.choice(len(left_bottom), 1, replace=False)
    selected_right_bottom = np.random.choice(len(right_bottom), 1, replace=False)

    for pos in selected_left_top:
        mask[left_top[pos]] = 1
    for pos in selected_right_top:
        mask[right_top[pos]] = 1
    for pos in selected_left_bottom:
        mask[left_bottom[pos]] = 1
    for pos in selected_right_bottom:
        mask[right_bottom[pos]] = 1

    mask = np.tile(mask, (64, 3, 1, 1))
    # print(mask)

    return mask


def ResNet18(num_classes=10, normalize=None, device = torch.device(0), is_n_repeat = True, pos = 0, eot = False, lb = 2048, reNum=5):
    mask= generate_mask()
    if is_n_repeat:
        return ResNetPartmask(RandonBasicBlock35, BasicBlock, [2,2,2,2], mask=mask, normalize = normalize, num_classes=num_classes, device = device)
        # return ResNetPartmask(RandonBasicBlock21, BasicBlock, [2,2,2,2], mask=mask, normalize = normalize, num_classes=num_classes, device = device, reNum=reNum)
    return ResNetPartmask(RandonBasicBlock, BasicBlock, [2,2,2,2], mask=mask, normalize = normalize, num_classes=num_classes, device = device, reNum=reNum)

def ResNet18_1layer(num_classes=10, normalize=None, device = torch.device(0), is_n_repeat = True, pos = 0, eot = False, lb = 2048, reNum=5):
    mask= generate_mask()
    if is_n_repeat:
        # return ResNetPartmask5(RandonBasicBlock2, BasicBlock, [2,2,2,2], mask=mask, normalize = normalize, num_classes=num_classes, device = device)
        return ResNetPartmask5(RandonBasicBlock35, BasicBlock, [2,2,2,2], mask=mask, normalize = normalize, num_classes=num_classes, device = device)
    return ResNetPartmask5(RandonBasicBlock, BasicBlock, [2,2,2,2], mask=mask, normalize = normalize, num_classes=num_classes, device = device)