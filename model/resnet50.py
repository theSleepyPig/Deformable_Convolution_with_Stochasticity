import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
#from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from model.layer import Conv2d
import os

from typing import Optional, List, Tuple, Union
from torch. nn. modules. utils import _single, _pair, _triple, _reverse_repeat_tuple
import torch.nn.functional as F

__all__ = ['ResNet', 'ResNet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1
    ) -> None:
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1
    ) -> None:
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck2(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        mask=None, device=None
    ) -> None:
        super(Bottleneck2, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = MaskedConv2d(width, width, kernel_size=5, stride=stride, padding=2, bias=False, mask=mask, device=device)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        self.print_flag = True
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def set_rand_mask1(self):
        mask = np.zeros((5, 5))
        
        top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]  

        selected_top = np.random.choice(len(top), 4, replace=False)

        for pos in selected_top:
            mask[top[pos]] = 1

        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (self.conv1.weight.shape[0], 1, 1, 1))
        self.conv2.mask = torch.tensor(mask, dtype=torch.float32).to(self.conv1.weight.device)
        
        if self.print_flag:
            print(self.conv2.mask)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印
            

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

        # weight = weight.to(input.device)
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
    
    
class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 200,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        rp=False,
        rp_out_channel=0) -> None:
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.rp = rp

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        
        # add random parameters
        self.w_shape = (1, 3, 224, 224)
        if self.rp:
            self.rand_weight = torch.normal(0, 1, self.w_shape).cuda()
            Mu_ = torch.ones(self.w_shape).cuda()
            SD_ = torch.ones(self.w_shape).cuda()
            self.Mu_ = nn.Parameter(Mu_)
            self.SD_ = nn.Parameter(SD_)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)


    def random_rp_matrix(self):
        for name, param in self.named_parameters():
            if 'rp' in name and 'conv' not in name:
                kernel_size = param.data.size()[-1]
                param.data = torch.normal(mean=0.0, std=1/kernel_size, size=param.data.size()).to('cuda')

    def rp_forward(self, x, out, kernel):
        rp_out = kernel(x)
        if out is None:
            return rp_out
        else:
            out = torch.cat([out, rp_out], dim=1)
            return out

    def _forward_impl(self, x: Tensor) -> Tensor:
        if self.rp:
            self.rand_weight = self.rand_weight.to(self.SD_.device)
            x = x * (self.rand_weight * self.SD_ + self.Mu_)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    def set_rands(self):
        min_SD_ = float(self.SD_.min())
        if min_SD_ < 1:
            with torch.no_grad():
                self.SD_[self.SD_ < 1] += 1 - min_SD_
        self.rand_weight = torch.normal(0, 1, self.w_shape).cuda()


class ResNetmask(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 200,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        rp=False,
        rp_out_channel=0) -> None:
        super(ResNetmask, self).__init__()
        
        mask=None
        device= None

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.rp = rp
        block2 = Bottleneck2

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        
        # add random parameters
        self.w_shape = (1, 3, 224, 224)
        # if self.rp:
        #     self.rand_weight = torch.normal(0, 1, self.w_shape).cuda()
        #     Mu_ = torch.ones(self.w_shape).cuda()
        #     SD_ = torch.ones(self.w_shape).cuda()
        #     self.Mu_ = nn.Parameter(Mu_)
        #     self.SD_ = nn.Parameter(SD_)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer2(block2, 64, layers[0], mask= mask, device=device)
        self.layer2 = self._make_layer2(block2, 128, layers[1], stride=2, mask=mask, device=device,
                                       dilate=replace_stride_with_dilation[0])
        
        self.layer32 = self._make_layer2(block2, 256, 1, stride=2,
                                       dilate=replace_stride_with_dilation[1], mask=mask, device=device)
        
        self.layer3 = self._make_layer(block, 256, 5, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)


    def _make_layer2(self, block: Type[Union[BasicBlock, Bottleneck2]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, mask=None, device=None) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, mask=mask, device=device,))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, mask=mask, device=device,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def random_rp_matrix(self):
        for name, param in self.named_parameters():
            if 'rp' in name and 'conv' not in name:
                kernel_size = param.data.size()[-1]
                param.data = torch.normal(mean=0.0, std=1/kernel_size, size=param.data.size()).to('cuda')

    def rp_forward(self, x, out, kernel):
        rp_out = kernel(x)
        if out is None:
            return rp_out
        else:
            out = torch.cat([out, rp_out], dim=1)
            return out

    def _forward_impl(self, x: Tensor) -> Tensor:
        if self.rp:
            self.rand_weight = self.rand_weight.to(self.SD_.device)
            x = x * (self.rand_weight * self.SD_ + self.Mu_)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer32(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    def set_rands(self):
        min_SD_ = float(self.SD_.min())
        if min_SD_ < 1:
            with torch.no_grad():
                self.SD_[self.SD_ < 1] += 1 - min_SD_
        self.rand_weight = torch.normal(0, 1, self.w_shape).cuda()
        
    def set_rand_mask(self):
        for layer in self.layer1:
            layer.set_rand_mask1()
        for layer in self.layer2:
            layer.set_rand_mask1() 
        for layer in self.layer32:
            layer.set_rand_mask1()

def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNetmask(block, layers, **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        model_path = os.path.join('pretrained', 'resnet50_coco_pretrained.pth')
        state_dict = torch.load(model_path)
        """print("-------------------Pretrained Model--------------------\n", len(state_dict.keys()))
        for i in model.state_dict().keys():
            if not (i in state_dict.keys()):
                print(i)"""
        model.load_state_dict(state_dict, strict=False)

    return model


def ResNet50(pretrained=False, rp=False, rp_out_channel=0):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress=False, rp=rp, rp_out_channel=rp_out_channel)

def ResNet50_1(pretrained=False, rp=False, rp_out_channel=0):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress=False, rp=rp, rp_out_channel=rp_out_channel)