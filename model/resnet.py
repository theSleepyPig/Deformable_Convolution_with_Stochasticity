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

            q_lt = torch.cat(
                [torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                dim=-1).long()
            q_rb = torch.cat(
                [torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                dim=-1).long()
            q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
            q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

            # clip p
            p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)],
                          dim=-1)

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

class VarBasicBlock(nn.Module):
    '''BasicBlock.'''
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(VarBasicBlock, self).__init__()
        self.bn1 = BatchNorm2d(planes)
        self.conv1 = DeformConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                DeformConv2d(in_planes, self.expansion*planes,
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

class ResNet2(nn.Module):
    def __init__(self, block1, block2, num_blocks, num_classes=10, normalize=None):
        super(ResNet2, self).__init__()
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
    
def ResNet18(num_classes=10, normalize=None, device = torch.device(0), pos = 0, eot = False, lb = 2048):
    return ResNet( BasicBlock, [2,2,2,2], normalize = normalize, num_classes=num_classes)
    # return ResNet(BasicBlock, [2,2,2,2], normalize = normalize, num_classes=num_classes)