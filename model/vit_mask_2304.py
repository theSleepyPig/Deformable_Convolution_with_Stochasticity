from torchvision.models.vision_transformer import VisionTransformer, ViT_B_16_Weights
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional, Callable, List
from functools import partial
import numpy as np
from model.randpos_multi_resnet import MaskedConv2d
from torchvision.ops.misc import ConvNormActivation
import math

from torchvision.ops.misc import ConvNormActivation, MLP
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from collections import namedtuple



class VisionTransformer_DCN(VisionTransformer):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 10,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        is_conv_stem_configs: Optional[List] = False,
        DCS_kernel_size: int = 7,
        **kwargs
    ):
        normalize = kwargs.pop('normalize', False)
        super().__init__(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout,
                         num_classes, representation_size, norm_layer, conv_stem_configs=None)
        


        # self.layerNum = layerNum
        # self.randType = randType
        # Change Conv 2 DCS
        if is_conv_stem_configs:
                # 定义配置的 namedtuple，增加了 padding
            ConvStemConfig = namedtuple('ConvStemConfig', ['out_channels', 'kernel_size', 'stride', 'padding'])
            # 创建配置列表
            conv_stem_configs = [
                ConvStemConfig(out_channels=hidden_dim, kernel_size=DCS_kernel_size, stride=4, padding=3),
                ConvStemConfig(out_channels=hidden_dim, kernel_size=DCS_kernel_size, stride=4, padding=3)
            ]
            
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    MaskedConv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                    )
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Sequential(
                MaskedConv2d(
                    in_channels=3, out_channels=hidden_dim, kernel_size=DCS_kernel_size, padding = 3, stride=4
                ),
                MaskedConv2d(
                    in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=DCS_kernel_size, padding = 3, stride=4
                )
            )

        if isinstance(self.conv_proj[0], MaskedConv2d):
            # Init the patchify stem
            fan_in = self.conv_proj[0].in_channels * self.conv_proj[0].kernel_size[0] * self.conv_proj[0].kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj[0].weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj[0].bias is not None:
                nn.init.zeros_(self.conv_proj[0].bias)

        if isinstance(self.conv_proj[1], MaskedConv2d):
            # Init the patchify stem
            fan_in = self.conv_proj[1].in_channels * self.conv_proj[1].kernel_size[0] * self.conv_proj[1].kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj[1].weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj[1].bias is not None:
                nn.init.zeros_(self.conv_proj[1].bias)
                
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        self.print_flag = True
        
        self.last_selected1 = None  # 用来存储上次选择的点
        self.last_selected2 = None
        
        self.groups = [
            [4, 16, 18, 20, 30, 32, 34, 46],  # blue
            [10, 12, 22, 24, 26, 28, 38, 40],   # red
            [17, 19, 31, 33],             # grey
            [9, 11, 13, 23, 25, 27, 37, 39, 41]   # green...
        ]

        self.mask_size = (7, 7)  # Mask size is 5x5
        print("230623062306")
        
        self.print_counter = 0
        self.max_prints = 2

    def set_rand_mask(self):
        """
        Randomly select one point from each group to form a mask.
        """
        # Initialize a 7*7 zero mask
        mask = np.zeros(self.mask_size)

        # For each group, randomly select one point
        for group in self.groups:
            selected_position = np.random.choice(group) - 1  # Convert to 0-based index
            row, col = divmod(selected_position, 7)
            mask[row, col] = 1.0

        # Convert mask to torch tensor and expand it to match the convolutional weights
        selected_mask = torch.tensor(mask, dtype=torch.float32)
        expanded_mask = selected_mask.repeat(self.conv_proj[0].weight.shape[0], 1, 1, 1)  # expand mask to match the convolutional weights
        self.conv_proj[0].mask = expanded_mask
        
        #TODO: 4 experiments: full random(no strategy)-4, full random(no strategy)-1, group(2304/2592)-4, group(2304/2592)-1
    # 第二层
        # Initialize a zero mask
        mask = np.zeros(self.mask_size)

        # For each group, randomly select one point
        for group in self.groups:
            selected_position = np.random.choice(group) - 1  # Convert to 0-based index
            row, col = divmod(selected_position, 7)
            mask[row, col] = 1.0

        # Convert mask to torch tensor and expand it to match the convolutional weights
        selected_mask = torch.tensor(mask, dtype=torch.float32)
        expanded_mask = selected_mask.repeat(self.conv_proj[1].weight.shape[0], 1, 1, 1)  # expand mask to match the convolutional weights
        self.conv_proj[1].mask = expanded_mask
            
        if self.print_counter < self.max_prints:
            print(self.conv_proj[0].mask) 
            print(self.conv_proj[1].mask)
            self.print_counter += 1
            
            




class MaskedConv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size = 3,
        stride = 1,
        padding = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            groups = groups,
            norm_layer = norm_layer,
            activation_layer = activation_layer,
            dilation = dilation,
            inplace = inplace,
            bias = bias,
            conv_layer = MaskedConv2d,
        )


def _vision_transformer2(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> VisionTransformer_DCN:
    if weights is not None:
        # _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    model = VisionTransformer_DCN(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        # original_weights = weights.get_state_dict(progress=progress)
        # print("Original keys and their sizes:")
        # for key, value in original_weights.items():
        #     print(f"{key}: {value.size()}")
            # 获取预训练权重
        pretrained_state_dict = weights.get_state_dict(progress=True)
        updated_pretrained_dict = remap_state_dict(pretrained_state_dict)
        
    # 调用自定义加载函数
        load_custom_state_dict(model, updated_pretrained_dict)
        # model.load_state_dict(weights.get_state_dict(progress=progress))
        
        
        # partial_weights = {k: v for k, v in weights.get_state_dict(progress=progress).items() if 'classifier' not in k}
        # load_partial_state_dict(model, partial_weights)

    return model


_COMMON_META: Dict[str, Any] = {
    "categories": _IMAGENET_CATEGORIES,
}

_COMMON_SWAG_META = {
    **_COMMON_META,
    "recipe": "https://github.com/facebookresearch/SWAG",
    "license": "https://github.com/facebookresearch/SWAG/blob/main/LICENSE",
}

@register_model()
@handle_legacy_interface(weights=("pretrained", ViT_B_16_Weights.IMAGENET1K_V1))
def vit_b_16_mask(*, weights: Optional[ViT_B_16_Weights] = ViT_B_16_Weights.IMAGENET1K_V1, progress: bool = True, **kwargs: Any) -> VisionTransformer_DCN:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_16_Weights
        :members:
    """
    weights = ViT_B_16_Weights.verify(weights)
    print(1111)
    print(weights)

    return _vision_transformer2(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )


def load_partial_state_dict(model, state_dict):
    new_state_dict = model.state_dict()

    
    excluded_layers = [
        "conv_proj.0.weight", "conv_proj.0.bias",
        "conv_proj.1.weight", "conv_proj.1.bias",
        "conv_proj.weight", "conv_proj.bias"  
    ]

    for name, param in state_dict.items():
        if name in excluded_layers:
            print(f"Skipping layer: {name} due to exclusion.")
            continue
        
        if name in new_state_dict and new_state_dict[name].size() == param.size():
            new_state_dict[name].copy_(param)
        else:
            print(f"Skipping layer: {name} due to size mismatch.")

    model.load_state_dict(new_state_dict, strict=False)


def load_custom_state_dict(model, state_dict):
    model_state = model.state_dict()
    
    # 定义要排除的层
    excluded_layers = [
        "conv_proj.0.weight", "conv_proj.0.bias",  
        "conv_proj.1.weight", "conv_proj.1.bias",   
        "conv_proj.weight", "conv_proj.bias"        
    ]

    # 保证只有尺寸和名称都匹配的层才被加载
    for name, param in state_dict.items():
        if name in excluded_layers:
            print(f"Skipping layer: {name} due to exclusion.")
            continue
        if name in model_state and model_state[name].size() == param.size():
            model_state[name].copy_(param)
        else:
            if name not in model_state:
                print(f"Skipping layer: {name} because it is not found in the model.")
            elif model_state[name].size() != param.size():
                print(f"Skipping layer: {name} due to size mismatch.")

    model.load_state_dict(model_state, strict=False)
    
def remap_state_dict(pretrained_dict):
    """
    Remap keys in pretrained_dict to match the model's layer names.
    """
    key_mapping = {
        # Mapping for all 12 layers (encoder_layer_0 to encoder_layer_11)
    }
    
    for i in range(12):
        key_mapping.update({
            f'encoder.layers.encoder_layer_{i}.mlp.linear_1.weight': f'encoder.layers.encoder_layer_{i}.mlp.0.weight',
            f'encoder.layers.encoder_layer_{i}.mlp.linear_1.bias': f'encoder.layers.encoder_layer_{i}.mlp.0.bias',
            f'encoder.layers.encoder_layer_{i}.mlp.linear_2.weight': f'encoder.layers.encoder_layer_{i}.mlp.3.weight',
            f'encoder.layers.encoder_layer_{i}.mlp.linear_2.bias': f'encoder.layers.encoder_layer_{i}.mlp.3.bias',
        })
        
    new_state_dict = {}
    for key, value in pretrained_dict.items():
        new_key = key_mapping.get(key, key)  # Use the new key if it exists in the mapping, otherwise use the original key
        new_state_dict[new_key] = value
    return new_state_dict