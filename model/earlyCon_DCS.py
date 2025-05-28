import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from model.randpos_multi_resnet import MaskedConv2d
import numpy as np

# helpers
def make_tuple(t):
    """
    return the input if it's already a tuple.
    return a tuple of the input if the input is not already a tuple.
    """
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=7, dim_head=64, dropout=0.):
        """
        reduced the default number of heads by 1 per https://arxiv.org/pdf/2106.14881v2.pdf
        """
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class EarlyConvViT(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        """
        3x3 conv, stride 1, 5 conv layers per https://arxiv.org/pdf/2106.14881v2.pdf
        """
        super().__init__()

        # n_filter_list = (channels, 48, 96, 192, 384)  # hardcoding for now because that's what the paper used
        # n_filter_list = (48, 96, 192, 384)  # hardcoding for now because that's what the paper used
        n_filter_list = (96, 192, 384)

        self.conv_layers = nn.Sequential(
                # nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1)),
                nn.Sequential(nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1)),
                # nn.Sequential(nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1)),
                nn.Sequential(MaskedConv2d(48, 96, kernel_size=5, stride=2, padding=2)),
            *[nn.Sequential(
                nn.Conv2d(in_channels=n_filter_list[i],
                          out_channels=n_filter_list[i + 1],
                          kernel_size=3,  # hardcoding for now because that's what the paper used
                          stride=2,  # hardcoding for now because that's what the paper used
                          padding=1),  # hardcoding for now because that's what the paper used
            )
                for i in range(len(n_filter_list)-1)
            ])

        self.conv_layers.add_module("conv_1x1", torch.nn.Conv2d(in_channels=n_filter_list[-1], 
                                    out_channels=dim, 
                                    stride=1,  # hardcoding for now because that's what the paper used 
                                    kernel_size=1,  # hardcoding for now because that's what the paper used 
                                    padding=0))  # hardcoding for now because that's what the paper used
        self.conv_layers.add_module("flatten image", 
                                    Rearrange('batch channels height width -> batch (height width) channels'))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_filter_list[-1] + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.print_flag = True  
        self.last_selected = None  # 用来存储上次选择的点
        
    

    def forward(self, img):
        x = self.conv_layers(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    def set_rand_mask(self):
        mask = np.zeros((5, 5))
        
        top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]

        # selected_top = np.random.choice(len(top), 2, replace=False)
        
        # for pos in selected_top:
        #     mask[top[pos]] = 1
        
        if self.last_selected is not None:
            available_choices = [idx for idx in range(len(top)) if idx not in self.last_selected]
        else:
            available_choices = list(range(len(top)))
            
        selected_top = np.random.choice(available_choices, 2, replace=False)
        self.last_selected = selected_top  
        
        for pos in selected_top:
            mask[top[pos]] = 1
            
        # masked_conv = self.conv_layers[0][0]  # 第一个 Sequential 中的第一层就是 MaskedConv2d
        masked_conv = self.conv_layers[1][0]
        out_channels = masked_conv.weight.shape[0]

        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (out_channels, 1, 1, 1))
        mask = torch.tensor(mask, dtype=torch.float32).to(masked_conv.weight.device)
        
        masked_conv.mask = mask
        
        if self.print_flag:
            print(masked_conv.mask)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印
            
    def set_rand_mask1(self):
        mask = np.zeros((5, 5))
        
        top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]

        selected_top = np.random.choice(len(top), 2, replace=False)
        
        for pos in selected_top:
            mask[top[pos]] = 1
        
        # if self.last_selected is not None:
        #     available_choices = [idx for idx in range(len(top)) if idx not in self.last_selected]
        # else:
        #     available_choices = list(range(len(top)))
            
        # selected_top = np.random.choice(available_choices, 2, replace=False)
        # self.last_selected = selected_top  
        
        # for pos in selected_top:
        #     mask[top[pos]] = 1
            
        # masked_conv = self.conv_layers[0][0]  # 第一个 Sequential 中的第一层就是 MaskedConv2d
        masked_conv = self.conv_layers[1][0]
        out_channels = masked_conv.weight.shape[0]

        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (out_channels, 1, 1, 1))
        mask = torch.tensor(mask, dtype=torch.float32).to(masked_conv.weight.device)
        
        masked_conv.mask = mask
        
        if self.print_flag:
            print(masked_conv.mask)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印


class CNNEmbedding_DCS(nn.Module):
    def __init__(self, hidden_dim=192, image_size=224, patch_size=16):
        super().__init__()
        dim = hidden_dim
        n_filter_list = (96, 192, 384)

        self.conv_layers = nn.Sequential(
                # nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1)),
                nn.Sequential(nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1)),
                # nn.Sequential(nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1)),
                nn.Sequential(MaskedConv2d(48, 96, kernel_size=5, stride=2, padding=2)),
            *[nn.Sequential(
                nn.Conv2d(in_channels=n_filter_list[i],
                          out_channels=n_filter_list[i + 1],
                          kernel_size=3,  # hardcoding for now because that's what the paper used
                          stride=2,  # hardcoding for now because that's what the paper used
                          padding=1),  # hardcoding for now because that's what the paper used
            )
                for i in range(len(n_filter_list)-1)
            ])

        self.conv_layers.add_module("conv_1x1", torch.nn.Conv2d(in_channels=n_filter_list[-1], 
                                    out_channels=dim, 
                                    stride=1,  # hardcoding for now because that's what the paper used 
                                    kernel_size=1,  # hardcoding for now because that's what the paper used 
                                    padding=0))  # hardcoding for now because that's what the paper used
        self.conv_layers.add_module("flatten image", 
                                    Rearrange('batch channels height width -> batch (height width) channels'))
        # self.pos_embedding = nn.Parameter(torch.randn(1, n_filter_list[-1] + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(0.)
        
        # ✅ 兼容 HuggingFace ViT：必须要有 projection
        self.projection = nn.Identity()
        self.projection.weight = nn.Parameter(torch.empty(1))  # dummy 用于 dtype 查询
        
        self.print_flag = True  
        self.last_selected = None  # 用来存储上次选择的点

    def forward(self, img, interpolate_pos_encoding=False):
        # B = x.shape[0]
        # x = self.conv_layers(x)  # shape: (B, hidden_dim, H', W')
        # x = x.flatten(2).transpose(1, 2)  # -> (B, num_patches, hidden_dim)

        # cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        # x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, hidden_dim)
        # x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.conv_layers(img)
        # b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b).to(x.device)
        
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)].to(x.device)
        
            # ✅ 正确处理 cls_token 和 pos_embedding 的设备一致性
        # cls_tokens = self.cls_token.expand(b, -1, -1).to(x.device)  # (B, 1, dim)
        # pos_embedding = self.pos_embedding[:, :n+1, :].to(x.device)  # (1, N+1, dim)

        # x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, dim)
        # x = x + pos_embedding
        # x = self.dropout(x)
        return x
    
    def set_rand_mask(self):
        mask = np.zeros((5, 5))
        
        top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]

        # selected_top = np.random.choice(len(top), 2, replace=False)
        
        # for pos in selected_top:
        #     mask[top[pos]] = 1
        
        if self.last_selected is not None:
            available_choices = [idx for idx in range(len(top)) if idx not in self.last_selected]
        else:
            available_choices = list(range(len(top)))
            
        selected_top = np.random.choice(available_choices, 2, replace=False)
        self.last_selected = selected_top  
        
        for pos in selected_top:
            mask[top[pos]] = 1
            
        # masked_conv = self.conv_layers[0][0]  # 第一个 Sequential 中的第一层就是 MaskedConv2d
        masked_conv = self.conv_layers[1][0]
        out_channels = masked_conv.weight.shape[0]

        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (out_channels, 1, 1, 1))
        mask = torch.tensor(mask, dtype=torch.float32).to(masked_conv.weight.device)
        
        masked_conv.mask = mask
        
        if self.print_flag:
            print(masked_conv.mask)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印
            
    def set_rand_mask1(self):
        mask = np.zeros((5, 5))
        
        top = [(2, 2), (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (3, 2), (4, 2), (0, 3), (0, 4), (1, 3), (1, 4), (0, 2), (1, 2), (3, 3), (3, 4), (4, 3), (4, 4), (2, 3), (2, 4)]

        selected_top = np.random.choice(len(top), 2, replace=False)
        
        for pos in selected_top:
            mask[top[pos]] = 1
        
        # if self.last_selected is not None:
        #     available_choices = [idx for idx in range(len(top)) if idx not in self.last_selected]
        # else:
        #     available_choices = list(range(len(top)))
            
        # selected_top = np.random.choice(available_choices, 2, replace=False)
        # self.last_selected = selected_top  
        
        # for pos in selected_top:
        #     mask[top[pos]] = 1
            
        # masked_conv = self.conv_layers[0][0]  # 第一个 Sequential 中的第一层就是 MaskedConv2d
        masked_conv = self.conv_layers[1][0]
        out_channels = masked_conv.weight.shape[0]

        # 根据 conv1 权重的形状调整 mask
        mask = np.tile(mask, (out_channels, 1, 1, 1))
        mask = torch.tensor(mask, dtype=torch.float32).to(masked_conv.weight.device)
        
        masked_conv.mask = mask
        
        if self.print_flag:
            print(masked_conv.mask)
            self.print_flag = False  # 设置标志为 False，以便以后不再打印