import math
import os
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from functools import partial
from typing import List

from mmcv.runner import load_state_dict
from .base_backbone import BaseBackbone
from .layers import trunc_normal_, resize_pos_embed_v2

from .vit import Block
from .myutil import top_pool
from .la_max import LANet
from .vit_pooling import PoolingBlock


class PoolingViT(BaseBackbone):
    """ 
    
    """
    def __init__(self, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer_eps=1e-5, freeze=False,
                 input_type='image',  pretrained=None, 
                 in_channels=[], patch_num=0,
                 attn_method='SUM_ABS_1',
                 cnn_pool_config=None,
                 vit_pool_configs=None,
                 multi_head_fusion=False,
                 sum_batch_mean=False,
                 **kwargs):
        super().__init__()
        if kwargs:
            print('Unused kwargs: ')
            print(kwargs)
        assert input_type  == 'feature', 'Only suit for hybrid model'
        self.sum_batch_mean = sum_batch_mean
        if sum_batch_mean:
            self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.multi_head_fusion = multi_head_fusion
        self.num_heads = num_heads
        if multi_head_fusion:
            assert vit_pool_configs is None, 'MultiHeadFusion only support original ViT Block, by now'

        self.input_type = input_type
        norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.projs = nn.ModuleList([nn.Conv2d(in_channels[i], embed_dim, 1,) for i in range(len(in_channels))])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, patch_num, embed_dim), requires_grad=True)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.attn_method = attn_method
        self.cnn_pool_config = cnn_pool_config
        if attn_method == 'LA':
            # self.attn_f = LANet(in_channels[-1], 16)
            self.attn_f = LANet(embed_dim, 16)
            
        elif attn_method == 'SUM':
            self.attn_f = lambda x: torch.sum(x, dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_1':
            self.attn_f = lambda x: torch.sum(torch.abs(x), dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_2':
            self.attn_f = lambda x: torch.sum(torch.pow(torch.abs(x), 2), dim=1).unsqueeze(1)
        elif attn_method == 'MAX':
            self.attn_f = lambda x: torch.max(x, dim=1)[0].unsqueeze(1)
        elif attn_method == 'MAX_ABS_1':
            self.attn_f = lambda x: torch.max(torch.abs(x), dim=1)[0].unsqueeze(1)
        elif attn_method == 'Random':
            self.attn_f = lambda x: x[:, torch.randint(high=x.shape[1], size=(1,))[0], ...].unsqueeze(1)
        else:
            raise ValueError("Unknown attn_method")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if vit_pool_configs is None:

            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, head_fusion=multi_head_fusion,
                    )
                for i in range(depth)])
        else:
            vit_keep_rates = vit_pool_configs['keep_rates']
            self.blocks = nn.ModuleList([
                PoolingBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                    pool_config=dict(keep_rate=vit_keep_rates[i], **vit_pool_configs),
                    )
            for i in range(depth)])
        # print('viisthere?')
        # print(self.blocks)

        self.norm = norm_layer(embed_dim)

        self.s2_pooling = nn.MaxPool2d(kernel_size=2)

        if pretrained:
            self.init_weights(pretrained, patch_num)
        else:
            trunc_normal_(self.patch_pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
        if freeze:
            self.apply(self._freeze_weights)

        # print('viisthere?')
        # print(self.blocks)


    def init_weights(self, pretrained, patch_num=0):
        # logger = get_root_logger()
        # logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        print(f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pos_embed = state_dict['pos_embed']     # [1, 197, 768] for base
        patch_pos_embed = pos_embed[:, 1:, :]

        if patch_num != pos_embed.shape[1] - 1:
            pass
            # logger.warning(f'interpolate pos_embed from {patch_pos_embed.shape[1]} to {patch_num}')
            print(f'interpolate pos_embed from {patch_pos_embed.shape[1]} to {patch_num}')
            pos_embed_new = resize_pos_embed_v2(patch_pos_embed, patch_num, 0)
        else:   # remove cls_token
            print('does not need to resize!')
            pos_embed_new = patch_pos_embed
        del state_dict['pos_embed']
        state_dict['patch_pos_embed'] = pos_embed_new
        state_dict['cls_pos_embed'] = pos_embed[:, 0, :].unsqueeze(1)

        if self.multi_head_fusion:
            # convert blocks.0.attn.qkv.weight to blocks.0.attn.qkv.0.weight
            num_groups = self.blocks[0].attn.group_number
            d = self.embed_dim // num_groups
            print('d', d)
            for k in list(state_dict.keys()):
                if k.startswith('blocks.'):
                    keys = k.split('.')
                    if  not (keys[2] == 'attn' and keys[3] == 'qkv'):
                        continue
                    for i in range(num_groups):
                        new_key = f'blocks.{keys[1]}.attn.qkv.{i}.weight'
                        new_value = state_dict[k][i*3*d:(i+1)*3*d, i*d: i*d+d]
                        state_dict[new_key] = new_value

                    del state_dict[k]

        for k in ('patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias'):
            del state_dict[k]
        load_state_dict(self, state_dict, strict=False)
        # self.load_state_dict(state_dict,strict=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _freeze_weights(self, m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
        for param in m.parameters():
            param.requires_grad = False

    def forward_features(self, x):
        assert len(x) == 1, '目前只支持1个 stage'
        assert isinstance(x, list) or isinstance(x, tuple)
        if len(x) == 2: # S2, S3
            x[0] = self.s2_pooling(x[0])
        elif len(x) == 3:
            x[0] = nn.MaxPool2d(kernel_size=4)(x[0])
            x[1] = self.s2_pooling(x[1])
        if os.getenv('DEBUG_MODE') == '1':
            print(x[0].shape)

        x = [self.projs[i](x[i]) for i in range(len(x))]
        # x = x[0]
        B, C, H, W = x[-1].shape
        attn_map = self.attn_f(x[-1]) # [B, 1, H, W]
        if self.attn_method == 'LA':
            x[-1] = x[-1] * attn_map    #  to have gradient
        x = [i.flatten(2).transpose(2, 1) for i in x]
        # x = self.projs[0](x).flatten(2).transpose(2, 1)
        # disable the first row and columns
        # attn_map[:, :, 0, :] = 0.
        # attn_map[:, :, :, 0] = 0.
        attn_weight = attn_map.flatten(2).transpose(2, 1)

        # attn_weight = torch.rand(attn_weight.shape, device=attn_weight.device)
        
        x = torch.stack(x).sum(dim=0)   # S1 + S2 + S3
        x = x + self.patch_pos_embed
        
        B, N, C = x.shape
        
        if self.cnn_pool_config is not None:
            keep_indexes = top_pool(attn_weight, dim=C, **self.cnn_pool_config)
            if keep_indexes is not None:
                x = x.gather(dim=1, index=keep_indexes)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        cls_tokens = cls_tokens + self.cls_pos_embed

        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)    # (B, N, dim)
        if os.environ.get('DEBUG_MODE', '0') == '1':
            print('output', x.shape)
        x = x[:, 0]
        if self.sum_batch_mean:
            x = x + x.mean(dim=0) * self.alpha
        loss = dict()
        return x, loss, attn_map

    def forward(self, x, **kwargs):
        # print('viisthere?')
        # print(self.blocks)
        x, loss, attn_map = self.forward_features(x)
        return dict(x=x, loss=dict(VitDiv_loss=loss), attn_map=attn_map)