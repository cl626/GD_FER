# use for Max(multi LA) -> HybridEmbd

import random
from typing import List
import torch
from torch import nn, Tensor
from mmcv.runner import load_state_dict
# from mmcls.utils import get_root_logger

from .myutil import top_pool
from .layers import resize_pos_embed, trunc_normal_


class LANet(nn.Module):
    def __init__(self, channel_num, ratio=16):
        super().__init__()
        assert channel_num % ratio == 0, f"input_channel{channel_num} must be exact division by ratio{ratio}"
        self.channel_num = channel_num
        self.ratio = ratio
        self.relu = nn.ReLU(inplace=True)

        self.LA_conv1 = nn.Conv2d(channel_num, int(channel_num / ratio), kernel_size=1)
        self.bn1 = nn.BatchNorm2d(int(channel_num / ratio))
        self.LA_conv2 = nn.Conv2d(int(channel_num / ratio), 1, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        LA = self.LA_conv1(x)
        LA = self.bn1(LA)
        LA = self.relu(LA)
        LA = self.LA_conv2(LA)
        LA = self.bn2(LA)
        LA = self.sigmoid(LA)
        return LA
        # LA = LA.repeat(1, self.channel_num, 1, 1)
        # x = x*LA

        # return x
