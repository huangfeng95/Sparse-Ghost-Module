#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
import torch
import torch.nn as nn
from .network_blocks import BaseConv, CSPLayer, DWConv


class C3MixGhost(CSPLayer):
    # C3 module with MixGhostBottleneck()
    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__(in_channels, out_channels, n, shortcut, expansion)
        c_ = int(out_channels * expansion)  # hidden channels
        self.m = nn.Sequential(*[MixGhostBottleneck(c_, c_) for _ in range(n)])


class MixGhostBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(MixGhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act="silu") if s == 2 else nn.Identity(),  # dw
                                  MixGhostConv(c_, c2, 1, 1, act="silu"))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act="silu"),
                                      MixGhostConv(c1, c2, 1, 1, act="silu")) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MixGhostConv(nn.Module):
     def __init__(
         self, in_channels, out_channels, ksize, stride, groups=1, act="silu"
     ):
         super().__init__()
         c_ = out_channels // 2  # hidden channels
         _c = c_ // 2
         self.cv1 = BaseConv(in_channels, c_, ksize, stride, math.gcd(in_channels, c_), False, act)
         self.cv2 = BaseConv(c_, _c, 7, 1, _c, False, act)
         self.cv3 = BaseConv(c_, _c, 9, 1, _c, False, act)
     def forward(self, x):
         y = self.cv1(x)
         return torch.cat([y, self.cv2(y), self.cv3(y)], 1)

