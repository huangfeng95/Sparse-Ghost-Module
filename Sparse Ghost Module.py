# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.cuda import amp

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync



class C3SparseGhost(C3):
    # C3 module with SparseGhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SparseGhostBottleneck(c_, c_) for _ in range(n)))


class SparseGhostConv(nn.Module):
    # SparseGhost Convolution
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 4  # hidden channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c1,
                      (k, 1),
                      (s, 1),
                      ((k) // 2, 0),
                      bias=False, groups=c1
                      ),
            nn.BatchNorm2d(c1),
            nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity()),
            nn.Conv2d(c1, c_,
                      (1, k),
                      (1, s),
                      (0, (k) // 2),
                      bias=False, groups=1
                      ),
            nn.BatchNorm2d(c_),
            nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 
                (5, 1), 
                (1, 1), 
                ((5)//2, 0), 
                bias=False, groups=1
            ),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=False) if act is True else (act if isinstance(act, nn.Module) else nn.Identity()),
            nn.Conv2d(c_, 3*c_,
                (1, 5),
                (1, 1),
                (0, (5)//2),
                bias=False, groups=c_
            ),
            nn.BatchNorm2d(3*c_),
            nn.SiLU(inplace=False) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class SparseGhostBottleneck(nn.Module):
    # SparseGhost Bottleneck
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 4
        self.conv = nn.Sequential(
            SparseGhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            SparseGhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
