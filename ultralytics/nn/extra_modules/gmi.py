# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Minimal RT-DETR-GMI modules used by ``RT-DETR-GMI.yaml``."""

# C2f_MambaOut wraps the Apache-2.0 MambaOut block.
# MFM is adapted from zhoushen1/DCMPNet (MIT); see THIRD_PARTY_NOTICES.md.

import torch
import torch.nn as nn

from ultralytics.nn.backbone.MambaOut import GatedCNNBlock_BCHW
from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv

__all__ = 'C2f_MambaOut', 'MFM'


class C2f_MambaOut(C2f):
    """C2f block whose internal bottlenecks are replaced by MambaOut Gated CNN blocks."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(GatedCNNBlock_BCHW(self.c) for _ in range(n))


class MFM(nn.Module):
    """Multi-input feature fusion module used in the RT-DETR-GMI neck/head."""

    def __init__(self, inc, dim, reduction=8):
        super().__init__()
        self.height = len(inc)
        hidden = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, dim * self.height, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)
        self.conv1x1 = nn.ModuleList(Conv(c, dim, 1) if c != dim else nn.Identity() for c in inc)

    def forward(self, in_feats_):
        in_feats = [layer(in_feats_[idx]) for idx, layer in enumerate(self.conv1x1)]
        b, c, h, w = in_feats[0].shape
        stacked = torch.cat(in_feats, dim=1).view(b, self.height, c, h, w)
        feats_sum = torch.sum(stacked, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(b, self.height, c, 1, 1))
        return torch.sum(stacked * attn, dim=1)
