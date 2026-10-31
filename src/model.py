"""Phase 5: dual-head 3D residual U-Net with attention gates (PLAN section 3).

Encoder: 5 levels, channels [16, 32, 64, 128, 256], each level one residual block of two
3x3x3 convs (InstanceNorm + LeakyReLU), downsampling by a stride-2 first conv (as MONAI's
UNet with num_res_units=2 in the owner's baseline).
Decoder: transposed conv up, additive attention gate on the skip (Oktay et al. 2018: gating
signal from the coarser level, strided theta on the skip, sigmoid coefficients upsampled
trilinearly), concatenation, residual block. `attention=False` gives plain skip concatenation
(the `noattn` ablation).
Heads: segmentation 1x1x1 conv -> 3 logits [ET, TC, WT]; classification = global average pool
of the bottleneck -> MLP 256 -> 64 -> 1.
`forward_cls` zeroes T1ce and FLAIR itself, so detection only ever sees T1 and T2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import detection_input  # noqa: E402


def _norm_act(c):
    return nn.Sequential(nn.InstanceNorm3d(c, affine=True), nn.LeakyReLU(0.01, inplace=True))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.na1 = _norm_act(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm3d(cout, affine=True)
        self.skip = (nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.InstanceNorm3d(cout, affine=True))
                     if (cin != cout or stride != 1) else nn.Identity())
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.norm2(self.conv2(self.na1(self.conv1(x)))) + self.skip(x))


class AttentionGate(nn.Module):
    """Additive attention gate. x: skip (C_x, S), g: gating signal from the coarser level (C_g, S/2)."""

    def __init__(self, c_x, c_g, c_int):
        super().__init__()
        self.theta = nn.Conv3d(c_x, c_int, 2, 2, bias=False)
        self.phi = nn.Conv3d(c_g, c_int, 1, bias=True)
        self.psi = nn.Conv3d(c_int, 1, 1, bias=True)
        self.out = nn.Sequential(nn.Conv3d(c_x, c_x, 1, bias=False), nn.InstanceNorm3d(c_x, affine=True))
        self.last_alpha = None

    def forward(self, x, g):
        a = torch.sigmoid(self.psi(F.relu(self.theta(x) + self.phi(g))))
        a = F.interpolate(a, size=x.shape[2:], mode="trilinear", align_corners=False)
        self.last_alpha = a.detach()
        return self.out(x * a)


class DualUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, channels=(16, 32, 64, 128, 256), attention=True,
                 cls_hidden=64, cls_dropout=0.0):
        super().__init__()
        c = list(channels)
        self.attention = attention
        self.enc = nn.ModuleList([ResBlock(in_channels, c[0])] +
                                 [ResBlock(c[i - 1], c[i], stride=2) for i in range(1, len(c))])
        self.up = nn.ModuleList([nn.ConvTranspose3d(c[i + 1], c[i], 2, 2) for i in range(len(c) - 1)])
        self.gate = nn.ModuleList([AttentionGate(c[i], c[i + 1], max(c[i] // 2, 8)) if attention else nn.Identity()
                                   for i in range(len(c) - 1)])
        self.dec = nn.ModuleList([ResBlock(2 * c[i], c[i]) for i in range(len(c) - 1)])
        self.seg_head = nn.Conv3d(c[0], out_channels, 1)
        self.cls_head = nn.Sequential(nn.Linear(c[-1], cls_hidden), nn.LeakyReLU(0.01, inplace=True),
                                      nn.Dropout(cls_dropout), nn.Linear(cls_hidden, 1))
        self.levels = len(c)

    def encode(self, x):
        feats = []
        for blk in self.enc:
            x = blk(x)
            feats.append(x)
        return feats

    def decode(self, feats):
        g = feats[-1]
        for i in reversed(range(self.levels - 1)):
            skip = self.gate[i](feats[i], g) if self.attention else feats[i]
            g = self.dec[i](torch.cat([self.up[i](g), skip], 1))
        return g

    def forward_seg(self, x):
        """All 4 channels -> segmentation logits (B, 3, D, H, W)."""
        return self.seg_head(self.decode(self.encode(x)))

    def forward_cls(self, x):
        """Detection logit (B,) from T1 and T2 only (T1ce and FLAIR are zeroed here)."""
        b = self.encode(detection_input(x))[-1]
        return self.cls_head(b.mean((2, 3, 4))).squeeze(1)

    def forward(self, x, task="seg"):
        # single entry point so nn.DataParallel can split either pass
        return self.forward_seg(x) if task == "seg" else self.forward_cls(x)


def build(cfg):
    m = cfg.get("model", {})
    return DualUNet(channels=tuple(m.get("channels", (16, 32, 64, 128, 256))), attention=m.get("attention", True),
                    cls_hidden=m.get("cls_hidden", 64), cls_dropout=m.get("cls_dropout", 0.0))


def n_params(model):
    return sum(p.numel() for p in model.parameters())
