"""PLAN2 section 3: PLAN's dual-head U-Net with a second segmentation pass and an optional adversarial head.

forward_det(x): one encoder pass on T1 + T2 only (T1ce/FLAIR zeroed for EVERY sample, so no sample-specific
zero channels) giving the detection logit (head on the pooled bottleneck, as PLAN) and the pass-B WT logit
map (same decoder and segmentation head, WT channel). With adv=True a gradient-reversal head on the pooled
features predicts BraTS vs IXI (Ganin et al. 2016); the training loss applies it within each label.
forward_seg(x) (pass A, all 4 channels, ET/TC/WT) and forward_cls(x) are inherited unchanged, so
Grad-CAM++, deletion and PLAN's controls work on this model as on PLAN's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import detection_input  # noqa: E402
from model import DualUNet  # noqa: E402


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class P2Net(DualUNet):
    def __init__(self, adv=False, adv_lambda=1.0, **kw):
        super().__init__(**kw)
        c = self.cls_head[0].in_features
        self.adv_lambda = adv_lambda
        self.adv_head = (nn.Sequential(nn.Linear(c, 64), nn.LeakyReLU(0.01, inplace=True), nn.Linear(64, 1))
                         if adv else None)

    def forward_det(self, x):
        """(detection logit (B,), pass-B WT logits (B, D, H, W), adversarial logit (B,) or None)."""
        feats = self.encode(detection_input(x))
        pooled = feats[-1].mean((2, 3, 4))
        logit = self.cls_head(pooled).squeeze(1)
        wt = self.seg_head(self.decode(feats))[:, 2]
        adv = (self.adv_head(_GradReverse.apply(pooled, self.adv_lambda)).squeeze(1)
               if self.adv_head is not None else None)
        return logit, wt, adv

    def forward(self, x, task="seg"):
        if task == "det":
            logit, wt, adv = self.forward_det(x)
            return (logit, wt) if adv is None else (logit, wt, adv)
        return super().forward(x, task)


def build(cfg):
    m = cfg.get("model", {})
    return P2Net(adv=m.get("adv", False), adv_lambda=m.get("adv_lambda", 1.0),
                 channels=tuple(m.get("channels", (16, 32, 64, 128, 256))), attention=m.get("attention", True),
                 cls_hidden=m.get("cls_hidden", 64), cls_dropout=m.get("cls_dropout", 0.0))
