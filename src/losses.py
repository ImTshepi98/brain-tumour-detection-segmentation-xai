"""Phase 5 losses: L = w_seg * L_seg + w_cls * L_cls (PLAN: w_seg 1, w_cls 0.5).
L_seg = sigmoid squared Dice + BCE on [ET, TC, WT], BraTS cases only (owner's baseline loss).
L_cls = BCE-with-logits on the detection logit, all cases (classes balanced by the sampler)."""
from __future__ import annotations

import torch.nn.functional as F
from monai.losses import DiceLoss

_dice = DiceLoss(sigmoid=True, squared_pred=True)


def seg_loss(logits, target):
    return _dice(logits, target) + F.binary_cross_entropy_with_logits(logits, target)


def cls_loss(logits, label, pos_weight=None):
    return F.binary_cross_entropy_with_logits(logits, label, pos_weight=pos_weight)
