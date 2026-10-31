"""Metrics shared by training-time validation and Phase 6 evaluation."""
from __future__ import annotations

import numpy as np
import torch

REGIONS = ("ET", "TC", "WT")


def postprocess(seg_logits, thr=0.5):
    """Sigmoid > thr per region, then nesting ET in TC in WT (as the owner's baseline)."""
    p = torch.sigmoid(seg_logits) > thr
    p[:, 1] |= p[:, 0]
    p[:, 2] |= p[:, 1]
    return p


def dice_per_case(pred, target):
    """pred, target bool (B, 3, ...) -> float array (B, 3); both empty gives 1 (BraTS convention)."""
    pred, target = pred.flatten(2), target.bool().flatten(2)
    inter = (pred & target).sum(-1).double()
    den = (pred.sum(-1) + target.sum(-1)).double()
    d = torch.where(den == 0, torch.ones_like(den), 2 * inter / den.clamp(min=1))
    return d.cpu().numpy()


def detection(y_true, prob, thr=0.5):
    y, p = np.asarray(y_true).astype(int), np.asarray(prob, float)
    pred = (p >= thr).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    tn, fn = int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    except ImportError:
        auroc = float("nan")
    return dict(threshold=thr, tp=tp, fp=fp, tn=tn, fn=fn, recall=recall, precision=precision, f1=f1,
                specificity=spec, auroc=auroc)


def selection_score(mean_dsc, f1, w_seg, w_cls):
    """best.pt criterion: 0.5*mean DSC + 0.5*F1 for the dual model; single-task runs use their own metric."""
    if w_seg > 0 and w_cls > 0:
        return 0.5 * mean_dsc + 0.5 * f1
    return mean_dsc if w_seg > 0 else f1
