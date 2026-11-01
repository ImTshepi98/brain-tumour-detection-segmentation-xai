"""PLAN2 metrics and GPU helpers shared by train_p2.py and eval_p2.py (definitions: PLAN2 section 4,
"Operational definitions", tag plan2-prereg)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss

import hemi

_dice_batch = DiceLoss(sigmoid=True, squared_pred=True, batch=True)


def seg_loss(logits, target):
    """Squared Dice computed over the whole batch (so empty targets of negative hemispheres do not each count
    as a failed case) + BCE. logits / target (B, C, ...)."""
    return _dice_batch(logits, target) + F.binary_cross_entropy_with_logits(logits, target)


def balanced_accuracy(rows, probs, thr):
    """{dataset: BA} with BA_d = (TPR over d's positive cells + TNR over d's negative cells) / 2."""
    out = {}
    for d in ("brats", "ixi"):
        pos = [p for r, p in zip(rows, probs) if r["dataset"] == d and r["label"] == 1]
        neg = [p for r, p in zip(rows, probs) if r["dataset"] == d and r["label"] == 0]
        if pos and neg:
            out[d] = float(0.5 * (np.mean(np.asarray(pos) >= thr) + np.mean(np.asarray(neg) < thr)))
    return out


def worst_group_ba(rows, probs, thr):
    ba = balanced_accuracy(rows, probs, thr)
    return min(ba.values()) if ba else float("nan")


def choose_threshold(rows, scores):
    """Maximise worst-group BA over midpoints between consecutive distinct scores (plus 0.5 for probabilities);
    ties go to the candidate closest to 0.5 (PLAN2 3, operational definitions)."""
    s = np.unique(np.asarray(scores, float))
    cand = list((s[1:] + s[:-1]) / 2) + [0.5, float(s[0]) - 1e-9, float(s[-1]) + 1e-9]
    best = max(cand, key=lambda t: (round(worst_group_ba(rows, scores, t), 12), -abs(t - 0.5)))
    return float(best), worst_group_ba(rows, scores, best)


def cell_rates(rows, probs, thr):
    """{cell: (n, share >= thr)}."""
    out = {}
    for c in hemi.CELLS:
        p = [q for r, q in zip(rows, probs) if r["cell"] == c]
        if p:
            out[c] = (len(p), float(np.mean(np.asarray(p) >= thr)))
    return out


def erode_batch(x, target, gen, max_iter=2):
    """Random brain-boundary erosion of 0..max_iter voxels per sample (PLAN2 2.4). WT voxels are kept; each
    channel is re-z-scored over its remaining non-zero voxels (the preprocessing rule). The midline face is
    the array edge and max_pool pads with -inf, so the cut face itself is never eroded."""
    k = torch.randint(0, max_iter + 1, (x.shape[0],), generator=gen, device=x.device)
    if not k.any():
        return x
    x = x.clone()
    brain = (x != 0).any(1, keepdim=True).float()
    for i in range(x.shape[0]):
        if k[i] == 0:
            continue
        m = brain[i:i + 1]
        for _ in range(int(k[i])):
            m = -F.max_pool3d(-m, 3, 1, 1)
        keep = (m[0, 0] > 0) | (target[i, 2] > 0.5)
        for c in range(x.shape[1]):
            ch = x[i, c]
            sel = keep & (ch != 0)
            if sel.sum() < 2:
                continue
            v = ch[sel]
            out = torch.zeros_like(ch)
            out[sel] = (v - v.mean()) / v.std(unbiased=False).clamp(min=1e-6)
            x[i, c] = out
    return x


def to_tensors(samples, device):
    """[(x, bits, cell)] -> x (B, 4, ...), target (B, 3, ...) [ET, TC, WT], label (B,), is_brats (B,) bool."""
    x = torch.from_numpy(np.stack([np.asarray(s[0], np.float32) for s in samples])).to(device)
    bits = torch.from_numpy(np.stack([s[1] for s in samples])).to(device)
    target = torch.stack([(bits >> k) & 1 for k in range(3)], 1).float()
    label = torch.tensor([hemi.POSITIVE[s[2]] for s in samples], dtype=torch.float32, device=device)
    is_brats = torch.tensor([hemi.DATASET[s[2]] == "brats" for s in samples], device=device)
    return x, target, label, is_brats
