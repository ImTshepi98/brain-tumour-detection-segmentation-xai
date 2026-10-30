"""Shared constants and label / pooling helpers used by prep, datasets and tests."""
from __future__ import annotations

import numpy as np

CHANNELS = ("t1", "t1ce", "t2", "flair")  # fixed input order [T1, T1ce, T2, FLAIR]
REGIONS = ("ET", "TC", "WT")              # bit0, bit1, bit2
SEED = 42


def seg_to_bits(seg):
    """BraTS native labels (1 NCR, 2 ED, 4 ET) -> uint8 bits: bit0 ET={4}, bit1 TC={1,4}, bit2 WT={1,2,4}."""
    seg = np.asarray(seg)
    et = seg == 4
    tc = np.isin(seg, (1, 4))
    wt = np.isin(seg, (1, 2, 4))
    return (et.astype(np.uint8) | (tc.astype(np.uint8) << 1) | (wt.astype(np.uint8) << 2))


def bits_to_channels(bits):
    """uint8 bits (...) -> float32 (3, ...) in order [ET, TC, WT]."""
    bits = np.asarray(bits)
    return np.stack([(bits >> k) & 1 for k in range(3)]).astype(np.float32)


def channels_to_bits(ch):
    """Binary (3, ...) [ET, TC, WT] -> uint8 bits."""
    ch = np.asarray(ch).astype(np.uint8)
    return ch[0] | (ch[1] << 1) | (ch[2] << 2)


def zscore_nonzero(arr, mean=None, std=None):
    """Owner's formula (signature brats2021-supplied-grid-zscore-v1): z-score over this
    modality's non-zero voxels (float64 mean/std, ddof 0), background stays 0, no clipping.
    Returns (normalised float32, mean, std)."""
    arr = np.asarray(arr, np.float32)
    active = arr != 0
    vals = arr[active]
    if mean is None:
        mean = float(vals.mean(dtype=np.float64))
        std = float(vals.std(dtype=np.float64))
    out = np.zeros(arr.shape, np.float32)
    out[active] = (vals - mean) / std
    return out, mean, std


def bbox(mask):
    """Inclusive-exclusive bounding box [(lo, hi), ...] of a boolean mask."""
    idx = np.nonzero(mask)
    return [(int(i.min()), int(i.max()) + 1) for i in idx]


def union_bbox(boxes):
    return [(min(b[d][0] for b in boxes), max(b[d][1] for b in boxes)) for d in range(len(boxes[0]))]


def pad_box_to_even(box, shape):
    """Grow each (lo, hi) so its length is even, staying inside shape."""
    out = []
    for (lo, hi), n in zip(box, shape):
        if (hi - lo) % 2:
            if hi < n:
                hi += 1
            else:
                lo -= 1
        out.append((lo, hi))
    return out


def crop(arr, box):
    sl = tuple(slice(lo, hi) for lo, hi in box)
    return arr[(Ellipsis,) + sl]


def _blocks(a):
    """(..., D, H, W) with even D, H, W -> (..., D/2, H/2, W/2, 8) block view."""
    *lead, d, h, w = a.shape
    a = a.reshape(*lead, d // 2, 2, h // 2, 2, w // 2, 2)
    n = len(lead)
    a = a.transpose(*range(n), n, n + 2, n + 4, n + 1, n + 3, n + 5)
    return a.reshape(*lead, d // 2, h // 2, w // 2, 8)


def pool_image(x, brain, min_frac=0.5):
    """2x average-pool (C, D, H, W) over brain voxels only; blocks whose brain fraction
    is below min_frac become 0 (background re-zeroed)."""
    xb = _blocks(np.asarray(x, np.float32))
    bb = _blocks(np.asarray(brain, np.float32))
    cnt = bb.sum(-1)
    s = (xb * bb[None]).sum(-1)
    out = np.where(cnt[None] > 0, s / np.maximum(cnt[None], 1), 0.0)
    out[:, cnt / 8.0 < min_frac] = 0.0
    return out.astype(np.float32), (cnt / 8.0 >= min_frac)


def pool_labels(bits, thr=0.5):
    """2x pool each binary region channel (mean >= thr) and return uint8 bits."""
    ch = bits_to_channels(bits)
    pooled = _blocks(ch).mean(-1) >= thr
    # keep nesting consistent: ET subset of TC subset of WT
    pooled[1] |= pooled[0]
    pooled[2] |= pooled[1]
    return channels_to_bits(pooled)


def pad_to(arr, target):
    """Zero-pad the last len(target) axes at the end to target size."""
    pads = [(0, 0)] * (arr.ndim - len(target))
    for n, t in zip(arr.shape[-len(target):], target):
        assert n <= t, (arr.shape, target)
        pads.append((0, t - n))
    return np.pad(arr, pads)


def round_up(n, m):
    return int(-(-n // m) * m)
