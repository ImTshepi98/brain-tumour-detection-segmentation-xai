"""Phase 4: split checks and data loading for the dual-head model.

Arrays (2 mm, 4x96x112x80 fp16, channels [T1, T1ce, T2, FLAIR]; IXI channels 1 and 3 zero) are
read once into RAM. When a border is given, the shared SynthStrip boundary is applied on load
(`common.shared_mask`, same rule for both cohorts, BraTS WT kept) and each channel is re-z-scored.
Augmentation runs on the GPU per batch: flips on every axis, per-channel intensity scale 0.9-1.1
and shift +/-0.1, Gaussian noise (sd 0.05); all on brain voxels only, background stays 0.
The test split is refused unless the caller passes allow_test=True (Phase 6 only).
"""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, shared_mask  # noqa: E402

EXPECTED = {"brats": {"train": 875, "val": 188, "test": 188}, "ixi": {"train": 404, "val": 86, "test": 87}}
DET_CHANNELS = (0, 2)  # T1, T2: the detection pass sees only these


def read_splits(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_splits(rows, expected=EXPECTED):
    """Every subject appears once, in exactly one of train/val/test; per-cohort counts match."""
    subjects = [r["subject"] for r in rows]
    assert len(subjects) == len(set(subjects)), "duplicate subject in splits"
    assert all(r["split"] in ("train", "val", "test") for r in rows), "unknown split"
    assert all(r["source"] in ("brats", "ixi") for r in rows), "unknown source"
    assert all(int(r["label"]) == (r["source"] == "brats") for r in rows), "label != source"
    counts = {s: {sp: 0 for sp in ("train", "val", "test")} for s in ("brats", "ixi")}
    for r in rows:
        counts[r["source"]][r["split"]] += 1
    if expected is not None:
        assert counts == expected, f"split counts {counts} != expected {expected}"
    return counts


def locate(rows, brats_dir, ixi_dirs, brats_sdt_dirs=()):
    """subject -> dict(x, y, sdt) file paths (sdt None when not found)."""
    out = {}
    bd = Path(brats_dir)
    for r in rows:
        s = r["subject"]
        if r["source"] == "brats":
            x = bd / "x" / f"{s}.npy"
            if x.exists():
                sdt = next((Path(d) / "sdt" / f"{s}.npy" for d in brats_sdt_dirs
                            if (Path(d) / "sdt" / f"{s}.npy").exists()), None)
                out[s] = dict(x=x, y=bd / "y" / f"{s}.npy", sdt=sdt)
        else:
            for d in map(Path, ixi_dirs):
                if (d / "x" / f"{s}.npy").exists():
                    sdt = d / "sdt" / f"{s}.npy"
                    out[s] = dict(x=d / "x" / f"{s}.npy", y=None, sdt=sdt if sdt.exists() else None)
                    break
    return out


def load_one(paths, border=None):
    """(x fp16 (4, D, H, W), bits uint8 (D, H, W)); IXI bits are all zero."""
    x = np.load(paths["x"])
    y = np.load(paths["y"]) if paths["y"] is not None else np.zeros(x.shape[1:], np.uint8)
    if border is not None:
        assert paths["sdt"] is not None, f"no SynthStrip distance map for {paths['x']}"
        x, _ = shared_mask(x, np.load(paths["sdt"]), border, y if paths["y"] is not None else None)
    return np.asarray(x, np.float16), np.asarray(y, np.uint8)


class Cohort:
    """One or more splits held in RAM as fp16 / uint8 arrays."""

    def __init__(self, rows, xs, ys):
        self.rows = rows
        self.ids = [r["subject"] for r in rows]
        self.source = [r["source"] for r in rows]
        self.site = [r["site"] for r in rows]
        self.label = np.array([int(r["label"]) for r in rows], np.int64)
        self.has_seg = np.array([int(r["has_seg"]) for r in rows], bool)
        self.x, self.y = xs, ys
        shapes = {a.shape for a in xs}
        assert len(shapes) <= 1, f"arrays differ in shape: {shapes}"
        self.shape = next(iter(shapes)) if shapes else None

    def __len__(self):
        return len(self.ids)

    @classmethod
    def load(cls, rows, splits, brats_dir, ixi_dirs, brats_sdt_dirs=(), border=None, workers=4,
             allow_missing=False, allow_test=False, subjects=None):
        assert allow_test or "test" not in splits, "test split is locked (PLAN B4)"
        rows = [r for r in rows if r["split"] in splits and (subjects is None or r["subject"] in subjects)]
        paths = locate(rows, brats_dir, ixi_dirs, brats_sdt_dirs)
        missing = [r["subject"] for r in rows if r["subject"] not in paths]
        assert allow_missing or not missing, f"{len(missing)} arrays missing, e.g. {missing[:3]}"
        rows = [r for r in rows if r["subject"] in paths]
        with ThreadPoolExecutor(max(1, workers)) as ex:
            loaded = list(ex.map(lambda r: load_one(paths[r["subject"]], border), rows))
        return cls(rows, [a for a, _ in loaded], [b for _, b in loaded])

    def subset(self, idx):
        return Cohort([self.rows[i] for i in idx], [self.x[i] for i in idx], [self.y[i] for i in idx])

    def batch(self, idx, device):
        """x float32 (B, 4, D, H, W), target float32 (B, 3, D, H, W) [ET, TC, WT], label float (B,), has_seg bool (B,)."""
        x = torch.from_numpy(np.stack([self.x[i] for i in idx])).to(device, non_blocking=True).float()
        bits = torch.from_numpy(np.stack([self.y[i] for i in idx])).to(device, non_blocking=True)
        target = torch.stack([(bits >> k) & 1 for k in range(3)], 1).float()
        label = torch.as_tensor(self.label[idx], dtype=torch.float32, device=device)
        has_seg = torch.as_tensor(self.has_seg[idx], device=device)
        return x, target, label, has_seg


def sample_epoch(labels, n, mode="balanced", seed=SEED, epoch=0):
    """Indices for one epoch. balanced: each draw picks a class with p=0.5, then a subject of that
    class uniformly (without replacement until the class is used up, then reshuffled).
    brats_only: a shuffle of the positive subjects. all: a shuffle of everything."""
    rng = np.random.default_rng(seed + 1000 * epoch)
    labels = np.asarray(labels)
    if mode == "all":
        return rng.permutation(len(labels))[:n] if n else rng.permutation(len(labels))
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if mode == "brats_only":
        p = rng.permutation(pos)
        return p[:n] if n else p
    assert mode == "balanced" and len(pos) and len(neg), mode
    pools = {1: [], 0: []}
    out = []
    for c in rng.random(n or len(labels)) < 0.5:
        c = int(c)
        if not pools[c]:
            pools[c] = list(rng.permutation(pos if c else neg))
        out.append(pools[c].pop())
    return np.array(out, np.int64)


def augment(x, target, gen, scale=0.1, shift=0.1, noise=0.05, flip=True):
    """GPU augmentation of a batch (returns new tensors); `gen` is a torch.Generator on x.device."""
    b, c = x.shape[:2]
    if flip:
        f = torch.rand(b, 3, generator=gen, device=x.device) < 0.5
        x, target = x.clone(), target.clone()
        for i in range(b):
            dims = [d + 1 for d in range(3) if f[i, d]]
            if dims:
                x[i] = x[i].flip(dims)
                target[i] = target[i].flip(dims)
    brain = (x != 0).any(1, keepdim=True)
    a =1 + (torch.rand(b, c, 1, 1, 1, generator=gen, device=x.device) * 2 - 1) * scale
    s = (torch.rand(b, c, 1, 1, 1, generator=gen, device=x.device) * 2 - 1) * shift
    n = torch.randn(x.shape, generator=gen, device=x.device) * noise
    present = (x != 0).flatten(2).any(-1)[..., None, None, None]  # absent channels (IXI T1ce/FLAIR) stay 0
    x = torch.where(brain & present, x * a + s + n, torch.zeros_like(x))
    return x, target


def detection_input(x):
    """Zero every channel except T1 and T2 (the detection pass, for ALL samples)."""
    keep = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    keep[list(DET_CHANNELS)] = 1
    return x * keep.view(1, -1, 1, 1, 1)
