"""PLAN2 2.1-2.2: hemisphere samples, their labels, and tumour / sham pasting in hemisphere space.

Axis 0 of the spatial grid is left-right (as evaluate.LR_AXIS). The midline m is the cut between voxels
m-1 and m that maximises the Dice overlap of the mean train brain mask with its mirror image (voxel
i <-> 2m-1-i). Left hemisphere = x[m-W:m]; right = x[m:m+W] flipped; both zero-padded to W, so the
midline is always the LAST index of a hemisphere ("canonical" frame). Labels (fixed in PLAN2 2.1):
positive if the hemisphere holds >= 125 WT voxels (1 mL at 2 mm); negative if it holds none and no WT
voxel lies within 2 voxels (4 mm) of the midline; otherwise excluded.
Each hemisphere is re-z-scored per channel over its own non-zero voxels.
Pasting reuses the intensity matching of evaluate.insert_tumour: per channel present in both, the pasted
tissue keeps its z-score relative to its source's tumour-free brain, is rescaled to the host brain's
mean/SD, and the host channel is then re-z-scored. Because both hemispheres are canonical, a donor's
tumour lands at the same distance from the midline in the host; the sham source is the donor's OTHER
hemisphere at the same canonical voxels (its healthy mirror tissue).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import zscore_nonzero  # noqa: E402

WIDTH = 48
MIN_POS_VOX = 125   # 1.0 mL at 2 mm
MARGIN = 2          # 4 mm
SIDES = ("L", "R")
CELLS = ("B+r", "B+p", "B-r", "B-s", "I+p", "I-r", "I-s")
POSITIVE = {"B+r": 1, "B+p": 1, "B-r": 0, "B-s": 0, "I+p": 1, "I-r": 0, "I-s": 0}
DATASET = {c: ("brats" if c[0] == "B" else "ixi") for c in CELLS}


def find_midline(mean_mask, thr=0.5):
    """Cut index m (between voxels m-1 and m along axis 0) maximising the mirror Dice of mean_mask > thr."""
    m = np.asarray(mean_mask) > thr
    n = m.shape[0]
    best, best_d = None, -1.0
    for c in range(n // 4, 3 * n // 4 + 1):
        src = 2 * c - 1 - np.arange(n)
        valid = (src >= 0) & (src < n)
        mir = np.zeros_like(m)
        mir[valid] = m[src[valid]]
        d = 2 * (m & mir).sum() / max(int(m.sum() + mir.sum()), 1)
        if d > best_d:
            best, best_d = c, float(d)
    return best, best_d


def _index(m, side, width):
    return np.arange(m - width, m) if side == "L" else np.arange(m + width - 1, m - 1, -1)


def cut(a, m, side, width=WIDTH):
    """Hemisphere of a (..., X, H, W) in the canonical frame (midline at the last index of axis -3)."""
    a = np.asarray(a)
    idx = _index(m, side, width)
    valid = (idx >= 0) & (idx < a.shape[-3])
    out = np.zeros(a.shape[:-3] + (width,) + a.shape[-2:], a.dtype)
    out[..., np.flatnonzero(valid), :, :] = np.take(a, idx[valid], axis=-3)
    return out


def uncut(h, m, side, n):
    """Inverse of cut along axis -3 into an array of length n (voxels outside the hemisphere are 0)."""
    h = np.asarray(h)
    idx = _index(m, side, h.shape[-3])
    valid = (idx >= 0) & (idx < n)
    out = np.zeros(h.shape[:-3] + (n,) + h.shape[-2:], h.dtype)
    out[..., idx[valid], :, :] = h[..., np.flatnonzero(valid), :, :]
    return out


def side_label(bits, m, side, width=WIDTH, min_vox=MIN_POS_VOX, margin=MARGIN):
    """(label 1 / 0 / -1 = excluded, WT voxels in this hemisphere)."""
    wt = (np.asarray(bits) & 4) > 0
    n = int(cut(wt, m, side, width).sum())
    if n >= min_vox:
        return 1, n
    if n == 0 and not wt[max(m - margin, 0):m + margin].any():
        return 0, 0
    return -1, n


def crop_loss(brain, m, width=WIDTH):
    """Share of brain voxels that fall outside both hemisphere crops."""
    brain = np.asarray(brain, bool)
    lost = brain[:max(m - width, 0)].sum() + brain[m + width:].sum()
    return float(lost / max(int(brain.sum()), 1))


def renorm(h):
    """Per-channel z-score over the hemisphere's own non-zero voxels (absent channels stay 0)."""
    h = np.asarray(h, np.float32)
    out = np.zeros_like(h)
    for c in range(h.shape[0]):
        if h[c].any():
            out[c] = zscore_nonzero(h[c])[0]
    return out


def hemispheres(x, bits, m, width=WIDTH):
    """[(side, x_h float32 renormed, bits_h uint8, label, wt_voxels)] for both sides of one scan (IXI:
    bits None, label 0)."""
    out = []
    for side in SIDES:
        lab, n = side_label(bits, m, side, width) if bits is not None else (0, 0)
        out.append((side, renorm(cut(x, m, side, width)),
                    cut(bits, m, side, width) if bits is not None else np.zeros((width,) + np.shape(x)[-2:], np.uint8),
                    lab, n))
    return out


def paste(host, donor, donor_bits, sham_src=None, sham_bits=None):
    """Paste the donor hemisphere's WT region into the host hemisphere (same canonical voxels, inside the
    host brain). sham_src = the donor's other hemisphere: its healthy tissue at those voxels is pasted
    instead (voxels where it is outside its brain or inside its own WT are skipped).
    Returns (x float32, target bits uint8: donor bits inside the pasted region, empty for a sham,
    fraction of donor WT voxels pasted, pasted volume in mL at 2 mm)."""
    host, donor = np.asarray(host, np.float32), np.asarray(donor, np.float32)
    hb, db = (host != 0).any(0), (donor != 0).any(0)
    dbits = np.asarray(donor_bits)
    wt = (dbits & 4) > 0
    region = wt & hb
    src, ref = donor, db & ~wt
    if sham_src is not None:
        src = np.asarray(sham_src, np.float32)
        sb = (src != 0).any(0)
        swt = (np.asarray(sham_bits) & 4) > 0 if sham_bits is not None else np.zeros_like(sb)
        region &= sb & ~swt
        ref = sb & ~swt
    out = host.copy()
    target = np.zeros(hb.shape, np.uint8)
    if not region.any():
        return out, target, 0.0, 0.0
    for c in range(host.shape[0]):
        hv, dn = host[c][hb], src[c][ref]
        if not hv.any() or not dn.any():
            continue
        z = (src[c][region] - dn.mean(dtype=np.float64)) / dn.std(dtype=np.float64)
        out[c][region] = z * hv.std(dtype=np.float64) + hv.mean(dtype=np.float64)
        out[c][hb] = zscore_nonzero(out[c][hb])[0]
    if sham_src is None:
        target[region] = dbits[region]
    return out, target, float(region.sum() / wt.sum()), float(region.sum() * 8 / 1000)


class HemiSet:
    """Hemispheres of one or more splits held in RAM: a pool with BOTH hemispheres of every subject
    (fp16 x, uint8 bits) and the list of kept samples (excluded hemispheres stay in the pool only, as sham
    sources). Sample k: pool index `pool[k]`, contralateral pool index `contra[k]`, cell B+r / B-r / I-r."""

    def __init__(self, xs, bits, meta):
        self.px, self.pbits, self.pmeta = xs, bits, meta
        self.pool = [j for j, mt in enumerate(meta) if mt["label"] >= 0]
        self.contra = [j ^ 1 for j in self.pool]          # pool holds (L, R) pairs at (2i, 2i+1)
        self.cell = [("B" if meta[j]["source"] == "brats" else "I") + ("+r" if meta[j]["label"] == 1 else "-r")
                     for j in self.pool]
        self.counts = {}
        for mt in meta:
            key = f"{mt['split']}|{mt['source']}|{('excluded', 'negative', 'positive')[mt['label'] + 1]}"
            self.counts[key] = self.counts.get(key, 0) + 1

    def __len__(self):
        return len(self.pool)

    def of(self, cell):
        return [k for k, c in enumerate(self.cell) if c == cell]

    def meta(self, k):
        return self.pmeta[self.pool[k]]

    def x(self, k):
        return self.px[self.pool[k]]

    def bits(self, k):
        return self.pbits[self.pool[k]]

    @classmethod
    def load(cls, rows, splits, m, brats_dir, ixi_dirs, brats_sdt_dirs=(), border=None, width=WIDTH, workers=4,
             allow_test=False, subjects=None):
        """Streams one scan at a time (full scans are never all in RAM)."""
        from concurrent.futures import ThreadPoolExecutor
        import datasets as ds
        assert allow_test or "test" not in splits, "test split is locked (PLAN B4)"
        rows = [r for r in rows if r["split"] in splits and (subjects is None or r["subject"] in subjects)]
        paths = ds.locate(rows, brats_dir, ixi_dirs, brats_sdt_dirs)
        missing = [r["subject"] for r in rows if r["subject"] not in paths]
        assert not missing, f"{len(missing)} arrays missing, e.g. {missing[:3]}"

        def one(r):
            x, y = ds.load_one(paths[r["subject"]], border)
            bits = y if r["source"] == "brats" else None
            out = []
            for side, xh, bh, lab, n in hemispheres(x, bits, m, width):
                out.append((xh.astype(np.float16), bh, dict(subject=r["subject"], side=side, source=r["source"],
                                                            site=r["site"], split=r["split"], label=lab,
                                                            wt_voxels=n, crop_loss=crop_loss((x != 0).any(0), m, width))))
            return out
        xs, bits, meta = [], [], []
        with ThreadPoolExecutor(max(1, workers)) as ex:
            for pair in ex.map(one, rows):
                for xh, bh, mt in pair:
                    xs.append(xh)
                    bits.append(bh)
                    meta.append(mt)
        return cls(xs, bits, meta)
