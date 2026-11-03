"""PLAN2 inference for the GUI: one scan -> both hemispheres -> per-hemisphere detection and segmentation,
stitched back into the scan grid. Same cut, canonical flip and re-z-scoring as training (hemi.py).

Detection (T1 + T2 only), per hemisphere:
  head  probability >= the head threshold frozen on val (PLAN2 worst-group rule)
  seg   pass-B WT volume >= v* (mL), also frozen on val
Segmentation: pass A (ET/TC/WT) needs all four sequences; pass B (WT) runs on T1 + T2.
Explanations of one hemisphere's detection logit: Grad-CAM++ (enc[-2]), occlusion, region SHAP.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hemi  # noqa: E402
import metrics  # noqa: E402
import p2_xai  # noqa: E402
import shortcut_checks as sc  # noqa: E402
from eval_p2 import load as load_checkpoint  # noqa: E402
from infer import VOXEL_ML, check_input  # noqa: E402

MIDLINE = 41                       # P1: mirror-Dice midline of the mean train brain mask (DECISIONS 1 Nov)
DEFAULT_THRESHOLDS = (0.5, 1.544)  # val: head probability, pass-B WT mL (results_val.json of p2-main)
SIDE_NAME = {"L": "left (axis-0 low side)", "R": "right (axis-0 high side)"}


def load(ckpt, device="cpu"):
    """(model, state, (head threshold, v* mL)); thresholds from thresholds.json or results_val.json next to
    the checkpoint, else the p2-main val values."""
    model, st = load_checkpoint(ckpt, torch.device(device))
    thr = DEFAULT_THRESHOLDS
    for p in (Path(ckpt).parent / "thresholds.json", Path(ckpt).parent / "results_val.json"):
        if p.exists():
            r = json.loads(p.read_text())
            thr = (float(r["threshold_head"]), float(r["threshold_seg"]))
            break
    return model, st, thr


def hemisphere(x, side):
    return hemi.renorm(hemi.cut(np.asarray(x, np.float32), MIDLINE, side))


@torch.no_grad()
def analyse(model, x, thr, device="cpu"):
    """x (4, X, H, W) BraTS-like 2 mm. Returns per-side results and stitched full-scan maps."""
    can_seg = check_input(x, (96, 112, 80))  # the midline index belongs to this grid
    n = x.shape[1]
    seg = np.zeros((3,) + x.shape[1:], bool) if can_seg else None
    wt_b = np.zeros(x.shape[1:], bool)
    sides = {}
    t = time.perf_counter()
    for side in hemi.SIDES:
        h = hemisphere(x, side)
        xt = torch.from_numpy(h[None]).to(device)
        lg, wt, *_ = model.forward_det(xt)
        wb = (torch.sigmoid(wt[0]) > 0.5).cpu().numpy()
        prob, ml = float(torch.sigmoid(lg[0])), float(wb.sum() * VOXEL_ML)
        wt_b |= hemi.uncut(wb, MIDLINE, side, n)
        r = dict(prob=prob, logit=float(lg[0]), head_tumour=prob >= thr[0], wt_ml_b=ml, seg_tumour=ml >= thr[1],
                 x=h, wt_b=wb)
        if can_seg:
            p = metrics.postprocess(model.forward_seg(xt))[0].cpu().numpy()
            seg |= hemi.uncut(p, MIDLINE, side, n)
            r["volumes_ml"] = {k: float(p[i].sum() * VOXEL_ML) for i, k in enumerate(metrics.REGIONS)}
        sides[side] = r
    ms = (time.perf_counter() - t) * 1000
    out = dict(sides=sides, seg=seg, wt_b=wt_b, can_segment=can_seg, ms=ms, thresholds=thr,
               tumour_head=any(s["head_tumour"] for s in sides.values()),
               tumour_seg=any(s["seg_tumour"] for s in sides.values()))
    if can_seg:
        out["volumes_ml"] = {k: float(seg[i].sum() * VOXEL_ML) for i, k in enumerate(metrics.REGIONS)}
    return out


def to_scan(hmap, side, n):
    """Hemisphere-frame map back to the scan grid."""
    return hemi.uncut(np.asarray(hmap, np.float32), MIDLINE, side, n)


def gradcam(model, h, device="cpu"):
    m = sc.gradcam_det(model, torch.from_numpy(np.asarray(h, np.float32)[None]).to(device), model.enc[-2])
    return np.squeeze(m).astype(np.float32)


def occlusion(model, h, device="cpu", stride=sc.OCC_STRIDE, bs=4):
    """Occlusion map (logit drop); stride 8 = the evaluation setting, 16 = faster preview."""
    m = sc.occlusion(model, np.asarray(h, np.float32), device, bs=bs, stride=stride)[0]
    return m / m.max() if m.max() > 0 else m


def shap_regions(model, h, wt_b, device="cpu", evals=p2_xai.SHAP_EVALS, bs=4, perms=None):
    """Region SHAP of the detection logit with readable region names, sorted by |value|. perms (GUI) sets the
    number of permutations instead of the evaluation budget.
    Returns ([(name, phi_logit, phi_ml, n_voxels)], region label map, number of tumour components)."""
    h = np.asarray(h, np.float32)
    regions, k = p2_xai.shap_regions(h, np.asarray(wt_b, bool))
    base = p2_xai.baseline(h)
    if perms:
        evals = perms * (int(regions.max()) + 2)
    phi_l, phi_v, _ = p2_xai.shap_values(model, h, regions, base, device, evals=evals, bs=bs)
    rows = []
    for r in range(int(regions.max()) + 1):
        vox = regions == r
        if r == 0:
            name = "outside brain"
        elif r <= k:
            name = f"predicted tumour, component {r}"
        else:
            c = np.argwhere(vox).mean(0) * 2  # mm in the hemisphere frame; midline at the last index of axis 0
            name = f"brain cube {((hemi.WIDTH - 1) * 2 - c[0]):.0f} mm from midline, y {c[1]:.0f}, z {c[2]:.0f} mm"
        rows.append((name, float(phi_l[r]), float(phi_v[r]), int(vox.sum())))
    rows.sort(key=lambda t: -abs(t[1]))
    return rows, regions, k
