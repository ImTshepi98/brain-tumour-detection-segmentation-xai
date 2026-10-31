"""Phase 8: single-case inference used by the GUI (and case export for it).

analyse(): detection probability and decision from T1/T2; segmentation (ET/TC/WT) only when T1ce
and FLAIR are present; region volumes in mL (2 mm voxels = 8 mm^3); optional Grad-CAM++ map;
inference time.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
from evaluate import load_checkpoint  # noqa: E402
from gradcam3d import detection_cam, segmentation_cam  # noqa: E402

VOXEL_ML = 8.0 / 1000.0


def load_case(path):
    """.npz from preprocess_one/export_cases (x, y, meta) or a bare (4, D, H, W) .npy."""
    path = Path(path)
    if path.suffix == ".npy":
        x = np.load(path)
        return x, np.zeros(x.shape[1:], np.uint8), {}
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    return d["x"], d["y"] if "y" in d else np.zeros(d["x"].shape[1:], np.uint8), meta


def check_input(x, expected_shape=None):
    x = np.asarray(x)
    assert x.ndim == 4 and x.shape[0] == 4, f"expected (4, D, H, W) in order [T1, T1ce, T2, FLAIR], got {x.shape}"
    if expected_shape is not None:
        assert tuple(x.shape[1:]) == tuple(expected_shape), f"expected spatial shape {tuple(expected_shape)}"
    assert np.isfinite(x).all(), "non-finite values"
    assert x[0].any() and x[2].any(), "T1 and T2 are required"
    return bool(x[1].any() and x[3].any())  # segmentation possible


def analyse(model, x, thr=0.5, cam=None, device="cpu"):
    """cam: None, 'detection' or 'segmentation' (WT). Returns a dict of numpy results."""
    can_seg = check_input(x)
    xt = torch.from_numpy(np.asarray(x, np.float32)[None]).to(device)
    t = time.perf_counter()
    with torch.no_grad():
        prob = float(torch.sigmoid(model.forward_cls(xt))[0])
        pred = metrics.postprocess(model.forward_seg(xt))[0].cpu().numpy() if can_seg else None
    ms = (time.perf_counter() - t) * 1000
    out = dict(prob=prob, tumour=prob >= thr, threshold=thr, can_segment=can_seg, ms=ms, seg=pred, cam=None)
    if pred is not None:
        out["volumes_ml"] = {r: float(pred[k].sum() * VOXEL_ML) for k, r in enumerate(metrics.REGIONS)}
    if cam == "detection":
        out["cam"] = detection_cam(model, xt)[0][0].cpu().numpy()
    elif cam == "segmentation" and can_seg:
        out["cam"] = segmentation_cam(model, xt, "WT")[0][0].cpu().numpy()
    return out


def load(ckpt, device="cpu"):
    model, st = load_checkpoint(ckpt, torch.device(device))
    thr = 0.5
    for p in (Path(ckpt).parent / "threshold.json", Path(ckpt).parent / "eval" / "results_val.json"):
        if p.exists():
            thr = float(json.loads(p.read_text())["threshold"])
            break
    return model, st, thr


def export_cases(cohort, ids, out_dir):
    """Save selected cohort cases (already through the shared boundary) as GUI .npz files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in ids:
        i = cohort.ids.index(s)
        x = cohort.x[i]
        meta = dict(id=s, source=cohort.source[i], site=cohort.site[i], label=int(cohort.label[i]),
                    has_seg=bool(cohort.has_seg[i]), shared_boundary=True)
        np.savez_compressed(out / f"{s}.npz", x=x, y=cohort.y[i], meta=json.dumps(meta))
    return out
