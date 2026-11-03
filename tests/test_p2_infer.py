"""PLAN2 GUI inference: hemispheres are analysed separately and stitched back into the scan grid."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import p2_infer  # noqa: E402
from p2_model import build  # noqa: E402


def test_analyse_and_explain_shapes():
    torch.manual_seed(0)
    model = build({"model": {"channels": (4, 8, 8, 8, 8), "cls_hidden": 8}}).eval()
    rng = np.random.default_rng(0)
    x = np.zeros((4, 96, 112, 80), np.float32)
    x[:, 20:76, 20:92, 10:70] = rng.normal(size=(4, 56, 72, 60))
    r = p2_infer.analyse(model, x, (0.5, 1.0))
    assert set(r["sides"]) == {"L", "R"} and r["can_segment"]
    assert r["seg"].shape == (3, 96, 112, 80) and r["wt_b"].shape == (96, 112, 80)
    for s in r["sides"].values():
        assert s["x"].shape == (4, 48, 112, 80) and 0 <= s["prob"] <= 1
    cam = p2_infer.gradcam(model, r["sides"]["R"]["x"])
    assert cam.shape == (48, 112, 80)
    full = p2_infer.to_scan(cam, "R", 96)
    assert full[:41].max() == 0          # a right-hemisphere map never lands left of the midline
    x2 = x.copy()
    x2[1] = 0                            # no T1ce: detection only
    assert not p2_infer.analyse(model, x2, (0.5, 1.0))["can_segment"]


def test_shap_region_names_and_permutations():
    torch.manual_seed(0)
    model = build({"model": {"channels": (4, 8, 8, 8, 8), "cls_hidden": 8}}).eval()
    h = np.zeros((4, 48, 112, 80), np.float32)
    h[:, 10:40, 30:80, 20:60] = 1.0
    wt = np.zeros((48, 112, 80), bool)
    wt[20:28, 40:48, 30:38] = True
    rows, regions, k = p2_infer.shap_regions(model, h, wt, perms=1, bs=8)
    assert k == 1 and len(rows) == int(regions.max()) + 1
    names = [n for n, *_ in rows]
    assert "outside brain" in names and "predicted tumour, component 1" in names
