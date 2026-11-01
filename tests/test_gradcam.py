import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import explain  # noqa: E402
from gradcam3d import GradCAMpp, detection_cam, segmentation_cam  # noqa: E402
from model import DualUNet  # noqa: E402


def test_gradcampp_matches_formula():
    torch.manual_seed(0)
    conv = nn.Conv3d(1, 3, 1)
    x = torch.rand(1, 1, 4, 4, 4)
    cam = GradCAMpp(conv)
    a = conv(x)
    s = (a * torch.tensor([1.0, -2.0, 0.5]).view(1, 3, 1, 1, 1)).sum()  # linear score: g constant per channel
    got = cam.compute(s, (4, 4, 4))
    cam.remove()
    a = a.detach()
    g = torch.tensor([1.0, -2.0, 0.5]).view(1, 3, 1, 1, 1).expand_as(a)
    alpha = g ** 2 / (2 * g ** 2 + a.sum((2, 3, 4), keepdim=True) * g ** 3)
    w = (alpha * torch.relu(g)).sum((2, 3, 4), keepdim=True)
    ref = torch.relu((w * a).sum(1))
    ref = (ref - ref.min()) / (ref.max() - ref.min())
    assert torch.allclose(got, ref, atol=1e-5)


def test_cams_on_model():
    torch.manual_seed(0)
    m = DualUNet(channels=(4, 8, 16, 32, 64)).eval()
    x = torch.randn(2, 4, 32, 32, 32)
    det, p = detection_cam(m, x)
    assert det.shape == (2, 32, 32, 32) and p.shape == (2,) and det.min() >= 0 and det.max() <= 1
    seg, pred = segmentation_cam(m, x, "WT")
    assert seg.shape == (2, 32, 32, 32) and pred.shape == (2, 32, 32, 32)
    assert not any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())  # no stray grads
    with torch.no_grad():
        m.seg_head.bias.fill_(-100.0)  # empty prediction -> zero map, no error
    seg0, pred0 = segmentation_cam(m, x, "ET")
    assert not pred0.any() and seg0.abs().sum() == 0


def test_mass_stats_and_deletion():
    brain = np.zeros((32, 32, 32), bool)
    brain[4:28, 4:28, 4:28] = True
    tumour = np.zeros_like(brain)
    tumour[12:18, 12:18, 12:18] = True
    cam = tumour.astype(float)
    s = explain.mass_stats(cam, brain, tumour)
    assert s["tumour_mass"] == 1.0 and s["shell_mass"] == 0.0 and s["outside_mass"] == 0.0
    assert s["tumour_ratio"] > 1 and 0 < s["shell_vol"] < 1
    m = DualUNet(channels=(4, 8, 16, 32, 64)).eval()
    x = np.where(brain, np.random.default_rng(0).normal(size=(4, 32, 32, 32)), 0).astype(np.float32)
    p0, d = explain.deletion(m, x, cam, brain, "cpu")
    assert 0 < p0 < 1 and set(d) == {f"drop_{k}_{f}" for k in ("cam", "random") for f in (10, 20, 30)}


def test_select_cases():
    cases = [dict(id=f"B{i}", label=1, prob=str(p), dsc_WT=str(d))
             for i, (p, d) in enumerate([(0.9, 0.9), (0.2, 0.1), (0.8, 0.5), (0.95, 0.7)])]
    cases += [dict(id="H0", label=0, prob="0.1", dsc_WT=""), dict(id="H1", label=0, prob="0.7", dsc_WT=""),
              dict(id="H2", label=0, prob="0.2", dsc_WT="")]
    got = explain.select_cases(cases, {"H0": 50.0, "H1": 80.0, "H2": 20.0})
    assert got[:2] == ["H1", "B1"] and "B0" in got and len(got) == 6 and len(set(got)) == 6


def test_explain_debug(tmp_path):
    r = explain.main(["--debug", "--out", str(tmp_path)])
    assert r["meta"]["n"] > 0 and r["meta"]["overlay_cases"]
    assert (tmp_path / "xai_metrics_val.md").exists()
    assert all((tmp_path / "overlays" / f"{c}.png").exists() for c in r["meta"]["overlay_cases"])


def test_select_cases_without_detection():
    # segonly: cases_<split>.csv has no prob column and the threshold is None
    cases = [dict(id=f"B{i}", label=1, dsc_WT=str(d)) for i, d in enumerate([0.9, 0.1, 0.5])]
    cases += [dict(id="H1", label=0, dsc_WT=""), dict(id="H0", label=0, dsc_WT="")]
    got = explain.select_cases(cases, {"H0": 50.0, "H1": 80.0}, thr=None)
    assert got[:3] == ["B0", "B2", "B1"] and set(got[3:]) == {"H0", "H1"}
