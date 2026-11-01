import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate as ev  # noqa: E402
import synthetic as syn  # noqa: E402


def test_hd95_brats_conventions():
    z = torch.zeros(3, 8, 8, 8, dtype=torch.bool)
    one = z.clone()
    one[:, 2:5, 2:5, 2:5] = True
    shifted = z.clone()
    shifted[:, 3:6, 2:5, 2:5] = True
    assert ev.hd95(z, z).tolist() == [0, 0, 0]
    assert ev.hd95(one, z).tolist() == [ev.HD95_MAX] * 3 and ev.hd95(z, one).tolist() == [ev.HD95_MAX] * 3
    assert np.allclose(ev.hd95(one, one), 0) and np.allclose(ev.hd95(shifted, one), 2.0)  # 1 voxel = 2 mm


def test_choose_threshold_max_f1():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.55, 0.6, 0.7, 0.8, 0.9]
    t = ev.choose_threshold(y, p)
    assert t == pytest.approx(0.65) and ev.metrics.detection(y, p, t)["f1"] == 1.0  # midpoint of the gap
    assert ev.choose_threshold([0, 0, 1, 1], [0.1, 0.7, 0.6, 0.9]) == 0.5  # F1 tie: closest to 0.5
    # main-r5 val: separable, but 6-decimal rounding put the cut on the wrong side of both edge cases
    y, p = [0, 0, 1, 1], [1.7e-05, 0.0002614514378365129, 0.03331190347671509, 0.99999]
    t = ev.choose_threshold(y, p)
    assert p[1] < t < p[2] and ev.metrics.detection(y, p, t)["f1"] == 1.0


def test_groups_split_ixi_by_band_and_site():
    recs = [dict(id="a", label=0, site="HH", prob=0.1), dict(id="b", label=0, site="IOP", prob=0.9),
            dict(id="c", label=0, site="HH", prob=0.2), dict(id="d", label=1, site="unknown", prob=0.8)]
    g = ev.groups(recs, 0.5, {"a": 50.0, "b": 30.0, "c": None})
    assert g["ixi_in_band"]["n"] == 1 and g["ixi_below_band"]["fp"] == 1 and g["ixi_no_age"]["n"] == 1
    assert g["ixi_site_HH"]["specificity"] == 1.0 and g["brats_all"]["recall"] == 1.0


def test_mirror_tumour_uses_contralateral_tissue():
    rng = np.random.default_rng(3)
    x = np.zeros((4, 24, 24, 24), np.float32)
    x[:, 2:22, 2:22, 2:22] = rng.normal(0, 1, (4, 20, 20, 20))
    bits = np.zeros((24, 24, 24), np.uint8)
    bits[4:8, 8:12, 8:12] = 7  # tumour on one side of the left-right axis
    x[:, bits > 0] += 3.0
    xm, frac = ev.mirror_tumour(x, bits)
    wt = bits > 0
    assert frac == 1.0 and np.array_equal(xm[:, ~wt], x[:, ~wt])
    assert np.array_equal(xm[:, 4:8, 8:12, 8:12], x[:, 19:15:-1, 8:12, 8:12])  # i -> 2 + 21 - i
    # a central tumour mirrors onto itself: filled with the healthy-brain median instead
    bits2 = np.zeros_like(bits)
    bits2[10:14, 8:12, 8:12] = 7
    xm2, frac2 = ev.mirror_tumour(x, bits2)
    assert frac2 < 1 and np.isfinite(xm2).all()


def test_insert_tumour_matches_host_and_keeps_contrast():
    rng = np.random.default_rng(5)
    host = np.zeros((4, 24, 24, 24), np.float32)
    host[[0, 2], 3:21, 3:21, 3:21] = rng.normal(0, 1, (2, 18, 18, 18))  # IXI-like: T1ce/FLAIR absent
    donor = np.zeros((4, 24, 24, 24), np.float32)
    donor[:, 2:22, 2:22, 2:22] = rng.normal(0, 1, (4, 20, 20, 20))
    bits = np.zeros((24, 24, 24), np.uint8)
    bits[1:7, 8:12, 8:12] = 7  # WT partly outside the host brain (x < 3)
    donor[:, bits > 0] += 3.0  # tumour 3 SD above the donor's healthy brain
    xi, inside, ml = ev.insert_tumour(host, donor, bits)
    region = (bits > 0) & (host != 0).any(0)
    assert inside == pytest.approx(region.sum() / (bits > 0).sum()) and ml == pytest.approx(region.sum() * 8 / 1000)
    assert not xi[[1, 3]].any()  # absent channels stay absent
    hb = (host != 0).any(0)
    assert not (((xi != 0).any(0)) & ~hb).any()  # nothing pasted outside the host brain
    for c in (0, 2):
        v = xi[c][hb]
        assert abs(v.mean()) < 1e-4 and abs(v.std() - 1) < 1e-3  # re-z-scored over the host brain
        assert xi[c][region].mean() - xi[c][hb & ~region].mean() > 2.5  # contrast kept (~3 SD)


def test_sham_insertion_pastes_healthy_mirror_tissue_at_same_voxels():
    rng = np.random.default_rng(6)
    host = np.zeros((4, 24, 24, 24), np.float32)
    host[[0, 2], 3:21, 3:21, 3:21] = rng.normal(0, 1, (2, 18, 18, 18))
    donor = np.zeros((4, 24, 24, 24), np.float32)
    donor[:, 2:22, 2:22, 2:22] = rng.normal(0, 1, (4, 20, 20, 20))
    bits = np.zeros((24, 24, 24), np.uint8)
    bits[4:8, 8:12, 8:12] = 7
    donor[:, bits > 0] += 3.0
    xt, ft, mlt = ev.insert_tumour(host, donor, bits)
    xs, fs, mls = ev.insert_tumour(host, donor, bits, sham=True)
    region = (bits > 0) & (host != 0).any(0)
    changed = (xs != host).any(0) & ~region
    assert fs == ft == 1.0 and mls == mlt  # mirror of a one-sided tumour is all healthy brain
    assert not xs[[1, 3]].any()
    hb = (host != 0).any(0)
    for c in (0, 2):
        assert xt[c][region].mean() - xt[c][hb & ~region].mean() > 2.5  # tumour keeps its contrast
        assert abs(xs[c][region].mean() - xs[c][hb & ~region].mean()) < 0.5  # sham tissue looks healthy
        # pasted values are the donor's mirror (i -> 2 + 21 - i), affinely matched
        src = donor[c, 19:15:-1, 8:12, 8:12].ravel()
        assert np.corrcoef(src, xs[c, 4:8, 8:12, 8:12].ravel())[0, 1] > 0.999
    # a central tumour mirrors onto itself: nothing valid to paste
    bits2 = np.zeros_like(bits)
    bits2[10:14, 8:12, 8:12] = 7
    _, f2, _ = ev.insert_tumour(host, donor, bits2, sham=True)
    assert f2 < 0.6


def test_insertion_donors_fixed_and_cover_all_healthy():
    class Cohort(list):
        pass
    c = Cohort(range(6))
    c.label = np.array([1, 1, 0, 0, 0, 1])
    c.has_seg = c.label.astype(bool)
    full, empty = np.full((4, 4, 4), 7, np.uint8), np.zeros((4, 4, 4), np.uint8)
    c.y = [full, empty, empty, empty, empty, full]
    d = ev.insertion_donors(c)
    assert d == ev.insertion_donors(c) and set(d) == {2, 3, 4} and set(d.values()) <= {0, 5}  # 1 has no WT


def test_erode_brain_shrinks_and_rezscores():
    x, _, brain = syn.make_case((24, 24, 24), tumour=False, rng=np.random.default_rng(4))
    e = ev.erode_brain(x)
    b2 = (e != 0).any(0)
    assert b2.sum() < brain.sum() and not (b2 & ~brain).any()
    assert abs(e[0][b2].mean()) < 1e-4 and abs(e[0][b2].std() - 1) < 1e-3 and not e[1].any()


def test_evaluate_val_then_test_once(tmp_path):
    res = ev.main(["--debug", "--out", str(tmp_path), "--controls"])
    assert res["split"] == "val" and res["threshold_rule"] == "max_f1_on_val"
    assert {"tumour_mirrored_brats", "tumour_inserted_ixi", "eroded_ixi"} <= set(res["controls"])
    assert res["controls"]["tumour_inserted_ixi"]["n"] > 0 and res["controls"]["sham_inserted_ixi"]["n"] > 0
    assert (tmp_path / "cases_val.csv").exists() and (tmp_path / "figures" / "roc_val.png").exists()
    with pytest.raises(AssertionError, match="locked"):
        ev.main(["--debug", "--out", str(tmp_path), "--split", "test"])
    with pytest.raises(AssertionError, match="threshold"):
        ev.main(["--debug", "--out", str(tmp_path), "--split", "test", "--confirm-test"])
    assert not (tmp_path / "test_lock.json").exists()
    t = ev.main(["--debug", "--out", str(tmp_path), "--split", "test", "--confirm-test",
                 "--threshold-from", str(tmp_path / "results_val.json")])
    assert t["split"] == "test" and t["threshold"] == res["threshold"] and t["threshold_rule"] == "frozen_from_val"
    assert json.loads((tmp_path / "test_lock.json").read_text())["threshold"] == res["threshold"]
    with pytest.raises(AssertionError, match="second read"):
        ev.main(["--debug", "--out", str(tmp_path), "--split", "test", "--confirm-test",
                 "--threshold-from", str(tmp_path / "results_val.json")])


def test_wilson_exact_at_ends():
    # n=48 (val site Guys) gave an upper bound a hair below 1.0 and crashed the error-bar figure
    assert all(ev.wilson(n, n)[1] == 1.0 and ev.wilson(0, n)[0] == 0.0 for n in range(1, 300))
    lo, hi = ev.wilson(40, 48)
    assert 0 < lo < 40 / 48 < hi < 1
