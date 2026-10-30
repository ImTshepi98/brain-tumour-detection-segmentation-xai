import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import qc_domain  # noqa: E402
import synthetic as syn  # noqa: E402


def _write(tmp_path, n=12, shape=(32, 32, 32)):
    rng = np.random.default_rng(42)
    bd, idir = tmp_path / "b", tmp_path / "i"
    for d in (bd / "x", bd / "y", idir / "x"):
        d.mkdir(parents=True)
    rows = []
    c = [s / 2 for s in shape]
    for i in range(n):
        brain = syn._ellipsoid(shape, c, [s * rng.uniform(0.3, 0.38) for s in shape])
        x = np.zeros((4, *shape), np.float16)
        for ch in range(4):
            x[ch][brain] = rng.normal(0, 1, int(brain.sum()))
        y = np.zeros(shape, np.uint8)
        y[syn._ellipsoid(shape, [c[0] + 3, c[1], c[2]], [4, 4, 4]) & brain] = 7
        s = f"BraTS2021_{i:05d}"
        np.save(bd / "x" / f"{s}.npy", x)
        np.save(bd / "y" / f"{s}.npy", y)
        rows.append(dict(subject=s, source="brats", site="unknown", split="train" if i < 9 else "val"))
        # IXI: smoother + skewed intensities so the classifier has something to find
        xi = np.zeros((4, *shape), np.float16)
        for ch in (0, 2):
            xi[ch][brain] = rng.gamma(2, 1, int(brain.sum())) - 2
        s = f"IXI{i:03d}"
        np.save(idir / "x" / f"{s}.npy", xi)
        rows.append(dict(subject=s, source="ixi", site=["Guys", "HH"][i % 2], split="train" if i < 9 else "val"))
    rows.append(dict(subject="BraTS2021_09999", source="brats", site="unknown", split="test"))
    sp = tmp_path / "splits.csv"
    with open(sp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return bd, idir, sp


def test_qc_domain_end_to_end(tmp_path):
    bd, idir, sp = _write(tmp_path)
    # the test-split subject has an array present: it must still be ignored
    np.save(bd / "x" / "BraTS2021_09999.npy", np.ones((4, 32, 32, 32), np.float16))
    out = tmp_path / "out"
    res = qc_domain.main(["--brats-dir", str(bd), "--ixi-dirs", str(idir), "--splits", str(sp),
                          "--out", str(out), "--workers", "1", "--threshold", "0.4"])  # tiny n: HGB cannot split, AUC 0.5
    assert res["n_brats"] == 12 and res["n_ixi"] == 12
    assert res["iterations_run"] and 0 <= res["base"]["plan"]["auc_mean"] <= 1
    feats = list(csv.DictReader(open(out / "features_base.csv")))
    assert len(feats) == 24 and all(r["split"] != "test" for r in feats)
    assert json.loads((out / "cv_results.json").read_text())["base"]["plan+extended"]["n_features"] > 70


def test_histogram_match_maps_to_reference():
    rng = np.random.default_rng(0)
    img = rng.gamma(2, 1, (10, 10, 10)).astype(np.float32)
    mask = np.ones_like(img, bool)
    ref = np.quantile(rng.normal(0, 1, 5000), qc_domain.QS)
    m = qc_domain.match(img, mask, ref)
    assert abs(np.median(m) - np.median(ref)) < 0.05 and np.all(np.diff(m.ravel()[np.argsort(img.ravel())]) >= -1e-6)
