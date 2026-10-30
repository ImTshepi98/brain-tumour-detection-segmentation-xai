import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ants = pytest.importorskip("ants")
import prep_brats_2mm as prep_b  # noqa: E402
import prep_ixi_2mm as prep_i  # noqa: E402
import synthetic as syn  # noqa: E402


def _fake_head(shape, rng, scale):
    """Head = brain ellipsoid + bright skull shell; different voxel size/orientation from BraTS."""
    brain = syn._ellipsoid(shape, [s / 2 for s in shape], [s * 0.33 for s in shape])
    head = syn._ellipsoid(shape, [s / 2 for s in shape], [s * 0.45 for s in shape])
    a = np.zeros(shape, np.float32)
    a[head] = 300
    a[brain] = rng.normal(600 * scale, 40, int(brain.sum()))
    return a


def test_prep_ixi_end_to_end_synthetic(tmp_path):
    shape = (32, 32, 24)
    src = tmp_path / "brats"
    syn.write_fake_brats_tar(src, n_cases=6, shape=shape, n_trainval=4)
    rng = np.random.default_rng(0)
    ixi_root = tmp_path / "ixi"
    ixi_root.mkdir()
    subs = ["IXI002", "IXI012"]
    for s in subs:
        for mod, sc in (("T1", 1.0), ("T2", 0.7)):
            aff = np.diag([1.1, 1.1, 1.3, 1.0])
            # real layout: '<name>.nii' is a directory holding the NIfTI file (IXI012 uses it)
            dest = ixi_root / f"{s}-Guys-0828-{mod}.nii.gz"
            if s == "IXI012":
                d = ixi_root / f"{s}-Guys-0828-{mod}.nii"
                d.mkdir()
                dest = d / f"{s}-Guys-0828-SCAN-{mod}.nii"
            nib.save(nib.Nifti1Image(_fake_head((30, 30, 22), rng, sc), aff), dest)
    splits = tmp_path / "splits.csv"
    with open(splits, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "id", "source", "site", "split", "label", "has_seg"])
        w.writeheader()
        for i in range(6):
            s = f"BraTS2021_{i:05d}"
            w.writerow(dict(subject=s, id=s if i < 4 else "", source="brats", site="unknown",
                            split=["train", "train", "train", "val", "test", "test"][i], label=1, has_seg=1))
        for s, sp in zip(subs, ("train", "test")):
            w.writerow(dict(subject=s, id="", source="ixi", site="Guys", split=sp, label=0, has_seg=0))
    bout = tmp_path / "b2mm"
    prep_b.main(["--tar", str(src / "BraTS2021_Training_Data.tar"), "--splits", str(splits),
                 "--stats", str(src / "modality_stats.csv"), "--out", str(bout),
                 "--expected-shape", *map(str, shape), "--workers", "1"])
    out = tmp_path / "i2mm"
    summ = prep_i.main(["--ixi-root", str(ixi_root), "--splits", str(splits), "--crop", str(bout / "crop.json"),
                        "--brats-tar", str(src / "BraTS2021_Training_Data.tar"), "--skullstrip", "threshold",
                        "--out", str(out), "--workers", "1", "--template-cases", "3",
                        "--expected-shape", *map(str, shape)])
    assert summ["n_done"] == 2 and summ["n_failed_this_run"] == 0
    xb = np.load(bout / "x" / "BraTS2021_00000.npy")
    for s in subs:
        x = np.load(out / "x" / f"{s}.npy")
        assert x.shape == xb.shape and x.dtype == np.float16
        assert not x[1].any() and not x[3].any() and x[0].any() and x[2].any()
        rec = json.loads((out / "rows" / f"{s}.json").read_text())
        assert rec["mask_dice_template"] > 0.3
        sdt = np.load(out / "sdt" / f"{s}.npy").astype(np.float32)
        brain = (x != 0).any(0)
        assert sdt.shape == x.shape[1:] and np.median(sdt[brain]) < 0 and sdt[~brain].min() > -3
        assert rec["brain_ml_b-2"] < rec["brain_ml_b+1"]
    # resume: nothing left to do
    assert prep_i.main(["--ixi-root", str(ixi_root), "--splits", str(splits), "--crop", str(bout / "crop.json"),
                        "--brats-tar", str(src / "BraTS2021_Training_Data.tar"), "--skullstrip", "threshold",
                        "--out", str(out), "--workers", "1", "--template-cases", "3",
                        "--expected-shape", *map(str, shape)])["n_done"] == 2


def test_qc_flags():
    import qc_ixi
    recs = [dict(subject=f"IXI{i:03d}", warped_brain_ml=1400 + i, native_brain_ml=1300 + i, mask_dice_template=0.95,
                 t1_template_corr=0.9, affine_det=1.0, brain_vox_outside_box=0, brain_vox_1mm=1000)
            for i in range(20)]
    recs[3]["warped_brain_ml"] = 3000
    recs[5]["mask_dice_template"] = 0.6
    recs[7]["affine_det"] = 2.0
    r = qc_ixi.flag_rows(recs, {"IXI009"})
    flagged = {s for s, v in r.items() if v}
    assert flagged == {"IXI003", "IXI005", "IXI007", "IXI009"}
